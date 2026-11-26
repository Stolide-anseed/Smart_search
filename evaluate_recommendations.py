import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter

from softline_search.query import Condition, ParsedQuery
from softline_search.recommendation import RecommendationAgent


SCENARIOS = ("ups_battery", "laptop_bag", "office_antivirus")


def _query(platform: str = "", devices: int | None = None) -> ParsedQuery:
    text = "офисный пакет" + (f" под {platform}" if platform else "") + (f" на {devices} ПК" if devices else "")
    conditions = []
    if platform:
        conditions.append(Condition("platform", "eq", platform, platform))
    if devices is not None:
        conditions.append(Condition("devices", "ge", Decimal(devices), str(devices)))
    return ParsedQuery(text, text, text, tuple(conditions), ())


CONTEXTS = {
    "default": (None, ()),
    "windows": (_query("windows"), ()),
    "one_device": (_query(devices=1), ()),
    "windows_one": (_query("windows", 1), ()),
    "windows_ten": (_query("windows", 10), ()),
    "foreign_os": (_query("windows server", 1), ()),
    "conflict": (_query("windows", 1), (Condition("platform", "eq", "linux", "Linux"),)),
}


def _memory() -> dict:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                *[(name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile",
                )],
            ]

        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if read(process(), ctypes.byref(counters), counters.cb):
            return {"rss_bytes": counters.working, "peak_rss_bytes": counters.peak}
    elif Path("/proc/self/statm").is_file():
        from resource import RUSAGE_SELF, getrusage

        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return {"rss_bytes": pages * os.sysconf("SC_PAGE_SIZE"),
                "peak_rss_bytes": getrusage(RUSAGE_SELF).ru_maxrss * 1024}
    return {}


def _measure(agent: RecommendationAgent, anchor: str, context: str, scenario: str | None) -> dict:
    query, filters = CONTEXTS[context]
    started = perf_counter()
    result = agent.recommend(anchor, query, filters=filters)
    elapsed = (perf_counter() - started) * 1000
    diagnostic = agent.recommend(anchor, query, filters=filters, debug=True).diagnostics
    return {
        "anchor": anchor, "name": diagnostic["name"], "article": diagnostic["article"], "context": context,
        "expected_scenario": scenario, "scenario_id": result.scenario_id,
        "status": result.status, "reason_code": result.reason_code, "missing_context": result.missing_context,
        "latency_ms": elapsed,
        "candidates": diagnostic["candidates"], "confirmed": diagnostic["confirmed"],
        "unknown": diagnostic["unknown"], "rejected": diagnostic["rejected"],
        "candidate_reasons": diagnostic["reason_counts"],
        "routing": diagnostic["routing"], "timings_ms": diagnostic["timings_ms"],
        "missing_battery_articles": diagnostic["missing_battery_articles"],
        "items": [
            {"id": item.offer.offer_id, "article": item.offer.vendor_code, "name": item.offer.name,
             "available": item.available, "has_price": item.has_price, "reason": item.reason, "sources": item.sources}
            for item in result.items
        ],
    }


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"tested": 0}
    latencies = sorted(row["latency_ms"] for row in rows)
    failures = Counter(row["reason_code"] for row in rows if not row["items"] and "reason_code" in row)
    candidate_reasons = Counter()
    for row in rows:
        candidate_reasons.update(row.get("candidate_reasons", {}))
    covered = sum(bool(row["items"]) for row in rows)
    available = sum(any(item["available"] and item["has_price"] for item in row["items"]) for row in rows)
    return {
        "tested": len(rows), "covered": covered, "coverage": covered / len(rows),
        "available_covered": available, "available_coverage": available / len(rows),
        "mean_recommendations": mean(len(row["items"]) for row in rows),
        "p50_ms": latencies[ceil(0.50 * len(rows)) - 1], "p95_ms": latencies[ceil(0.95 * len(rows)) - 1],
        "max_ms": latencies[-1], "empty_reason_counts": dict(failures), "candidate_reason_counts": dict(candidate_reasons),
    }


def _audit(rows: dict, gold: list[dict]) -> dict:
    routed, tested, true_positives, false_positives, unjudged, known_pairs = 0, 0, 0, 0, 0, 0
    details = []
    for entry in gold:
        key = (entry["anchor"], entry.get("context", "default"))
        if key not in rows:
            details.append({"anchor": key[0], "context": key[1], "missing_run": True})
            continue
        row = rows[key]
        matches = entry["scenario_id"] == row["scenario_id"]
        expected, incompatible = set(entry.get("compatible", ())), set(entry.get("incompatible", ()))
        selected = {item["id"] for item in row["items"]}
        correct = selected & expected
        wrong = selected if entry["scenario_id"] is None or entry.get("expect_empty") else selected & incompatible
        other = selected - correct - wrong
        details.append({"anchor": key[0], "context": key[1], "scenario_correct": matches,
                        "confirmed": sorted(correct), "incompatible": sorted(wrong), "unjudged": sorted(other),
                        "missed_reference_pairs": sorted(expected - correct)})
        tested += 1
        routed += matches
        true_positives += len(correct)
        false_positives += len(wrong)
        unjudged += len(other)
        known_pairs += len(expected)
    judged = true_positives + false_positives
    return {
        "tested": tested, "scenario_matches": routed, "scenario_accuracy": routed / tested if tested else None,
        "true_positives": true_positives, "false_positives": false_positives, "unjudged": unjudged,
        "precision": true_positives / judged if judged else None,
        "known_reference_pairs": known_pairs, "reference_pair_recall": true_positives / known_pairs if known_pairs else None,
        "cases": details,
    }


def evaluate(catalog: Path, baseline_path: Path | None, gold: list[dict]) -> dict:
    started = perf_counter()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path else None
    size = catalog.stat().st_size
    if baseline and (baseline["catalog"], baseline["catalog_size"]) != (catalog.name, size):
        raise ValueError("XML-снимок для сравнения должен совпадать с baseline.")
    memory = {"before_catalog": _memory()}
    agent = RecommendationAgent(catalog)
    memory["prepared"] = _memory()
    rows = []
    groups = {}
    for scenario in SCENARIOS:
        for context in CONTEXTS if scenario == "office_antivirus" else ("default",):
            batch = [_measure(agent, anchor, context, scenario) for anchor in agent.anchor_ids(scenario)]
            groups[f"{scenario}/{context}"] = batch
            rows.extend(batch)
    by_key = {(row["anchor"], row["context"]): row for row in rows}
    audit_rows = []
    for entry in gold:
        key = (entry["anchor"], entry.get("context", "default"))
        if key not in by_key:
            try:
                by_key[key] = _measure(agent, *key, entry["scenario_id"])
            except KeyError:
                continue
            audit_rows.append(by_key[key])
    report = {
        "version": "v2", "date": datetime.now().astimezone().isoformat(timespec="seconds"), "catalog": catalog.name,
        "catalog_size": size, "build_ms": agent.index_stats["build_ms"], "index": agent.index_stats,
        "summary": {group: _summarize(batch) for group, batch in groups.items()}, "latency": _summarize(rows),
        "gold": _audit(by_key, gold), "rows": rows, "audit_rows": audit_rows,
    }
    if baseline:
        old_by_key = {(row["anchor"], row["context"]): row for row in baseline["rows"]}
        comparison = {}
        for group, batch in groups.items():
            old = [old_by_key[(row["anchor"], row["context"])] for row in batch
                   if (row["anchor"], row["context"]) in old_by_key]
            new = [row for row in batch if (row["anchor"], row["context"]) in old_by_key]
            old_stats, new_stats = _summarize(old), _summarize(new)
            delta = {key: new_stats[key] - old_stats[key] for key in (
                "coverage", "available_coverage", "p50_ms", "p95_ms", "max_ms", "mean_recommendations",
            ) if key in old_stats}
            comparison[group] = {"baseline": old_stats, "v2": new_stats, "delta": delta,
                                 "v2_without_baseline": len(batch) - len(new)}
        memory["baseline"] = baseline["memory"]
        report["comparison"] = comparison
        report["baseline_broad_summary"] = baseline["summary"]
        report["baseline_build_ms"] = baseline["build_ms"]
        report["baseline_gold"] = _audit(old_by_key, gold)
    memory["after_evaluation"] = _memory()
    report["memory"] = memory
    report["elapsed_seconds"] = perf_counter() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка допродаж на XML-каталоге по сопоставимым выборкам.")
    parser.add_argument("--catalog", type=Path, default=Path("Mindbox.yml.gz.xml"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/softline_recommendations.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/softline-v2/recommendations-v2.json"))
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold.is_file() else []
    report = evaluate(args.catalog, args.baseline, gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for group, summary in report["summary"].items():
        print(f"{group}  {summary.get('covered', 0)}/{summary['tested']}")
    print(args.output)


if __name__ == "__main__":
    main()
