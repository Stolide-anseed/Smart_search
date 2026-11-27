"""Measure two cross-sell agents on the same available offers."""

import argparse
import json
from collections import Counter
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from softline_search.autonomous_agent import AutonomousRecommendationAgent
from softline_search.index import SearchIndex
from softline_search.llm_agent_v3 import LLMRecommendationAgentV3
from softline_search.llm_agent_v3_2 import LLMRecommendationAgentV32
from softline_search.llm_provider import LLMSettings
from softline_search.recommendation import RecommendationAgent
from softline_search.vendor_evidence import VendorEvidenceService


CASES = ("493495", "505902", "370916", "534617", "526951")
AGENTS = {"v3.1": LLMRecommendationAgentV3, "v3.2": LLMRecommendationAgentV32}


def _row(anchor, result, catalog, repeat: int = 1) -> dict:
    return {
        "repeat": repeat,
        "id": anchor.offer_id,
        "name": anchor.name,
        "category": catalog.categories.get(anchor.category_id, anchor.category_id),
        "mode": result.mode,
        "reason_code": result.reason_code,
        "execution_status": result.execution_status,
        "failure": result.fallback_reason,
        "elapsed_ms": result.elapsed_ms,
        "usage": result.usage,
        "llm_calls": result.usage.get("llm_calls", 0),
        "tool_calls": result.usage.get("tool_calls", 0),
        "catalog_searches": result.usage.get("catalog_searches"),
        "search_retries": result.usage.get("search_retries", 0),
        "diagnostics": result.diagnostics or {},
        "searches": (result.diagnostics or {}).get("searches", []),
        "tools": result.tools,
        "items": [
            {
                "id": item.offer.offer_id,
                "name": item.offer.name,
                "article": item.offer.vendor_code,
                "category": catalog.categories.get(item.offer.category_id, item.offer.category_id),
                "relationship": getattr(item, "relationship", "") or getattr(item, "direction_label", ""),
                "reason": item.reason,
                "verification_status": getattr(item, "verification_status", "CONFIRMED"),
                "available": item.available,
                "has_price": item.has_price,
            }
            for item in result.items
        ],
    }


def live_run(args, settings: LLMSettings) -> dict:
    versions = AGENTS if args.version is None else {args.version: AGENTS[args.version]}
    rows = {version: [] for version in versions}
    with ExitStack() as stack:
        index = stack.enter_context(closing(SearchIndex(args.index)))
        catalog = index.prepare_query_catalog(args.catalog)
        index.search_candidates("ноутбук", limit=1)
        rules = RecommendationAgent(args.catalog)
        cache = Path(stack.enter_context(TemporaryDirectory()))
        agents = {
            name: stack.enter_context(closing(agent(index, args.catalog, rules, settings,
                                                    VendorEvidenceService(cache / f"{name}.sqlite3"))))
            for name, agent in versions.items()
        }
        available = {offer.offer_id for offer in index.offers}
        offers_to_run = (args.offer or CASES) * args.repeat
        for position, identifier in enumerate(offers_to_run):
            if identifier not in available:
                raise ValueError(f"Предложение {identifier} отсутствует в поисковом индексе.")
            order = list(agents) if position % 2 == 0 else list(reversed(agents))
            for version in order:
                result = agents[version].recommend(identifier, limit=4, debug=True)
                row = _row(rules.get_offer(identifier), result, catalog, position // len(args.offer or CASES) + 1)
                rows[version].append(row)
                print(version, identifier, len(row["items"]), row["failure"] or row["reason_code"],
                      round(row["elapsed_ms"]), flush=True)
    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.model,
        "limits": {"timeout_seconds": settings.timeout_seconds, "iterations": settings.max_iterations,
                   "tool_calls": settings.max_tool_calls},
        "catalog": args.catalog.name,
        "rows": rows,
    }


def summarize(rows: list[dict], judgments: dict) -> dict:
    latencies = sorted(row["elapsed_ms"] for row in rows)
    relevant = judged = count = 0
    for row in rows:
        for item in row["items"]:
            count += 1
            label = judgments.get(f'{row["id"]}/{item["id"]}', {}).get("relevant")
            if type(label) is bool:
                judged += 1
                relevant += int(label)
    candidate_sets = {}
    for row in rows:
        candidate_sets.setdefault(row["id"], []).append(tuple(sorted(item["id"] for item in row["items"])))
    stability = [max(Counter(sets).values()) / len(sets) for sets in candidate_sets.values() if len(sets) > 1]
    return {
        "cases": len(rows),
        "covered": sum(bool(row["items"]) for row in rows),
        "items": count,
        "judged": judged,
        "relevant": relevant,
        "relevance_precision": relevant / judged if judged else None,
        "p50_ms": round(median(latencies), 1),
        "p95_ms": round(latencies[ceil(len(latencies) * .95) - 1], 1),
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in rows),
        "mean_distinct_categories": round(sum(len({item["category"] for item in row["items"]}) for row in rows)
                                           / max(1, len(rows)), 2),
        "mean_category_distinctness": round(sum(
            len({item["category"] for item in row["items"]}) / len(row["items"]) if row["items"] else 0
            for row in rows) / max(1, len(rows)), 3),
        "repeat_candidate_set_stability": round(sum(stability) / len(stability), 3) if stability else None,
        "search_retries": sum(row["search_retries"] for row in rows),
        "failures": dict(Counter(row["failure"] for row in rows if row["failure"])),
        "empty_reasons": dict(Counter(row["failure"] or row["reason_code"] for row in rows if not row["items"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Запустить обе версии с настроенным платным API.")
    parser.add_argument("--catalog", type=Path, default=Path("Mindbox.yml.gz.xml"))
    parser.add_argument("--index", type=Path, default=Path(".search-index.sqlite3"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--offer", action="append", help="ID тестового товара. Параметр можно повторять.")
    parser.add_argument("--repeat", type=int, default=1, help="Число прогонов каждого товара.")
    parser.add_argument("--version", choices=tuple(AGENTS), help="Повторно измерить одну версию из отчёта.")
    parser.add_argument("--judgments", type=Path, help="JSON независимой ручной разметки пар.")
    parser.add_argument("--output", type=Path, default=Path("outputs/softline-v3/comparison-v32.json"))
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        parser.error("--repeat должен быть от 1 до 10.")
    if args.live:
        settings = LLMSettings.from_env(args.env)
        if not settings.enabled or not settings.api_key:
            parser.error("Для --live нужен включённый агент и настроенный ключ API.")
        previous = json.loads(args.output.read_text(encoding="utf-8")) if args.version and args.output.is_file() else None
        if previous:
            expected = (args.offer or list(CASES)) * args.repeat
            if (previous["model"] != settings.model or previous["catalog"] != args.catalog.name
                    or previous["limits"] != {"timeout_seconds": settings.timeout_seconds,
                                               "iterations": settings.max_iterations,
                                               "tool_calls": settings.max_tool_calls}
                    or any([row["id"] for row in rows] != expected for rows in previous["rows"].values())):
                parser.error("Версия модели, лимиты и тестовые товары должны совпадать с сохранённым прогоном.")
        report = live_run(args, settings)
        if previous:
            report["rows"] = {**previous["rows"], **report["rows"]}
            report["baseline_run_at"] = previous.get("baseline_run_at", previous["run_at"])
    else:
        report = json.loads(args.output.read_text(encoding="utf-8"))
    judgments = (
        json.loads(args.judgments.read_text(encoding="utf-8")) if args.judgments else report.get("judgments", {})
    )
    report["judgments"] = judgments
    report["summary"] = {version: summarize(rows, judgments) for version, rows in report["rows"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
