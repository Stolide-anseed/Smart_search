"""Compare the post-review recommendation pipeline on the same catalog cases."""

import argparse
import json
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate_post_p0 import _agent_metrics, _case_key, _cost, _measure, _read, _summary
from softline_search.index import SearchIndex
from softline_search.llm_agent import LLMRecommendationAgent
from softline_search.llm_provider import LLMSettings
from softline_search.recommendation import RecommendationAgent
from softline_search.vendor_evidence import VendorEvidenceService


def _report(rows: list[dict], gold: list[dict], rates: dict | None) -> dict:
    usage = _agent_metrics(rows)
    cost = _cost(rows, rates) if usage["usage"].get("llm_calls") else {"average_per_click": 0}
    office = next(row for row in rows if _case_key(row) == ("473582", "windows_one"))
    return {
        **_summary(rows, gold), "agent": usage, "cost": cost, "rows": rows,
        "commercial_check": {
            "office_windows_one_offers": [item["id"] for item in office["items"]],
            "compact_confirmed_first": bool(office["items"] and office["items"][0]["id"] == "398852"),
            "unverified_or_dominated_offers": [item["id"] for item in office["items"]
                                              if item["id"] in {"518959", "399141"}],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("Mindbox.yml.gz.xml"))
    parser.add_argument("--index", type=Path, default=Path(".search-index.sqlite3"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/softline-fix/baseline-review.json"))
    parser.add_argument("--rates", type=Path, default=Path("outputs/softline-p1/proxyapi-rates.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true", help="Выполнить C/D с текущей моделью и платным API.")
    parser.add_argument("--timeout", type=float, default=10, help="Верхняя граница сетевого подбора в секундах.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    baseline = _read(args.baseline)
    cases = baseline["cases"]
    keys = {_case_key(case) for case in cases}
    fixtures = Path(__file__).parent / "tests" / "fixtures"
    gold = {
        _case_key(case): case for file in ("softline_recommendations.json", "softline_post_p0.json",
                                         "softline_review_fixes.json")
        for case in _read(fixtures / file) if _case_key(case) in keys
    }
    if keys != set(gold):
        parser.error("A baseline case is missing an independent reference label.")
    references = list(gold.values())
    settings = LLMSettings.from_env()
    settings = replace(settings, timeout_seconds=min(args.timeout, settings.timeout_seconds))
    rates = _read(args.rates).get(settings.model.removeprefix("openai/")) if args.rates.exists() else None
    rules = RecommendationAgent(args.catalog)
    local = _measure(rules.recommend, cases)
    report = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "catalog": args.catalog.name, "catalog_size": args.catalog.stat().st_size,
        "index": rules.index_stats, "model": settings.model, "online_budget_seconds": settings.timeout_seconds,
        "A_local": _report(baseline["local"], references, rates),
        "A_live": _report(baseline["live"], references, rates) if baseline.get("live") else None,
        "B_local": _report(local, references, rates),
        "reference_cases": references, "live_status": "not_requested",
    }
    comparable = len(cases) == len(baseline["local"]) and baseline["catalog_size"] == report["catalog_size"]
    if not comparable:
        parser.error("The baseline does not match the current catalog or the number of cases.")
    if args.live and settings.enabled and settings.api_key:
        with closing(SearchIndex(args.index)) as index, TemporaryDirectory(prefix="softline-fix-bench-") as directory:
            index.prepare_query_catalog(args.catalog)
            index.search_candidates("ноутбук", limit=1)
            evidence = VendorEvidenceService(Path(directory) / "facts.sqlite3")
            with closing(LLMRecommendationAgent(index, args.catalog, rules, settings, evidence)) as agent:
                for name in ("C_cold", "D_warm"):
                    print(f"{name} · {len(cases)} товаров", flush=True)
                    rows = _measure(agent.recommend, cases)
                    report[name] = _report(rows, references, rates)
                    for row in rows:
                        print(row["anchor"], row["mode"], len(row["items"]), row["execution_status"], flush=True)
            report["verified_online_facts"] = [
                {**fact.public_data(), "evidence_text": fact.evidence_text} for fact in evidence.active_facts()
            ]
        report["live_status"] = "completed"
    elif args.live:
        report["live_status"] = "llm_disabled" if not settings.enabled else "missing_api_key"
    output = args.output or Path("outputs/softline-fix") / (
        "recommendations.json" if args.live else "recommendations-local.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
