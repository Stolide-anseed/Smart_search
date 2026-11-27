import argparse
import json
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from evaluate_recommendations import CONTEXTS, SCENARIOS, _audit, _summarize
from softline_search.index import SearchIndex
from softline_search.llm_agent import LLMRecommendationAgent
from softline_search.llm_provider import LLMSettings
from softline_search.recommendation import RecommendationAgent
from softline_search.vendor_evidence import VendorEvidenceService


LIVE_CASES = (("501535", "default"), ("460304", "default"), ("82106", "default"), ("473582", "default"))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _case_key(case: dict) -> tuple[str, str]:
    return case["anchor"], case.get("context", "default")


def _measure(recommend, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        anchor, context = _case_key(case)
        query, filters = CONTEXTS[context]
        started = perf_counter()
        result = recommend(anchor, query, filters=filters)
        rows.append({
            "anchor": anchor, "context": context, "expected_scenario": case["scenario_id"],
            "scenario_id": result.scenario_id, "status": result.status, "reason_code": result.reason_code,
            "latency_ms": (perf_counter() - started) * 1000,
            "mode": getattr(result, "mode", None),
            "fallback_reason": getattr(result, "fallback_reason", None),
            "execution_status": getattr(result, "execution_status", None),
            "outcome_code": getattr(result, "outcome_code", None),
            "origins": getattr(result, "origins", ()),
            "usage": getattr(result, "usage", {}),
            "tools": getattr(result, "tools", ()),
            "items": [
                {"id": item.offer.offer_id, "article": item.offer.vendor_code, "available": item.available,
                 "has_price": item.has_price, "reason": item.reason, "sources": item.sources,
                 "total_price": str(item.total_price) if item.total_price is not None else None}
                for item in result.items
            ],
        })
    return rows


def _summary(rows: list[dict], gold: list[dict]) -> dict:
    groups = {
        f"{scenario}/{context}": _summarize([
            row for row in rows if (row["expected_scenario"], row["context"]) == (scenario, context)
        ])
        for scenario, context in dict.fromkeys((row["expected_scenario"], row["context"]) for row in rows)
    }
    return {"by_scenario": groups, "overall": _summarize(rows), "quality": _audit({
        _case_key(row): row for row in rows
    }, gold)}


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _agent_metrics(rows: list[dict]) -> dict:
    usage = Counter()
    calls = Counter()
    failures = Counter()
    for row in rows:
        usage.update(row["usage"])
        calls[row["mode"]] += 1
        if row["fallback_reason"]:
            failures[row["fallback_reason"]] += 1
    return {
        "modes": dict(calls), "fallbacks": dict(failures), "usage": dict(usage),
        "valid_arguments_rate": _share(usage["valid_tool_calls"], usage["tool_calls"]),
        "tool_success_rate": _share(usage["successful_tool_calls"], usage["tool_calls"]),
        "external_enrichment_rate": _share(usage["verified_facts"], usage["external_lookups"]),
        "fact_cache_hit_rate": _share(
            usage["cache_hits"] - usage["result_cache_hits"],
            usage["cache_lookups"] - usage["result_cache_lookups"],
        ),
        "result_cache_hit_rate": _share(calls["cached_result"], len(rows)),
        "fallback_rate": _share(calls["rule_based_fallback"], len(rows)),
        "timeout_rate": _share(failures["agent_timeout"] + failures["provider_timeout"], len(rows)),
        "mean_tool_calls": _share(usage["tool_calls"], len(rows)),
        "mean_iterations": _share(usage["iterations"], len(rows)),
    }


def _cost(rows: list[dict], rates: dict | None) -> dict | None:
    if not rates:
        return None
    usage = _agent_metrics(rows)["usage"]
    calls = usage.get("llm_calls", 0)
    if not calls or usage.get("llm_calls_with_usage", 0) != calls:
        return None
    cached = usage.get("cached_input_tokens", 0)
    uncached = usage.get("input_tokens", 0) - cached
    total = (
        uncached * Decimal(str(rates["input_per_million"]))
        + cached * Decimal(str(rates["cached_input_per_million"]))
        + usage.get("output_tokens", 0) * Decimal(str(rates["output_per_million"]))
    ) / 1_000_000
    searches = usage.get("web_searches", 0)
    if searches and rates.get("web_search_per_call") is None:
        return None
    total += searches * Decimal(str(rates.get("web_search_per_call") or 0))
    per_click = total / len(rows)
    return {
        "currency": rates["currency"], "source": rates["source"],
        "at_published_rates": float(total), "average_per_click": float(per_click),
        "month_at_500_searches_per_day": {
            f"{percent}%": float(per_click * 500 * 30 * percent / 100) for percent in (10, 25, 50)
        },
    }


def _live(index, catalog, rules, settings, models, gold, rates) -> dict:
    by_key = {_case_key(case): case for case in gold}
    cases = [by_key[key] for key in LIVE_CASES]
    results = {}
    for number, model in enumerate(models, 1):
        print(f"Модель {number} из {len(models)}. Холодный прогон.", flush=True)
        with TemporaryDirectory(prefix="softline-agent-comparison-") as directory:
            evidence = VendorEvidenceService(Path(directory) / "facts.sqlite3")
            with closing(LLMRecommendationAgent(index, catalog, rules, replace(settings, model=model), evidence)) as agent:
                cold = _measure(agent.recommend, cases)
                print(f"Модель {number} из {len(models)}. Повторный прогон.", flush=True)
                warm = _measure(agent.recommend, cases)
            results[model] = {
                name: {**_summary(rows, cases), "agent": _agent_metrics(rows), "rows": rows}
                for name, rows in (("cold", cold), ("repeat", warm))
            }
            results[model]["cost_cold"] = _cost(cold, rates.get(model))
            results[model]["cost_cold_and_repeat"] = _cost(cold + warm, rates.get(model))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнить локальный подбор, кеш характеристик и инструментального агента.")
    parser.add_argument("--catalog", type=Path, default=Path("Mindbox.yml.gz.xml"))
    parser.add_argument("--index", type=Path, default=Path(".search-index.sqlite3"))
    parser.add_argument("--cache", type=Path, default=Path(".vendor-facts.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/softline-p1/recommendations.json"))
    parser.add_argument("--live", action="store_true", help="Выполнить платные запросы для сравнения моделей.")
    parser.add_argument("--model", action="append", help="Модель для платного прогона, можно указать несколько раз.")
    parser.add_argument("--rates", type=Path, help="JSON с тарифами API по моделям.")
    args = parser.parse_args()
    fixtures = Path(__file__).parent / "tests" / "fixtures"
    gold = _read(fixtures / "softline_recommendations.json") + _read(fixtures / "softline_post_p0.json")
    rules = RecommendationAgent(args.catalog)
    evidence = VendorEvidenceService(args.cache)
    catalog_cases = [
        {"anchor": anchor, "context": context, "scenario_id": scenario}
        for scenario in SCENARIOS for anchor in rules.anchor_ids(scenario)
        for context in (CONTEXTS if scenario == "office_antivirus" else ("default",))
    ]
    cases = list({_case_key(case): case for case in catalog_cases + gold}.values())
    local = _measure(rules.recommend, cases)
    enriched = _measure(lambda anchor, query, **kw: rules.recommend_with_evidence(anchor, evidence, query, **kw), cases)
    report = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "catalog": args.catalog.name, "catalog_size": args.catalog.stat().st_size,
        "catalog_case_count": len(catalog_cases), "gold_case_count": len(gold),
        "rule_based": {**_summary(local, gold), "rows": local},
        "enriched_rule_based": {**_summary(enriched, gold), "rows": enriched},
        "local_external_cost_per_click": 0,
        "live_status": "not_requested",
    }
    if args.live:
        settings = LLMSettings.from_env()
        if not settings.enabled or not settings.api_key:
            report["live_status"] = "llm_disabled" if not settings.enabled else "missing_api_key"
        else:
            rates = _read(args.rates) if args.rates else {}
            models = args.model or list(dict.fromkeys((settings.model, "gpt-5.4-mini")))
            with closing(SearchIndex(args.index)) as index:
                index.prepare_query_catalog(args.catalog)
                index.search_candidates("ноутбук", limit=1)
                report["live"] = _live(index, args.catalog, rules, settings, models, gold, rates)
            report["live_status"] = "completed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
