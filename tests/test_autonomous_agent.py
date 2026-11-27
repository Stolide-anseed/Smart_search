import json
from collections import deque
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from xml.etree.ElementTree import Element, SubElement, tostring

from softline_search.autonomous_agent import DEFAULT_TOP_K, MAX_TOP_K, AutonomousRecommendationAgent
from softline_search.index import SearchHit, _catalog_families
from softline_search.llm_provider import LLMSettings, ProviderCall, ProviderFailure, ProviderResponse
from softline_search.query_catalog import QueryCatalog
from softline_search.recommendation import RecommendationAgent
from softline_search.recommendation_facts import Status, Validation


_PRINT_SUPPLIES = "Расходные материалы для печати"
_BAGS = "Переноска ноутбука"


def _call(identifier, name, **arguments):
    return ProviderResponse(calls=(ProviderCall(identifier, name, json.dumps(arguments, ensure_ascii=False)),))


def _search(identifier, query, category, direction):
    return _call(identifier, "search_catalog", query=query, category_id=category, direction=direction, limit=8)


def _selection(*pairs):
    return ProviderResponse(text=json.dumps({
        "selections": [{"candidate_id": identifier, "direction": direction} for identifier, direction in pairs],
    }, ensure_ascii=False))


class _Provider:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []

    def respond(self, **request):
        self.requests.append(request.copy())
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class AutonomousRecommendationAgentTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.catalog = Path(temporary.name) / "catalog.xml"
        feed = Element("yml_catalog")
        shop = SubElement(feed, "shop")
        categories = SubElement(shop, "categories")
        for identifier, name in (
            ("printers", "Принтеры"),
            ("print_supplies", "Картриджи"),
            ("paper", "Бумага для печати"),
            ("70824", "Ноутбуки"),
            ("83844", "Сумки для ноутбуков"),
        ):
            SubElement(categories, "category", id=identifier).text = name
        offers = SubElement(shop, "offers")
        entries = (
            ("printer", "printers", "Принтер OfficePrint", "printer", "20000", ()),
            ("toner", "print_supplies", "Картридж OfficePrint", "toner", "4000", ()),
            ("toner-unpriced", "print_supplies", "Картридж OfficePrint XL", "toner", "", ()),
            ("toner-prior", "print_supplies", "Картридж OfficePrint Plus", "toner", "4200", ()),
            ("paper", "paper", "Бумага А4", "paper", "700", ()),
            ("laptop", "70824", "Ноутбук Atlas", "laptop", "68000", (("Габариты корпуса", "350×235×17 мм"),)),
            ("bag", "83844", "Сумка для ноутбука Atlas", "bag", "2300", (("Внутренние размеры", "360×240×20 мм"),)),
            ("small-bag", "83844", "Маленький чехол для ноутбука", "small-bag", "2000",
             (("Внутренние размеры", "300×230×20 мм"),)),
            ("bag-without-specs", "83844", "Сумка для ноутбука City", "bag-city", "1300", ()),
        )
        for identifier, category, name, group, price, parameters in entries:
            element = SubElement(offers, "offer", id=identifier, group_id=group, available="true")
            for tag, value in (
                ("name", name), ("categoryId", category), ("vendor", "Example"), ("vendorCode", identifier),
                ("price", price), ("currencyId", "RUR"),
            ):
                SubElement(element, tag).text = value
            for label, value in parameters:
                SubElement(element, "param", name=label).text = value
        self.catalog.write_bytes(tostring(feed, encoding="utf-8"))
        self.rules = RecommendationAgent(self.catalog)
        by_id = {identifier: self.rules.get_offer(identifier) for identifier, *_ in entries}
        families = _catalog_families(tuple(by_id.values()))
        catalog = QueryCatalog(self.catalog, tuple(by_id.values()))

        def retrieve(query, *, category_ids, limit):
            return [SearchHit(offer, 1.0) for offer in by_id.values() if offer.category_id in category_ids][:limit]

        self.index = SimpleNamespace(
            prepare_query_catalog=Mock(return_value=catalog), search_candidates=Mock(side_effect=retrieve),
            family_key=lambda offer: families[offer.offer_id],
        )

    def agent(self, provider=None, **settings):
        agent = AutonomousRecommendationAgent(
            self.index, self.catalog, self.rules, replace(LLMSettings(), **settings), provider=provider,
        )
        self.addCleanup(agent.close)
        return agent

    def test_model_searches_two_self_selected_directions_and_grounds_items(self):
        provider = _Provider(
            _call("context", "get_product_details", offer_id="printer"),
            _search("toner", "картридж OfficePrint", "print_supplies", _PRINT_SUPPLIES),
            _search("paper", "бумага А4", "paper", "Бумага для печати"),
            _call("check", "check_compatibility_batch", offer_ids=["toner", "paper"]),
            _selection(("toner", _PRINT_SUPPLIES), ("paper", "Бумага для печати")),
        )
        result = self.agent(provider).recommend("printer", debug=True)
        self.assertEqual(result.mode, "autonomous_agent")
        context = json.loads(provider.requests[0]["conversation"][0]["content"])
        self.assertEqual(context["anchor"]["offer_id"], "printer")
        self.assertEqual(context["anchor"]["name"], "Принтер OfficePrint")
        self.assertEqual(context["top_k"], DEFAULT_TOP_K)
        self.assertNotIn("scenarios", context)
        self.assertEqual(self.index.search_candidates.call_count, 2)
        self.assertEqual([item.offer.offer_id for item in result.items], ["toner", "paper"])
        self.assertEqual([item.direction_label for item in result.items], [_PRINT_SUPPLIES, "Бумага для печати"])
        self.assertEqual(result.directions, tuple(item.direction_id for item in result.items))
        self.assertTrue(all(item.verification_status == Status.UNKNOWN and item.missing_criteria for item in result.items))
        self.assertEqual(result.diagnostics["live_validation"]["checked"], 2)
        self.assertEqual([event["tool"] for event in result.tools].count("search_catalog"), 2)
        self.assertIn("validate_autonomous_candidates", [event["tool"] for event in result.tools])
        batch = next(event["compatibility"] for event in result.tools if event["tool"] == "check_compatibility_batch")
        self.assertEqual(batch, {"checked": 2, "confirmed": 0, "rejected": 0, "unknown": 2, "errors": 0})

    def test_unretrieved_ids_and_posthoc_directions_are_dropped(self):
        provider = _Provider(
            _call("browse", "get_category_products", category_id="paper", limit=5),
            _search("toner", "картридж OfficePrint", "print_supplies", _PRINT_SUPPLIES),
            _selection(
                ("made-up", _PRINT_SUPPLIES), ("paper", _PRINT_SUPPLIES), ("printer", _PRINT_SUPPLIES),
                ("toner", "Неиспользованное направление"), ("toner-unpriced", _PRINT_SUPPLIES),
            ),
        )
        result = self.agent(provider).recommend("printer")
        self.assertEqual([item.offer.offer_id for item in result.items], ["toner-unpriced"])
        self.assertFalse(result.items[0].has_price)

    def test_model_choices_are_independently_checked_and_marked(self):
        provider = _Provider(
            _search("bags", "сумка для ноутбука Atlas", "83844", _BAGS),
            _selection(("small-bag", _BAGS), ("bag", _BAGS), ("bag-without-specs", _BAGS)),
        )
        result = self.agent(provider).recommend("laptop", debug=True)
        self.assertEqual(
            [(item.offer.offer_id, item.verification_status) for item in result.items],
            [("bag", Status.CONFIRMED), ("bag-without-specs", Status.UNKNOWN)],
        )
        self.assertEqual(result.diagnostics["live_validation"]["rejected"], 1)

    def test_commercial_order_and_family_dedup_apply_before_top_k(self):
        turns = (
            _search("toner", "картридж OfficePrint", "print_supplies", _PRINT_SUPPLIES),
            _search("paper", "бумага А4", "paper", "Бумага"),
            _selection(("toner-unpriced", _PRINT_SUPPLIES), ("toner", _PRINT_SUPPLIES), ("paper", "Бумага")),
        )
        for limit, expected in ((4, ["toner", "paper"]), (1, ["toner"])):
            with self.subTest(limit=limit):
                result = self.agent(_Provider(*turns)).recommend("printer", limit=limit)
                self.assertEqual([item.offer.offer_id for item in result.items], expected)

    def test_confirmed_variant_replaces_partial_variant_and_keeps_proven_quantity(self):
        provider = _Provider(
            _search("toner", "картридж OfficePrint", "print_supplies", _PRINT_SUPPLIES),
            _selection(("toner-prior", _PRINT_SUPPLIES), ("toner", _PRINT_SUPPLIES)),
        )
        results = (
            Validation(Status.UNKNOWN, "needs_check", missing=("compatibility",)),
            Validation(Status.CONFIRMED, "known_capacity", facts={"quantity": 3}),
        )
        with patch("softline_search.autonomous_agent.validate_cross_sell", side_effect=results):
            result = self.agent(provider).recommend("printer")
        item, = result.items
        self.assertEqual(item.offer.offer_id, "toner")
        self.assertEqual(item.purchase_quantity, 3)
        self.assertEqual(item.total_price, Decimal("12000"))

    def test_completed_empty_stays_empty(self):
        result = self.agent(_Provider(_selection())).recommend("laptop")
        self.assertEqual(result.mode, "autonomous_agent")
        self.assertEqual(result.items, [])
        self.assertEqual(result.execution_status, "completed_empty")

    def test_no_provider_or_failure_uses_local_fallback(self):
        provider = _Provider(ProviderFailure("rate_limited"))
        failed = self.agent(provider).recommend("laptop", limit=1)
        self.assertEqual(failed.mode, "rule_based_fallback")
        self.assertEqual(failed.fallback_reason, "rate_limited")
        self.assertEqual([item.offer.offer_id for item in failed.items], ["bag"])
        disabled = self.agent(_Provider(), enabled=False).recommend("laptop", limit=1)
        self.assertEqual(disabled.mode, "local")
        self.assertEqual([item.offer.offer_id for item in disabled.items], ["bag"])

    def test_public_top_k_range_is_enforced(self):
        agent = self.agent(_Provider())
        for value in (0, MAX_TOP_K + 1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                agent.recommend("printer", limit=value)
