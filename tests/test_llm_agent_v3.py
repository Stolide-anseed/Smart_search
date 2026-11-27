import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from xml.etree.ElementTree import Element, SubElement, tostring

from softline_search.demo import _recommendation_v3_products
from softline_search.index import SearchHit, _catalog_families
from softline_search.llm_agent_v3 import LLMRecommendationAgentV3
from softline_search.llm_provider import LLMSettings, ProviderCall, ProviderFailure, ProviderResponse
from softline_search.query_catalog import QueryCatalog
from softline_search.recommendation import RecommendationAgent


def search(query, relationship, *, category="", alternative="", call_id="1"):
    return ProviderResponse(calls=(ProviderCall(call_id, "search_catalog", json.dumps({
        "query": query, "alternative_query": alternative, "relationship": relationship,
        "category_id": category, "limit": 8,
    }, ensure_ascii=False)),))


def selection(*entries):
    return ProviderResponse(text=json.dumps({"selections": [
        {"candidate_id": identifier, "relationship": relationship, "reason": reason}
        for identifier, relationship, reason in entries
    ]}, ensure_ascii=False))


class V3AgentTests(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "catalog.xml"
        root = Element("yml_catalog")
        shop = SubElement(root, "shop")
        categories = SubElement(shop, "categories")
        for category, name in (
            ("70824", "Ноутбуки"), ("83844", "Сумки и чехлы для ноутбуков"),
            ("drawers", "Денежные ящики"), ("receipt", "Чековые принтеры"),
            ("ram", "Оперативная память"), ("boards", "Материнские платы"),
            ("printer", "Принтеры"), ("toner", "Картриджи"), ("paper", "Бумага"),
            ("laminators", "Ламинаторы"), ("film", "Плёнка для ламинирования"),
        ):
            SubElement(categories, "category", id=category).text = name
        offers = SubElement(shop, "offers")
        entries = (
            ("laptop", "70824", "Ноутбук Atlas", "laptop"),
            ("other_laptop", "70824", "Ноутбук City", "other_laptop"),
            ("bag", "83844", "Сумка Atlas для ноутбука", "bag"),
            ("bag_variant", "83844", "Сумка Atlas для ноутбука синяя", "bag"),
            ("drawer", "drawers", "Денежный ящик CD-300", "drawer"),
            ("receipt", "receipt", "Чековый принтер M-58", "receipt"),
            ("ram", "ram", "Оперативная память DDR4 16 ГБ", "ram"),
            ("board4", "boards", "Материнская плата A-400 DDR4", "board4"),
            ("board5", "boards", "Материнская плата B-500 DDR5", "board5"),
            ("printer", "printer", "Принтер OfficePrint", "printer"),
            ("toner", "toner", "Картридж OfficePrint для лазерной печати", "toner"),
            ("paper", "paper", "Бумага А4", "paper"),
            ("laminator", "laminators", "Ламинатор Lunar A4", "laminator"),
            ("film", "film", "Плёнка для ламинирования A4", "film"),
        )
        for identifier, category, name, family in entries:
            element = SubElement(offers, "offer", id=identifier, group_id=family, available="true")
            for tag, value in (("name", name), ("categoryId", category), ("vendor", "Example"),
                               ("vendorCode", identifier), ("price", "1000"), ("currencyId", "RUR")):
                SubElement(element, tag).text = value
        self.path.write_bytes(tostring(root, encoding="utf-8"))
        self.rules = RecommendationAgent(self.path)
        self.offers = tuple(self.rules.get_offer(identifier) for identifier, *_ in entries)
        catalog = QueryCatalog(self.path, self.offers)
        families = _catalog_families(self.offers)

        def retrieve(query, *, category_ids, limit):
            offers = [offer for offer in self.offers if not category_ids or offer.category_id in category_ids]
            return [SearchHit(offer, 1.0) for offer in offers[:limit]]

        self.index = SimpleNamespace(
            prepare_query_catalog=Mock(return_value=catalog), family_key=lambda offer: families[offer.offer_id],
            search_candidates=Mock(side_effect=retrieve),
        )

    def agent(self, *turns, **settings):
        responses = deque(turns)

        def respond(**_):
            result = responses.popleft()
            if isinstance(result, Exception):
                raise result
            return result

        provider = SimpleNamespace(respond=Mock(side_effect=respond))
        agent = LLMRecommendationAgentV3(self.index, self.path, self.rules, replace(LLMSettings(), **settings),
                                        provider=provider)
        self.addCleanup(agent.close)
        return agent

    def test_open_relationships_for_five_product_categories(self):
        cases = (
            ("laptop", "сумка", "83844", "Защита при переноске", "bag"),
            ("drawer", "чековый принтер", "receipt", "Кассовый узел", "receipt"),
            ("ram", "материнская плата DDR4", "boards", "Платформа сборки", "board4"),
            ("printer", "картриджи", "toner", "Расходник для печати", "toner"),
            ("laminator", "пленка для ламинирования", "film", "Плёнка для документов", "film"),
        )
        with (patch.object(self.rules, "applicable_scenarios", side_effect=AssertionError("legacy routing")),
              patch.object(self.rules, "applicable_directions", side_effect=AssertionError("legacy routing"))):
            for anchor, query, category, relation, candidate in cases:
                with self.subTest(anchor=anchor):
                    agent = self.agent(search(query, relation, category=category),
                                       selection((candidate, relation, "Дополняет назначение товара.")))
                    result = agent.recommend(anchor, debug=True)
                    self.assertEqual([item.candidate_id for item in result.items], [candidate])
                    self.assertEqual(result.items[0].relationship, relation)
                    self.assertEqual(result.items[0].verification_status, "UNKNOWN")
                    request = agent.provider.respond.call_args_list[0].kwargs
                    for payload in (request["instructions"], json.dumps(request["tools"]), json.dumps(request["final_schema"])):
                        self.assertNotIn("scenario_id", payload)
                    if anchor == "ram":
                        self.assertEqual(result.diagnostics["rejections"]["conflicting_memory_type"], 1)

    def test_retry_runs_hybrid_with_an_alternate_query_after_no_relevant_hit(self):
        agent = self.agent(
            search("несуществующий расходник", "Печать", category="toner", alternative="лазерный картридж"),
            selection(("toner", "Печать", "Расходник позволяет продолжать печать.")),
        )
        self.index.search_candidates.side_effect = [
            [SearchHit(self.rules.get_offer("toner"), 1)],
            [SearchHit(self.rules.get_offer("toner"), 1)],
        ]
        result = agent.recommend("printer", debug=True)
        self.assertEqual([call.args[0] for call in self.index.search_candidates.call_args_list],
                         ["несуществующий расходник", "лазерный картридж"])
        self.assertEqual(result.usage["search_retries"], 1)
        self.assertEqual([item.candidate_id for item in result.items], ["toner"])
        self.assertIn("catalog_search_retry", [e["tool"] for e in result.tools])

    def test_final_response_excludes_substitutes_ungrounded_ids_and_duplicate_families(self):
        agent = self.agent(
            search("сумка", "Переноска", category="83844"),
            selection(("other_laptop", "Переноска", "Аналог"),
                      ("unknown", "Переноска", "Вымышленный товар"),
                      ("bag", "Другое назначение", "Не найден по этому назначению"),
                      ("bag", "Переноска", "Совместим с любым ноутбуком"),
                      ("bag_variant", "Переноска", "Защита при переноске")),
        )
        result = agent.recommend("laptop", debug=True)
        self.assertEqual([item.candidate_id for item in result.items], ["bag"])
        self.assertNotIn("любым ноутбуком", result.items[0].reason)
        self.assertEqual(result.diagnostics["rejections"], {
            "candidate_not_retrieved_for_relationship": 2, "duplicate_family": 2,
        })
        response = _recommendation_v3_products(self.rules.get_offer("laptop"), result)
        self.assertNotIn("scenario_id", response)
        self.assertEqual(response["products"][0]["candidate_id"], "bag")
        self.assertEqual(response["products"][0]["relationship"], "Переноска")

    def test_empty_reasons_distinguish_catalog_coverage_and_model_selection(self):
        self.index.search_candidates.side_effect = None
        self.index.search_candidates.return_value = []
        agent = self.agent(search("лак", "Обслуживание", alternative="чистящий набор"), selection())
        empty = agent.recommend("printer", debug=True)
        self.assertEqual(empty.reason_code, "no_catalog_matches")
        self.assertEqual(empty.usage["catalog_searches"], 2)
        self.index.search_candidates.return_value = [SearchHit(self.rules.get_offer("toner"), 1)]
        agent = self.agent(search("картридж", "Печать"), selection())
        result = agent.recommend("printer")
        self.assertEqual(result.reason_code, "model_selected_none")

    def test_provider_error_deadline_and_call_limit_are_preserved(self):
        failure = self.agent(ProviderFailure("rate_limited")).recommend("printer")
        self.assertEqual((failure.reason_code, failure.execution_status), ("rate_limited", "provider_error"))
        self.assertFalse(failure.items)
        limited = self.agent(search("картридж", "Печать"), search("бумага", "Печать", call_id="2"),
                             max_tool_calls=1).recommend("printer")
        self.assertEqual(limited.reason_code, "tool_call_limit")
        self.assertEqual(limited.usage["tool_calls"], 1)
        release = Event()
        agent = self.agent(timeout_seconds=0.05)
        agent.provider.respond.side_effect = lambda **_: (release.wait(1), selection())[1]
        try:
            self.assertEqual(agent.recommend("printer").reason_code, "agent_timeout")
        finally:
            release.set()

    def test_top_k_and_obvious_irrelevance(self):
        agent = self.agent(search("тонер", "Запас", category="toner", alternative="картридж"), selection())
        result = agent.recommend("laptop", debug=True)
        self.assertEqual(result.reason_code, "no_relevant_candidates")
        self.assertGreaterEqual(result.diagnostics["rejections"]["different_target_product"], 1)
        agent = self.agent(search("картридж", "Печать", category="toner"),
                           search("бумага", "Печать", category="paper", call_id="2"),
                           selection(("toner", "Печать", "Расходник"), ("paper", "Печать", "Для документов")))
        self.assertEqual([item.candidate_id for item in agent.recommend("printer", limit=1).items], ["toner"])
