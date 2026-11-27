import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
from openai import OpenAI

from softline_search.llm_agent import LLMRecommendationAgent
from softline_search.llm_provider import (
    LLMSettings, OpenAIProvider, ProviderCall, ProviderFailure, ProviderResponse, ProviderUsage, WebResponse,
)
from softline_search.query import Condition, ParsedQuery
from softline_search.recommendation import RecommendationAgent
from softline_search.vendor_evidence import EvidenceResult, VendorDocument, VendorEvidenceService


class LiveEvaluationCasesTests(TestCase):
    def test_each_live_case_has_reference_labels(self):
        from evaluate_post_p0 import LIVE_CASES

        fixtures = Path(__file__).parent / "fixtures"
        paths = (fixtures / "softline_recommendations.json", fixtures / "softline_post_p0.json")
        keys = {
            (case["anchor"], case.get("context", "default"))
            for path in paths for case in json.loads(path.read_text(encoding="utf-8"))
        }
        self.assertEqual(set(LIVE_CASES) - keys, set())


def _call(call_id, name, **arguments):
    if name == "search_vendor_specs":
        arguments.setdefault("strategy", "exact_article")
    return ProviderCall(call_id, name, json.dumps(arguments))


def _selection(*ids):
    return ProviderResponse(text=json.dumps({
        "selections": [{"scenario_id": "laptop_bag", "candidate_id": identifier} for identifier in ids],
    }), usage=ProviderUsage(200, 10, 20, reported=True))


def _browse_candidates(call_id="browse"):
    return ProviderResponse(calls=(_call(
        call_id, "get_category_products", scenario_id="laptop_bag", category_id="", limit=5,
    ),), usage=ProviderUsage(220, 0, 30, reported=True))


class _ScriptedProvider:
    def __init__(self, *turns):
        self.turns = deque(turns)
        self.requests = []
        self.web_requests = []

    def respond(self, **request):
        self.requests.append(deepcopy(request))
        turn = self.turns.popleft()
        if isinstance(turn, Exception):
            raise turn
        return turn

    def search_web(self, query, domains, **request):
        self.web_requests.append((query, domains, request))
        return WebResponse(("https://support.example.test/manual",), ProviderUsage(80, 0, 15, 1, True))


class LLMRecommendationAgentTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog = Path(directory.name) / "feed.xml"
        feed = Element("yml_catalog")
        shop = SubElement(feed, "shop")
        categories = SubElement(shop, "categories")
        SubElement(categories, "category", id="70824").text = "Ноутбуки"
        SubElement(categories, "category", id="83844").text = "Сумки для ноутбуков"
        offers = SubElement(shop, "offers")
        for offer_id, category, name, attribute, value in (
            ("main", "70824", "Ноутбук", "Габариты корпуса", "350×235×17 мм"),
            ("fit", "83844", "Сумка для ноутбука", "Внутренние размеры", "360×240×20 мм"),
            ("too-small", "83844", "Чехол для ноутбука", "Внутренние размеры", "300×230×20 мм"),
            ("unmeasured", "70824", "Ноутбук без размеров", "Цвет", "чёрный"),
        ):
            offer = SubElement(offers, "offer", id=offer_id, available="true")
            for tag, item in (("categoryId", category), ("name", name), ("vendor", "Lenovo"),
                              ("vendorCode", offer_id), ("price", "1000"), ("currencyId", "RUR"),
                              ("description", "WHOLE_CATALOG_DESCRIPTION_MUST_STAY_LOCAL")):
                SubElement(offer, tag).text = item
            SubElement(offer, "param", name=attribute).text = value
        self.catalog.write_bytes(tostring(feed, encoding="utf-8"))
        self.rules = RecommendationAgent(self.catalog)

    def agent(self, provider=None, evidence=None, *, index=None, **settings):
        configured = replace(LLMSettings(), **settings)
        if provider is not None and evidence is None:
            evidence = SimpleNamespace(active_facts=lambda: (), cached=lambda *_: None)
        agent = LLMRecommendationAgent(index, self.catalog, self.rules, configured, evidence, provider)
        self.addCleanup(agent.close)
        return agent

    def exercise_agent_tools(self):
        patcher = patch.object(self.rules, "enrichment_targets", return_value=(("main", "dimensions_mm"),))
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def test_no_api_key_uses_rules_and_reports_real_mode(self):
        agent = self.agent()
        result = agent.recommend("main")
        self.assertEqual(result.mode, "local")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.execution_status, "completed_with_results")
        self.assertEqual(result.scenario_ids, ("laptop_bag",))
        self.assertEqual([item.offer.offer_id for item in result.items], ["fit"])
        self.assertEqual(result.usage["llm_calls"], 0)
        missing = agent.recommend("unmeasured")
        self.assertEqual(missing.mode, "rule_based_fallback")
        self.assertEqual(missing.fallback_reason, "missing_api_key")

    def test_tools_run_then_server_discards_incompatible_and_unknown_selected_ids(self):
        check = _call("fit-check", "check_compatibility", scenario_id="laptop_bag", candidate_id="fit")
        provider = _ScriptedProvider(
            _browse_candidates(), ProviderResponse(calls=(check,)), _selection("missing", "too-small", "fit"),
        )
        agent = self.agent(provider)
        private = "секретный текст, identity@example.test"
        query = ParsedQuery(private, private, private, (
            Condition("model", "eq", "PRIVATE_CUSTOMER_NAME", private),
            Condition("devices", "ge", Decimal(2), private),
        ), ())
        with patch.object(self.rules, "enrichment_targets", return_value=(("main", "dimensions_mm"),)):
            first = agent.recommend("main", query)
        second = agent.recommend("main", query)

        self.assertEqual(first.mode, "llm_agent")
        self.assertEqual(first.scenario_id, "laptop_bag")
        self.assertEqual([item.offer.offer_id for item in first.items], ["fit"])
        self.assertEqual([entry["tool"] for entry in first.tools],
                         ["rule_based_recommendation", "prepare_context", "get_category_products",
                          "check_compatibility", "agent_selection", "rule_based_recommendation"])
        self.assertEqual(first.usage["input_tokens"], 420)
        self.assertEqual(first.usage["cached_input_tokens"], 10)
        self.assertEqual(first.usage["llm_calls_with_usage"], 2)
        self.assertEqual(first.usage["tool_calls"], 2)
        self.assertEqual(second.mode, "local")
        self.assertEqual(second.usage["llm_calls"], 0)
        self.assertEqual(len(provider.requests), 3)
        sent = json.dumps(provider.requests)
        for sensitive in ("identity@example.test", "PRIVATE_CUSTOMER_NAME", "WHOLE_CATALOG_DESCRIPTION"):
            self.assertNotIn(sensitive, sent)
        self.assertIn("devices", sent)
        changed_query = replace(query, original="different")
        self.assertEqual(agent.recommend("main", changed_query).mode, "local")

    def test_empty_gpt_response_is_a_completed_empty_result_and_briefly_reused(self):
        provider = _ScriptedProvider(_selection())
        agent = self.agent(provider)
        first = agent.recommend("unmeasured")
        self.assertEqual(first.mode, "llm_agent")
        self.assertEqual(first.execution_status, "completed_empty")
        self.assertEqual(first.outcome_code, "missing_product_facts")
        self.assertIsNone(first.fallback_reason)
        second = agent.recommend("unmeasured")
        self.assertEqual(second.mode, "cached_result")
        self.assertFalse(second.items)
        self.assertEqual(len(provider.requests), 1)

    def test_simultaneous_same_context_shares_one_agent_run(self):
        entered, release, joined = Event(), Event(), Event()
        provider = _ScriptedProvider(_selection())
        respond = provider.respond

        def wait_for_second(**request):
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return respond(**request)

        provider.respond = wait_for_second
        agent = self.agent(provider)
        start_work = agent._enrichment_task

        def observe_shared_work(key, run):
            work = start_work(key, run)
            if not work[2]:
                joined.set()
            return work

        with patch.object(agent, "_enrichment_task", side_effect=observe_shared_work):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(agent.recommend, "unmeasured")
                try:
                    self.assertTrue(entered.wait(timeout=1))
                    second = pool.submit(agent.recommend, "unmeasured")
                    self.assertTrue(joined.wait(timeout=1))
                finally:
                    release.set()
                self.assertEqual(first.result().mode, "llm_agent")
                self.assertEqual(second.result().mode, "cached_result")
        self.assertEqual(len(provider.requests), 1)

    def test_unknown_tool_and_invalid_arguments_have_structured_errors(self):
        self.exercise_agent_tools()
        malformed = ProviderCall("malformed", "search_catalog", '{"query":"test","category_id":"","limit":true}')
        unknown = _call("unknown", "execute_python", source="not executable")
        provider = _ScriptedProvider(
            ProviderResponse(calls=(unknown, malformed), usage=ProviderUsage(reported=True)),
            _browse_candidates(), _selection("fit"),
        )
        result = self.agent(provider).recommend("main")
        self.assertEqual(result.mode, "llm_agent")
        self.assertEqual(result.usage["tool_calls"], 3)
        self.assertEqual(result.usage["valid_tool_calls"], 1)
        self.assertEqual(result.usage["successful_tool_calls"], 1)
        self.assertEqual({item.get("error_code") for item in result.tools} - {None},
                         {"unknown_tool", "invalid_arguments"})
        self.assertNotIn("execute_python", json.dumps(result.tools))
        outputs = [json.loads(item["output"]) for item in provider.requests[1]["conversation"]
                   if item.get("type") == "function_call_output"]
        self.assertEqual(outputs, [{"error": {"code": "unknown_tool"}}, {"error": {"code": "invalid_arguments"}}])

    def test_repeated_calls_terminate_and_return_fallback(self):
        self.exercise_agent_tools()
        provider = _ScriptedProvider(_browse_candidates("first"), _browse_candidates("second"))
        result = self.agent(provider).recommend("main")
        self.assertEqual(result.mode, "rule_based_fallback")
        self.assertEqual(result.fallback_reason, "repeated_tool_calls")
        self.assertEqual(result.usage["llm_calls"], 2)

    def test_search_catalog_uses_existing_hybrid_and_still_validates_items(self):
        self.exercise_agent_tools()
        class Index:
            def search_candidates(_, query, *, category_ids, limit):
                assert query == "ноутбук сумка"
                assert category_ids == ("83844",)
                assert limit == 2
                return [SimpleNamespace(offer=self.rules.get_offer(offer_id)) for offer_id in ("too-small", "fit")]

        provider = _ScriptedProvider(
            ProviderResponse(calls=(_call("search", "search_catalog", query="ноутбук сумка", category_id="83844", limit=2),)),
            _selection("too-small", "fit"),
        )
        result = self.agent(provider, index=Index()).recommend("main")
        self.assertEqual(result.mode, "llm_agent")
        self.assertEqual([item.offer.offer_id for item in result.items], ["fit"])

    def test_unsupported_scenario_does_not_call_provider(self):
        provider = _ScriptedProvider()
        result = self.agent(provider).recommend("fit")
        self.assertEqual(result.reason_code, "unsupported_scenario")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.mode, "local")
        self.assertEqual(provider.requests, [])

    def test_deadline_limits_a_slow_provider(self):
        complete = Event()
        provider = _ScriptedProvider()

        def slow(**_):
            complete.wait(timeout=2)
            return _selection()

        provider.respond = slow
        agent = self.agent(provider, timeout_seconds=0.06)
        start = monotonic()
        try:
            result = agent.recommend("unmeasured")
        finally:
            complete.set()
        self.assertEqual(result.fallback_reason, "agent_timeout")
        self.assertLess(monotonic() - start, 0.4)
        self.assertEqual(result.usage["llm_calls"], 1)

    def test_generic_error_does_not_disclose_provider_payload(self):
        provider = _ScriptedProvider(ProviderFailure("private error from an alternative provider"))
        result = self.agent(provider).recommend("unmeasured")
        self.assertEqual(result.fallback_reason, "provider_unavailable")
        self.assertNotIn("private", json.dumps(result.tools))

    def test_limit_prevents_calling_extra_tools(self):
        provider = _ScriptedProvider(_browse_candidates(), _browse_candidates("second"))
        result = self.agent(provider, max_tool_calls=1).recommend("unmeasured")
        self.assertEqual(result.fallback_reason, "tool_call_limit")
        self.assertEqual(result.usage["tool_calls"], 1)
        self.assertEqual(provider.requests[-1]["tools"], [])

    def test_missing_specs_use_bounded_hosted_search_and_then_validators(self):
        from softline_search.vendor_evidence import VendorFact

        fact = VendorFact(
            manufacturer="Lenovo", product_model="Ноутбук без размеров", article="unmeasured",
            attribute="dimensions_mm", value=("350", "235", "17"), unit="mm",
            source_url="https://psref.lenovo.com/Detail/?M=unmeasured",
            evidence_text="Dimensions (W×D×H): 350 × 235 × 17 mm", retrieved_at="2026-11-27T10:00:00+00:00",
            kind="laptop",
        )

        class Evidence:
            def cached(_, identity, attribute):
                return None

            def active_facts(_):
                return ()

            def search(_, identity, attribute, *, deadline, discover, strategy):
                assert attribute == "dimensions_mm"
                assert strategy == "exact_article"
                urls = discover("Lenovo unmeasured dimensions", ("psref.lenovo.com",), deadline)
                assert urls
                return SimpleNamespace(status="verified", fact=fact, cache_hit=False)

        first = _call("find-dimensions", "search_vendor_specs", offer_id="unmeasured", attribute="dimensions_mm")
        provider = _ScriptedProvider(ProviderResponse(calls=(first,)), _browse_candidates(), _selection())
        result = self.agent(provider, Evidence()).recommend("unmeasured")
        self.assertEqual(result.mode, "llm_agent")
        self.assertEqual(result.usage["web_searches"], 1)
        self.assertEqual(result.usage["verified_facts"], 1)
        self.assertIn(fact.source_url, result.items[0].sources)
        self.assertNotIn(fact.evidence_text, json.dumps(provider.requests, ensure_ascii=False))

    def test_zero_web_budget_stays_local(self):
        lookup = _call("find-dimensions", "search_vendor_specs", offer_id="unmeasured", attribute="dimensions_mm")
        provider = _ScriptedProvider(ProviderResponse(calls=(lookup,)), _selection())
        evidence = SimpleNamespace(cached=lambda *args: None, active_facts=lambda: ())
        result = self.agent(provider, evidence, max_web_searches=0).recommend("unmeasured")
        self.assertEqual(result.mode, "local")
        self.assertEqual(result.usage["external_lookups"], 0)
        self.assertFalse(result.items)
        self.assertFalse(provider.requests)

    def test_hosted_search_actions_exhaust_the_budget_even_in_one_response(self):
        class Provider(_ScriptedProvider):
            def search_web(self, query, domains, **request):
                response = super().search_web(query, domains, **request)
                return replace(response, usage=replace(response.usage, web_searches=2))

        class Evidence:
            def cached(self, *args):
                return None

            def active_facts(self):
                return ()

            def search(self, identity, attribute, *, deadline, discover, strategy):
                discover("official dimensions", ("psref.lenovo.com",), deadline)
                assert discover("alternate official document", ("psref.lenovo.com",), deadline) == ()
                return EvidenceResult(None, "search_no_results")

        lookup = _call("find-dimensions", "search_vendor_specs", offer_id="unmeasured", attribute="dimensions_mm")
        provider = Provider(ProviderResponse(calls=(lookup,)), _selection())
        result = self.agent(provider, Evidence(), max_web_searches=2).recommend("unmeasured")
        self.assertEqual(len(provider.web_requests), 1)
        self.assertEqual(result.usage["web_searches"], 2)
        self.assertIn("search_no_results", [entry.get("error_code") for entry in result.tools])

    def test_requests_for_an_unrelated_attribute_never_start_a_lookup(self):
        lookup = _call("invalid-attribute", "search_vendor_specs", offer_id="unmeasured", attribute="battery_articles")
        provider = _ScriptedProvider(ProviderResponse(calls=(lookup,)), _browse_candidates(), _selection("fit"))
        result = self.agent(provider).recommend("unmeasured")
        self.assertEqual(result.mode, "llm_agent")
        self.assertEqual(result.usage["external_lookups"], 0)
        self.assertIn("unsupported_lookup", [entry.get("error_code") for entry in result.tools])

    def test_missing_key_fallback_can_use_persisted_verified_evidence_without_network(self):
        name = self.catalog.with_suffix(".sqlite3")
        html = ("<h1>Lenovo unmeasured</h1><table><tr><th>Dimensions (W×D×H)</th>"
                "<td>350 × 235 × 17 mm</td></tr></table>").encode()

        def fetch(url, domains, deadline):
            return VendorDocument(url, "text/html", html)

        seeder = VendorEvidenceService(name, fetch=fetch)
        identity = self.rules.product_identity("unmeasured")
        confirmed = seeder.search(identity, "dimensions_mm", deadline=monotonic() + 1)
        self.assertIsNotNone(confirmed.fact)

        def offline(*_):
            self.fail("fallback tried the network")

        warmed = VendorEvidenceService(name, fetch=offline)
        result = self.agent(evidence=warmed).recommend("unmeasured", debug=True)
        self.assertEqual(result.mode, "cached_result")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual([item.offer.offer_id for item in result.items], ["fit"])
        self.assertIn(confirmed.fact.source_url, result.items[0].sources)
        self.assertEqual(result.diagnostics["decision"]["status"], "CONFIRMED")

    def test_429_after_confirmed_candidate_keeps_facts_and_all_server_results(self):
        html = ("""<h1>Lenovo unmeasured</h1><table><tr><th>Dimensions (W x D x H)</th>
                   <td>350 x 235 x 17 mm</td></tr></table>""").encode()
        evidence = VendorEvidenceService(
            self.catalog.with_suffix(".sqlite3"),
            fetch=lambda url, *_: VendorDocument(url, "text/html", html),
        )
        lookup = _call("lookup", "search_vendor_specs", offer_id="unmeasured", attribute="dimensions_mm")
        check = _call("check", "check_compatibility", scenario_id="laptop_bag", candidate_id="fit")
        provider = _ScriptedProvider(
            ProviderResponse(calls=(lookup, _browse_candidates().calls[0], check)),
            ProviderFailure("rate_limited"),
        )
        agent = self.agent(provider, evidence)
        result = agent.recommend("unmeasured")
        self.assertEqual(result.mode, "rule_based_fallback")
        self.assertEqual(result.fallback_reason, "rate_limited")
        self.assertEqual(result.execution_status, "provider_error")
        self.assertEqual(result.outcome_code, "completed_with_results")
        self.assertEqual([item.offer.offer_id for item in result.items], ["fit"])
        self.assertIn("external_lookup", result.origins)
        self.assertTrue(any(entry.get("compatibility", {}).get("confirmed") == 1
                            for entry in result.tools if entry["tool"] == "check_compatibility"))
        second = agent.recommend("unmeasured")
        self.assertEqual(second.mode, "cached_result")
        self.assertEqual([item.offer.offer_id for item in second.items], ["fit"])
        self.assertEqual(len(provider.requests), 2)


class OpenAIProviderTests(TestCase):
    def test_env_file_defaults_and_secret_free_repr(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY='test-only-not-a-secret'\nAGENT_TIMEOUT_SECONDS=NaN\n"
                            "LLM_ENABLED=yes\nbase_url=https://api.proxyapi.ru/v1\n", encoding="utf-8-sig")
            with patch.dict(os.environ, {"LLM_MODEL": "gpt-5.4-mini"}, clear=True):
                settings = LLMSettings.from_env(path)
        self.assertEqual(settings.api_key, "test-only-not-a-secret")
        self.assertNotIn(settings.api_key, repr(settings))
        self.assertEqual(settings.model, "gpt-5.4-mini")
        self.assertEqual(settings.timeout_seconds, 10)
        self.assertEqual(settings.base_url, "https://api.proxyapi.ru/v1")

    def test_official_sdk_sends_responses_schema_and_handles_rate_limit(self):
        seen = []

        def send(request):
            seen.append(json.loads(request.content))
            return httpx.Response(429, json={"error": {"message": "internal provider message"}}, request=request)

        client = OpenAI(api_key="test-key", base_url="https://api.proxyapi.ru/v1",
                        http_client=httpx.Client(transport=httpx.MockTransport(send)))
        self.addCleanup(client.close)
        provider = OpenAIProvider(LLMSettings(), client=client)
        schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}
        with self.assertRaisesRegex(ProviderFailure, "^rate_limited$"):
            provider.respond(instructions="test", conversation=[], tools=[], final_schema=schema,
                             timeout=0.5, max_output_tokens=32)
        request, = seen
        self.assertEqual(request["model"], "openai/gpt-6-luna")
        self.assertFalse(request["store"])
        self.assertTrue(request["text"]["format"]["strict"])

    def test_hosted_web_adapter_keeps_only_official_structured_sources(self):
        requests = []

        def send(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, request=request, json={
                "id": "response", "object": "response", "status": "completed", "output": [
                    {"type": "web_search_call", "id": "search", "status": "completed", "action": {
                        "type": "search", "query": "Lenovo exact article", "sources": [
                            {"type": "url", "url": "https://psref.lenovo.com/Detail/?M=article"},
                            {"type": "url", "url": "https://psref.lenovo.com.example.test/spec"},
                        ],
                    }},
                ], "usage": {"input_tokens": 120, "output_tokens": 20, "total_tokens": 140,
                             "input_tokens_details": {"cached_tokens": 8}, "output_tokens_details": {"reasoning_tokens": 0}},
            })

        client = OpenAI(api_key="test-key", base_url="https://api.proxyapi.ru/v1",
                        http_client=httpx.Client(transport=httpx.MockTransport(send)))
        self.addCleanup(client.close)
        provider = OpenAIProvider(LLMSettings(model="openai/gpt-5.4-mini"), client=client)
        result = provider.search_web("Lenovo exact article", ("psref.lenovo.com",), timeout=1, max_output_tokens=64)
        self.assertEqual(result.usage.web_searches, 1)
        self.assertEqual(result.usage.cached_input_tokens, 8)
        self.assertEqual(result.urls, ("https://psref.lenovo.com/Detail/?M=article",))
        self.assertEqual(requests[0]["model"], "openai/gpt-5.4-mini")
        self.assertEqual(requests[0]["max_tool_calls"], 1)
        self.assertEqual(requests[0]["tools"][0]["filters"]["allowed_domains"], ["psref.lenovo.com"])

    def test_sdk_parses_real_responses_shapes_for_tool_call_and_final_json(self):
        arguments = '{"query":"","category_id":"83844","limit":5}'
        call = {"type": "function_call", "call_id": "call_test", "name": "search_catalog",
                "arguments": arguments}
        final_text = '{"selections":[]}'
        outputs = deque(((call,), ({"type": "message", "id": "msg_final", "role": "assistant",
                                   "status": "completed", "content": [{"type": "output_text",
                                                                       "text": final_text, "annotations": []}]},)))
        requests = []

        def send(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, request=request, json={
                "id": "response", "object": "response", "status": "completed", "output": outputs.popleft(),
            })

        client = OpenAI(api_key="test-key", base_url="https://api.proxyapi.ru/v1",
                        http_client=httpx.Client(transport=httpx.MockTransport(send)))
        self.addCleanup(client.close)
        provider = OpenAIProvider(LLMSettings(), client=client)
        options = {
            "instructions": "test", "tools": [], "timeout": 1, "max_output_tokens": 256,
            "final_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }
        first = provider.respond(conversation=[{"role": "user", "content": "test"}], **options)
        self.assertEqual(first.calls, (ProviderCall("call_test", "search_catalog", arguments),))
        output = {"type": "function_call_output", "call_id": "call_test", "output": '{"products":[]}'}
        second = provider.respond(conversation=[{"role": "user", "content": "test"}, call, output], **options)
        self.assertEqual(second.text, final_text)
        self.assertEqual(requests[-1]["input"][-1], output)
