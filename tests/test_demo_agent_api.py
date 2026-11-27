import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

from softline_search.catalog import Offer
from softline_search.demo import serve_demo
from softline_search.index import CheckedSearch, ProductDifference, SearchHit, SimilarHit
from softline_search.llm_agent import AgentRecommendationResult
from softline_search.llm_agent_v3 import MODE, V3Recommendation, V3RecommendationResult
from softline_search.llm_provider import LLMSettings
from softline_search.recommendation_facts import Status
from softline_search.query import ParsedQuery
from softline_search.recommendation import Recommendation


@dataclass(frozen=True, slots=True)
class _PartialRecommendation(Recommendation):
    verification_status: str = "UNKNOWN"
    missing_criteria: tuple[str, ...] = ("bag_dimensions",)


class DemoAgentAPITests(TestCase):
    def test_search_and_similar_continue_while_an_agent_request_waits(self):
        self._check_http("baseline")

    def test_v3_is_served_by_default_with_relationships_and_empty_reasons(self):
        self._check_http("v3")

    def _check_http(self, version):
        source = Offer("1", "laptop", "70824", "Ноутбук Lenovo", "Lenovo", "L1", "", "85000", "RUR", "")
        similar = Offer("2", "alternative", "70824", "Ноутбук HP", "HP", "L2", "", "74000", "RUR", "")
        bag = Offer("3", "bag", "83844", "Сумка HP", "HP", "B1", "", "2000", "RUR", "")
        parsed = ParsedQuery("ноутбук", "ноутбук", "ноутбук", (), ())
        index = SimpleNamespace(
            offers=(source, similar, bag),
            prepare_query_catalog=Mock(return_value=SimpleNamespace(
                categories={"70824": "Ноутбуки", "83844": "Сумки"},
                is_software_category=lambda category: False,
            )),
            parse_query=Mock(return_value=parsed),
            search_checked=Mock(return_value=CheckedSearch(parsed, [SearchHit(source, 1.0)], [], {})),
            similar=Mock(return_value=[SimilarHit(
                similar, 1.0, "Цена ниже", (ProductDifference("Цена", "85000", "74000"),),
            )]),
            suggest=Mock(return_value=["Ноутбук Lenovo"]),
        )
        request_started = Event()
        release_request = Event()

        def recommend(*args, **kwargs):
            if kwargs["limit"] == 2:
                if version == "v3":
                    return V3RecommendationResult([], "missing_api_key", Status.UNKNOWN, "not_run", MODE,
                                                  "missing_api_key", None, (), {}, 0, source)
                return AgentRecommendationResult(
                    "laptop_bag", [Recommendation(bag, True, True, "Проверены габариты", ("https://www.hp.com/spec",))],
                    scenario_ids=("laptop_bag",), mode="rule_based_fallback", fallback_reason="missing_api_key",
                )
            request_started.set()
            if not release_request.wait(5):
                raise RuntimeError("test request timeout")
            if version == "v3":
                item = V3Recommendation(bag, "Переноска", "Для переноски ноутбука", True, True)
                return V3RecommendationResult([item], "partial_validation", Status.UNKNOWN,
                                              "completed_with_results", MODE, None, None, (), {}, 0, source)
            return AgentRecommendationResult(
                None,
                [_PartialRecommendation(bag, True, True, "Сумка для переноски ноутбука",
                                        direction_id="carry", direction_label="Переноска")],
                directions=("Переноска",), mode="autonomous_agent", execution_status="completed_with_results",
                tools=({"tool": "validate_autonomous_candidates", "status": "ok", "elapsed_ms": 1.0, "count": 1},),
            )

        agent = SimpleNamespace(recommend=Mock(side_effect=recommend), close=Mock())
        ready = Event()
        servers = []

        class TestServer(ThreadingHTTPServer):
            def serve_forever(self, poll_interval=0.05):
                servers.append(self)
                ready.set()
                super().serve_forever(poll_interval=poll_interval)

        target = ("softline_search.llm_agent_v3.LLMRecommendationAgentV3" if version == "v3"
                  else "softline_search.autonomous_agent.AutonomousRecommendationAgent")
        with (
            TemporaryDirectory() as directory,
            patch("softline_search.demo.ThreadingHTTPServer", TestServer),
            patch("softline_search.recommendation.RecommendationAgent"),
            patch("softline_search.vendor_evidence.VendorEvidenceService", return_value=None),
            patch("softline_search.llm_provider.LLMSettings.from_env", return_value=LLMSettings(enabled=False))
            as load_settings,
            patch(target, return_value=agent),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            catalog = Path(directory) / "snapshot.xml"
            catalog.touch()
            env_path = Path(directory) / "settings.env"
            kwargs = {"env_path": env_path}
            if version == "baseline":
                kwargs["agent_version"] = version
            thread = Thread(target=serve_demo, args=(index, catalog, "bm25", 0), kwargs=kwargs, daemon=True)
            thread.start()
            self.assertTrue(ready.wait(3))
            server = servers[0]

            def get(route):
                with urlopen(f"http://127.0.0.1:{server.server_port}" + route, timeout=2) as response:
                    return json.load(response)

            try:
                pending = executor.submit(get, "/api/recommend?offer_id=1&q=laptop")
                self.assertTrue(request_started.wait(2))
                self.assertEqual(get("/api/search?q=laptop")["products"][0]["id"], "1")
                result = get("/api/similar?offer_id=1")
                self.assertEqual(result["products"][0]["id"], "2")
                self.assertEqual(result["products"][0]["differences"][0]["selected"], "85000")
                self.assertEqual(get("/api/suggest?q=notebook")["suggestions"], ["Ноутбук Lenovo"])
                with self.assertRaises(HTTPError) as error:
                    get("/api/recommend?offer_id=not-in-feed")
                self.assertEqual(error.exception.code, 404)
                for invalid in ("0", "9", "1.5", "", "2&limit=3"):
                    with self.subTest(limit=invalid), self.assertRaises(HTTPError) as error:
                        get("/api/recommend?offer_id=1&limit=" + invalid)
                    self.assertEqual(error.exception.code, 400)
                release_request.set()
                recommendation = pending.result(timeout=3)
                self.assertEqual(recommendation["mode"], MODE if version == "v3" else "autonomous_agent")
                self.assertEqual(recommendation["source"]["id"], "1")
                item = recommendation["products"][0]
                self.assertEqual(item["verification_status"], "UNKNOWN")
                if version == "v3":
                    self.assertNotIn("scenario_id", recommendation)
                    self.assertEqual(item["candidate_id"], "3")
                    self.assertEqual(item["relationship"], "Переноска")
                else:
                    self.assertEqual(recommendation["directions"], [{"id": "carry", "label": "Переноска", "count": 1}])
                    self.assertEqual(item["missing_criteria"], ["bag_dimensions"])
                self.assertNotIn("diagnostics", recommendation)
                agent.recommend.assert_called_once_with("1", parsed, filters=(), debug=False, limit=4)
                fallback = get("/api/recommend?offer_id=1&limit=2&debug=1")
                if version == "v3":
                    self.assertEqual(fallback["reason_code"], "missing_api_key")
                    self.assertEqual(fallback["products"], [])
                else:
                    self.assertEqual(fallback["mode"], "rule_based_fallback")
                    self.assertEqual(fallback["scenario_id"], "laptop_bag")
                    self.assertEqual(fallback["fallback_reason"], "missing_api_key")
                    self.assertEqual(fallback["products"][0]["verification_status"], "CONFIRMED")
                    self.assertEqual(fallback["products"][0]["sources"], ["https://www.hp.com/spec"])
                agent.recommend.assert_called_with("1", None, filters=(), debug=True, limit=2)
            finally:
                release_request.set()
                server.shutdown()
                thread.join(timeout=3)
        agent.close.assert_called_once()
        load_settings.assert_called_once_with(env_path)
