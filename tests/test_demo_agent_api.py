import json
from concurrent.futures import ThreadPoolExecutor
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
from softline_search.llm_provider import LLMSettings
from softline_search.query import ParsedQuery
from softline_search.recommendation import Recommendation


class DemoAgentAPITests(TestCase):
    def test_search_and_similar_continue_while_an_agent_request_waits(self):
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
            request_started.set()
            if not release_request.wait(5):
                raise RuntimeError("test request timeout")
            return AgentRecommendationResult(
                "laptop_bag", [Recommendation(bag, True, True, "Проверены габариты", ("https://www.hp.com/spec",))],
                scenario_ids=("laptop_bag",), mode="llm_agent",
                tools=({"tool": "check_compatibility", "status": "ok", "elapsed_ms": 1.0, "count": 1},),
            )

        agent = SimpleNamespace(recommend=recommend, close=Mock())
        ready = Event()
        servers = []

        class TestServer(ThreadingHTTPServer):
            def serve_forever(self, poll_interval=0.05):
                servers.append(self)
                ready.set()
                super().serve_forever(poll_interval=poll_interval)

        with (
            TemporaryDirectory() as directory,
            patch("softline_search.demo.ThreadingHTTPServer", TestServer),
            patch("softline_search.recommendation.RecommendationAgent"),
            patch("softline_search.vendor_evidence.VendorEvidenceService", return_value=None),
            patch("softline_search.llm_provider.LLMSettings.from_env", return_value=LLMSettings(enabled=False)),
            patch("softline_search.llm_agent.LLMRecommendationAgent", return_value=agent),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            catalog = Path(directory) / "snapshot.xml"
            catalog.touch()
            thread = Thread(target=serve_demo, args=(index, catalog, "bm25", 0), daemon=True)
            thread.start()
            self.assertTrue(ready.wait(3))
            server = servers[0]

            def get(route):
                with urlopen(f"http://127.0.0.1:{server.server_port}" + route, timeout=2) as response:
                    return json.load(response)

            try:
                pending = executor.submit(get, "/api/recommend?offer_id=1")
                self.assertTrue(request_started.wait(2))
                self.assertEqual(get("/api/search?q=laptop")["products"][0]["id"], "1")
                result = get("/api/similar?offer_id=1")
                self.assertEqual(result["products"][0]["id"], "2")
                self.assertEqual(result["products"][0]["differences"][0]["selected"], "85000")
                self.assertEqual(get("/api/suggest?q=notebook")["suggestions"], ["Ноутбук Lenovo"])
                with self.assertRaises(HTTPError) as error:
                    get("/api/recommend?offer_id=not-in-feed")
                self.assertEqual(error.exception.code, 404)
                release_request.set()
                recommendation = pending.result(timeout=3)
                self.assertEqual(recommendation["mode"], "llm_agent")
                self.assertEqual(recommendation["scenario_id"], "laptop_bag")
                self.assertEqual(recommendation["source"]["id"], "1")
                self.assertEqual(recommendation["products"][0]["sources"], ["https://www.hp.com/spec"])
                self.assertNotIn("diagnostics", recommendation)
            finally:
                release_request.set()
                server.shutdown()
                thread.join(timeout=3)
        agent.close.assert_called_once()
