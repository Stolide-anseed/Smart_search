import json
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring

from softline_search.cross_sell import (
    DIRECTIONS, ArticleCompatibility, CapacityCompatibility, CandidateInterfaces,
    DimensionCompatibility, InterfaceCompatibility, InterfaceOption, LicenseCompatibility,
    PlatformCompatibility, PortFacts, PowerCompatibility, ProtocolCompatibility,
    classify_candidate, requested_direction_ids,
)
from softline_search.llm_agent import LLMRecommendationAgent
from softline_search.llm_provider import LLMSettings, ProviderCall, ProviderResponse, ProviderUsage
from softline_search.query import ParsedQuery
from softline_search.recommendation import RecommendationAgent
from softline_search.recommendation_facts import Status


def _parsed(text: str) -> ParsedQuery:
    return ParsedQuery(text, text.casefold(), text.casefold(), (), ())


def _candidate(*kinds, requires=(), platforms=()):
    return CandidateInterfaces(
        "test", tuple(InterfaceOption(kind, kind, "test") for kind in kinds),
        tuple(requires), tuple(platforms), ("test",),
    )


class ValidatorTests(TestCase):
    def setUp(self):
        self.ports = PortFacts(
            {"usb_a": 2, "usb_c": 1, "hdmi": 1, "displayport": 1, "audio_35": 1},
            sources=("XML",),
        )

    def test_interface_confirms_only_a_known_matching_port(self):
        result = InterfaceCompatibility.validate(self.ports, _candidate("usb_a"))
        self.assertEqual(result.status, Status.CONFIRMED)
        self.assertEqual(result.reason_code, "interface_match")
        self.assertIn("USB Type-A", result.reason)
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("usb_c")).status, Status.CONFIRMED)
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("audio_35")).status, Status.CONFIRMED)
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("hdmi")).status, Status.CONFIRMED)
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("displayport")).status, Status.CONFIRMED)

    def test_missing_attribute_never_confirms(self):
        ports = PortFacts({}, sources=("XML",))
        result = InterfaceCompatibility.validate(ports, _candidate("usb_a"))
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(result.reason_code, "missing_anchor_interface")
        self.assertIn("usb_a", result.missing)
        self.assertEqual(InterfaceCompatibility.validate(ports, _candidate()).status, Status.UNKNOWN)
        self.assertEqual(InterfaceCompatibility.validate(ports, _candidate()).reason_code, "missing_candidate_interface")

    def test_explicitly_absent_or_foreign_interfaces_are_rejected(self):
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("ps2")).status, Status.UNKNOWN)
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("rj9")).status, Status.REJECTED)
        self.assertEqual(InterfaceCompatibility.validate(self.ports, _candidate("dvi")).status, Status.REJECTED)
        ports = PortFacts({}, frozenset({"ps2"}), sources=("spec",))
        self.assertEqual(InterfaceCompatibility.validate(ports, _candidate("ps2")).status, Status.REJECTED)

    def test_one_supported_option_wins_over_unknown_options(self):
        result = InterfaceCompatibility.validate(self.ports, _candidate("ps2", "usb_c"))
        self.assertEqual(result.status, Status.CONFIRMED)
        self.assertEqual(result.facts["matched_interface"], "usb_c")

    def test_protocol_requires_explicit_dp_alt_or_power_delivery(self):
        candidate = _candidate("usb_c", requires=("usb_c_dp_alt",))
        self.assertEqual(ProtocolCompatibility.validate(self.ports, candidate).status, Status.UNKNOWN)
        unknown = replace(self.ports, usb_c_dp_alt=False)
        self.assertEqual(ProtocolCompatibility.validate(unknown, candidate).status, Status.REJECTED)
        supported = replace(self.ports, usb_c_dp_alt=True)
        self.assertEqual(ProtocolCompatibility.validate(supported, candidate).status, Status.CONFIRMED)

    def test_power_delivery_uses_the_same_unknown_rule(self):
        candidate = _candidate("usb_c", requires=("usb_c_pd",))
        self.assertEqual(PowerCompatibility.validate(self.ports, candidate).status, Status.UNKNOWN)
        self.assertEqual(PowerCompatibility.validate(replace(self.ports, usb_c_pd=False), candidate).status,
                         Status.REJECTED)
        self.assertEqual(PowerCompatibility.validate(replace(self.ports, usb_c_pd=True), candidate).status,
                         Status.CONFIRMED)

    def test_dimension_validator_rejects_oversize_and_keeps_missing_unknown(self):
        self.assertEqual(DimensionCompatibility.validate((350, 235, 17), (360, 240, 20)).status, Status.CONFIRMED)
        self.assertEqual(DimensionCompatibility.validate((350, 235, 17), (340, 240, 20)).status, Status.REJECTED)
        self.assertEqual(DimensionCompatibility.validate(None, (340, 240, 20)).status, Status.UNKNOWN)
        self.assertEqual(DimensionCompatibility.validate((350, 235, 17), None).status, Status.UNKNOWN)

    def test_platform_validator_only_uses_explicit_restrictions(self):
        self.assertEqual(PlatformCompatibility.validate("windows", ()).status, Status.CONFIRMED)
        self.assertEqual(PlatformCompatibility.validate("windows", ("windows", "macos")).status, Status.CONFIRMED)
        self.assertEqual(PlatformCompatibility.validate("windows", ("macos",)).status, Status.REJECTED)
        self.assertEqual(PlatformCompatibility.validate(None, ("windows",)).status, Status.UNKNOWN)

    def test_article_and_capacity_validators(self):
        self.assertEqual(ArticleCompatibility.validate("RBC10", "rbc10").status, Status.CONFIRMED)
        self.assertEqual(ArticleCompatibility.validate("RBC10", "RBC20").status, Status.REJECTED)
        self.assertEqual(ArticleCompatibility.validate(None, "RBC10").status, Status.UNKNOWN)
        self.assertEqual(CapacityCompatibility.validate(5, 10).status, Status.CONFIRMED)
        self.assertEqual(CapacityCompatibility.validate(5, 4).status, Status.REJECTED)
        self.assertEqual(CapacityCompatibility.validate(5, None).status, Status.UNKNOWN)

    def test_license_validator_combines_platform_and_capacity(self):
        self.assertEqual(LicenseCompatibility.validate("windows", ("windows",), 5, 10).status, Status.CONFIRMED)
        self.assertEqual(LicenseCompatibility.validate("windows", ("macos",), 5, 10).status, Status.REJECTED)
        self.assertEqual(LicenseCompatibility.validate("windows", ("windows",), 11, 10).status, Status.REJECTED)

    def test_query_direction_detection_is_explicit_and_stable(self):
        self.assertEqual(requested_direction_ids("Подбери мышь для ноутбука"), ("laptop_mouse",))
        self.assertEqual(requested_direction_ids("нужна гарнитура"), ("laptop_headset",))
        self.assertEqual(requested_direction_ids("монитор и клавиатура"),
                         ("laptop_keyboard", "laptop_monitor"))
        self.assertEqual(requested_direction_ids("сумка для ноутбука"), ("laptop_bag",))

    def test_candidate_classification_excludes_presenters_mats_and_plain_headphones(self):
        from softline_search.recommendation_facts import read_fields

        def fields(**params):
            element = Element("offer")
            for name, value in params.items():
                SubElement(element, "param", name=name).text = value
            return read_fields(element)

        self.assertEqual(classify_candidate("input_device", fields(**{"Тип устройства": "Мышь"})), "mouse")
        self.assertEqual(classify_candidate("input_device", fields(**{"Тип устройства": "Клавиатура"})), "keyboard")
        self.assertEqual(classify_candidate("input_device", fields(**{"Тип устройства": "Клавиатура+мышь"})),
                         "keyboard_mouse")
        self.assertIsNone(classify_candidate("input_device", fields(**{"Тип устройства": "Презентер"})))
        self.assertIsNone(classify_candidate("headset", fields(**{"Тип": "Микрофон"})))
        self.assertIsNone(classify_candidate("headset", fields(**{"Тип": "Наушники"})))
        self.assertEqual(classify_candidate("headset", fields(**{"Тип": "Bluetooth-гарнитура"})), "headset")


class _Catalog:
    def __init__(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "feed.xml"
        feed = Element("yml_catalog")
        shop = SubElement(feed, "shop")
        categories = SubElement(shop, "categories")
        for category_id, name in (
            ("70824", "Ноутбуки"), ("83844", "Сумки для ноутбуков"), ("82833", "Мыши, клавиатуры"),
            ("88831", "Гарнитуры"), ("80843", "Мониторы"), ("84206", "USB-концентраторы"),
        ):
            SubElement(categories, "category", id=category_id).text = name
        offers = SubElement(shop, "offers")

        def offer(identifier, category, name, vendor, code, price, params=None, description=""):
            element = SubElement(offers, "offer", id=identifier, available="true")
            for tag, text in (("categoryId", category), ("name", name), ("vendor", vendor),
                              ("vendorCode", code), ("price", price), ("currencyId", "RUR"),
                              ("description", description)):
                SubElement(element, tag).text = text
            for key, value in (params or {}).items():
                SubElement(element, "param", name=key).text = value

        offer("laptop", "70824", "Ноутбук Test 14", "Test", "L1", "50000", {
            "Габариты корпуса": "350×235×17 мм", "Диагональ экрана": "14",
            "Разъемы Type-C": "1", "Разъемы USB 2.0": "2", "Разъемы HDMI": "1",
            "Разъемы DisplayPort": "1", "Audio mini-jack": "1", "Операционная система": "Windows 11",
        })
        offer("bag", "83844", "Сумка для ноутбука", "BagCo", "B1", "2000",
              {"Внутренние размеры": "360×240×20 мм"})
        offer("mouse-usb", "82833", "Мышь USB", "A4", "M1", "1000",
              {"Тип устройства": "Мышь", "Интерфейс": "USB 2.0"})
        offer("mouse-ps2", "82833", "Мышь PS/2", "A4", "M2", "900",
              {"Тип устройства": "Мышь", "Интерфейс": "PS/2"})
        offer("mouse-bt", "82833", "Мышь Bluetooth", "Logitech", "M3", "2000",
              {"Тип устройства": "Мышь", "Интерфейс": "Bluetooth 5.2"})
        offer("mouse-c", "82833", "Мышь Type-C", "Logitech", "M4", "1500",
              {"Тип устройства": "Мышь", "Интерфейс": "Type-C"})
        offer("keyboard-usb", "82833", "Клавиатура USB", "Logitech", "K1", "1200",
              {"Тип устройства": "Клавиатура", "Интерфейс": "USB 2.0"})
        offer("keyboard-ps2", "82833", "Клавиатура PS/2", "Defender", "K2", "800",
              {"Тип устройства": "Клавиатура", "Интерфейс": "PS/2"})
        offer("combo", "82833", "Клавиатура+мышь USB", "Defender", "C1", "1800",
              {"Тип устройства": "Клавиатура+мышь", "Интерфейс": "USB 2.0"})
        offer("headset-jack", "88831", "Гарнитура Jack", "A4", "H1", "1000",
              {"Тип": "Гарнитура", "Подключение": "Проводное", "Интерфейс подключения": "Mini-jack 3.5 мм"})
        offer("headset-usb", "88831", "Гарнитура USB", "Logitech", "H2", "1500",
              {"Тип": "Гарнитура", "Подключение": "Проводное", "Интерфейс подключения": "USB 2.0 Type-A"})
        offer("headset-rj9", "88831", "Гарнитура RJ9", "Fanvil", "H3", "900",
              {"Тип": "Гарнитура", "Подключение": "Проводное", "Интерфейс подключения": "RJ9"})
        offer("headset-bt", "88831", "Bluetooth-гарнитура", "Logitech", "H4", "3000",
              {"Тип": "Bluetooth-гарнитура", "Подключение": "Беспроводное",
               "Беспроводное соединение": "Bluetooth 5.2", "Интерфейс подключения": "Type-C"})
        offer("monitor-hdmi", "80843", "Монитор HDMI", "AOC", "D1", "10000",
              {"Интерфейсы": "HDMI, DisplayPort"})
        offer("monitor-dp", "80843", "Монитор DP", "Dell", "D2", "11000", {"Интерфейсы": "DisplayPort"})
        offer("monitor-dvi", "80843", "Монитор DVI", "AOC", "D3", "8000", {"Интерфейсы": "DVI-D"})
        offer("monitor-usbc", "80843", "Монитор USB-C", "AOC", "D4", "12000", {"Интерфейсы": "USB Type-C"})
        offer("hub-explicit", "84206", "USB-концентратор Type-C", "TP-LINK", "U1", "2000",
              {"Входной интерфейс": "USB Type-C"})
        offer("hub-unknown", "84206", "USB-концентратор", "TP-LINK", "U2", "1500", {},
              "Входящие1 × USB 3.0, Выходящие4 × USB 3.0")
        self.path.write_bytes(tostring(feed, encoding="utf-8"))
        self.agent = RecommendationAgent(self.path)

    def close(self):
        self.directory.cleanup()


class UniversalRecommendationTests(TestCase):
    def setUp(self):
        self.catalog = _Catalog()
        self.addCleanup(self.catalog.close)
        self.agent = self.catalog.agent

    def ids(self, result):
        return [item.offer.offer_id for item in result.items]

    def test_mouse_direction_confirms_usb_and_type_c_and_excludes_ps2_and_unknown_bluetooth(self):
        result = self.agent.recommend("laptop", directions=("laptop_mouse",), debug=True)
        self.assertEqual(result.scenario_id, "laptop_mouse")
        self.assertEqual(result.directions, ("laptop_mouse",))
        self.assertEqual(set(self.ids(result)), {"mouse-usb", "mouse-c"})
        self.assertTrue(all(item.direction_label == "Мышь для ноутбука" for item in result.items))
        self.assertEqual(result.diagnostics["live_validation"]["rejected"], 0)
        self.assertEqual(result.diagnostics["live_validation"]["unknown"], 2)
        self.assertIn("USB Type-A", result.items[0].reason)

    def test_keyboard_and_combo_directions_use_the_same_validator(self):
        keyboard = self.agent.recommend("laptop", directions=("laptop_keyboard",))
        self.assertEqual(set(self.ids(keyboard)), {"keyboard-usb", "combo"})
        combo = self.agent.recommend("laptop", directions=("laptop_keyboard_mouse",))
        self.assertEqual(set(self.ids(combo)), {"combo"})

    def test_headset_direction_uses_wired_interfaces_and_ignores_charging_type_c(self):
        result = self.agent.recommend("laptop", directions=("laptop_headset",), debug=True)
        self.assertEqual(set(self.ids(result)), {"headset-jack", "headset-usb"})
        self.assertEqual(result.diagnostics["live_validation"]["rejected"], 1)
        self.assertEqual(result.diagnostics["live_validation"]["unknown"], 1)

    def test_monitor_direction_confirms_video_inputs_and_rejects_foreign_connector(self):
        result = self.agent.recommend("laptop", directions=("laptop_monitor",), debug=True)
        self.assertEqual(set(self.ids(result)), {"monitor-hdmi", "monitor-dp"})
        self.assertNotIn("monitor-dvi", self.ids(result))
        self.assertNotIn("monitor-usbc", self.ids(result))
        self.assertEqual(result.diagnostics["live_validation"]["rejected"], 1)
        self.assertEqual(result.diagnostics["live_validation"]["unknown"], 1)

    def test_hub_requires_an_explicit_upstream_connector(self):
        result = self.agent.recommend("laptop", directions=("laptop_usb_hub",), debug=True)
        self.assertEqual(self.ids(result), ["hub-explicit"])
        self.assertEqual(result.diagnostics["live_validation"]["unknown"], 1)

    def test_check_compatibility_and_batch_use_the_same_server_validators(self):
        self.assertEqual(self.agent.check_compatibility("laptop", "mouse-usb", "laptop_mouse").status,
                         Status.CONFIRMED)
        # An absent PS/2 attribute is not proof of absence: the pair stays UNKNOWN.
        self.assertEqual(self.agent.check_compatibility("laptop", "mouse-ps2", "laptop_mouse").status,
                         Status.UNKNOWN)
        self.assertEqual(self.agent.check_compatibility("laptop", "mouse-bt", "laptop_mouse").status,
                         Status.UNKNOWN)
        batch = self.agent.check_compatibility_batch(
            "laptop", "laptop_mouse", ("mouse-usb", "mouse-ps2", "mouse-bt"),
        )
        self.assertEqual([item["status"] for item in batch],
                         [Status.CONFIRMED, Status.UNKNOWN, Status.UNKNOWN])

    def test_select_confirmed_discards_unknown_and_incompatible_pairs(self):
        items = self.agent.select_confirmed("laptop", [
            ("laptop_mouse", "mouse-usb"), ("laptop_mouse", "mouse-ps2"), ("laptop_mouse", "mouse-bt"),
        ])
        self.assertEqual([item.offer.offer_id for item in items], ["mouse-usb"])
        self.assertEqual(items[0].direction_id, "laptop_mouse")

    def test_compatibility_requirements_expose_ports_and_missing_attributes(self):
        result = self.agent.compatibility_requirements("laptop", "laptop_mouse")
        self.assertEqual(result["direction_id"], "laptop_mouse")
        self.assertEqual(result["anchor_ports"]["ports"]["usb_a"], 2)
        self.assertEqual(result["candidate_count"], 4)
        self.assertIn("usb_c_dp_alt", result["missing_anchor_attributes"])
        self.assertTrue(result["examples"])

    def test_scenario_candidates_returns_confirmed_before_unknown(self):
        candidates = self.agent.scenario_candidates("laptop", "laptop_mouse", limit=10)
        ids = [offer.offer_id for offer in candidates]
        self.assertIn("mouse-usb", ids)
        self.assertIn("mouse-bt", ids)
        self.assertLess(ids.index("mouse-usb"), ids.index("mouse-bt"))

    def test_query_routing_selects_only_named_directions(self):
        mouse = self.agent.recommend("laptop", _parsed("мышь для ноутбука"))
        self.assertEqual(mouse.directions, ("laptop_mouse",))
        self.assertTrue(all(item.direction_id == "laptop_mouse" for item in mouse.items))
        bag = self.agent.recommend("laptop", _parsed("сумка для ноутбука"))
        self.assertEqual(bag.directions, ())
        self.assertEqual(self.ids(bag), ["bag"])

    def test_multiple_directions_are_ranked_and_deduplicated(self):
        result = self.agent.recommend("laptop", _parsed("мышь и клавиатура для ноутбука"))
        self.assertEqual(result.directions, ("laptop_mouse", "laptop_keyboard"))
        ids = self.ids(result)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("mouse-usb", ids)
        self.assertIn("keyboard-usb", ids)

    def test_vendor_diversity_caps_each_brand_at_two(self):
        # The synthetic catalog has two A4 mice, two Logitech mice and two Logitech keyboards.
        result = self.agent.recommend("laptop", directions=("laptop_mouse", "laptop_keyboard"))
        vendors = [item.offer.vendor for item in result.items]
        self.assertLessEqual(vendors.count("Logitech"), 2)
        self.assertLessEqual(vendors.count("A4"), 2)

    def test_legacy_bag_result_is_unchanged(self):
        result = self.agent.recommend("laptop")
        self.assertEqual(result.scenario_id, "laptop_bag")
        self.assertEqual(result.directions, ())
        self.assertEqual(self.ids(result), ["bag"])
        self.assertEqual(result.items[0].direction_id, "")


class _ScriptedProvider:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    def respond(self, **request):
        self.requests.append(request)
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def search_web(self, query, domains, **request):
        raise AssertionError("web search is not expected")


def _call(call_id, name, **arguments):
    return ProviderCall(call_id, name, json.dumps(arguments))


def _usage():
    return ProviderUsage(100, 0, 10, reported=True)


class UniversalLLMTests(TestCase):
    def setUp(self):
        self.catalog = _Catalog()
        self.addCleanup(self.catalog.close)
        self.rules = self.catalog.agent

    def agent(self, provider=None):
        settings = replace(LLMSettings(), enabled=True, max_web_searches=0)
        evidence = type("Evidence", (), {"active_facts": lambda self: (), "cached": lambda self, *_: None})()
        agent = LLMRecommendationAgent(None, self.catalog.path, self.rules, settings, evidence, provider)
        self.addCleanup(agent.close)
        return agent

    def test_llm_selects_new_direction_and_server_rechecks_every_candidate(self):
        provider = _ScriptedProvider(
            ProviderResponse(calls=(
                _call("dirs", "get_cross_sell_directions", offer_id="laptop"),
                _call("bag", "get_category_products", category_id="", scenario_id="laptop_bag", limit=5),
                _call("req", "get_compatibility_requirements", offer_id="laptop", scenario_id="laptop_mouse"),
            ), usage=_usage()),
            ProviderResponse(calls=(
                _call("batch", "check_compatibility_batch", scenario_id="laptop_mouse",
                      offer_ids=["mouse-usb", "mouse-ps2", "mouse-bt"]),
            ), usage=_usage()),
            ProviderResponse(text=json.dumps({"selections": [
                {"scenario_id": "laptop_bag", "candidate_id": "bag"},
                {"scenario_id": "laptop_mouse", "candidate_id": "mouse-usb"},
                {"scenario_id": "laptop_mouse", "candidate_id": "mouse-ps2"},
            ]}), usage=_usage()),
        )
        query = _parsed("подбери аксессуары для ноутбука")
        result = self.agent(provider).recommend("laptop", query)

        self.assertEqual(result.mode, "llm_agent")
        self.assertEqual(result.scenario_ids, ("laptop_bag", "laptop_mouse"))
        # LLM selections are hints: the server reruns the direction and keeps every confirmed match.
        self.assertEqual(set(item.offer.offer_id for item in result.items), {"bag", "mouse-usb", "mouse-c"})
        tools = [entry["tool"] for entry in result.tools]
        self.assertIn("get_cross_sell_directions", tools)
        self.assertIn("get_compatibility_requirements", tools)
        self.assertIn("check_compatibility_batch", tools)
        self.assertIn("agent_selection", tools)

    def test_provider_failure_preserves_confirmed_local_bag(self):
        provider = _ScriptedProvider(RuntimeError("provider down"))
        query = _parsed("подбери аксессуары для ноутбука")
        result = self.agent(provider).recommend("laptop", query)
        self.assertEqual(result.mode, "rule_based_fallback")
        self.assertEqual([item.offer.offer_id for item in result.items], ["bag"])
        self.assertIsNotNone(result.fallback_reason)

    def test_no_direction_planning_for_explicit_bag_query(self):
        provider = _ScriptedProvider()
        result = self.agent(provider).recommend("laptop", _parsed("сумка для ноутбука"))
        self.assertEqual(result.mode, "local")
        self.assertFalse(provider.requests)

    def test_unknown_direction_selection_is_discarded(self):
        provider = _ScriptedProvider(
            ProviderResponse(text=json.dumps({"selections": [
                {"scenario_id": "laptop_mouse", "candidate_id": "missing"},
            ]}), usage=_usage()),
        )
        query = _parsed("подбери аксессуары для ноутбука")
        result = self.agent(provider).recommend("laptop", query)
        self.assertEqual(result.mode, "llm_agent")
        self.assertEqual([item.offer.offer_id for item in result.items], ["bag"])
