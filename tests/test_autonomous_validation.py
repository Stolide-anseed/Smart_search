from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring

from softline_search.autonomous_validation import validate_cross_sell
from softline_search.query_catalog import QueryCatalog
from softline_search.recommendation import RecommendationAgent
from softline_search.recommendation_facts import Status, Validation


class AutonomousValidationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.xml"
        feed = Element("yml_catalog")
        shop = SubElement(feed, "shop")
        categories = SubElement(shop, "categories")
        for identifier, name in (
            ("70824", "Ноутбуки"), ("83844", "Сумки и чехлы для ноутбуков"),
            ("80843", "Мониторы"), ("printer", "Принтеры и расходные материалы"),
            ("apps", "Специализированное ПО"), ("82080", "Источники бесперебойного питания"),
        ):
            SubElement(categories, "category", id=identifier).text = name
        offers = SubElement(shop, "offers")
        entries = (
            ("laptop", "70824", "Ноутбук Lenovo", "82X7004BPS", "", ()),
            ("ups", "82080", "ИБП Example 1", "UPS-1", "", ()),
            ("fit_bag", "83844", "Сумка для ноутбука", "", "", (
                ("Размеры внутреннего отделения", "400 × 270 × 25 мм"),
            )),
            ("small_bag", "83844", "Сумка для ноутбука", "", "", (
                ("Размеры внутреннего отделения", "350 × 240 × 20 мм"),
            )),
            ("unknown_bag", "83844", "Сумка для ноутбука", "", "", ()),
            ("monitor", "80843", "Монитор Example X1", "X1", "display", ()),
            ("monitor_variant", "80843", "Монитор Example X1 Pro", "X1-Pro", "display", ()),
            ("other_monitor", "80843", "Монитор Example X2", "X2", "", ()),
            ("mount", "80843", "Кронштейн для монитора", "VESA-1", "", ()),
            ("printer", "printer", "Принтер Example Print 100", "PRINT-100", "", ()),
            ("cartridge", "printer", "Картридж для принтера Example Print 100", "T-100", "", ()),
            ("database", "apps", "Данные Example Database", "DB-1", "", (
                ("Тип продукта", "Софт"), ("Тип программного обеспечения", "СУБД"),
            )),
            ("other_db", "apps", "Работа с данными Other DB", "DB-2", "", (
                ("Тип продукта", "Софт"), ("Тип программного обеспечения", "СУБД"),
            )),
            ("backup", "apps", "Архив данных Example", "ARC-1", "", (
                ("Тип продукта", "Софт"), ("Тип программного обеспечения", "Резервное копирование"),
            )),
        )
        for identifier, category, name, article, group, attrs in entries:
            offer = SubElement(offers, "offer", id=identifier, available="true", group_id=group)
            for key, value in (
                ("categoryId", category), ("name", name), ("vendorCode", article),
                ("vendor", "Lenovo" if identifier == "laptop" else "Example"),
                ("price", "100"), ("currencyId", "RUR"), ("url", f"https://example.test/{identifier}"),
            ):
                SubElement(offer, key).text = value
            for attr, value in attrs:
                SubElement(offer, "param", name=attr).text = value
        catalog_path.write_bytes(tostring(feed, encoding="utf-8"))
        cls.rules = RecommendationAgent(catalog_path)
        cls.offers = {identifier: cls.rules.get_offer(identifier) for identifier, *_ in entries}
        cls.catalog = QueryCatalog(catalog_path, tuple(cls.offers.values()))

    def validate(self, anchor: str, candidate: str):
        return validate_cross_sell(self.offers[anchor], self.offers[candidate], self.catalog, self.rules)

    def test_novel_direction_allows_consumable_in_shared_catalogue_category(self):
        self.assertFalse(self.rules.applicable_scenarios("printer"))
        self.assertFalse(self.rules.applicable_directions("printer"))
        result = self.validate("printer", "cartridge")
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(result.reason_code, "partial_catalog_check")
        self.assertEqual(result.missing, ("compatibility",))
        self.assertIn("уточните совместимость", result.reason)

    def test_rejects_declared_printing_consumable_for_unrelated_device(self):
        for device in ("laptop", "ups", "monitor"):
            with self.subTest(device=device):
                result = self.validate(device, "cartridge")
                self.assertEqual(result.status, Status.REJECTED)
                self.assertEqual(result.reason_code, "different_target_product")

    def test_uses_specific_software_type_to_detect_a_substitute(self):
        self.assertEqual(self.validate("database", "other_db").reason_code, "substitute_product")
        self.assertEqual(self.validate("database", "backup").status, Status.UNKNOWN)

    def test_recognises_accessory_under_primary_product_category(self):
        self.assertEqual(self.catalog.product_types(self.offers["mount"]), frozenset({"monitor"}))
        self.assertEqual(self.validate("monitor", "mount").status, Status.UNKNOWN)
        alternate = self.validate("monitor", "other_monitor")
        self.assertEqual(alternate.status, Status.REJECTED)
        self.assertEqual(alternate.reason_code, "substitute_product")

    def test_applies_known_compatibility_and_keeps_missing_specs_provisional(self):
        confirmed = self.validate("laptop", "fit_bag")
        self.assertEqual(confirmed.status, Status.CONFIRMED)
        self.assertEqual(confirmed.reason_code, "dimensions_match")

        conflict = self.validate("laptop", "small_bag")
        self.assertEqual(conflict.status, Status.REJECTED)
        self.assertEqual(conflict.reason_code, "incompatible_dimensions")

        partial = self.validate("laptop", "unknown_bag")
        self.assertEqual(partial.status, Status.UNKNOWN)
        self.assertEqual(partial.reason_code, "missing_bag_dimensions")
        self.assertIn("bag_dimensions", partial.missing)
        self.assertTrue(partial.reason)

    def test_conflict_takes_priority_over_another_matching_check(self):
        with (
            patch.object(self.rules, "applicable_directions", return_value=("match", "conflict")),
            patch.object(self.rules, "is_candidate", return_value=True),
            patch.object(self.rules, "check_compatibility", side_effect=(
                Validation(Status.CONFIRMED, "interface_match"),
                Validation(Status.REJECTED, "incompatible_platform"),
            )),
        ):
            result = self.validate("monitor", "mount")
        self.assertEqual(result.status, Status.REJECTED)
        self.assertEqual(result.reason_code, "incompatible_platform")

    def test_rejects_same_item_family_and_catalogue_miss(self):
        self.assertEqual(self.validate("monitor", "monitor").reason_code, "same_offer")
        self.assertEqual(self.validate("monitor", "monitor_variant").reason_code, "same_product_family")
        absent = replace(self.offers["mount"], offer_id="invented")
        result = validate_cross_sell(self.offers["monitor"], absent, self.catalog, self.rules)
        self.assertEqual(result.reason_code, "unknown_offer")
        self.assertEqual(result.status, Status.REJECTED)
