from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from openpyxl import Workbook

from softline_search.encoder import EMBEDDING_SIZE
from softline_search.evaluation import evaluate, metrics
from softline_search.index import SearchIndex, build_index
from softline_search.query import Condition
from softline_search.query_catalog import SOFTWARE_CATEGORY_ID


class QuerySearchTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cls.catalog = root / "catalog.xml"
        cls.index_path = root / "index.sqlite3"

        data = Element("yml_catalog")
        shop = SubElement(data, "shop")
        categories = SubElement(shop, "categories")
        SubElement(categories, "category", id="872").text = "Офисные программы"
        SubElement(categories, "category", id="work", parentId="872").text = "Программы для работы"
        SubElement(categories, "category", id="cloud").text = "Облачная инфраструктура"
        for category, label in (
            ("ram", "Оперативная память"), ("laptop", "Ноутбуки"),
            ("other", "Оптические приводы"), ("shredder", "Шредеры"),
            ("office", "Офисные пакеты программ Microsoft Office"), ("monitor", "Мониторы"),
            ("gpu", "Видеокарты"), ("lenovo", "Ноутбуки Lenovo"),
        ):
            entry = SubElement(categories, "category", id=category)
            entry.text = label
            if category == "office":
                entry.set("parentId", "work")
        offers = SubElement(shop, "offers")

        def add(offer_id, name, category, price, vendor="Example", attrs=(), description="", group="", code=""):
            offer = SubElement(offers, "offer", id=offer_id, available="true", group_id=group)
            for tag, value in (
                ("name", name), ("categoryId", category), ("price", price), ("vendor", vendor),
                ("currencyId", "RUR"), ("description", description), ("vendorCode", code),
            ):
                SubElement(offer, tag).text = value
            for key, value in attrs:
                SubElement(offer, "param", name=key).text = value

        laptop_memory = (("Ёмкость одного модуля", "8"), ("Назначение памяти", "для ноутбука"))
        add("ddr4", "Оперативная память DDR4 8 ГБ", "ram", "8000", attrs=(*laptop_memory, ("Тип памяти", "DDR4")))
        add("unknown", "Оперативная память для ноутбука 8 ГБ", "ram", "7500", attrs=laptop_memory)
        add("ddr3", "Оперативная память DDR3L 8 ГБ", "ram", "6500", attrs=(*laptop_memory, ("Тип памяти", "DDR3L")))
        add("memory16", "Оперативная память DDR4 16GB (1Gb*8)", "ram", "9500", attrs=(("Назначение памяти", "для ноутбука"),))
        add("notebook", "Ноутбук ASUS 16 ГБ", "laptop", "60000", vendor="ASUS")
        add("tablet", "Планшет Example P1", "laptop", "30000")
        add("design_gpu", "Видеокарта ModelOne", "gpu", "24000", attrs=(("Назначение", "проектирование"),))
        add("design_copy", "Видеокарта ModelOne", "gpu", "24500", attrs=(("Назначение", "проектирование"),))
        add("gaming_gpu", "Видеокарта ModelTwo", "gpu", "25000", attrs=(("Назначение", "игры"),))
        add("design_alternative", "Видеокарта ModelThree", "gpu", "26000", attrs=(("Назначение", "проектирование"),))
        for offer_id, model, processor in (
            ("think_16", "Thinkbook 16 G8", "Intel Core 7 240H"),
            ("think_16_other", "Thinkbook 16 G8", "Intel Core 5 210H"),
            ("think_14", "Thinkbook 14 G8", "Intel Core 7 240H"),
        ):
            add(offer_id, f"Ноутбук Lenovo {model} {processor}", "lenovo", "95000", vendor="Lenovo",
                attrs=(("Модель процессора", processor), ("Серия ноутбука", "ThinkBook")), code=offer_id)
        add("drive", "Привод ASUS", "other", "4000", vendor="ASUS", description="Ноутбук ASUS")
        add("office12", "Microsoft Office 2024 на 12 ПК, бессрочная", "office", "18000", vendor="Microsoft",
            code="EP2-06605")
        add("office1", "Microsoft Office 2024 на 1 ПК, бессрочная", "office", "4000", vendor="Microsoft")
        add("office_subscription", "Microsoft Office 2024 на 20 ПК, подписка", "office", "5000", vendor="Microsoft")
        add("office_unknown", "Microsoft Office 2024 Standard, бессрочная", "office", "9000", vendor="Microsoft")
        add("project", "Microsoft Project Standard 2024", "office", "12000", vendor="Microsoft")
        add("office_home", "Microsoft Office для дома 2024 на 1 ПК, бессрочная", "office", "15000", vendor="Microsoft")
        add("cloud_service", "Облачная инфраструктура", "cloud", "9000", attrs=(("Тип продукта", "Софт"),))
        add("hardware", "Оборудование Example", "office", "9000", attrs=(("Тип продукта", "Оборудование"),))
        add("shredder", "Шредер Example", "shredder", "13000")
        add("m1", "Монитор Example A1", "monitor", "9000", group="m1")
        add("m1_variant", "Монитор Example A1, RGB", "monitor", "11000", group="m1")
        add("m2_white", "Монитор Example B2", "monitor", "9000", code="B2/WHITE")
        add("m2_black", "Монитор Example B2", "monitor", "10000", code="B2/BLACK")
        add("m3", "Монитор Example C3", "monitor", "15000", code="C3")
        add("m3_duplicate", "Монитор C3 Example", "monitor", "16000", code="c3")
        add("m4", "Монитор Example D4", "monitor", "15000", code="D4")
        add("m5", "Монитор Example E5", "monitor", "17000", code="E5")
        add("m1_reserve", "Монитор Example A1, датчик", "monitor", "0", group="m1")
        for code in ("F6", "G7", "H8"):
            add(code, f"Монитор Example {code}", "monitor", "0", code=code)
        cls.catalog.write_bytes(tostring(data, encoding="utf-8"))

        vectors = np.full((len(offers), EMBEDDING_SIZE), 1 / 32, dtype="<f2")
        with patch("softline_search.index._catalog_embeddings", return_value=vectors):
            build_index(cls.catalog, cls.index_path, progress=lambda _: None)

    def test_layout_is_corrected_using_catalog_words(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            parsed = index.parse_query("ihtlths", self.catalog)
            self.assertEqual(parsed.normalized, "шредеры")
            self.assertEqual(parsed.changes[0].kind, "раскладка")

    def test_budget_and_product_type_apply_to_direct_and_reserve_results(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            results = index.search_checked("ноутбук асус до полтинника", self.catalog, mode="bm25")
            self.assertEqual(results.query.search_text, "ноутбук ASUS")
            self.assertEqual(results.hits, [])
            self.assertEqual(results.reserve, [])

    def test_known_conflict_is_excluded_and_unknown_goes_to_reserve(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            result = index.search_checked("память для ноута ддр4 на восьмерку", self.catalog, mode="bm25")
            self.assertEqual([hit.offer.offer_id for hit in result.hits], ["ddr4"])
            self.assertEqual([hit.offer.offer_id for hit in result.reserve], ["unknown"])
            self.assertEqual(result.unconfirmed, {"unknown": ("DDR4",)})

    def test_license_count_is_a_minimum_and_subscription_is_excluded(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            result = index.search_checked("Microsoft Office 2024 на 10 ПК без подписки", self.catalog, mode="bm25")
            self.assertEqual([hit.offer.offer_id for hit in result.hits], ["office12"])
            self.assertEqual([hit.offer.offer_id for hit in result.reserve], ["office_unknown"])

    def test_currency_is_not_confused_with_rubles_and_power_is_not_a_price(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            parsed = index.parse_query("ноутбук ASUS до 1000 $", self.catalog)
            self.assertIn("price_usd", {condition.field for condition in parsed.conditions})
            result = index.search_checked("ноутбук ASUS до 1000 $", self.catalog, mode="bm25")
            self.assertEqual(result.hits, [])
            self.assertEqual([hit.offer.offer_id for hit in result.reserve], ["notebook"])
            power = index.parse_query("блок питания до 1200 Вт", self.catalog)
            self.assertEqual({condition.field for condition in power.conditions}, {"product_type", "power_w"})

    def test_checked_search_selects_four_families_and_deduplicates_articles(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            offers = {offer.offer_id: offer for offer in index.offers}
            self.assertEqual(index.family_key(offers["m1"]), index.family_key(offers["m1_variant"]))
            self.assertEqual(index.family_key(offers["m2_white"]), index.family_key(offers["m2_black"]))
            self.assertEqual(index.family_key(offers["m3"]), index.family_key(offers["m3_duplicate"]))
            result = index.search_checked("монитор Example до 20 тыс. ₽", self.catalog, mode="bm25")
            self.assertEqual(len(result.hits), 4)
            self.assertEqual(len({index.family_key(hit.offer) for hit in result.hits}), 4)
            self.assertEqual(result.reserve, [])

    def test_reserve_fills_only_free_family_slots(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            result = index.search_checked("монитор Example до 10 тыс. ₽", self.catalog, mode="bm25")
            self.assertEqual(len(result.hits), 2)
            self.assertEqual(len(result.reserve), 2)
            self.assertEqual(len({index.family_key(hit.offer) for hit in result.hits + result.reserve}), 4)
            self.assertNotIn("m1_reserve", result.unconfirmed)

    def test_explicit_price_filter_keeps_eligible_families_and_excludes_unknown_prices(self) -> None:
        price = Condition("price_rub", "le", Decimal(9000), "До 9000 ₽")
        with closing(SearchIndex(self.index_path)) as index:
            rows = {offer.offer_id: row for row, offer in enumerate(index.offers)}
            order = ("m1_variant", "m2_white", "m1", "m1_reserve", "F6", "m3")
            ranks = [(rows[offer_id], 1.0) for offer_id in order]
            with patch.object(index, "_rankings", return_value=([], ranks)):
                result = index.search_checked(
                    "монитор Example", self.catalog, mode="bm25", filters=(price,)
                )
            self.assertEqual([hit.offer.offer_id for hit in result.hits], ["m2_white", "m1"])
            self.assertEqual(result.reserve, [])

    def test_explicit_filters_and_query_conditions_must_both_match(self) -> None:
        laptop = Condition("category_id", "eq", "laptop", "Категория — Ноутбуки")
        with closing(SearchIndex(self.index_path)) as index:
            result = index.search_checked("монитор Example", self.catalog, mode="bm25", filters=(laptop,))
            self.assertEqual(result.hits, [])
            self.assertEqual(result.reserve, [])

    def test_software_filter_keeps_descendant_offers_before_selecting_top_families(self) -> None:
        software = Condition("software_category", "eq", SOFTWARE_CATEGORY_ID, "Категория — ПО")
        with closing(SearchIndex(self.index_path)) as index:
            rows = {offer.offer_id: row for row, offer in enumerate(index.offers)}
            order = ("m1", "m2_white", "m3", "m4", "cloud_service", "hardware", "office12", "office1")
            ranks = [(rows[offer_id], 1.0) for offer_id in order]
            with patch.object(index, "_rankings", return_value=([], ranks)):
                baseline = index.search_checked("решение", self.catalog, mode="bm25")
                result = index.search_checked("решение", self.catalog, mode="bm25", filters=(software,))
            self.assertEqual([hit.offer.offer_id for hit in baseline.hits], list(order[:4]))
            self.assertEqual([hit.offer.offer_id for hit in result.hits], ["office12", "office1"])
            self.assertEqual(result.reserve, [])

    def test_prepared_catalog_keeps_text_only_unknowns_in_reserve(self) -> None:
        price = Condition("price_rub", "le", Decimal(9000), "До 9000 ₽")
        with closing(SearchIndex(self.index_path)) as index:
            index.prepare_query_catalog(self.catalog)
            with patch("softline_search.query_catalog.catalog_elements", side_effect=AssertionError("XML reread")):
                result = index.search_checked(
                    "память для ноута ддр4 на восьмерку", self.catalog, mode="bm25", filters=(price,)
                )
            self.assertEqual([hit.offer.offer_id for hit in result.hits], ["ddr4"])
            self.assertEqual([hit.offer.offer_id for hit in result.reserve], ["unknown"])
            self.assertEqual(result.unconfirmed, {"unknown": ("DDR4",)})

    def test_evaluation_matches_a_suitable_variant_of_the_target_family(self) -> None:
        path = self.catalog.with_name("family_eval.xlsx")
        workbook = Workbook()
        workbook.active.append(["query", "target_url", "target_type", "target_id"])
        workbook.active.append(["монитор Example до 10 тыс. ₽", "https://example.test/m1", "offer", "m1_variant"])
        workbook.save(path)
        workbook.close()
        with closing(SearchIndex(self.index_path)) as index:
            rank = [(row, 1.0) for row, offer in enumerate(index.offers) if offer.offer_id == "m1"]
            with patch.object(index, "_rankings", return_value=(rank, rank)):
                rows = evaluate(index, path, catalog_path=self.catalog)
            scores = metrics(rows)["hybrid"]
            self.assertEqual(scores["HitRate@4"], 1)
            self.assertEqual(scores["Recall@4"], 1)

    def test_lexical_expansion_respects_price_and_reports_shortage(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            result = index.search_checked("мониторами до 10 тыс. ₽", self.catalog, mode="bm25")
            self.assertEqual({hit.offer.offer_id for hit in result.hits}, {"m1", "m2_white"})
            self.assertEqual(set(result.alternatives), {"m1", "m2_white"})
            self.assertEqual(result.shortage_reason, "insufficient_confirmed_families")
            uncertain = index.search_checked("мониторами с 1920x1080 до 10 тыс. ₽", self.catalog, mode="bm25")
            self.assertFalse(uncertain.alternatives)

    def test_similar_uses_stored_vectors_and_preserves_budget_and_families(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            with patch("softline_search.index.E5Encoder", side_effect=AssertionError("E5 loaded for similar")):
                alternatives = index.similar("m1", self.catalog, "монитор до 15 тыс. ₽", limit=10)
            self.assertEqual(len(alternatives), 3)
            self.assertEqual(len({index.family_key(item.offer) for item in alternatives}), 3)
            self.assertTrue(all(index.family_key(item.offer) != ("group", "m1") for item in alternatives))
            self.assertTrue(all(item.offer.category_id == "monitor" and 0 < Decimal(item.offer.price) <= 15000
                                for item in alternatives))
            self.assertTrue(any(item.label == "Цена" for hit in alternatives for item in hit.differences))
            with self.assertRaises(KeyError):
                index.similar("no-such-id", self.catalog)

    def test_similar_checks_type_and_verified_purpose_within_category(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            self.assertFalse(index.similar("notebook", self.catalog))
            self.assertEqual(
                [hit.offer.offer_id for hit in index.similar("design_gpu", self.catalog)], ["design_alternative"]
            )
            self.assertEqual([hit.offer.offer_id for hit in index.similar("think_16", self.catalog)], ["think_14"])

    def test_similar_office_excludes_standalone_apps_and_home_only_licenses(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            offers = index.similar("office12", self.catalog)
            self.assertEqual({hit.offer.offer_id for hit in offers}, {"office1", "office_subscription", "office_unknown"})

    def test_similar_ignores_selected_sku_and_keeps_explicit_license_and_budget(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            offers = index.similar("office12", self.catalog, "EP2-06605 до 6000 ₽ без подписки")
            self.assertEqual([hit.offer.offer_id for hit in offers], ["office1"])

    def test_exact_article_keeps_matches_and_adds_separately_marked_analogues(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            query = "EP2-06605 без подписки до 20 тыс. ₽"
            result = index.search_checked(query, self.catalog, mode="bm25")
            self.assertEqual([hit.offer.offer_id for hit in result.hits], ["office12"])
            self.assertEqual({hit.offer.offer_id for hit in result.analogues}, {"office1", "office_unknown"})
            self.assertEqual(result.reserve, [])
            constrained = index.search_checked("EP2-06605 до 9 тыс. ₽", self.catalog, mode="bm25")
            self.assertEqual(constrained.hits, [])
            self.assertEqual(constrained.analogues, [])

    def test_suggestions_reuse_catalog_prefixes_and_normalizer_without_encoder(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            with patch("softline_search.index.E5Encoder", side_effect=AssertionError("E5 loaded for suggest")):
                self.assertFalse(index.suggest("м", self.catalog))
                self.assertIn("Шредеры", index.suggest("ihtl", self.catalog))
                slang = index.suggest("моник", self.catalog)
                self.assertTrue(slang)
                self.assertTrue(all("монитор" in text.casefold() for text in slang))
                self.assertIn("Монитор Example B2", index.suggest("b2/w", self.catalog))
                self.assertTrue(index.suggest("thinkb", self.catalog))
                self.assertLessEqual(len(index.suggest("мон", self.catalog, limit=100)), 5)

    def test_shared_index_allows_search_and_similar_from_request_threads(self) -> None:
        with closing(SearchIndex(self.index_path)) as index, ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                executor.submit(index.search_checked, "монитор", self.catalog, mode="bm25"),
                executor.submit(index.similar, "m1", self.catalog, "монитор до 15 тыс. ₽"),
                executor.submit(index.similar, "design_gpu", self.catalog),
            ]
            self.assertEqual(len(tasks[0].result(timeout=5).hits), 4)
            self.assertEqual(len(tasks[1].result(timeout=5)), 3)
            self.assertEqual(len(tasks[2].result(timeout=5)), 1)

    def test_agent_retrieval_scopes_each_hybrid_ranking(self) -> None:
        with closing(SearchIndex(self.index_path)) as index:
            ranking = [(row, 1.0) for row in range(len(index.offers))]
            with patch.object(index, "_rankings", return_value=(ranking, list(reversed(ranking)))) as ranks:
                hits = index.search_candidates("монитор", category_ids=("monitor",), limit=3)
                missing = index.search_candidates("монитор", category_ids=("unknown",), limit=3)
            self.assertEqual(len(hits), 3)
            self.assertTrue(all(hit.offer.category_id == "monitor" for hit in hits))
            self.assertFalse(missing)
            ranks.assert_called_once_with("монитор", "hybrid")

    def test_new_metrics_include_unmapped_queries_and_measure_reserve(self) -> None:
        path = self.catalog.with_name("unmapped_eval.xlsx")
        workbook = Workbook()
        workbook.active.append(["query", "target_url", "target_type", "target_id"])
        workbook.active.append(["монитор до 10 тыс. ₽", "https://example.test/unavailable", "offer", "unavailable"])
        workbook.save(path)
        workbook.close()
        with closing(SearchIndex(self.index_path)) as index:
            order = {"m1", "m2_white", "F6", "G7"}
            ranking = [(row, 1.0) for row, offer in enumerate(index.offers) if offer.offer_id in order]
            with patch.object(index, "_rankings", return_value=(ranking, ranking)):
                rows = evaluate(index, path, catalog_path=self.catalog)
            scores = metrics(rows)["hybrid"]
            self.assertEqual(scores["FourProductRate"], 0.0)
            self.assertEqual(scores["FourProductRateWithAlternatives"], 0.0)
            self.assertEqual(scores["FourDisplayedProductRate"], 1.0)
            self.assertEqual(scores["ConstraintSatisfactionRate"], 0.5)
            self.assertGreaterEqual(scores["p95_ms"], 0)
            self.assertEqual(rows[0].search_observations["hybrid"].shortage_reason, "unconfirmed_requirements")
            self.assertNotIn("HitRate@4", scores)
