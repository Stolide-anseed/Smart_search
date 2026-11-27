from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring

from softline_search.query import Condition, ParsedQuery
from softline_search.recommendation import RecommendationAgent


class RecommendationAgentTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "feed.xml"
        self.feed = Element("yml_catalog")
        self.shop = SubElement(self.feed, "shop")
        self.categories = SubElement(self.shop, "categories")
        self.offers = SubElement(self.shop, "offers")

    def offer(
        self, offer_id, category, name, *, vendor="Example", article="", group="", price="5000",
        available="true", fields=(), description="",
    ):
        offer = SubElement(self.offers, "offer", id=offer_id, group_id=group, available=available)
        for tag, value in (
            ("categoryId", category), ("name", name), ("vendor", vendor), ("vendorCode", article), ("price", price),
            ("currencyId", "RUR"), ("url", f"https://example.test/{offer_id}"), ("description", description),
        ):
            SubElement(offer, tag).text = value
        for key, value in fields:
            SubElement(offer, "param", name=key).text = value

    def agent(self):
        self.path.write_bytes(tostring(self.feed, encoding="utf-8"))
        return RecommendationAgent(self.path)

    def office(self):
        self.offer("office", "1145", "Microsoft Office для дома и бизнеса на 1 ПК",
                   description="Платформа: Windows • Поставка: цифровая")

    def endpoint(self, offer_id, devices="5", *, platform="Windows", group="", license_type="Полная версия",
                 organization="", customer="Юрлицо", additional_fields=()):
        count = f"{devices} шт." if devices.isdigit() else devices
        self.offer(
            offer_id, "650", "PRO32 Endpoint Security", group=group,
            fields=(("Тип организации", organization), ("Тип клиента", customer), *additional_fields),
            description=f"Платформа: {platform} • Тип лицензии: {license_type} • К-во узлов {count}",
        )

    def test_scenario_shortlist_prefers_smaller_sufficient_licenses(self):
        self.office()
        self.endpoint("a-larger", "25")
        self.endpoint("z-sufficient", "5")
        agent = self.agent()

        options = agent.scenario_candidates("office", "office_antivirus", limit=1, category_id="650")
        self.assertEqual([offer.offer_id for offer in options], ["z-sufficient"])
        self.assertEqual(agent.scenario_candidates("office", "office_antivirus", category_id="1145"), ())

    def test_selects_all_confirmed_battery_families_in_the_priority_tier(self):
        self.offer("ups", "82080", "ИБП APC", vendor="APC", group="own",
                   fields=tuple(("Совместимая батарея", f"RBC{num}") for num in (10, 20, 30, 40, 50, 60)))
        self.offer("a", "82014", "Батарея APC", vendor="APC", article="APCRBC10", group="single")
        self.offer("b", "82014", "Батарея APC", vendor="APC", article="RBC10")
        self.offer("c", "82014", "Батарея APC", vendor="APC", article="APCRBC20", group="single")
        self.offer("d", "82014", "Батарея APC", vendor="APC", article="APCRBC30", group="own")
        self.offer("e", "82014", "Батарея APC", vendor="APC", article="APCRBC40", price="0")
        self.offer("f", "82014", "Батарея APC", vendor="APC", article="APCRBC50")
        self.offer("g", "82014", "Батарея APC", vendor="APC", article="APCRBC60")
        self.offer("other_brand", "82014", "Батарея", article="RBC40")
        self.offer("partial", "82014", "Батарея APC", vendor="APC", article="RBC400")

        agent = self.agent()
        result = agent.recommend("ups")
        self.assertEqual(result.scenario_id, "ups_battery")
        self.assertEqual([item.offer.offer_id for item in result.items], ["a", "f", "g"])
        self.assertIn("RBC10", result.items[0].reason)

    def test_preserves_the_status_of_a_confirmed_fallback(self):
        self.offer("ups", "82080", "ИБП APC", vendor="APC",
                   fields=(("Совместимая батарея", "RBC141"),))
        self.offer("battery", "82014", "Батарея APC", vendor="APC", article="APCRBC141",
                   available="false", price="0")
        item, = self.agent().recommend("ups").items
        self.assertFalse(item.available)
        self.assertFalse(item.has_price)

    def test_validates_bag_dimensions_and_exact_model_matches(self):
        self.offer("laptop", "70824", "Ноутбук Lenovo", vendor="Lenovo", article="82X7004BPS")
        self.offer("curated", "83844", "Сумка HP", vendor="HP Inc.", article="A08JWAA")
        self.offer("measured", "83844", "Сумка",
                   fields=(("Размеры внутреннего отделения", "36 × 24 × 2 см"),))
        self.offer("thin", "83844", "Сумка",
                   fields=(("Внутренние размеры", "400 x 270 x 15 мм"),))
        self.offer("outside", "83844", "Сумка",
                   fields=(("Габариты", "400 x 270 x 30 мм"), ("Максимальная диагональ", "16")))
        self.offer("exact_model", "83844", "Сумка",
                   fields=(("Совместимые модели", "82X7004BPS; ANOTHER"),))
        self.offer("model_prefix", "83844", "Сумка", fields=(("Совместимые модели", "82X7004"),))
        self.offer("model_conflict", "83844", "Сумка",
                   fields=(("Совместимые модели", "82X7004BPS"), ("Внутренние размеры", "350 x 240 x 20 мм")))

        result = self.agent().recommend("laptop")
        self.assertEqual(result.scenario_id, "laptop_bag")
        self.assertEqual([item.offer.offer_id for item in result.items], ["exact_model", "curated", "measured"])
        self.assertIn("359,3 × 235 × 17,9 мм", result.items[1].reason)

    def test_applies_updated_software_filters_to_the_same_prepared_offer(self):
        self.office()
        self.offer("office_unknown_size", "1145", "Microsoft Office для бизнеса",
                   description="Платформа: Windows")
        self.endpoint("five", "5", group="endpoint")
        self.endpoint("ten", "10", group="endpoint")
        self.endpoint("versioned", "5", group="endpoint", platform="Windows 10/11")
        self.endpoint("linux", "5", platform="Linux")
        self.endpoint("server", "5", platform="Windows Server 2022")
        self.endpoint("range", "от 5 до 20")
        self.endpoint("conflicting_count", "10", additional_fields=(("К-во узлов", "5"),))
        self.endpoint("conflicting_platform", "10", additional_fields=(("Поддерживаемая ОС", "Linux"),))
        self.endpoint("migration", "10", license_type="Миграция")
        self.endpoint("academic", "10", organization="Образовательная")
        self.endpoint("noncommercial", "10", additional_fields=(("Тип клиента", "Некоммерческая организация"),))
        self.offer("nonendpoint", "650", "Антивирус для почтового сервера",
                   fields=(("Назначение", "Почтовые сервера"), ("Тип клиента", "Юрлицо")),
                   description="Платформа: Windows • Тип лицензии: Полная версия • К-во узлов 5 шт.")
        agent = self.agent()

        with patch("softline_search.recommendation.catalog_elements", side_effect=AssertionError("Repeated XML read")):
            first = agent.recommend("office")
            six_devices = (Condition("devices", "ge", Decimal(6), "6 рабочих мест"),)
            limited = agent.recommend("office", filters=six_devices)
            larger = agent.recommend("office_unknown_size", filters=six_devices)
            version = agent.recommend("office", filters=(Condition("platform", "eq", "windows 11", "Windows 11"),))
            bare_text = "Microsoft Office Windows 11"
            bare = agent.recommend("office", ParsedQuery(bare_text, bare_text, bare_text, (), ()))
            preference_text = "Microsoft Office желательно на 10 ПК"
            preferred = agent.recommend("office", ParsedQuery(
                preference_text, preference_text, preference_text,
                (Condition("devices", "ge", Decimal(10), "на 10 ПК", required=False),), (),
            ))
            conflict = agent.recommend("office", filters=(Condition("platform", "eq", "linux", "Linux"),))

        self.assertEqual(first.scenario_id, "office_antivirus")
        self.assertEqual([item.offer.offer_id for item in first.items], ["five"])
        self.assertEqual(limited.items, [])
        self.assertEqual(limited.reason_code, "conflicting_context")
        self.assertEqual([item.offer.offer_id for item in larger.items], ["ten"])
        self.assertEqual([item.offer.offer_id for item in version.items], ["versioned"])
        self.assertEqual([item.offer.offer_id for item in bare.items], ["versioned"])
        self.assertEqual([item.offer.offer_id for item in preferred.items], ["ten"])
        self.assertEqual(conflict.items, [])

    def test_rejects_conflicting_context_and_private_use(self):
        self.office()
        self.endpoint("five")
        agent = self.agent()
        query = ParsedQuery("офис под Windows", "офис под Windows", "офис под Windows",
                            (Condition("platform", "eq", "windows", "под Windows"),), ())
        private = ParsedQuery("офис для дома", "офис для дома", "офис для дома", (), ())

        self.assertEqual(agent.recommend("office", query, filters=(
            Condition("platform", "eq", "macos", "macOS"),
        )).items, [])
        self.assertEqual(agent.recommend("office", private).items, [])

    def test_prepares_all_bags_and_checks_axes_with_units_and_explicit_order(self):
        self.offer("laptop", "70824", "Ноутбук с размерами корпуса",
                   description="<p>Габариты корпуса (Ш×Г×В): 36×25×2 см</p>")
        self.offer("fit", "83844", "Сумка", fields=(("Внутренние размеры (В×Ш×Г)", "21×361×251 мм"),))
        for offer_id, size in (("width", "359×260×22 мм"), ("depth", "370×249×22 мм"),
                               ("thickness", "370×260×19 мм")):
            self.offer(offer_id, "83844", "Чехол", fields=(("Внутренние размеры", size),))
        self.offer("outer", "83844", "Сумка", description="Внешние размеры: 440×290×40 мм")
        self.offer("conflict", "83844", "Сумка", fields=(("Внутренние размеры", "380×260×23 мм"),),
                   description="Внутренние размеры: 400×270×24 мм")

        result = self.agent().recommend("laptop", debug=True)
        self.assertEqual([item.offer.offer_id for item in result.items], ["fit"])
        self.assertEqual(result.diagnostics["candidates"], 6)
        self.assertEqual(result.diagnostics["confirmed"], 1)
        self.assertEqual(result.diagnostics["unknown"], 2)
        self.assertEqual(result.diagnostics["rejected"], 3)
        self.assertEqual(result.diagnostics["reason_counts"],
                         {"incompatible_dimensions": 3, "missing_bag_dimensions": 1, "conflicting_facts": 1})
        proof = result.diagnostics["candidate_examples"]["dimensions_match"]["validation"]
        self.assertEqual(proof["facts"]["maximum_mm"], ["361", "251", "21"])

    def test_official_dimensions_resolve_conflicts_for_several_chassis_articles(self):
        for offer_id, article in (("first", "21SG0033GQ"), ("second", "21SG0034GQ"), ("unknown", "21SG0033")):
            self.offer(offer_id, "70824", "Ноутбук Lenovo ThinkBook 14 G8 IRL", vendor="Lenovo", article=article,
                       fields=(("Диагональ экрана", "16"), ("Габариты корпуса", "500×240×20 мм")))
        self.offer("bag", "83844", "Сумка HP", vendor="HP Inc.", article="A08JWAA")
        agent = self.agent()

        for offer_id in ("first", "second"):
            with self.subTest(offer_id=offer_id):
                result = agent.recommend(offer_id, debug=True)
                self.assertEqual([item.offer.offer_id for item in result.items], ["bag"])
                self.assertTrue(result.diagnostics["anchor_facts"]["diagonal"]["conflict"])
                self.assertEqual(result.diagnostics["anchor_facts"]["diagonal"]["value"], "14")
                self.assertTrue(any("psref.lenovo.com" in source for source in result.items[0].sources))
        self.assertEqual(agent.recommend("unknown").reason_code, "incompatible_dimensions")

    def test_parses_licensed_device_ranges_only_with_proven_purchase_units(self):
        self.offer("multi", "1145", "Microsoft Office для бизнеса", description="Платформа: Windows")
        self.endpoint("fixed", "10 устройств")
        self.endpoint("ambiguous", "10–14 устройств")
        self.endpoint("unit_range", "от 5 до 20 узлов", additional_fields=(
            ("Условия покупки", "Цена за 1 устройство"), ("Минимальный объем покупки", "Минимум 10 лицензий"),
        ))
        self.endpoint("named_users", "10 пользователей")
        self.endpoint("no_extra", "5 ПК")
        self.endpoint("extra", "5 ПК", additional_fields=(("Возможность дозакупки", "Да"),))
        agent = self.agent()
        count = (Condition("devices", "ge", Decimal(9), "9 рабочих мест"),)
        response = agent.recommend("multi", filters=count, debug=True)
        by_id = {item.offer.offer_id: item for item in response.items}
        self.assertEqual(set(by_id), {"fixed", "unit_range", "extra"})
        self.assertEqual(by_id["extra"].purchase_quantity, 2)
        self.assertEqual(by_id["unit_range"].purchase_quantity, 10)
        self.assertEqual(by_id["unit_range"].total_price, Decimal(50000))
        self.assertEqual(response.diagnostics["reason_counts"]["unknown_license_capacity"], 1)
        self.assertEqual(response.diagnostics["reason_counts"]["conflicting_facts"], 1)
        self.assertEqual(response.diagnostics["reason_counts"]["insufficient_devices"], 1)
        self.assertEqual(agent.recommend("multi", filters=(Condition("devices", "ge", Decimal(21), "21"),))
                         .items[0].offer.offer_id, "extra")

    def test_platform_constraints_distinguish_versions_server_linux_and_alternatives(self):
        self.offer("multi", "1145", "Microsoft Office для бизнеса на 5 ПК",
                   description="Платформа: Windows 10, Windows 11, Astra Linux")
        self.endpoint("all", platform="Win 10/11, Astra Linux")
        self.endpoint("limited", platform="Windows 10")
        self.endpoint("server", platform="Windows Server 2022, Astra Linux")
        self.endpoint("linux", platform="Windows 10/11, Linux")
        self.endpoint("negated", platform="Windows 10/11, Astra Linux не поддерживается")
        self.endpoint("contradiction", platform="Windows 10/11, Astra Linux",
                      additional_fields=(("Поддерживаемая ОС", "Windows 10/11, не поддерживает Astra Linux"),))
        agent = self.agent()
        def by_id(result):
            return {item.offer.offer_id for item in result.items}
        self.assertEqual(by_id(agent.recommend("multi")), {"all"})

        for phrase, expected in (
            ("офис под Windows 11", {"all", "linux", "negated", "contradiction"}),
            ("офис под Windows 10 или Astra Linux", {"all", "limited", "server", "linux", "negated", "contradiction"}),
            ("офис под Windows 10 и Astra Linux", {"all"}),
        ):
            with self.subTest(query=phrase):
                query = ParsedQuery(phrase, phrase, phrase, (), ())
                self.assertEqual(by_id(agent.recommend("multi", query)), expected)

    def test_routes_only_applicable_office_packages_and_rejects_missing_context(self):
        for offer_id, name in (
            ("family", "Microsoft 365 Family"), ("home", "OfficeSuite Personal"),
            ("mdm", "Управление устройствами MDM"), ("virtual", "Виртуальный офис"),
            ("mail", "Microsoft Exchange, почтовый сервер"), ("bundle", "Microsoft Office + антивирус"),
            ("infra", "Управление инфраструктурой"),
        ):
            self.offer(offer_id, "1145", name, fields=(("Платформа", "Windows"),))
        self.offer("unknown", "1145", "Microsoft Office для бизнеса", fields=(("Количество пользователей", "10"),))
        self.endpoint("valid")
        self.offer("monitor", "70824", "Example монитор для ноутбука")

        agent = self.agent()
        for offer_id in ("family", "home", "mdm", "virtual", "mail", "bundle", "infra", "monitor"):
            with self.subTest(offer_id=offer_id):
                response = agent.recommend(offer_id, debug=True)
                self.assertIsNone(response.scenario_id)
                self.assertEqual(response.reason_code, "unsupported_scenario")
                self.assertFalse(response.diagnostics["routing"]["applicable"])
        response = agent.recommend("unknown")
        self.assertEqual(set(response.missing_context), {"platform", "devices"})
        self.assertEqual(response.status, "UNKNOWN")

    def test_requires_an_os_if_a_crossplatform_package_has_an_incomplete_platform_list(self):
        self.offer("office", "1145", "MobiOffice Crossplatform на 1 ПК",
                   description="Платформа: Windows")
        self.endpoint("confirmed")
        self.offer("unit_conflict", "650", "PRO32 Endpoint Security на 5 пользователей",
                   description="Платформа: Windows • Тип лицензии: Полная версия • К-во узлов 5 шт. • Тип клиента: Юрлицо")
        agent = self.agent()

        self.assertEqual(agent.recommend("office").missing_context, ("platform",))
        response = agent.recommend("office", filters=(Condition("platform", "eq", "windows", "Windows"),), debug=True)
        self.assertEqual([item.offer.offer_id for item in response.items], ["confirmed"])
        self.assertEqual(response.diagnostics["reason_counts"], {"conflicting_facts": 1})

    def test_checks_a_documented_desktop_limit_without_assuming_the_requested_device_count(self):
        self.offer("multi", "1145", "MobiOffice Multi-User (6 users) Crossplatform",
                   vendor="MobiSystems", article="MOMU1", fields=(("Платформа", "Windows"),))
        self.endpoint("large", "10")
        agent = self.agent()
        windows = Condition("platform", "eq", "windows", "Windows")

        self.assertEqual(agent.recommend("multi", filters=(windows,)).missing_context, ("devices",))
        response = agent.recommend("multi", filters=(windows, Condition("devices", "ge", Decimal(10), "10")),
                                   debug=True)
        self.assertEqual(response.reason_code, "conflicting_context")
        self.assertEqual(response.diagnostics["decision"]["facts"]["desktop_limit"], 6)

    def test_ups_tracks_missing_articles_and_keeps_only_commercially_eligible_batteries(self):
        self.offer("ups", "82080", "ИБП APC", vendor="APC",
                   description="<p>Совместимые батареи: APCRBC141 и RBC210</p>")
        self.offer("none", "82080", "ИБП APC", vendor="APC", fields=(("Сменная батарея", "RBC111"),))
        self.offer("other_ups", "82080", "ИБП Example", fields=(("Сменная батарея", "RBC141"),))
        self.offer("prefix", "82014", "Батарея APC", vendor="APC", article="RBC1410")
        self.offer("other_brand", "82014", "Батарея Example", article="RBC141")
        self.offer("reserve", "82014", "Батарея APC", vendor="APC", article="RBC141", available="false", price="0")
        agent = self.agent()

        self.assertEqual([item.offer.offer_id for item in agent.recommend("ups").items], ["reserve"])
        self.assertEqual(agent.recommend("ups", debug=True).diagnostics["missing_battery_articles"], ["RBC210"])
        self.assertEqual(agent.recommend("none").reason_code, "battery_not_in_catalog")
        self.assertIsNone(agent.recommend("other_ups").scenario_id)

    def test_prioritizes_the_selected_product_category(self):
        SubElement(self.categories, "category", id="display").text = "Мониторы"
        self.offer("monitor", "display", "Монитор для ноутбука")
        self.offer("unknown_ups", "misc", "Источник APC", vendor="APC",
                   fields=(("Совместимая батарея", "RBC141"),))
        self.offer("battery", "82014", "Батарея APC", vendor="APC", article="APCRBC141")
        request = ParsedQuery("ибп", "ибп", "ибп", (Condition("product_type", "eq", "ups", "ИБП"),), ())
        agent = self.agent()

        self.assertIsNone(agent.recommend("monitor", request).scenario_id)
        self.assertEqual([item.offer.offer_id for item in agent.recommend("unknown_ups", request).items], ["battery"])
        with self.assertRaises(KeyError):
            agent.recommend("missing")
    def test_asus_identity_is_available_without_xml_model_and_requires_exact_sku_binding(self):
        for identifier, article, model in (
            ("zenbook", "90NB14W1-M01Z50", "UX3405CA-ST1336"),
            ("other_zenbook", "A-200", "UX3405MA-PP101"),
            ("unknown_zenbook", "A-201", "UX3405CA-ST1336"),
            ("vivobook", "90NB10N3-M02KZ0", "X1605VA-MB2106"),
            ("expertbook", "90NX0801-M00EU0", "B1503CVA"),
            ("tuf", "A-202", "FA808UM-S8050"),
        ):
            self.offer(identifier, "70824", f"Ноутбук ASUS {model}", vendor="ASUS", article=article)
        agent = self.agent()
        for identifier, expected, verified in (
            ("zenbook", "UX3405CA", True), ("other_zenbook", "UX3405MA", False),
            ("unknown_zenbook", "UX3405CA", False), ("vivobook", "X1605VA", True),
            ("expertbook", "B1503CVA", True), ("tuf", "FA808UM", False),
        ):
            identity = agent.product_identity(identifier)
            with self.subTest(identifier=identifier):
                self.assertEqual(identity.offer_id, identifier)
                self.assertEqual(identity.chassis_model, expected)
                self.assertEqual(bool(identity.chassis_source), verified)
                facts = agent.product_details(identifier)["facts"]["dimensions_mm"]
                self.assertEqual(facts["value"] is not None, verified)

    def test_endpoint_commercial_rank_checks_quantity_buyer_and_minimum(self):
        self.offer("office", "1145", "Microsoft Office для бизнеса", description="Платформа: Windows")
        common = (("Тип клиента", "Юрлицо"), ("Срок лицензии", "12 месяцев"))
        for identifier, edition, nodes, price in (
            ("advanced", "Advanced", 5, "15750"),
            ("standard", "Standard", 36, "68400"),
        ):
            self.offer(identifier, "650", f"PRO32 Endpoint Security {edition} на {nodes} узлов",
                       vendor="PRO32", price=price, group=identifier, fields=common,
                       description=f"Платформа: Windows • Тип лицензии: Полная версия • К-во узлов {nodes} шт.")
        self.offer("base", "650", "PRO32 Office Security Base на 1 устройство", vendor="PRO32", price="3790",
                   article="PRO32-PUS-EKEY-3Y-1-NS-RU-AKS", fields=common,
                   description="Платформа: Windows • Тип лицензии: Полная версия • К-во узлов 1 шт.")
        self.endpoint("range-no-minimum", "от 1 до 40 узлов", additional_fields=(
            ("Единица цены", "за узел"),
        ))
        self.endpoint("renewal", "1", license_type="Продление")
        self.endpoint("migration", "1", license_type="Миграция")
        agent = self.agent()
        for devices, expected in ((1, "advanced"), (5, "advanced"), (36, "standard")):
            with self.subTest(devices=devices):
                filters = (Condition("devices", "ge", Decimal(devices), str(devices)),)
                result = agent.recommend("office", filters=filters, debug=True)
                self.assertEqual([item.offer.offer_id for item in result.items], [expected])
                self.assertIn("юридическое лицо", result.items[0].reason)
                self.assertEqual(result.diagnostics["reason_counts"]["unknown_minimum_purchase"], 2)
                self.assertEqual(result.diagnostics["reason_counts"]["incompatible_license_type"], 2)
                b2b = agent.recommend("office", filters=(*filters, Condition("customer_type", "eq", "b2b", "b2b")))
                self.assertTrue(b2b.items)
                b2c = agent.recommend("office", filters=(*filters, Condition("customer_type", "eq", "b2c", "b2c")))
                self.assertEqual(b2c.reason_code, "restricted_customer")
                self.assertFalse(b2c.items)
