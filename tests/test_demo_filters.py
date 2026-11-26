from decimal import Decimal
from unittest import TestCase
from urllib.parse import parse_qs

from softline_search.demo import _BadRequest, _filters


class DemoFiltersTests(TestCase):
    def test_valid_filter_parameters_become_typed_conditions(self) -> None:
        parameters = {
            "category_id": ["1145"], "vendor": ["«Microsoft»"], "price_to": ["1000,50"],
            "platform": ["Windows 11"], "devices": ["5"],
        }
        conditions = _filters(parameters, {"1145": "Офисные пакеты"}, {"«Microsoft»"})
        self.assertEqual(
            [(condition.field, condition.operator, condition.value) for condition in conditions],
            [
                ("category_id", "eq", "1145"), ("brand", "eq", "microsoft"),
                ("price_rub", "le", Decimal("1000.50")), ("platform", "eq", "windows 11"),
                ("devices", "ge", Decimal(5)),
            ],
        )

    def test_invalid_filters_fail_during_request_parsing(self) -> None:
        for query in (
            "category_id=unknown", "vendor=Unknown", "platform=MS-DOS",
            "price_to=-1", "price_to=NaN", "price_to=1e3", "price_to=0",
            "devices=2.5", "devices=0", "devices=1&devices=2",
        ):
            with self.subTest(query=query), self.assertRaises(_BadRequest):
                _filters(parse_qs(query), {"1145": "Офисные пакеты"}, {"«Microsoft»"})

    def test_virtual_software_category_is_distinct_from_exact_category(self) -> None:
        condition, = _filters({"category_id": ["software"]}, {"software": "ПО"}, set())
        self.assertEqual((condition.field, condition.value, condition.source), (
            "software_category", "software", "Категория — ПО"
        ))
