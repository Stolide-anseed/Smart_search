from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from xml.etree.ElementTree import Element, SubElement, tostring

from softline_search.recommendation import recommend_batteries


class BatteryRecommendationTests(TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "feed.xml"
        self.catalog = Element("yml_catalog")
        self.offers = SubElement(SubElement(self.catalog, "shop"), "offers")

    def add_offer(self, offer_id, code, *, vendor="APC", category="82014", available="true", price="5000", ref=""):
        offer = SubElement(self.offers, "offer", id=offer_id, available=available)
        for tag, value in (
            ("name", code), ("categoryId", category), ("vendor", vendor), ("vendorCode", code),
            ("price", price), ("currencyId", "RUR"), ("url", f"https://example.test/{offer_id}"),
        ):
            SubElement(offer, tag).text = value
        if ref:
            SubElement(offer, "param", name="Совместимая батарея").text = ref

    def recommendations(self, selected):
        self.path.write_bytes(tostring(self.catalog, encoding="utf-8"))
        return recommend_batteries(self.path, selected)[1]

    def test_requires_the_complete_article_and_the_correct_vendor_and_category(self) -> None:
        self.add_offer("ups", "SRT2200RMXLI", category="82080", ref="RBC141")
        self.add_offer("prefix", "APCRBC1410")
        self.add_offer("other_vendor", "APCRBC141", vendor="Example")
        self.add_offer("other_category", "APCRBC141", category="82080")
        self.add_offer("unavailable", "APCRBC141", available="false")
        self.add_offer("unpriced", "APCRBC141", price="0")
        self.add_offer("matched", "APCRBC141")
        self.add_offer("same_battery", "rbc141")

        items = self.recommendations("https://example.test/ups")
        self.assertEqual([item.offer.offer_id for item in items], ["matched"])
        self.assertEqual(items[0].battery_reference, "RBC141")

    def test_keeps_a_compatible_fallback_with_its_availability_and_price_status(self) -> None:
        self.add_offer("ups", "SRT2200RMXLI", category="82080", ref="RBC141")
        self.add_offer("matched", "APCRBC141", available="false", price="0.000")

        items = self.recommendations("ups")
        self.assertEqual([item.offer.offer_id for item in items], ["matched"])
        self.assertFalse(items[0].available)
        self.assertFalse(items[0].has_price)

    def test_returns_no_recommendation_without_an_explicit_reference(self) -> None:
        self.add_offer("ups", "SRT2200RMXLI", category="82080", ref="Другое или отсутствует")
        self.add_offer("battery", "APCRBC141")

        self.assertEqual(self.recommendations("ups"), [])
