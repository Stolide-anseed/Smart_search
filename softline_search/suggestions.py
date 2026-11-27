from bisect import bisect_left
from collections import Counter
from heapq import nsmallest
from html import unescape

from .catalog import Offer
from .query_normalization import query_key


class CatalogSuggestions:
    def __init__(self, offers: tuple[Offer, ...], categories: dict[str, str]) -> None:
        frequency: Counter[str] = Counter()
        entries: set[tuple[str, str]] = set()
        canonical_labels: dict[str, str] = {}

        def add(text: str) -> str:
            label = " ".join(unescape(text).split()[:9]).strip(" ,.-")
            if 2 <= len(label) <= 110:
                key = query_key(label)
                label = canonical_labels.setdefault(key, label)
                frequency[label] += 1
                entries.add((key, label))
                return label
            return ""

        for offer in offers:
            add(offer.vendor)
            add(categories.get(offer.category_id, ""))
            label = add(categories.get(offer.group_id, "") or offer.name)
            if not label:
                continue
            code = query_key(offer.vendor_code.strip())
            if 3 <= len(code) <= 40:
                entries.add((code, label))
            vendor = query_key(unescape(offer.vendor).strip())
            if len(vendor) < 3:
                continue
            title = query_key(label)
            head, found, model = title.partition(vendor)
            model = model.strip(" -–,./")
            if found and len(head) <= 32 and len(model) >= 3:
                entries.add((model, label))

        self._entries = sorted(entries)
        self._frequency = frequency

    def match(self, prefix: str, limit: int) -> list[str]:
        key = query_key(prefix)
        start = bisect_left(self._entries, (key, ""))
        end = bisect_left(self._entries, (key + "\U0010ffff", ""))
        labels = {label for _, label in self._entries[start:end]}
        return nsmallest(
            limit, labels,
            key=lambda label: (not query_key(label).startswith(key), -self._frequency[label], len(label), label),
        )
