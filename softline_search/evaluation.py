import json
from dataclasses import dataclass
from math import log2
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from openpyxl import load_workbook

from .catalog import Offer
from .index import DEMO_RESULT_COUNT, MODES, RESULT_LIMIT, SearchIndex, SearchMode


_TARGET_FIELDS = {"offer": "offer_id", "group": "group_id", "category": "category_id"}


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    query: str
    target_url: str
    query_type: str
    target_type: str
    target_id: str
    target_name: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: str
    target_url: str
    mapped: bool
    hit_positions: dict[SearchMode, tuple[int, ...]]
    relevant_count: int
    first_hybrid_result: str
    query_type: str = ""
    target_type: str = ""
    target_name: str = ""
    cutoff: int = RESULT_LIMIT

    @property
    def ranks(self) -> dict[SearchMode, int | None]:
        return {mode: positions[0] if positions else None for mode, positions in self.hit_positions.items()}


def _page_key(url: str) -> tuple[str, str]:
    page = urlsplit(url)
    return page.netloc.casefold(), page.path.rstrip("/")


def _target_offers(
    target_url: str, offers: tuple[Offer, ...], product_groups: dict[tuple[str, str], str]
) -> set[str]:
    key = _page_key(target_url)
    categories = set(parse_qs(urlsplit(target_url).query).get("filter[category][]", []))
    if categories:
        return {
            offer.offer_id
            for offer in offers
            if offer.category_id in categories and _page_key(offer.vendor_url) == key
        }

    exact = {offer.offer_id for offer in offers if _page_key(offer.url) == key}
    if exact:
        return exact
    if key in product_groups:
        return {offer.offer_id for offer in offers if offer.group_id == product_groups[key]}
    return {offer.offer_id for offer in offers if _page_key(offer.vendor_url) == key}


def _cell_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _read_examples(workbook_path: Path) -> list[EvaluationExample]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    examples = []
    try:
        rows = iter(workbook.active.iter_rows(values_only=True))
        header = [_cell_text(name).casefold() for name in next(rows, ())]
        target_columns = {"target_type", "target_id"}.intersection(header)
        if len(target_columns) == 1:
            raise ValueError("Для проверки по ID в Excel нужны столбцы target_type и target_id.")
        for row_number, row in enumerate(rows, start=2):
            fields = dict(zip(header, row))
            query_value = fields.get("query", row[0] if row else None)
            url_value = fields.get("target_url", row[1] if len(row) > 1 else None)
            if query_value is None and url_value is None:
                continue
            query = _cell_text(query_value)
            url = _cell_text(url_value)
            if not query or not urlsplit(url).netloc:
                raise ValueError(f"Строка {row_number} в Excel должна содержать запрос и целевой URL.")
            target_type = _cell_text(fields.get("target_type")).casefold()
            target_id = _cell_text(fields.get("target_id"))
            if target_columns and (target_type not in _TARGET_FIELDS or not target_id):
                raise ValueError(f"Строка {row_number} содержит некорректный тип или ID эталона.")
            examples.append(EvaluationExample(
                query=query,
                target_url=url,
                query_type=_cell_text(fields.get("query_type")).casefold(),
                target_type=target_type,
                target_id=target_id,
                target_name=_cell_text(fields.get("target_name")),
            ))
    finally:
        workbook.close()
    return examples


def evaluate(index: SearchIndex, workbook_path: Path, *, catalog_path: Path | None = None) -> list[QueryResult]:
    # Product landing page URLs are not present in the offer feed.
    mapping = Path(__file__).with_name("evaluation_targets.json")
    product_groups = {
        _page_key(url): group for url, group in json.loads(mapping.read_text(encoding="utf-8")).items()
    }
    by_query: dict[str, list[EvaluationExample]] = {}
    for example in _read_examples(workbook_path):
        by_query.setdefault(example.query, []).append(example)

    indexed_offers = {offer.offer_id: offer for offer in index.offers}
    cutoff = RESULT_LIMIT if catalog_path is None else DEMO_RESULT_COUNT
    rows = []
    for query, references in by_query.items():
        first_reference = references[0]
        if any(ref.query_type != first_reference.query_type for ref in references):
            raise ValueError(f"У запроса {query!r} в Excel указаны разные типы запроса.")
        target_types = {ref.target_type for ref in references}
        target_ids = set()
        for ref in references:
            if ref.target_type == "offer":
                target_ids.add(ref.target_id)
            elif ref.target_type:
                target_field = _TARGET_FIELDS[ref.target_type]
                target_ids.update(
                    offer.offer_id for offer in index.offers if getattr(offer, target_field) == ref.target_id
                )
            else:
                target_ids.update(_target_offers(ref.target_url, index.offers, product_groups))

        hit_positions = {}
        first_hybrid_result = ""
        matched_ids = target_ids & indexed_offers.keys()
        mapped = bool(matched_ids)
        target_families = (
            {index.family_key(indexed_offers[offer_id]) for offer_id in matched_ids}
            if catalog_path is not None else set()
        )
        if mapped:
            for mode in MODES:
                hits = (index.search(query, mode=mode) if catalog_path is None
                        else index.search_checked(query, catalog_path, mode=mode, limit=cutoff).hits)
                hit_positions[mode] = tuple(
                    rank for rank, hit in enumerate(hits, start=1)
                    if (hit.offer.offer_id in target_ids if catalog_path is None
                        else index.family_key(hit.offer) in target_families)
                )
                if mode == "hybrid" and hits:
                    first_hybrid_result = hits[0].offer.name
        rows.append(QueryResult(
            query=query,
            target_url=first_reference.target_url,
            mapped=mapped,
            hit_positions=hit_positions,
            relevant_count=len(target_ids) if catalog_path is None else len(target_families),
            first_hybrid_result=first_hybrid_result,
            query_type=first_reference.query_type,
            target_type=first_reference.target_type if len(target_types) == 1 else "mixed",
            target_name=first_reference.target_name,
            cutoff=cutoff,
        ))
    return rows


def metrics(rows: list[QueryResult]) -> dict[SearchMode, dict[str, float]]:
    mapped = [row for row in rows if row.mapped]
    if not mapped:
        return {}
    # Product groups and categories represent alternatives for a hit, not an exhaustive set of labels.
    exact_labels = all(row.target_type == "offer" for row in mapped)
    cutoff = mapped[0].cutoff
    discounts = [1 / log2(rank + 1) for rank in range(1, cutoff + 1)]
    scores = {}
    for mode in MODES:
        positions = [row.hit_positions[mode] for row in mapped]
        scores[mode] = {
            "HitRate@1": sum(1 in ranks for ranks in positions) / len(mapped),
            f"HitRate@{cutoff}": sum(bool(ranks) for ranks in positions) / len(mapped),
            f"MRR@{cutoff}": sum(1 / ranks[0] for ranks in positions if ranks) / len(mapped),
        }
        if exact_labels:
            scores[mode][f"Recall@{cutoff}"] = sum(
                len(ranks) / row.relevant_count for ranks, row in zip(positions, mapped)
            ) / len(mapped)
            scores[mode][f"nDCG@{cutoff}"] = sum(
                sum(discounts[rank - 1] for rank in ranks) / sum(discounts[:row.relevant_count])
                for ranks, row in zip(positions, mapped)
            ) / len(mapped)
    return scores
