import re
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import astuple, dataclass
from decimal import Decimal, InvalidOperation
from html import unescape
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np

from .catalog import Document, Offer, lexical_terms, read_catalog
from .encoder import EMBEDDING_SIZE, MODEL_ID, MODEL_REVISION, E5Encoder
from .query import Condition, ParsedQuery
from .query_catalog import QueryCatalog
from .query_normalization import query_key


SearchMode = Literal["hybrid", "e5", "bm25"]
FamilyKey = tuple[str, ...]
MODES: tuple[SearchMode, ...] = ("hybrid", "e5", "bm25")
RESULT_LIMIT = 10
INITIAL_CANDIDATES = 100
RRF_OFFSET = 60
E5_WEIGHT = 0.7
BM25_WEIGHT = 0.3
DEMO_RESULT_COUNT = 4
_FILTER_ORDER = {"article": 0, "category_id": 1, "brand": 2, "product_type": 3, "price_rub": 4, "model": 5}

_APPLICATION_ID = int.from_bytes(b"SLSI", "big")
_SCHEMA_VERSION = 1
_VECTOR_DTYPE = np.dtype("<f2")
_SCHEMA = """
CREATE TABLE offers (
    offer_id TEXT NOT NULL UNIQUE,
    group_id TEXT NOT NULL,
    category_id TEXT NOT NULL,
    name TEXT NOT NULL,
    vendor TEXT NOT NULL,
    vendor_code TEXT NOT NULL,
    vendor_url TEXT NOT NULL,
    price TEXT NOT NULL,
    currency TEXT NOT NULL,
    url TEXT NOT NULL
);
CREATE VIRTUAL TABLE lexical USING fts5(
    title, code, details, tokenize='unicode61 remove_diacritics 0'
);
CREATE TABLE vectors (embedding BLOB NOT NULL);
CREATE TABLE metadata (model_id TEXT NOT NULL, revision TEXT NOT NULL);
"""


@dataclass(frozen=True, slots=True)
class SearchHit:
    offer: Offer
    score: float


@dataclass(frozen=True, slots=True)
class CheckedSearch:
    query: ParsedQuery
    hits: list[SearchHit]
    reserve: list[SearchHit]
    unconfirmed: dict[str, tuple[str, ...]]


def _open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _check_index_type(connection: sqlite3.Connection) -> None:
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    if application_id != _APPLICATION_ID:
        raise ValueError("Файл не является индексом Softline Search.")


def _catalog_embeddings(
    documents: list[Document], device: str, batch_size: int, progress: Callable[[str], None]
) -> np.ndarray:
    import torch

    encoder = E5Encoder(device)
    progress(f"Кодируем предложения моделью {MODEL_ID} на {encoder.device}.")
    texts = [document.text for document in documents]
    # Batching texts of similar length reduces padding in the encoder.
    order = sorted(range(len(texts)), key=lambda row: len(texts[row]))
    vectors = np.empty((len(texts), EMBEDDING_SIZE), dtype=_VECTOR_DTYPE)
    offset = 0
    next_report = 1000

    while offset < len(order):
        rows = order[offset : offset + batch_size]
        try:
            vectors[rows] = encoder.encode_passages([texts[row] for row in rows])
        except torch.OutOfMemoryError:
            if encoder.device != "cuda" or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            progress(f"Уменьшаем размер пакета для CUDA до {batch_size}.")
        else:
            offset += len(rows)
            if offset >= next_report or offset == len(order):
                progress(f"Закодировано {offset:,} / {len(order):,} предложений.")
                next_report = offset + 1000
            continue
        torch.cuda.empty_cache()

    return vectors


def build_index(
    catalog_path: Path,
    index_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 64,
    progress: Callable[[str], None] = print,
) -> int:
    if batch_size < 1:
        raise ValueError("Размер пакета должен быть положительным.")
    if index_path.exists():
        with closing(_open_read_only(index_path)) as previous:
            _check_index_type(previous)

    progress(f"Читаем каталог {catalog_path.name}.")
    documents = read_catalog(catalog_path)
    if not documents:
        raise ValueError("В каталоге нет предложений с available=\"true\".")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=index_path.parent, prefix=".search-index-build-") as work_dir:
        temporary_index = Path(work_dir) / "index.sqlite3"
        with closing(sqlite3.connect(temporary_index)) as connection, connection:
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.executescript(_SCHEMA)
            connection.execute("INSERT INTO metadata VALUES (?, ?)", (MODEL_ID, MODEL_REVISION))
            connection.executemany(
                "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (astuple(document.offer) for document in documents),
            )
            # SQLite assigns the offer rowids in insertion order starting from 1.
            connection.executemany(
                "INSERT INTO lexical (rowid, title, code, details) VALUES (?, ?, ?, ?)",
                (
                    (
                        row,
                        " ".join(lexical_terms(document.title)),
                        " ".join(lexical_terms(document.offer.vendor_code)),
                        " ".join(lexical_terms(document.details)),
                    )
                    for row, document in enumerate(documents, start=1)
                ),
            )
            connection.execute("INSERT INTO lexical(lexical) VALUES ('optimize')")
            progress(f"BM25 проиндексировал {len(documents):,} предложений.")

            vectors = _catalog_embeddings(documents, device, batch_size, progress)
            connection.executemany(
                "INSERT INTO vectors VALUES (?)", ((vector.tobytes(),) for vector in vectors)
            )
        temporary_index.replace(index_path)
    return len(documents)


def _fuse_ranks(
    semantic: list[tuple[int, float]], lexical: list[tuple[int, float]], depth: int
) -> list[tuple[int, float]]:
    scores = {}
    for ranking, weight in ((semantic, E5_WEIGHT), (lexical, BM25_WEIGHT)):
        for rank, (row, _) in enumerate(ranking[:depth], start=1):
            scores[row] = scores.get(row, 0.0) + weight / (RRF_OFFSET + rank)
    return sorted(scores.items(), key=lambda match: (-match[1], match[0]))


def _baseline_family(offer: Offer) -> FamilyKey:
    return ("group", offer.group_id) if offer.group_id else ("offer", offer.offer_id)


def _catalog_families(offers: tuple[Offer, ...]) -> dict[str, FamilyKey]:
    articles: dict[str, FamilyKey] = {}
    groups_by_article: dict[FamilyKey, set[str]] = {}
    model_by_article: dict[FamilyKey, FamilyKey] = {}
    for offer in offers:
        vendor = query_key(unescape(offer.vendor)).strip()
        code = query_key(unescape(offer.vendor_code)).strip()
        if not vendor or not code:
            continue
        article = ("article", vendor, offer.category_id, code)
        if offer.group_id:
            groups_by_article.setdefault(article, set()).add(offer.group_id)
            continue
        articles[offer.offer_id] = article
        model, separator, _ = code.partition("/")
        # A slash marks a variant only if the base model also appears on its own in the title.
        if separator and any(char.isalpha() for char in model) and any(char.isdecimal() for char in model):
            title = query_key(unescape(offer.name))
            if re.search(r"(?<![\w./-])" + re.escape(model) + r"(?![\w./-])", title):
                model_by_article[article] = ("article", vendor, offer.category_id, model)

    families = {offer.offer_id: _baseline_family(offer) for offer in offers}
    for offer_id, article in articles.items():
        known_groups = groups_by_article.get(article, set())
        if len(known_groups) == 1:
            families[offer_id] = ("group", next(iter(known_groups)))
        else:
            families[offer_id] = model_by_article.get(article, article)
    return families


class SearchIndex:
    def __init__(self, path: Path, device: str = "auto") -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Индекс {path} не найден. Сначала запустите build-index.")
        connection = _open_read_only(path)
        try:
            _check_index_type(connection)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != _SCHEMA_VERSION:
                raise ValueError("Версия индекса изменилась. Выполните build-index ещё раз.")
            model = connection.execute("SELECT model_id, revision FROM metadata").fetchone()
            if model != (MODEL_ID, MODEL_REVISION):
                raise ValueError("Версия модели изменилась. Выполните build-index ещё раз.")
            offers = connection.execute(
                "SELECT offer_id, group_id, category_id, name, vendor, vendor_code, "
                "vendor_url, price, currency, url FROM offers ORDER BY rowid"
            )
            self.offers = tuple(Offer(*row) for row in offers)
        except Exception:
            connection.close()
            raise
        self._connection = connection
        self._device = device
        self._encoder = None
        self._vectors = None
        self._query_catalog = None
        self._query_parser = None
        self._query_catalog_path = None
        self._catalog_family_keys: dict[str, FamilyKey] | None = None

    def close(self) -> None:
        self._connection.close()

    def _lexical_rank(self, query: str) -> list[tuple[int, float]]:
        words = lexical_terms(query)
        # Include closed compounds such as MaxPatrol for queries written with a space.
        joined = [left + right for left, right in pairwise(words) if left.isalpha() and right.isalpha()]
        terms = list(dict.fromkeys(words + joined))[:64]
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        return self._connection.execute(
            "SELECT rowid - 1, -bm25(lexical, 2.0, 4.0, 1.0) AS score "
            "FROM lexical WHERE lexical MATCH ? ORDER BY score DESC, rowid",
            (match,),
        ).fetchall()

    def _semantic_rank(self, query: str) -> list[tuple[int, float]]:
        if self._encoder is None:
            self._encoder = E5Encoder(self._device)
        if self._vectors is None:
            vectors = np.empty((len(self.offers), EMBEDDING_SIZE), dtype=np.float32)
            rows = self._connection.execute("SELECT embedding FROM vectors ORDER BY rowid")
            count = 0
            for count, (blob,) in enumerate(rows, start=1):
                vectors[count - 1] = np.frombuffer(blob, dtype=_VECTOR_DTYPE)
            if count != len(self.offers):
                raise ValueError("Число векторов не соответствует каталогу. Пересоберите индекс.")
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            self._vectors = vectors
        scores = self._vectors @ self._encoder.encode_query(query)
        order = np.argsort(-scores, kind="stable")
        return [(int(row), float(scores[row])) for row in order]

    def family_key(self, offer: Offer) -> FamilyKey:
        if self._catalog_family_keys is None:
            self._catalog_family_keys = _catalog_families(self.offers)
        return self._catalog_family_keys[offer.offer_id]

    def _best_groups(
        self, ranking: Iterable[tuple[int, float]], limit: int, family_key: Callable[[Offer], FamilyKey]
    ) -> list[SearchHit]:
        hits = []
        seen = set()
        for row, score in ranking:
            offer = self.offers[row]
            key = family_key(offer)
            if key in seen:
                continue
            seen.add(key)
            hits.append(SearchHit(offer, score))
            if len(hits) == limit:
                break
        return hits

    def search(
        self, query: str, *, mode: SearchMode = "hybrid", limit: int = RESULT_LIMIT
    ) -> list[SearchHit]:
        query = " ".join(query.split())
        if not query or limit < 1:
            raise ValueError("Укажите непустой запрос и положительное число результатов.")
        semantic, lexical = self._rankings(query, mode)
        return self._ranked_hits(semantic, lexical, mode, limit)

    def _rankings(
        self, query: str, mode: SearchMode
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        if mode not in MODES:
            raise ValueError(f"Неизвестный способ поиска: {mode}")
        semantic = self._semantic_rank(query) if mode != "bm25" else []
        lexical = self._lexical_rank(query) if mode != "e5" else []
        return semantic, lexical

    def _ranked_hits(
        self, semantic: list[tuple[int, float]], lexical: list[tuple[int, float]], mode: SearchMode, limit: int,
        family_key: Callable[[Offer], FamilyKey] = _baseline_family,
    ) -> list[SearchHit]:
        if mode == "bm25":
            return self._best_groups(lexical, limit, family_key)
        if mode == "e5":
            return self._best_groups(semantic, limit, family_key)

        depth = max(INITIAL_CANDIDATES, limit)
        maximum = max(len(semantic), len(lexical))
        while True:
            hits = self._best_groups(_fuse_ranks(semantic, lexical, depth), limit, family_key)
            if len(hits) >= limit or depth >= maximum:
                return hits
            depth *= 2

    def prepare_query_catalog(self, catalog_path: Path) -> QueryCatalog:
        from .query import QueryParser
        from .query_normalization import QueryNormalizer

        path = catalog_path.resolve()
        if self._query_catalog_path == path and self._query_catalog is not None:
            return self._query_catalog
        catalog = QueryCatalog(path, self.offers)
        normalizer = QueryNormalizer(self.offers, list(catalog.categories.values()))
        group_names = {
            catalog.categories[offer.group_id] for offer in self.offers if offer.group_id in catalog.categories
        }
        category_names = {
            offer.category_id: catalog.categories[offer.category_id]
            for offer in self.offers
            if offer.category_id in catalog.categories
        }
        parser = QueryParser(normalizer, {offer.vendor for offer in self.offers}, group_names, category_names)
        self._query_catalog = catalog
        self._query_parser = parser
        self._query_catalog_path = path
        return catalog

    def parse_query(self, query: str, catalog_path: Path) -> ParsedQuery:
        self.prepare_query_catalog(catalog_path)
        return self._query_parser.parse(query)

    def search_checked(
        self,
        query: str,
        catalog_path: Path,
        *,
        mode: SearchMode = "hybrid",
        limit: int = DEMO_RESULT_COUNT,
        filters: tuple[Condition, ...] = (),
    ) -> CheckedSearch:
        if limit < 1:
            raise ValueError("Укажите положительное число результатов.")
        parsed = self.parse_query(query, catalog_path)
        catalog = self._query_catalog
        if catalog is None:
            raise RuntimeError("Каталог условий не загружен.")
        semantic, lexical = self._rankings(parsed.search_text, mode)
        required = sorted(
            (condition for condition in parsed.conditions if condition.required),
            key=lambda condition: _FILTER_ORDER.get(condition.field, 6),
        )
        preferences = [condition for condition in parsed.conditions if not condition.required]
        if not required and not filters:
            direct = self._ranked_hits(semantic, lexical, mode, limit, self.family_key)
            return CheckedSearch(parsed, self._order_preferences(direct, preferences), [], {})

        confirmed = set()
        pending = {}
        candidates = {row for row, _ in semantic} | {row for row, _ in lexical}
        for row in candidates:
            offer = self.offers[row]
            if filters and not all(catalog.check(offer, condition) is True for condition in filters):
                continue
            missing = []
            for condition in required:
                result = catalog.check(offer, condition)
                if result is False:
                    break
                if result is None:
                    # Reserve candidates must still be plausible answers to the requested product identity.
                    if condition.field in {"product_type", "category_id", "article", "model", "brand"}:
                        break
                    missing.append(condition.source)
            else:
                if missing:
                    pending[row] = tuple(dict.fromkeys(missing))
                else:
                    confirmed.add(row)

        def ranked(rows, count):
            return self._ranked_hits(
                [entry for entry in semantic if entry[0] in rows],
                [entry for entry in lexical if entry[0] in rows], mode, count, self.family_key,
            )

        direct = self._order_preferences(ranked(confirmed, limit), preferences)
        reserve = []
        reasons = {}
        shortage = min(DEMO_RESULT_COUNT, limit) - len(direct)
        if shortage > 0 and pending:
            shown_families = {self.family_key(hit.offer) for hit in direct}
            pending = {row: missing for row, missing in pending.items()
                       if self.family_key(self.offers[row]) not in shown_families}
            reserve = ranked(pending, shortage)
            by_id = {self.offers[row].offer_id: missing for row, missing in pending.items()}
            reasons = {hit.offer.offer_id: by_id[hit.offer.offer_id] for hit in reserve}
        return CheckedSearch(parsed, direct, reserve, reasons)

    def _order_preferences(self, hits: list[SearchHit], preferences) -> list[SearchHit]:
        if not preferences or self._query_catalog is None:
            return hits
        cheaper = any(condition.field == "prefer_lower_price" for condition in preferences)

        def key(hit: SearchHit) -> tuple[int, Decimal]:
            matched = sum(self._query_catalog.check(hit.offer, condition) is True for condition in preferences
                          if condition.field != "prefer_lower_price")
            if cheaper and hit.offer.currency in {"RUR", "RUB"}:
                try:
                    price = Decimal(hit.offer.price)
                    if price.is_finite() and price > 0:
                        return -matched, price
                except InvalidOperation:
                    pass
            return -matched, Decimal("Infinity")

        return sorted(hits, key=key)
