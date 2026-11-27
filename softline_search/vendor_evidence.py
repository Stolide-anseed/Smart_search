"""Fetch and cache technical facts supported by official, product-specific evidence."""

import argparse
import json
import re
import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from io import BytesIO
from itertools import islice
from pathlib import Path
from threading import Lock
from time import monotonic, time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .recommendation_facts import BAG_LIMITS, LAPTOP_DIMENSIONS, _parse_dimensions, normalized, product_key
from .recommendation_licenses import _platform_set


DEFAULT_CACHE = Path(".vendor-facts.sqlite3")
DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
VERIFIED_ATTRIBUTES = ("dimensions_mm", "battery_articles", "platforms")
_DOMAINS = {
    "apc": ("www.se.com", "download.se.com", "iportal.se.com", "www.apc.com", "shop.se.com"),
    "lenovo": ("psref.lenovo.com", "www.lenovo.com", "support.lenovo.com", "pcsupport.lenovo.com", "download.lenovo.com"),
    "hp": ("www.hp.com", "support.hp.com", "h20195.www2.hp.com", "www8.hp.com"),
    "asus": ("www.asus.com", "dlcdnwebimgs.asus.com"),
    "acer": ("www.acer.com", "global-download.acer.com"),
    "dell": ("www.dell.com", "dl.dell.com"),
    "msi": ("www.msi.com", "storage-asset.msi.com"),
    "gigabyte": ("www.gigabyte.com", "download.gigabyte.com"),
    "microsoft": ("www.microsoft.com", "support.microsoft.com", "learn.microsoft.com"),
    "mobisystems": ("mobisystems.com", "support.mobisystems.com"),
    "kaspersky": ("www.kaspersky.com", "support.kaspersky.com", "www.kaspersky.ru", "support.kaspersky.ru"),
    "pro32": ("pro32.com", "www.pro32.com", "static.pro32.com", "support.pro32.com"),
    "dr.web": ("www.drweb.ru", "support.drweb.ru", "download.drweb.com"),
}
_ALIASES = {
    "schneider electric": "apc", "apc by schneider electric": "apc",
    "dell technologies": "dell", "hewlett packard": "hp", "hp inc": "hp",
    "лаборатория касперского": "kaspersky",
}
_TYPES = {
    "laptop": ("dimensions_mm",), "notebook_bag": ("dimensions_mm",),
    "ups": ("battery_articles",), "office": ("platforms",), "antivirus_business": ("platforms",),
}
_SUBJECT_LABELS = {"sku", "mpn", "product number", "model", "model number", "артикул", "код товара", "модель"}
_BATTERY_LABELS = (
    "replacement battery", "replacement battery module", "replacement battery pack",
    "replacement battery cartridge", "compatible replacement battery",
    "совместимая батарея", "сменный аккумулятор", "сменная батарея",
)
_PLATFORM_LABELS = (
    "supported operating systems", "supported os", "supported platforms",
    "поддерживаемые операционные системы", "поддерживаемая операционная система", "поддерживаемые платформы",
)
_UNITS = {"dimensions_mm": "mm", "battery_articles": "article", "platforms": "os"}
SEARCH_STRATEGIES = ("exact_article", "verified_model", "official_technical_specifications", "official_regional_page")
_ATTRIBUTE_QUERIES = {
    "dimensions_mm": "dimensions width depth height",
    "battery_articles": "compatible replacement battery part number",
    "platforms": "supported operating systems",
}
_BATTERY_VALUE = re.compile(r"((?:APC)?RBC[0-9]{1,5})(?:\s+\([1-9][0-9]?\))?", re.I | re.ASCII)
_AXES = r"[wdhшгвт]\s*[×xх]\s*[wdhшгвт]\s*[×xх]\s*[wdhшгвт]"
_DIM_TRIPLE = r"\d{1,4}(?:[.,]\d+)?\s*[×xх*]\s*\d{1,4}(?:[.,]\d+)?\s*[×xх*]\s*\d{1,4}(?:[.,]\d+)?\s*(?:mm|cm|мм|см)"
_DIM_VALUE = re.compile(rf"(?P<value>{_DIM_TRIPLE})(?:\s*\([\d., ×xх*]+(?:inches?|″|\")\))?", re.I)
_NUMBER = r"\d{1,4}(?:[.,]\d+)?"
_THICKNESS_RANGE = rf"({_NUMBER})(?:\s*[~～〜–—-]\s*({_NUMBER}))?"
_ASUS_DIMENSIONS = re.compile(
    rf"({_NUMBER})\s*[×x]\s*({_NUMBER})\s*[×x]\s*{_THICKNESS_RANGE}\s*(mm|cm)(?:\s*\([^()]+\))?", re.I,
)
_ASUS_JP_DIMENSIONS = re.compile(
    rf"幅\s*({_NUMBER})\s*mm\s*[×x]\s*奥行き\s*({_NUMBER})\s*mm\s*[×x]\s*高さ\s*{_THICKNESS_RANGE}\s*(mm)",
)
_LAPTOP_LABEL = re.compile(rf"(?:chassis |product |physical |laptop |system |overall )?dimensions?\s*\((?:{_AXES})\)$")
_BAG_LABELS = (*BAG_LIMITS, "maximum laptop dimensions", "recommended maximum laptop dimensions",
               "internal laptop compartment dimensions", "notebook compartment dimensions")
_CONDITIONAL = re.compile(
    r"\b(?:except|requirements? vary|server only|not|unsupported|unsupported on|arm|minimum|"
    r"required update|or newer|or later|или новее|кроме|без|не поддерж\w*)\b", re.I,
)
_ARTICLE = re.compile(r"[a-z0-9][a-z0-9._/#-]{1,63}", re.ASCII | re.I)
_MAX_HTML = 2_000_000
_MAX_PDF = 6_000_000
_MAX_PAGES = 15
_MAX_FETCH_URLS = 4
_LOOKUP_RETRY_SECONDS = 30
_SHORT_CACHE_ENTRIES = 256


def _manufacturer_key(vendor: str) -> str:
    key = product_key(vendor, "")[0]
    return _ALIASES.get(key, key)


def _eligible_attribute(kind: str, attribute: str) -> bool:
    return attribute in _TYPES.get(kind, ())


@dataclass(frozen=True, slots=True)
class ProductIdentity:
    manufacturer: str
    product_model: str
    article: str
    kind: str
    exact_model: bool = False
    offer_id: str = ""
    chassis_model: str = ""
    chassis_source: str = ""
    spec_source: str = ""

    @property
    def domains(self) -> tuple[str, ...]:
        return _DOMAINS.get(_manufacturer_key(self.manufacturer), ())

    @property
    def key(self) -> tuple[str, str]:
        article = normalized(self.article)
        return _manufacturer_key(self.manufacturer), article or (normalized(self.product_model) if self.exact_model else "")

    @property
    def subject(self) -> str:
        return self.article or (self.product_model if self.exact_model else "")


@dataclass(frozen=True, slots=True)
class VendorFact:
    manufacturer: str
    product_model: str
    article: str
    attribute: str
    value: tuple[str, ...]
    unit: str
    source_url: str
    evidence_text: str
    retrieved_at: str
    kind: str
    verification_status: str = "CONFIRMED"
    identity_source_url: str = ""
    chassis_model: str = ""

    def public_data(self) -> dict:
        result = asdict(self)
        del result["evidence_text"]
        return result

    def applies_to(self, identity: ProductIdentity) -> bool:
        source_identity = ProductIdentity(self.manufacturer, self.product_model, self.article, identity.kind, True)
        same_chassis = (
            identity.chassis_source and self.chassis_model and self.attribute == "dimensions_mm"
            and self.kind == "laptop" and source_identity.key[0] == identity.key[0] == "asus"
            and normalized(self.chassis_model) == normalized(identity.chassis_model)
            and urlsplit(self.source_url).path.rstrip("/").endswith("/techspec")
        )
        return (
            self.verification_status == "CONFIRMED" and bool(self.evidence_text) and bool(identity.subject)
            and self.kind == identity.kind and (source_identity.key == identity.key or same_chassis)
            and _eligible_attribute(identity.kind, self.attribute)
            and self.unit == _UNITS[self.attribute] and bool(self.source_url)
            and _official_url(self.source_url, identity.domains) == self.source_url
            and (not self.identity_source_url
                 or _official_url(self.identity_source_url, identity.domains) == self.identity_source_url)
            and _valid_value(self.attribute, self.value)
        )


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    fact: VendorFact | None
    status: str
    cache_hit: bool = False
    download_attempts: int = 0
    download_ms: float = 0
    extraction_ms: float = 0


@dataclass(frozen=True, slots=True)
class VendorDocument:
    url: str
    content_type: str
    content: bytes


def _official_url(url: str, domains: tuple[str, ...]) -> str | None:
    if not isinstance(url, str) or len(url) > 1500 or any(char in url for char in "\\\t\r\n"):
        return None
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.hostname not in domains
                or parts.username is not None or parts.password is not None or parts.port not in {None, 443}):
            return None
    except ValueError:
        return None
    # Preserve only public lookup fields. Vendor pages can add analytics/session fields to redirects.
    allowed = {"psref.lenovo.com": {"m"}, "download.se.com": {"p_doc_ref"}}.get(parts.hostname, ())
    query = [(name, vals[0]) for name, vals in parse_qs(parts.query).items()
             if name.casefold() in allowed and len(vals) == 1 and _ARTICLE.fullmatch(vals[0])]
    return urlunsplit(("https", parts.hostname, parts.path or "/", urlencode(query), ""))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_vendor_document(url: str, domains: tuple[str, ...], deadline: float) -> VendorDocument:
    opener = build_opener(ProxyHandler({}), _NoRedirect)
    for _ in range(4):
        target = _official_url(url, domains)
        if not target:
            raise ValueError("source_domain_not_allowed")
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("vendor_timeout")
        request = Request(target, headers={"Accept": "text/html, application/pdf", "User-Agent": "Mozilla/5.0"})
        try:
            response = opener.open(request, timeout=min(remaining, 5))
        except HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308} or not exc.headers.get("Location"):
                exc.close()
                raise
            url = urljoin(target, exc.headers["Location"])
            exc.close()
            continue
        with response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain", "application/pdf"}:
                raise ValueError("unsupported_document")
            maximum = _MAX_PDF if content_type == "application/pdf" else _MAX_HTML
            content_length = response.headers.get("Content-Length", "")
            if content_length.isdigit() and int(content_length) > maximum:
                raise ValueError("document_too_large")
            data = bytearray()
            while len(data) <= maximum:
                if monotonic() >= deadline:
                    raise TimeoutError("vendor_timeout")
                chunk = response.read1(min(65536, maximum + 1 - len(data)))
                if not chunk:
                    return VendorDocument(target, content_type, bytes(data))
                data.extend(chunk)
            raise ValueError("document_too_large")
    raise ValueError("too_many_redirects")


class _SpecsHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.rows: list[tuple[str, ...]] = []
        self.products: list[dict] = []
        self._capture = ""
        self._text: list[str] = []
        self._cells: list[str] = []
        self._term = ""
        self._skipped = ""

    def handle_starttag(self, tag, attrs):
        if self._skipped:
            return
        if tag in {"style", "script", "template", "noscript"}:
            if tag == "script" and dict(attrs).get("type") == "application/ld+json":
                self._capture = "script"
                self._text = []
            else:
                self._skipped = tag
            return
        if tag == "tr":
            self._cells = []
        if tag in {"title", "h1", "th", "td", "dt", "dd"}:
            self._capture = tag
            self._text = []
        elif self._capture and tag in {"p", "br", "li"}:
            self._text.append(" ")

    def handle_data(self, data):
        if self._capture and not self._skipped:
            self._text.append(data)

    def handle_endtag(self, tag):
        if self._skipped:
            if self._skipped == tag:
                self._skipped = ""
            return
        if tag == "tr" and self._cells:
            self.rows.append(tuple(self._cells))
            self._cells = []
        if self._capture != tag:
            return
        raw = "".join(self._text)
        value = " ".join(raw.split())
        self._capture = ""
        if tag in {"title", "h1"} and value:
            self.headings.append(value)
        elif tag in {"th", "td"}:
            self._cells.append(value)
        elif tag == "dt":
            self._term = value
        elif tag == "dd" and self._term:
            self.rows.append((self._term, value))
            self._term = ""
        elif tag == "script" and len(raw) <= 200_000:
            try:
                data = json.loads(raw)
                nodes = data.get("@graph", [data]) if isinstance(data, dict) else data
                if isinstance(nodes, list):
                    self.products.extend(node for node in nodes if isinstance(node, dict)
                                         and node.get("@type") == "Product")
            except (ValueError, RecursionError):
                pass


def _identifier_in(text: str, identifier: str) -> bool:
    token = normalized(identifier)
    return bool(token and re.search(r"(?<![\w./#-])" + re.escape(token) + r"(?![\w./#-])", normalized(text)))


def _identity_in(text: str, identity: ProductIdentity) -> bool:
    if _identifier_in(text, identity.article):
        return True
    return identity.exact_model and _identifier_in(text, identity.product_model)


def _valid_value(attribute: str, values: tuple[str, ...]) -> bool:
    if not values or len(values) > 16 or any(not isinstance(value, str) or len(value) > 80 for value in values):
        return False
    if attribute == "dimensions_mm":
        if len(values) != 3:
            return False
        try:
            return all(Decimal(value).is_finite() and 0 < Decimal(value) < 2_000 for value in values)
        except InvalidOperation:
            return False
    if attribute == "battery_articles":
        return all(re.fullmatch(r"(?:APC)?RBC[0-9]{1,5}", value, re.ASCII) for value in values)
    return attribute == "platforms" and set(values) == _platform_set(", ".join(values))


class _AsusSpecsTable(HTMLParser):
    """Read the model and specification cells in their original column order."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, list[str]]] = []
        self._depth = 0
        self._table_depth = self._row_depth = self._cell_depth = None
        self._row_type = ""
        self._title = ""
        self._cells: list[str] = []
        self._parts: list[str] = []
        self._is_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "br" and self._cell_depth is not None:
            self._parts.append("\n")
        if tag != "div":
            return
        self._depth += 1
        attributes = dict(attrs)
        if attributes.get("id") == "productTableBody":
            self._table_depth = self._depth
        if self._table_depth is None:
            return
        css = attributes.get("class") or ""
        if self._row_depth is None and ("__rowTableModel___" in css or "__rowTable___" in css):
            self._row_depth = self._depth
            self._row_type = "model" if "__rowTableModel___" in css else "spec"
            self._title = ""
            self._cells = []
        if self._row_depth is not None and self._cell_depth is None:
            title = "__rowTableTitle___" in css
            if title or "__rowTableItems___" in css:
                self._cell_depth = self._depth
                self._is_title = title
                self._parts = []

    def handle_data(self, data):
        if self._cell_depth is not None:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag != "div":
            return
        if self._depth == self._cell_depth:
            value = "".join(self._parts).strip()
            if self._is_title:
                self._title = value
            else:
                self._cells.append(value)
            self._cell_depth = None
        if self._depth == self._row_depth:
            self.rows.append((self._row_type, self._title, self._cells))
            self._row_depth = None
        if self._depth == self._table_depth:
            self._table_depth = None
        self._depth -= 1


def _asus_dimension_value(value: str) -> tuple[Decimal, Decimal, Decimal] | None:
    match = _ASUS_DIMENSIONS.fullmatch(value) or _ASUS_JP_DIMENSIONS.fullmatch(value)
    if match is None:
        return None
    width, depth, low, high, unit = match.groups()
    scale = 10 if unit.lower() == "cm" else 1
    thicknesses = (low, high or low)
    return (Decimal(width.replace(",", ".")) * scale,
            Decimal(depth.replace(",", ".")) * scale,
            max(Decimal(thickness.replace(",", ".")) for thickness in thicknesses) * scale)


def _asus_table_evidence(document: VendorDocument, identity: ProductIdentity):
    parser = _AsusSpecsTable()
    parser.feed(document.content.decode("utf-8", errors="replace"))
    headers = [cells for kind, _, cells in parser.rows if kind == "model" and cells]
    if len(headers) != 1:
        return None, "document_extraction_failed"
    models = headers[0]
    columns = [
        index for index, model in enumerate(models)
        if _identifier_in(model, identity.article)
        or (identity.chassis_source and _identifier_in(model, identity.chassis_model))
    ]
    if not columns:
        return None, "identity_mismatch"
    dimensions = []
    for kind, title, cells in parser.rows:
        if (kind != "spec" or len(cells) != len(models)
                or not (re.fullmatch(r"dimensions?\s*\(w\s*[x×]\s*d\s*[x×]\s*h\)", normalized(title))
                        or normalized(title).startswith("サイズ (突起部除く)"))):
            continue
        for index in columns:
            for line in cells[index].splitlines():
                if value := _asus_dimension_value(line.strip()):
                    dimensions.append((value, f"{models[index]} — {title} — {line.strip()}"))
    if not dimensions:
        return None, "document_extraction_failed"
    if len({size[:2] for size, _ in dimensions}) > 1:
        return None, "conflicting_facts"
    size, evidence = max(dimensions, key=lambda item: item[0][2])
    return (tuple(format(value.normalize(), "f") for value in size), evidence), "verified"


def _extract_label_value(
    label: str, value: str, identity: ProductIdentity, attribute: str, source_host: str,
) -> tuple[str, ...] | None:
    label = re.sub(r"\s*\[\d+\]\s*$", "", normalized(label)).rstrip(":")
    value = normalized(value)
    if attribute == "dimensions_mm":
        if identity.kind == "notebook_bag":
            bag_label = any(label == name or label.startswith(name + " (") for name in _BAG_LABELS)
            hp_order = source_host == "www.hp.com" and label == "recommended max laptop dimension (metric)"
            if not bag_label or (not hp_order and not re.search(_AXES, label)):
                return None
        elif not _LAPTOP_LABEL.fullmatch(label) and not (
            re.search(_AXES, label) and any(label.startswith(name + " (") for name in LAPTOP_DIMENSIONS)
        ):
            return None
        match = _DIM_VALUE.fullmatch(value)
        dims = _parse_dimensions(match["value"], label) if match else None
        return tuple(format(val.normalize(), "f") for val in dims) if dims else None
    if attribute == "battery_articles" and label in _BATTERY_LABELS:
        codes = [_BATTERY_VALUE.fullmatch(part.strip()) for part in re.split(r"[,;/]", value)]
        if all(codes):
            return tuple(sorted({code[1].upper().removeprefix("APC") for code in codes}))
    if attribute == "platforms" and label in _PLATFORM_LABELS and not _CONDITIONAL.search(value):
        return tuple(sorted(_platform_set(value))) or None
    return None


def _html_evidence(document: VendorDocument, identity: ProductIdentity, attribute: str):
    parser = _SpecsHTML()
    parser.feed(document.content.decode("utf-8", errors="replace"))
    # Product-level structured metadata gives a scope independent of the surrounding page.
    subjects = []
    for product in parser.products:
        names = [product.get(key, "") for key in ("sku", "mpn", "name")]
        identifiers = [value for value in names[:2] if isinstance(value, str) and value]
        subject = next((line for line in (identifiers or names[2:])
                        if isinstance(line, str) and _identity_in(line, identity)), "")
        subjects.append(subject)
        props = product.get("additionalProperty")
        if subject and isinstance(props, list):
            for prop in props:
                if isinstance(prop, dict) and isinstance(prop.get("name"), str) and isinstance(prop.get("value"), str):
                    yield subject, prop["name"], prop["value"]
    if len(subjects) > 1 or any(not subject for subject in subjects):
        return
    heading = next((line for line in parser.headings
                    if _identity_in(line, identity)
                    and not re.search(r"\b(?:compared|versus|comparison|compatible with|fits|vs\.?|сравнение|подходит для)\b",
                                      normalized(line))), "")
    if not heading:
        return
    identities = [row[1] for row in parser.rows if len(row) == 2 and normalized(row[0]) in _SUBJECT_LABELS]
    if identities and any(not _identity_in(subject, identity) for subject in identities):
        return
    # An exact model and its product page may cover several articles, but refuse a comparison table.
    for row in parser.rows:
        if len(row) == 2:
            yield heading, row[0], row[1]


def _pdf_evidence(document: VendorDocument, identity: ProductIdentity, attribute: str):
    # Limit parsing to small official specifications. PDFs with no structured proof stay unknown.
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        pdf = PdfReader(BytesIO(document.content))
        if pdf.is_encrypted or len(pdf.pages) > _MAX_PAGES:
            return
        pages = [page.extract_text(extraction_mode="layout") for page in pdf.pages]
    except (PdfReadError, ValueError, RecursionError, TypeError, OSError):
        return
    if attribute == "battery_articles":
        for page in pages:
            if not re.search(r"\b(?:ups model|model number|model)\s+.*replacement batter", page, re.I):
                continue
            for line in page.splitlines():
                match = re.fullmatch(r"\s*(\S+)\s+((?:APC)?RBC\d{1,5})(?:\s+.*)?", line, re.I | re.ASCII)
                if match and normalized(match[1]) == normalized(identity.article):
                    yield match[1], "replacement battery", match[2]
        return
    if (attribute != "dimensions_mm" or not pages or urlsplit(document.url).hostname != "psref.lenovo.com"
            or not re.fullmatch(r"/syspool/Sys/PDF/.+_Spec\.pdf", urlsplit(document.url).path, re.I)):
        return
    heading = next((line.strip() for line in pages[0].splitlines()[:12] if _identity_in(line, identity)), "")
    if not heading:
        return
    # Multi-model tables usually have one labeled column per product. Accept only a single
    # dimensions row whose cells are a label and a value for the document's subject.
    for page in pages:
        lines = page.splitlines()
        for i, line in enumerate(lines):
            parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
            if len(parts) == 2:
                yield heading, *parts
            elif i + 1 < len(lines) and lines[i + 1].strip():
                yield heading, line.strip(), lines[i + 1].strip()


def _verify_document(document: VendorDocument, identity: ProductIdentity, attribute: str):
    url = _official_url(document.url, identity.domains)
    if not url or not identity.subject or not _eligible_attribute(identity.kind, attribute):
        return None, "identity_mismatch"
    host = urlsplit(url).hostname
    matches = {}
    chassis_model = ""
    if document.content_type == "application/pdf" and len(document.content) <= _MAX_PDF:
        try:
            evidence = tuple(_pdf_evidence(document, identity, attribute))
        except ImportError:
            return None, "document_extraction_failed"
    elif document.content_type in {"text/html", "text/plain", "application/xhtml+xml"} and len(document.content) <= _MAX_HTML:
        try:
            if (host == "www.asus.com" and identity.kind == "laptop" and attribute == "dimensions_mm"
                    and urlsplit(url).path.rstrip("/").endswith("/techspec")):
                parsed, status = _asus_table_evidence(document, identity)
                if not parsed:
                    return None, status
                matches[parsed[0]] = parsed[1]
                if identity.chassis_source and _identifier_in(parsed[1].split(" — ", 1)[0], identity.chassis_model):
                    chassis_model = identity.chassis_model
                evidence = ()
            else:
                evidence = tuple(_html_evidence(document, identity, attribute))
        except (ValueError, AssertionError):
            return None, "document_extraction_failed"
    else:
        return None, "document_extraction_failed"
    for subject, label, value in evidence:
        result = _extract_label_value(label, value, identity, attribute, host)
        if result and _valid_value(attribute, result):
            matches.setdefault(result, f"{subject} — {label} — {value}"[:600])
    if len(matches) != 1:
        return None, "conflicting_facts" if matches else "document_extraction_failed"
    value, excerpt = next(iter(matches.items()))
    if not _valid_value(attribute, value):
        return None, "document_extraction_failed"
    return VendorFact(identity.manufacturer, identity.product_model, identity.article, attribute, value,
                      _UNITS[attribute], url, excerpt[:600], datetime.now(UTC).isoformat(), identity.kind,
                      chassis_model=chassis_model), "verified"


def verify_document(document: VendorDocument, identity: ProductIdentity, attribute: str) -> VendorFact | None:
    return _verify_document(document, identity, attribute)[0]


def _document_urls(identity: ProductIdentity) -> tuple[str, ...]:
    vendor = _manufacturer_key(identity.manufacturer)
    article = identity.article
    if vendor == "asus" and identity.chassis_source and identity.spec_source:
        return (identity.spec_source,)
    if vendor == "apc" and identity.kind == "ups" and _ARTICLE.fullmatch(article):
        return (f"https://www.se.com/uk/en/product/{quote(article, safe='')}/",)
    if vendor == "lenovo" and identity.kind == "laptop":
        candidates = []
        if identity.exact_model:
            model = re.sub(r"^Lenovo\s+", "", identity.product_model, flags=re.I).strip()
            series = next((family for family in ("ThinkBook", "ThinkPad", "IdeaPad", "Yoga", "Legion", "LOQ")
                           if model.casefold().startswith(family.casefold() + " ")), "Lenovo")
            if re.fullmatch(r"[\w -]{5,65}", model):
                if series != "Lenovo":
                    model = series + model[len(series):]
                slug = quote(model.replace(" ", "_"), safe="_-")
                if series == "Lenovo":
                    slug = f"Lenovo_{slug}"
                candidates.append(f"https://psref.lenovo.com/syspool/Sys/PDF/{series}/{slug}/{slug}_Spec.PDF")
        if _ARTICLE.fullmatch(article):
            candidates.append(f"https://psref.lenovo.com/Detail/?M={quote(article, safe='')}")
        return tuple(candidates)
    return ()


def _lenovo_identity_url(identity: ProductIdentity) -> str | None:
    if (_manufacturer_key(identity.manufacturer) != "lenovo" or identity.exact_model or identity.kind != "laptop"
            or not re.fullmatch(r"[a-z0-9]{10}", identity.article, re.I | re.ASCII)
            or not re.fullmatch(r"IdeaPad 3 \d{2}[a-z]{3}\d{1,2}", identity.product_model, re.I | re.ASCII)):
        return None
    model = identity.product_model.lower().replace(" ", "-")
    article = identity.article.lower()
    return f"https://pcsupport.lenovo.com/us/en/products/laptops-and-netbooks/3-series/{model}/{article[:4]}/{article}/repair"


def _is_lenovo_catalog_model(document: VendorDocument, identity: ProductIdentity) -> bool:
    if document.content_type != "text/html" or urlsplit(document.url).hostname != "pcsupport.lenovo.com":
        return False
    # The title alone mirrors URL parameters. Read Lenovo's product record, which
    # includes the exact MTM and a manufacturer SKU specification for real variants.
    html = document.content.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"product"\s*:\s*(?=\{)', html):
        try:
            product, _ = decoder.raw_decode(html, match.end())
        except (ValueError, RecursionError):
            continue
        name = product.get("Name")
        ids = product.get("IDs")
        if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], str)
                and normalized(ids[0]) == normalized(identity.article) and isinstance(name, str)
                and _identifier_in(name, identity.article) and _identifier_in(name, identity.product_model)
                and product.get("Type") == "Product.Model" and product.get("Specification")):
            return True
    return False


class VendorEvidenceService:
    def __init__(
        self, cache_path: Path = DEFAULT_CACHE, *, ttl_seconds: float = DEFAULT_TTL_SECONDS,
        fetch: Callable[[str, tuple[str, ...], float], VendorDocument] = fetch_vendor_document,
        now: Callable[[], float] = time,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.cache_path = cache_path
        self._ttl = ttl_seconds
        self._fetch = fetch
        self._now = now
        self._lock = Lock()
        self._in_flight: dict[tuple[str, str, str], Future[EvidenceResult]] = {}
        self._failures: OrderedDict[tuple, tuple[float, str]] = OrderedDict()
        with closing(self._connect()) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS vendor_fact (
                    manufacturer TEXT NOT NULL, subject TEXT NOT NULL, attribute TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, expires_at REAL NOT NULL,
                    PRIMARY KEY (manufacturer, subject, attribute)
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.cache_path, timeout=0.25, autocommit=True)

    def _read(self, identity: ProductIdentity, attribute: str) -> VendorFact | None:
        with closing(self._connect()) as connection:
            row = connection.execute("""
                SELECT payload FROM vendor_fact
                WHERE manufacturer = ? AND subject = ? AND attribute = ? AND kind = ? AND expires_at > ?
            """, (*identity.key, attribute, identity.kind, self._now())).fetchone()
        return _decode_fact(row[0], identity) if row else None

    def cached(self, identity: ProductIdentity, attribute: str) -> VendorFact | None:
        if not identity.subject or not _eligible_attribute(identity.kind, attribute):
            return None
        try:
            exact = self._read(identity, attribute)
            if exact or not identity.chassis_source or attribute != "dimensions_mm":
                return exact
            shared = [fact for fact in self.active_facts() if fact.applies_to(identity) and fact.attribute == attribute]
            return shared[0] if shared and len({fact.value for fact in shared}) == 1 else None
        except sqlite3.Error:
            return None

    def active_facts(self) -> tuple[VendorFact, ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT payload, kind FROM vendor_fact WHERE expires_at > ?", (self._now(),)
                ).fetchall()
        except sqlite3.Error:
            return ()
        return tuple(fact for payload, kind in rows if (fact := _decode_fact(payload, kind=kind)))

    def _save(self, identity: ProductIdentity, fact: VendorFact) -> None:
        with closing(self._connect()) as connection:
            connection.execute("""
                INSERT INTO vendor_fact(manufacturer, subject, attribute, kind, payload, expires_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(manufacturer, subject, attribute) DO UPDATE
                SET kind=excluded.kind, payload=excluded.payload, expires_at=excluded.expires_at
            """, (*identity.key, fact.attribute, identity.kind, json.dumps(asdict(fact), ensure_ascii=False),
                  self._now() + self._ttl))

    def search(
        self, identity: ProductIdentity, attribute: str, *, deadline: float,
        discover: Callable[[str, tuple[str, ...], float], Iterable[str]] | None = None, refresh: bool = False,
        strategy: str = "exact_article",
    ) -> EvidenceResult:
        if (not identity.subject or not identity.domains or not _eligible_attribute(identity.kind, attribute)
                or strategy not in SEARCH_STRATEGIES):
            return EvidenceResult(None, "unsupported_lookup")
        if deadline <= monotonic():
            return EvidenceResult(None, "timeout")
        key = (*identity.key, attribute)
        failure_key = (*key, strategy, discover is not None)
        with self._lock:
            if not refresh and (fact := self.cached(identity, attribute)):
                return EvidenceResult(fact, "verified", True)
            if not refresh and (failure := self._failures.get(failure_key)):
                expires, status = failure
                if expires > monotonic():
                    return EvidenceResult(None, status, True)
                del self._failures[failure_key]
            pending = self._in_flight.get(key)
            if not pending:
                owner = self._in_flight[key] = Future()
        if pending:
            try:
                result = pending.result(timeout=max(0, deadline - monotonic()))
                return EvidenceResult(result.fact, result.status, result.fact is not None)
            except TimeoutError:
                return EvidenceResult(None, "timeout")
        result = EvidenceResult(None, "lookup_interrupted")
        try:
            result = self._lookup(identity, attribute, deadline, discover, strategy)
            if result.fact:
                self._save(identity, result.fact)
            return result
        except sqlite3.Error:
            result = replace(result, status="cache_unavailable")
            return result
        finally:
            with self._lock:
                if result.fact:
                    self._failures.pop(failure_key, None)
                elif result.status in {"source_unavailable", "search_no_results", "document_extraction_failed",
                                       "identity_mismatch", "conflicting_facts", "timeout"}:
                    self._failures[failure_key] = (monotonic() + _LOOKUP_RETRY_SECONDS, result.status)
                    self._failures.move_to_end(failure_key)
                    if len(self._failures) > _SHORT_CACHE_ENTRIES:
                        self._failures.popitem(last=False)
                owner.set_result(result)
                del self._in_flight[key]

    def _lookup(self, identity, attribute, deadline, discover, strategy) -> EvidenceResult:
        visited = set()
        attempted = 0
        download_ms = extraction_ms = 0.0
        outcomes = set()

        def read(raw):
            nonlocal attempted, download_ms
            url = _official_url(raw, identity.domains)
            if not url or url in visited or attempted == _MAX_FETCH_URLS or monotonic() >= deadline:
                return None
            visited.add(url)
            attempted += 1
            started = monotonic()
            try:
                return self._fetch(url, identity.domains, deadline)
            except ValueError:
                outcomes.add("document_extraction_failed")
                return None
            except (TimeoutError, URLError, OSError):
                outcomes.add("source_unavailable")
                return None
            finally:
                download_ms += (monotonic() - started) * 1000

        identity_source = ""
        if attribute == "dimensions_mm" and (identity_url := _lenovo_identity_url(identity)):
            document = read(identity_url)
            if document and _is_lenovo_catalog_model(document, identity):
                verified_url = _official_url(document.url, identity.domains)
                if verified_url:
                    identity = replace(identity, exact_model=True)
                    identity_source = verified_url

        def verify(urls):
            nonlocal extraction_ms
            for raw in islice(urls, _MAX_FETCH_URLS * 3):
                if attempted == _MAX_FETCH_URLS or monotonic() >= deadline:
                    break
                document = read(raw)
                if not document or monotonic() >= deadline:
                    continue
                started = monotonic()
                fact, status = _verify_document(document, identity, attribute)
                extraction_ms += (monotonic() - started) * 1000
                if fact and monotonic() < deadline:
                    return replace(fact, identity_source_url=identity_source)
                outcomes.add(status)
            return None

        fact = verify(_document_urls(identity))
        model = identity.chassis_model if identity.chassis_source else identity.product_model if identity.exact_model else ""
        exact = identity.article or model
        description = _ATTRIBUTE_QUERIES[attribute]
        model_query = f"{model} technical specifications {description}" if model else ""
        alternate = f"{model} official regional technical specifications {description}" if model else ""
        article_query = f"{exact} {description}"
        queries = {
            "exact_article": (article_query, model_query),
            "verified_model": (model_query, article_query),
            "official_technical_specifications": (model_query, alternate),
            "official_regional_page": (alternate, model_query),
        }
        if discover:
            for query in dict.fromkeys((*queries[strategy], article_query)):
                if fact or attempted == _MAX_FETCH_URLS or monotonic() >= deadline:
                    break
                if query:
                    # Search selects URLs. Only fetched and scoped manufacturer text supplies values.
                    fact = verify(discover(f"{identity.manufacturer} {query}"[:250], identity.domains, deadline))
        status = "verified" if fact else "timeout" if monotonic() >= deadline else next(
            (code for code in ("conflicting_facts", "identity_mismatch", "document_extraction_failed", "source_unavailable")
             if code in outcomes), "search_no_results",
        )
        return EvidenceResult(fact, status, download_attempts=attempted, download_ms=round(download_ms, 3),
                              extraction_ms=round(extraction_ms, 3))


def _decode_fact(payload: str, identity: ProductIdentity | None = None, *, kind: str = "") -> VendorFact | None:
    try:
        values = json.loads(payload)
        if not isinstance(values, dict) or not isinstance(values.get("value"), list):
            return None
        values["value"] = tuple(values["value"])
        fact = VendorFact(**values)
        subject = identity or ProductIdentity(fact.manufacturer, fact.product_model, fact.article, kind, True)
        return fact if fact.applies_to(subject) else None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def enrich_selected(rules, evidence: VendorEvidenceService, offer_ids: Iterable[str], *, refresh=False, timeout=8):
    for offer_id in offer_ids:
        identity = rules.product_identity(offer_id)
        details = rules.product_details(offer_id)
        for attribute in details["missing_attributes"]:
            result = evidence.search(identity, attribute, deadline=monotonic() + timeout, refresh=refresh)
            yield {
                "offer_id": offer_id, "attribute": attribute, "status": result.status,
                "cache_hit": result.cache_hit, "fact": result.fact.public_data() if result.fact else None,
            }


def main() -> None:
    from .recommendation import RecommendationAgent

    parser = argparse.ArgumentParser(description="Prepare verified product facts before running the demo")
    parser.add_argument("--catalog", type=Path, default=Path("Mindbox.yml.gz.xml"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--offer-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    rules = RecommendationAgent(args.catalog)
    evidence = VendorEvidenceService(args.cache)
    if any(rules.get_offer(offer_id) is None for offer_id in args.offer_id):
        parser.error("Unknown --offer-id")
    ids = args.offer_id or list(islice((
        offer_id for scenario in ("laptop_bag", "ups_battery", "office_antivirus")
        for offer_id in rules.anchor_ids(scenario) if rules.product_details(offer_id)["missing_attributes"]
    ), args.limit))
    for row in enrich_selected(rules, evidence, ids, refresh=args.refresh):
        print(json.dumps(row, ensure_ascii=True))


if __name__ == "__main__":
    main()
