import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from xml.etree.ElementTree import Element, iterparse


@dataclass(frozen=True, slots=True)
class Offer:
    offer_id: str
    group_id: str
    category_id: str
    name: str
    vendor: str
    vendor_code: str
    vendor_url: str
    price: str
    currency: str
    url: str


@dataclass(frozen=True, slots=True)
class Document:
    offer: Offer
    title: str
    details: str

    @property
    def text(self) -> str:
        return ". ".join(part for part in (self.title, self.details) if part)


_WORD = re.compile(r"[^\W_]+")
_COMPOUND = re.compile(r"[^\W_]+(?:[./_-][^\W_]+)+")
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_PLACEHOLDERS = {"другое или отсутствует", "не определено"}


def lexical_terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    terms = _WORD.findall(normalized)
    for compound in _COMPOUND.finditer(normalized):
        value = compound.group()
        if any(char.isdecimal() for char in value):
            terms.append("".join(_WORD.findall(value)))
    return terms


def _category_path(category_id: str, categories: dict[str, tuple[str, str]]) -> str:
    names = []
    seen = set()
    while category_id in categories and category_id not in seen:
        seen.add(category_id)
        name, category_id = categories[category_id]
        if name:
            names.append(name)
    return " > ".join(reversed(names))


def _field(element: Element, tag: str) -> str:
    return " ".join((element.findtext(tag) or "").split())


def read_offer(element: Element) -> Offer:
    return Offer(
        offer_id=element.attrib["id"],
        group_id=(element.get("group_id") or _field(element, "group_id")).strip(),
        category_id=_field(element, "categoryId"),
        name=_field(element, "name") or _field(element, "model"),
        vendor=_field(element, "vendor"),
        vendor_code=_field(element, "vendorCode"),
        vendor_url=_field(element, "vendorUrl"),
        price=_field(element, "price"),
        currency=_field(element, "currencyId"),
        url=_field(element, "url"),
    )


def _read_document(element: Element, categories: dict[str, tuple[str, str]]) -> Document:
    offer = read_offer(element)
    description = " ".join(_HTML_TAG.sub(" ", unescape(_field(element, "description"))).split())
    attributes = []
    for param in element.findall("param"):
        value = " ".join((param.text or "").split())
        if value and value.casefold() not in _PLACEHOLDERS:
            name = (param.get("name") or "").strip()
            attributes.append(f"{name}: {value}" if name else value)
    group_name, _ = categories.get(offer.group_id, ("", ""))
    if group_name.casefold() in offer.name.casefold():
        group_name = ""
    details = ". ".join(
        part
        for part in (
            group_name,
            offer.vendor,
            _category_path(offer.category_id, categories),
            description,
            ", ".join(dict.fromkeys(attributes)),
        )
        if part
    )
    return Document(offer, offer.name or offer.offer_id, details)


def catalog_elements(path: Path) -> Iterator[Element]:
    offers_element = None
    with path.open("rb") as stream:
        for event, element in iterparse(stream, events=("start", "end")):
            if event == "start":
                if element.tag == "offers":
                    offers_element = element
            elif element.tag == "category":
                yield element
            elif element.tag == "offer":
                yield element
                if offers_element is not None:
                    offers_element.remove(element)


def read_catalog(path: Path) -> list[Document]:
    categories = {}
    documents = []
    for element in catalog_elements(path):
        if element.tag == "category":
            categories[element.attrib["id"]] = (
                (element.text or "").strip(),
                element.get("parentId", ""),
            )
        elif element.get("available", "").lower() == "true":
            documents.append(_read_document(element, categories))
    return documents
