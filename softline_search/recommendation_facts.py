import json
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from html import unescape
from pathlib import Path
from typing import Generic, TypeVar
from xml.etree.ElementTree import Element

from .query import number
from .query_normalization import query_key


T = TypeVar("T")
Dimensions = tuple[Decimal, Decimal, Decimal]  # Width, depth, thickness in millimeters.
_EMPTY = {"", "другое или отсутствует", "не определено", "неизвестно", "нет данных", "—"}
_LINE_BREAK = re.compile(r"</(?:p|div|li|tr)>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^<>]*>")
_UNPUNCTUATED = re.compile(
    r"^(к-во (?:узлов|устройств|пользователей)|количество (?:узлов|устройств|пк|пользователей)|"
    r"число рабочих мест)\s+((?:от\s+)?\d.*)$"
)
_NUMBER_UNIT = r"(\d+(?:[.,]\d+)?)\s*(мм|mm|см|cm)?"
_SEPARATOR = r"\s*[×xх*]\s*"
_DIMENSIONS = re.compile(_SEPARATOR.join([_NUMBER_UNIT] * 3))
_AXES = re.compile(_SEPARATOR.join([r"([шгдтвwdhtl])"] * 3))
_AXIS_NAMES = dict(zip("шгдтвwdhtl", ("w", "d", "d", "h", "h", "w", "d", "h", "h", "d"), strict=True))
LAPTOP_DIMENSIONS = (
    "габариты ноутбука", "размеры ноутбука", "габариты корпуса", "размеры корпуса",
    "габариты без упаковки", "размеры без упаковки", "габариты", "размеры",
)
BAG_LIMITS = (
    "внутренние размеры", "внутренние габариты", "внутренние размеры сумки",
    "внутренние размеры отделения", "размеры отделения для ноутбука",
    "размеры внутреннего отделения", "максимальные размеры ноутбука",
    "максимальные габариты ноутбука", "рекомендуемые максимальные размеры ноутбука",
    "recommended max laptop dimension", "laptop compartment dimensions",
)


class Status(StrEnum):
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class Fact(Generic[T]):
    value: T | None = None
    sources: tuple[str, ...] = ()
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class Validation:
    status: Status
    reason_code: str
    reason: str = ""
    facts: dict | None = None
    missing: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


def resolve(values: list[tuple[T, str]], official: list[tuple[T, str]] | None = None) -> Fact[T]:
    reliable = official or values
    distinct = {value for value, _ in reliable}
    all_sources = tuple(dict.fromkeys(source for _, source in [*reliable, *values]))
    if len(distinct) != 1:
        return Fact(sources=all_sources, conflict=len(distinct) > 1)
    value = next(iter(distinct))
    sources = tuple(dict.fromkeys(source for _, source in reliable))
    return Fact(value, sources, any(other != value for other, _ in values))


def normalized(text: str) -> str:
    return query_key(" ".join(unescape(unescape(text)).split()))


def product_key(vendor: str, article: str) -> tuple[str, str]:
    vendor = normalized(vendor).strip(" «»\"")
    if vendor in {"hp inc", "hp inc."}:
        vendor = "hp"
    return vendor, normalized(article)


@dataclass(slots=True)
class Fields:
    parameters: dict[str, list[str]]
    description: dict[str, list[str]]
    title: str
    model: str
    text: str

    def values(self, *labels: str) -> list[str]:
        return [value for value, _ in self.evidence(*labels)]

    def evidence(self, *labels: str) -> list[tuple[str, str]]:
        return [
            (value, f"{source}: {label}")
            for mapping, source in ((self.parameters, "XML param"), (self.description, "XML описание"))
            for label in labels for value in mapping.get(label, ())
        ]


def read_fields(element: Element) -> Fields:
    params = defaultdict(list)
    for parameter in element.findall("param"):
        label = normalized(parameter.get("name", ""))
        value = normalized(parameter.text or "")
        if label and value not in _EMPTY:
            params[label].append(value)

    text = _LINE_BREAK.sub("\n", unescape(unescape(element.findtext("description") or "")))
    text = _TAG.sub("", text)
    description = defaultdict(list)
    for segment in re.split(r"[•\n]|;\s*(?=[^;:]{1,50}:)", text):
        segment = normalized(segment)
        label, separator, value = segment.partition(":")
        if separator and label and value.strip() not in _EMPTY:
            description[label.strip()].append(value.strip())
        elif match := _UNPUNCTUATED.fullmatch(segment):
            description[match[1]].append(match[2])
    title = element.findtext("name") or element.findtext("model") or ""
    return Fields(dict(params), dict(description), normalized(title), normalized(element.findtext("model") or ""), normalized(text))


def load_specs(path: Path) -> dict[tuple[str, str, str], list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal, parse_int=Decimal)
    index = defaultdict(list)
    for spec in records:
        if not spec.get("source"):
            raise ValueError("Для паспортной характеристики требуется URL производителя.")
        articles = [spec.get("article", ""), *spec.get("articles", [])]
        for article in dict.fromkeys(articles):
            if article:
                index[(spec["type"], *product_key(spec["vendor"], article))].append(spec)
    return dict(index)


def spec_values(specs: list[dict], key: str, parse) -> list[tuple]:
    return [(parse(spec[key]), spec["source"]) for spec in specs if key in spec]


def _parse_dimensions(value: str, label: str) -> Dimensions | None:
    axes = _AXES.search(label) or _AXES.search(value)
    if axes and axes.string == value:
        value = (value[:axes.start()] + value[axes.end():]).strip(" ():,;")
    match = _DIMENSIONS.fullmatch(value.strip(" ()"))
    if not match:
        return None
    values = list(zip(match.groups()[::2], match.groups()[1::2], strict=True))
    unit_hint = re.search(r"\b(мм|mm|см|cm)\b", label)
    units = {unit for _, unit in values if unit}
    if len(units) > 1 and any(not unit for _, unit in values):
        return None
    shared_unit = next(iter(units), unit_hint[1] if unit_hint else None)
    if not shared_unit:
        return None
    dims = tuple(number(value) * (10 if (unit or shared_unit) in {"см", "cm"} else 1) for value, unit in values)
    if axes:
        names = [_AXIS_NAMES[axis] for axis in axes.groups()]
        if set(names) != {"w", "d", "h"}:
            return None
        dims = tuple(dims[names.index(axis)] for axis in "wdh")
    return dims if all(value > 0 for value in dims) else None


def dimensions(fields: Fields, labels: tuple[str, ...], specs: list[dict]) -> Fact[Dimensions]:
    official = spec_values(specs, "dimensions_mm", tuple)
    if any(len(values) != 3 or any(not value.is_finite() or value <= 0 for value in values) for values, _ in official):
        raise ValueError("Паспортные габариты должны содержать три положительных значения в миллиметрах.")
    candidates = []
    for mapping, source in ((fields.parameters, "XML param"), (fields.description, "XML описание")):
        for label, values in mapping.items():
            if not any(label == name or label.startswith((name + " (", name + ",")) for name in labels):
                continue
            for value in values:
                if dims := _parse_dimensions(value, label):
                    candidates.append((dims, f"{source}: {label}"))
    for text, source in ((fields.title, "XML название"), (fields.model, "XML model")):
        for label in labels:
            if match := re.search(r"(?<!\w)" + re.escape(label) + r"\s*:\s*([^;]+)", text):
                if dims := _parse_dimensions(match[1], label):
                    candidates.append((dims, f"{source}: {label}"))
    return resolve(candidates, official)


def diagonal(fields: Fields, labels: tuple[str, ...], specs: list[dict]) -> Fact[Decimal]:
    candidates = []
    for value, source in fields.evidence(*labels):
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)(?:\s*(?:дюйм\w*|in|[\"″]))?", value)
        if match and number(match[1]) > 0:
            candidates.append((number(match[1]), source))
    return resolve(candidates, spec_values(specs, "diagonal_inches", Decimal))


def models(fields: Fields, labels: tuple[str, ...], specs: list[dict]) -> Fact[frozenset[str]]:
    candidates = [(frozenset(part.strip() for part in re.split(r"[,;]", value) if part.strip()), source)
                  for value, source in fields.evidence(*labels)]
    # Several param entries may enumerate independent compatible models.
    official = spec_values(specs, "compatible_models", lambda values: frozenset(normalized(value) for value in values))
    if official:
        return resolve(candidates, official)
    return Fact(frozenset.union(frozenset(), *(value for value, _ in candidates)),
                tuple(dict.fromkeys(source for _, source in candidates)))
