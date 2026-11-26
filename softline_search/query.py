import re
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Literal

from .query_normalization import QueryChange, QueryNormalizer, query_key


Operator = Literal["eq", "ne", "ge", "gt", "le", "lt"]


@dataclass(frozen=True, slots=True)
class Condition:
    field: str
    operator: Operator
    value: str | Decimal
    source: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    original: str
    normalized: str
    search_text: str
    conditions: tuple[Condition, ...]
    changes: tuple[QueryChange, ...]
    ambiguities: tuple[str, ...] = ()


# The same product terms identify explicit intent and classify catalog titles.
PRODUCT_TYPES = {
    "laptop": r"ноутбук\w*|лэптоп\w*|ультрабук\w*",
    "tablet": r"планшет\w*|планшетн\w+ пк",
    "notebook_bag": r"сумк\w* (?:для|под) ноутбук\w*|чехол (?:для|под) ноутбук\w*",
    "ram": r"оперативн\w+ памят\w*|модул\w* памят\w*|памят\w* (?:для|под) ноутбук\w*|(?:so[ -]?dimm|dimm)",
    "gpu": r"видеокарт\w*|видеоадаптер\w*",
    "ssd": r"ssd|твердотельн\w+ накопител\w*",
    "hdd": r"(?:жестк\w+ диск\w*|hdd)",
    "flash_drive": r"флешк\w*|флэш\S*|usb[ -]накопител\w*",
    "cpu": r"процессор\w*",
    "motherboard": r"материнск\w+ плат\w*|системн\w+ плат\w*",
    "psu": r"блок\w* питани\w*",
    "ups_battery": r"(?:сменн\w+\s+)?(?:батаре\w+|аккумулятор\w*) (?:для|к|на) (?:ибп|источник\w* бесперебойного питания)",
    "ups": r"ибп|источник\w* бесперебойного питания",
    "monitor": r"монитор\w*",
    "tv": r"телевизор\w*",
    "mouse": r"мыш[ьи]\w*",
    "keyboard": r"клавиатур\w*",
    "keyboard_set": r"клавиатур\w* и мыш\w*|комплект\w+ (?:из )?клавиатур\w* и мыш\w*",
    "router": r"маршрутизатор\w*|роутер\w*",
    "modem": r"модем\w*",
    "switch": r"коммутатор\w*",
    "webcam": r"веб[ -]?камер\w*",
    "ip_camera": r"(?:ip[ -]?)?камер\w+ видеонаблюдени\w*|видеокамер\w*|ip[ -]?камер\w*",
    "intercom": r"видеодомофон\w*|домофон\w*",
    "smartphone": r"смартфон\w*|мобильн\w+ телефон\w*|телефон\w*",
    "landline": r"проводн\w+ телефон\w*|(?:ip|sip|dect)[ -]телефон\w*|системн\w+ телефон\w*",
    "cable": r"кабел\w*|провод\w*",
    "charger": r"зарядн\w+ устройств\w*|зарядк\w*",
    "speakers": r"колонк\w*|акустическ\w+ систем\w*",
    "headphones": r"наушник\w*|гарнитур\w*",
    "gamepad": r"геймпад\w*|джойстик\w*|игров\w+ контроллер\w*",
    "chair": r"кресл\w*",
    "shredder": r"шредер\w*",
    "server": r"сервер\w*",
    "desktop": r"настольн\w+ компьютер\w*|системн\w+ блок\w*|компьютер\w*",
    "antivirus": r"антивирус\w*",
    "office": r"офисн\w+ (?:по|пакет\w*|программ\w*)|microsoft office|office\s+20\d{2}|офис",
}
TYPE_PATTERNS = {kind: re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)") for kind, pattern in PRODUCT_TYPES.items()}

_CATALOG_BRAND_ALIASES = {
    "kaspersky": ("kaspersky", "лаборатория касперского"),
    "microsoft": ("microsoft",), "dr.web": ("dr.web", "доктор веб"),
    "content ai": ("content ai", "контент ии", "abbyy"),
}
_NUM = r"(?:\d{1,3}(?:[ \u202f]\d{3})+|\d+(?:[.,]\d+)?)"
_COMPARISON = r"не\s+(?:более|менее|дороже|дешевле)|не\s+выше|не\s+ниже|минимум|максимум|свыше|меньше|больше|от|до"
_MODIFIERS: dict[str, Operator] = {
    "от": "ge", "не менее": "ge", "не ниже": "ge", "минимум": "ge", "не дешевле": "ge",
    "до": "le", "не более": "le", "не выше": "le", "максимум": "le", "не дороже": "le",
    "больше": "gt", "свыше": "gt", "меньше": "lt",
}
_THOUSANDS = r"(?:тыс\.?|тысяч(?:а|и|у)?|[кk])"
_CURRENCY = r"(?:₽|руб\.?|рубл(?:ей|я|ь)|р\.|\$|usd|доллар\w*|€|eur|евро)"
_BUDGET = re.compile(
    rf"(?<!\w)(?P<cmp>{_COMPARISON}|в пределах|бюджет(?:ом)?(?: до)?)\s+"
    rf"(?P<num>{_NUM})\s*(?P<thousands>{_THOUSANDS})?\s*(?P<currency>{_CURRENCY})?(?![\w.,])"
)
_PRICE_RANGE = re.compile(
    rf"(?<!\w)(?:от\s+(?P<lower>{_NUM})\s*(?P<lower_scale>{_THOUSANDS})?\s+до\s+|"
    rf"(?P<dash_lower>{_NUM})\s*(?P<dash_scale>{_THOUSANDS})?\s*[-–—]\s*)"
    rf"(?P<upper>{_NUM})\s*(?P<scale>{_THOUSANDS})?\s*(?P<currency>{_CURRENCY})?(?![\w.,])"
)
_UNIT = r'tб|тб|tb|гб|gb|гигабайт\w*|гиг(?:а|ов)?|ватт?\w*|вт|w|дюйм\w*|inch(?:es)?|["″]|см|cm|метр\w*|м|m'
_MEASURE = re.compile(rf"(?<![\w.])(?:(?P<cmp>{_COMPARISON})\s+)?(?P<num>{_NUM})\s*(?P<unit>{_UNIT})(?!\w)")
_MEASURE_RANGE = re.compile(
    rf"(?<![\w.])(?:от\s+(?P<lower>{_NUM})\s*(?P<lower_unit>{_UNIT})?\s+до\s+|"
    rf"(?P<dash_lower>{_NUM})\s*(?P<dash_unit>{_UNIT})?\s*[-–—]\s*)"
    rf"(?P<upper>{_NUM})\s*(?P<unit>{_UNIT})(?!\w)"
)
_MEASUREMENT_SUFFIX = re.compile(rf"\s*(?:{_UNIT}|пк|устройств\w*|пользовател\w*|узл\w*)(?!\w)")
_SEATS = re.compile(
    rf"(?<!\w)(?:(?P<cmp>{_COMPARISON})\s+|на\s+)?(?P<num>{_NUM})\s*"
    r"(?P<unit>пк|компьютер\w*|устройств\w*|рабоч(?:их|ее|ие) мест\w*|пользовател\w*|узл\w*)(?!\w)"
)
_TERM = re.compile(rf"(?<!\w)(?:(?P<cmp>{_COMPARISON})\s+|на\s+)(?:(?P<num>{_NUM})\s+)?(?P<unit>год\w*|лет|мес\.?|месяц\w*)(?!\w)")
_PLATFORM = re.compile(r"\b(?:для|под|без)\s+(windows(?:\s+(?:10|11))?|linux|astra linux|macos|android|ios)(?!\w)")
_LICENSE = re.compile(r"(?<!\w)(бессрочн\w*|раз и навсегда|без подписки|по подписке|с подпиской|подписка)(?!\w)")
_PREFERENCE = re.compile(r"(?:желательно|предпочтительно|по возможности)")
_CONNECTOR = re.compile(r"(?<![\w.])(?:usb[- ]?c|hdmi|displayport|rj[- ]?45)(?![\w.])")
MEMORY_TYPE_PATTERN = re.compile(r"(?<!\w)(?:g?ddr[3456][lx]?|lpddr[345]x?)(?!\w)")
_FORM_FACTORS = re.compile(r"(?<!\w)(?:m\.2|so[- ]?dimm)(?!\w)")
_VERSION = re.compile(r"(?<!\w)(?:верси[яию]|version)\s+(\d+(?:\.\d+){0,2})(?![\w.])")
_CONNECTION = re.compile(r"(?<!\w)(без провода|беспроводн\w*|проводн\w*)(?!\w)")


def number(value: str) -> Decimal:
    return Decimal("".join(value.split()).replace(",", "."))


def _price_field(currency: str | None) -> str:
    if currency in {"$", "usd"} or (currency and currency.startswith("доллар")):
        return "price_usd"
    if currency in {"€", "eur", "евро"}:
        return "price_eur"
    return "price_rub"


def _quantity(text: str, kind: str, unit: str, raw: str, start: int) -> tuple[str, Decimal] | None:
    value = number(raw)
    if unit in {"тб", "tb", "tб", "гб", "gb"} or unit.startswith("гиг"):
        tb = unit in {"тб", "tb", "tб"}
        points_to_disk = re.search(r"\b(?:ssd|hdd|диск\w*)\s*$", text[:start])
        field = "storage_gb" if kind in {"ssd", "hdd", "flash_drive"} or tb or points_to_disk else "memory_gb"
        return field, value * (1000 if tb else 1)
    if unit in {"вт", "w"} or unit.startswith("ват"):
        return "power_w", value
    if unit.startswith("дюйм") or unit in {"inch", "inches", '"', "″"}:
        return "diagonal_in", value
    if kind == "cable":
        return "length_cm", value * (1 if unit in {"см", "cm"} else 100)
    return None


def product_type(text: str) -> tuple[str, str]:
    """Return the main requested type rather than a device named after 'для'."""
    hits = []
    for kind, pattern in TYPE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            prefix = text[:match.start()]
            if re.search(r"(?:^|\s)(?:для|под|к|на)\s*$", prefix):
                continue
            hits.append((match.start(), -len(match.group()), kind, match.group()))
    if not hits:
        return "", ""
    _, _, kind, phrase = min(hits)
    return kind, phrase


@lru_cache(maxsize=2048)
def brand_keys(vendor: str) -> frozenset[str]:
    text = query_key(vendor).strip(" «»\"")
    keys = {canonical for canonical, aliases in _CATALOG_BRAND_ALIASES.items() if any(alias in text for alias in aliases)}
    if text:
        keys.add(text)
    return frozenset(keys)


def category_key(text: str) -> str:
    return re.sub(r"^(?:программы|программное обеспечение) для ", "по для ", query_key(text))


class QueryParser:
    def __init__(
        self, normalizer: QueryNormalizer, vendors: set[str], families: set[str], categories: dict[str, str]
    ) -> None:
        self.normalizer = normalizer
        brands = set(_CATALOG_BRAND_ALIASES)
        for vendor in vendors:
            brands.update(key for key in brand_keys(vendor) if 3 <= len(key) <= 30 and key not in {"сеть", "мир", "компьютер"})
        self._brand_pattern = re.compile(
            r"(?<![\w./-])(?:" + "|".join(re.escape(key) for key in sorted(brands, key=len, reverse=True)) + r")(?![\w./-])"
        )
        self._family_names = {query_key(name) for name in families if len(name) >= 4}
        self._category_ids: dict[str, list[str]] = {}
        for category_id, name in categories.items():
            self._category_ids.setdefault(category_key(name), []).append(category_id)

    def parse(self, query: str) -> ParsedQuery:
        if not query.strip():
            raise ValueError("Укажите непустой запрос.")
        normalized, changes = self.normalizer.normalize(query)
        changes = list(changes)
        kind, _ = product_type(query_key(normalized))
        rewrite = []
        if kind in {"ssd", "hdd", "flash_drive", "ram", "gpu"}:
            rewrite += [(r"(?<!\w)на терабайт(?!\w)", "на 1 ТБ"), (r"(?<=\d)\s+г(?!\w)", " ГБ")]
        if kind == "ram":
            rewrite.append((r"\bна восьм[её]рку\b", "на 8 ГБ"))
        if kind in {"notebook_bag", "monitor", "tv"}:
            rewrite.append((r"\b(\d{2})\s+и\s+(\d)\b", r"\1,\2 дюйма"))
        if kind == "cable":
            rewrite.append((r"\bметра три\b", "3 м"))
        for pattern, replacement in rewrite:
            def replace(match: re.Match[str]) -> str:
                after = match.expand(replacement)
                changes.append(QueryChange("единицы", match.group(), after))
                return after
            normalized = re.sub(pattern, replace, normalized, flags=re.IGNORECASE)
        text = query_key(normalized)
        conditions: list[Condition] = []
        alternatives = bool(re.search(r"\bили\b", text))

        def add(field: str, op: Operator, value: str | Decimal, start: int, end: int) -> None:
            clause = re.split(r"[,;.!?]|\bно\b", text[:start])[-1]
            conditions.append(Condition(
                field, op, value, normalized[start:end],
                field != "prefer_lower_price" and not _PREFERENCE.search(clause)
                and (not alternatives or field.startswith("price_")),
            ))

        kind, phrase = product_type(text)
        category_ids = self._category_ids.get(category_key(text), [])
        if len(category_ids) == 1:
            add("category_id", "eq", category_ids[0], 0, len(text))
        elif kind:
            start = text.index(phrase)
            add("product_type", "eq", kind, start, start + len(phrase))
        for match in self._brand_pattern.finditer(text):
            prefix = text[:match.start()]
            if not re.search(r"\b(?:для|под|к|на)\s*$", prefix):
                add("brand", "eq", match.group(), *match.span())
                break
        for token in re.finditer(r"[^\W_]+(?:[-._/][^\W_]+)*\+?", text):
            if token.group() in self.normalizer.codes and any(ch.isdecimal() for ch in token.group()):
                add("article", "eq", token.group(), *token.span())
                break
        # Exact catalog family names and explicit title tokens carry model intent.
        model = re.search(r"\bмодель\s+([a-zа-я][a-zа-я\d./+-]{2,})(?!\w)", text)
        if model:
            add("model", "eq", model[1], *model.span())
        elif kind == "gpu":
            model = re.search(r"(?<![\w.])(?:rx\s+|rtx\s+|gtx\s+)?\d{4}(?:\s+(?:xt|ti|super))?(?![\w.₽])", text)
            if model and not re.search(r"\b(?:до|от)\s*$", text[:model.start()]):
                add("model", "eq", model.group(), *model.span())
        elif kind == "office" and (suite := re.search(r"\bmicrosoft office\b|\bофис microsoft\b", text)):
            add("model", "eq", "microsoft office", *suite.span())
        elif not category_ids and text in self._family_names:
            add("model", "eq", text, 0, len(text))
        if version := _VERSION.search(text):
            add("version", "eq", version[1], *version.span())
        elif kind == "office" and (year := re.search(r"\b(?:19|20)\d{2}\b", text)):
            add("version", "eq", year.group(), *year.span())

        budget_spans = self._budgets(text, add)
        self._attributes(text, kind, add, budget_spans)
        search = normalized
        for start, end in sorted(budget_spans, reverse=True):
            search = search[:start] + " " + search[end:]
        search = " ".join(search.strip(" ,;-").split()) or normalized
        ambiguities = ("В запросе перечислены альтернативы",) if alternatives else ()
        return ParsedQuery(query, normalized, search, tuple(conditions), tuple(changes), ambiguities)

    @staticmethod
    def _budgets(text: str, add) -> list[tuple[int, int]]:
        spans = []
        named = re.search(r"\bдо\s+полтинника\b", text)
        if named:
            add("price_rub", "le", Decimal(50000), *named.span())
            spans.append(named.span())
        for match in _PRICE_RANGE.finditer(text):
            lower, upper = number(match["lower"] or match["dash_lower"]), number(match["upper"])
            lower_scale = match["lower_scale"] or match["dash_scale"] or match["scale"]
            if not (match["currency"] or lower_scale or lower >= 1000):
                continue
            if not (match["currency"] or lower_scale) and _MEASUREMENT_SUFFIX.match(text[match.end():]):
                continue
            field = _price_field(match["currency"])
            add(field, "ge", lower * (1000 if lower_scale else 1), *match.span())
            add(field, "le", upper * (1000 if match["scale"] or lower_scale else 1), *match.span())
            spans.append(match.span())
        for match in _BUDGET.finditer(text):
            if any(start <= match.start() < end for start, end in spans):
                continue
            value = number(match["num"])
            if not (match["thousands"] or match["currency"]):
                if value < 1000 or _MEASUREMENT_SUFFIX.match(text[match.end():]):
                    continue
            add(_price_field(match["currency"]), _MODIFIERS.get(match["cmp"], "le"),
                value * (1000 if match["thousands"] else 1), *match.span())
            spans.append(match.span())
        if match := re.search(r"\b(?:недорог\w*|бюджетн\w*)\b", text):
            add("prefer_lower_price", "eq", "yes", *match.span())
        return spans

    @staticmethod
    def _attributes(text: str, kind: str, add, budget_spans: list[tuple[int, int]]) -> None:
        spans = list(budget_spans)
        for match in _MEASURE_RANGE.finditer(text):
            if any(start <= match.start() < end for start, end in spans):
                continue
            lower = _quantity(text, kind, match["lower_unit"] or match["dash_unit"] or match["unit"], match["lower"] or match["dash_lower"], match.start())
            upper = _quantity(text, kind, match["unit"], match["upper"], match.start())
            if lower and upper and lower[0] == upper[0]:
                add(lower[0], "ge", lower[1], *match.span())
                add(upper[0], "le", upper[1], *match.span())
                spans.append(match.span())
        for match in _MEASURE.finditer(text):
            if any(start <= match.start() < end for start, end in spans):
                continue
            quantity = _quantity(text, kind, match["unit"], match["num"], match.start())
            if quantity:
                add(quantity[0], _MODIFIERS.get(match["cmp"], "eq"), quantity[1], *match.span())
        if kind == "ram" and (purpose := re.search(r"\b(?:для|под) ноутбук\w*\b", text)):
            add("memory_usage", "eq", "laptop", *purpose.span())
        for match in _SEATS.finditer(text):
            if kind not in {"", "office", "antivirus"} and "лиценз" not in text:
                continue
            field = "users" if match["unit"].startswith("пользовател") else "devices"
            add(field, _MODIFIERS.get(match["cmp"], "ge"), number(match["num"]), *match.span())
        for match in _TERM.finditer(text):
            value = number(match["num"] or "1")
            add("term_months", _MODIFIERS.get(match["cmp"], "eq"), value * (12 if match["unit"].startswith("год") or match["unit"] == "лет" else 1), *match.span())
        for match in _LICENSE.finditer(text):
            term = match.group()
            value = "perpetual" if term.startswith("бессроч") or term == "раз и навсегда" else "subscription"
            add("license_type", "ne" if term == "без подписки" else "eq", value, *match.span())
        for match in _PLATFORM.finditer(text):
            add("platform", "ne" if match.group().startswith("без ") else "eq", match[1], *match.span())
        if kind in {"laptop", "desktop", "tablet"} and (no_os := re.search(r"\bбез\s+(?:ос|операционной системы)\b", text)):
            add("installed_os", "eq", "none", *no_os.span())
        if kind in {"mouse", "keyboard", "headphones", "speakers", "gamepad"} and (connection := _CONNECTION.search(text)):
            value = "wired" if connection.group().startswith("провод") else "wireless"
            add("connection", "eq", value, *connection.span())
        for field, pattern in (("memory_type", MEMORY_TYPE_PATTERN), ("form_factor", _FORM_FACTORS), ("connector", _CONNECTOR)):
            for match in pattern.finditer(text):
                prefix = text[:match.start()]
                add(field, "ne" if re.search(r"\bбез\s*$", prefix) else "eq", match.group().replace(" ", "-").replace("sodimm", "so-dimm"), *match.span())
