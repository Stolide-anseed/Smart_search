import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html import unescape
from pathlib import Path

from .catalog import Offer, catalog_elements, lexical_terms
from .query import Condition, MEMORY_TYPE_PATTERN, brand_keys, number, product_type
from .query_normalization import query_key
from .recommendation_facts import Fields
from .recommendation_licenses import office_scope


_NUMBER = r"\d+(?:[.,]\d+)?"
_EMPTY_VALUES = {"другое или отсутствует", "не определено", "неизвестно"}
_GB = re.compile(rf"(?<![\w.])({_NUMBER})\s*(тб|tb|гб|gb)(?!\w)")
_MEMORY_CHIP_LAYOUT = re.compile(rf"\(\s*{_NUMBER}\s*(?:гб|gb)\s*\*\s*\d+\s*\)")
_KNOWN_CATEGORY_TYPES = {
    "ноутбуки": "laptop", "видеокарты": "gpu", "оперативная память": "ram",
    "мониторы": "monitor", "телевизоры": "tv", "процессоры": "cpu",
    "материнские платы": "motherboard", "блоки питания": "psu", "коммутаторы": "switch",
    "жесткие диски": "hdd", "твердотельные накопители": "ssd", "ssd-накопители": "ssd",
    "сумки и чехлы для ноутбуков": "notebook_bag", "шредеры": "shredder",
    "антивирусы для дома": "antivirus", "антивирусы для организаций": "antivirus",
    "офисные пакеты программ microsoft office": "office", "офисные программы": "office",
    "ип-видеокамеры": "ip_camera", "ip-видеокамеры": "ip_camera",
    "источники бесперебойного питания": "ups", "веб-камеры": "webcam",
    "беспроводные маршрутизаторы wi-fi": "router", "маршрутизаторы": "router",
    "внутренние твердотельные накопители (ssd)": "ssd", "внешние hdd": "hdd",
    "флешки usb": "flash_drive", "сменные батареи для ибп": "ups_battery",
    "ибп (источники бесперебойного питания)": "ups", "3g/lte-роутеры": "modem",
    "проводные телефоны": "landline", "ip-телефоны": "landline", "телефоны dect": "landline",
    "серверное по": "software", "rack-серверы": "server", "микросерверы": "server",
    "другое офисное по": "office", "российское офисное по (импортозамещение)": "office",
    "кабели, переходники, разъемы": "cable", "кабели": "cable", "вебкамеры": "webcam",
    "смартфоны": "smartphone", "зарядные устройства": "charger", "десктопы": "desktop",
    "игровые манипуляторы": "gamepad", "колонки": "speakers", "домофоны": "intercom",
    "наушники и микрофоны": "headphones",
}
_NUMERIC_PARAMS = {
    "memory_gb": ("емкость одного модуля", "объем памяти", "объем установленной озу"),
    "storage_gb": ("емкость диска", "объем установленного ssd", "объем установленного hdd"),
    "power_w": ("мощность бп", "мощность блока питания", "эффективная мощность", "мощность"),
    "diagonal_in": ("диагональ экрана", "диагональ", "максимальная диагональ"),
    "length_cm": ("длина кабеля",),
    "devices": ("к-во узлов", "к-во устройств"),
    "users": ("к-во пользователей до", "количество пользователей"),
    "term_months": ("срок действия",),
}
_TEXT_PARAMS = {
    "license_type": ("тип лицензии",), "platform": ("платформа", "поддерживаемая ос"),
    "memory_type": ("тип памяти",), "form_factor": ("форм-фактор",),
    "connector": ("входной разъем", "выходной разъем", "интерфейс", "интерфейсы"),
    "memory_usage": ("назначение памяти",), "version": ("версия",),
    "installed_os": ("операционная система",), "connection": ("тип подключения", "подключение"),
}
_ATTR_KEYS = {key for keys in (*_NUMERIC_PARAMS.values(), *_TEXT_PARAMS.values()) for key in keys} | {
    "тип устройства", "тип продукта", "назначение", "модель процессора", "серия ноутбука",
    "тип программного обеспечения", "состав комплекта",
}
_MODEL = re.compile(r"(?<!\w)(?:(?:rtx|gtx|rx)\s+)?\d{4}(?:\s+(?:ti|xt|super))?(?!\w)")
_PRICE_CURRENCIES = {"price_rub": {"RUR", "RUB"}, "price_usd": {"USD"}, "price_eur": {"EUR"}}
SOFTWARE_CATEGORY_ID = "software"
# The YML taxonomy keeps software in several roots. SaaS is included, while IaaS,
# service offerings and e-books also use the "софт" marker outside these roots.
_SOFTWARE_ROOTS = {
    "652", "5676", "416", "371", "182", "2052", "719", "4291", "2502", "188", "872",
    "226", "19145", "8", "5960", "11765", "327028", "440598", "4010", "639", "631",
}


@dataclass(frozen=True, slots=True)
class OfferDetails:
    description: str
    attributes: dict[str, list[str]]


def _parse_quantity(text: str, field: str) -> Decimal | None:
    match = re.search(_NUMBER, text)
    if not match:
        return None
    value = number(match.group())
    if field in {"storage_gb", "memory_gb"} and re.search(r"(?<!\w)(?:тб|tb)(?!\w)", text):
        value *= 1000
    if field == "term_months" and re.search(r"год\w*|\bлет\b", text):
        value *= 12
    if field == "length_cm" and re.search(r"(?<!\w)(?:м|m|метр\w*)(?!\w)", text):
        value *= 100
    return value


class QueryCatalog:
    def __init__(self, path: Path, offers: tuple[Offer, ...]) -> None:
        by_id = {offer.offer_id: offer for offer in offers}
        self.categories: dict[str, str] = {}
        category_parents: dict[str, str] = {}
        self.details: dict[str, OfferDetails] = {}
        self._cached: dict[tuple[str, str], set[str | Decimal]] = {}
        self._product_types: dict[str, set[str]] = {}
        for element in catalog_elements(path):
            offer_id = element.get("id", "")
            if element.tag == "category":
                self.categories[offer_id] = (element.text or "").strip()
                category_parents[offer_id] = (element.get("parentId") or "").strip()
                continue
            if offer_id not in by_id or element.findtext("categoryId") != by_id[offer_id].category_id:
                continue
            if (element.findtext("url") or "").strip() != by_id[offer_id].url:
                continue
            name = element.findtext("name") or element.findtext("model") or ""
            if " ".join(name.split()) != by_id[offer_id].name or (element.findtext("price") or "").strip() != by_id[offer_id].price:
                continue
            attrs: dict[str, list[str]] = {}
            for param in element.findall("param"):
                key = query_key(param.get("name") or "").strip()
                raw = query_key(param.text or "").strip()
                if key in _ATTR_KEYS and raw and raw not in _EMPTY_VALUES:
                    attrs.setdefault(key, []).append(raw)
            description = query_key(unescape(element.findtext("description") or ""))
            self.details[offer_id] = OfferDetails(description, attrs)
        if offers and len(self.details) != len(offers):
            raise ValueError("Каталог условий не совпадает с индексом. Укажите тот же XML или пересоберите индекс.")

        children: dict[str, list[str]] = {}
        for category_id, parent in category_parents.items():
            if parent:
                children.setdefault(parent, []).append(category_id)
        self._software_categories = _SOFTWARE_ROOTS.copy()
        pending = list(_SOFTWARE_ROOTS)
        while pending:
            for category_id in children.get(pending.pop(), ()):
                if category_id not in self._software_categories:
                    self._software_categories.add(category_id)
                    pending.append(category_id)

    def is_software_category(self, category_id: str) -> bool:
        return category_id == SOFTWARE_CATEGORY_ID or category_id in self._software_categories

    def _classify_product(self, offer: Offer) -> set[str]:
        name = query_key(offer.name)
        label = query_key(self.categories.get(offer.category_id, ""))
        category_type = _KNOWN_CATEGORY_TYPES.get(label, "")
        if category_type == "laptop" and re.match(r"планшет", name):
            return {"tablet"}
        if category_type == "modem":
            return {"modem", "router"}
        if label == "клавиатуры, мыши, комплекты":
            detail = self.details.get(offer.offer_id)
            types = {"мышь": "mouse", "клавиатура": "keyboard", "клавиатура+мышь": "keyboard_set"}
            values = detail.attributes.get("тип устройства", ()) if detail else ()
            if device_types := {types[value] for value in values if value in types}:
                return device_types
        if category_type:
            return {category_type}
        details = self.details.get(offer.offer_id)
        if details and "софт" in details.attributes.get("тип продукта", ()):
            return {"software"}
        device_type = product_type(name)[0] or product_type(label)[0]
        if device_type == "router" and re.search(r"\b(?:lte|4g)\b", name):
            return {"router", "modem"}
        return {device_type} if device_type else set()

    def _product_type(self, offer: Offer) -> set[str]:
        if offer.offer_id not in self._product_types:
            self._product_types[offer.offer_id] = self._classify_product(offer)
        return self._product_types[offer.offer_id]

    def product_types(self, offer: Offer) -> frozenset[str]:
        return frozenset(self._product_type(offer))

    def purposes(self, offer: Offer) -> frozenset[str]:
        detail = self.details.get(offer.offer_id)
        return frozenset(
            value for key in ("назначение", "назначение памяти") for value in detail.attributes.get(key, ())
        ) if detail else frozenset()

    def office_role(self, offer: Offer) -> str:
        detail = self.details.get(offer.offer_id)
        title = query_key(unescape(offer.name))
        _, role = office_scope(Fields(detail.attributes if detail else {}, {}, title, "", ""), True)
        if role == "not_office_suite":
            # A project planner and a diagram editor may share the office category.
            return " ".join(lexical_terms(title)[:2])
        return role

    def similar_model(self, offer: Offer) -> str:
        title = query_key(unescape(offer.name))
        detail = self.details.get(offer.offer_id)
        processors = detail.attributes.get("модель процессора", ()) if detail else ()
        series = detail.attributes.get("серия ноутбука", ()) if detail else ()
        if len(processors) == len(series) == 1 and "laptop" in self._product_type(offer):
            model, separator, _ = title.partition(processors[0])
            if separator and re.search(rf"(?<!\w){re.escape(series[0])}(?!\w)", model):
                title = model
        return " ".join(title.strip(" ,.-").split())

    def _values(self, offer: Offer, field: str) -> set[str | Decimal]:
        if field == "product_type":
            return self._product_type(offer)
        if field == "category_id":
            return {offer.category_id} if offer.category_id else set()
        if field == "software_category":
            details = self.details.get(offer.offer_id)
            kinds = details.attributes.get("тип продукта", ()) if details else ()
            if offer.category_id in self._software_categories and all(kind == "софт" for kind in kinds):
                return {SOFTWARE_CATEGORY_ID}
            return set()
        if field == "brand":
            return set(brand_keys(offer.vendor))
        if field == "article":
            return {query_key(offer.vendor_code)} if offer.vendor_code else set()
        if field in _PRICE_CURRENCIES:
            if offer.currency not in _PRICE_CURRENCIES[field] or not offer.price:
                return set()
            try:
                value = number(offer.price)
            except InvalidOperation:
                return set()
            return {value} if value.is_finite() and value > 0 else set()
        if field == "model":
            return {query_key(offer.name), query_key(self.categories.get(offer.group_id, ""))} - {""}

        title = query_key(offer.name)
        raw = self.details.get(offer.offer_id)
        description = raw.description if raw else ""
        params = raw.attributes if raw else {}
        context = title + " • " + description
        if field in _NUMERIC_PARAMS:
            if field == "memory_gb" and (kit := re.search(r"\b(\d+)\s*[xх×]\s*(\d+)\s*(?:гб|gb)\b", title)):
                return {number(kit[1]) * number(kit[2])}
            values = {
                value for key in _NUMERIC_PARAMS[field] for text in params.get(key, ())
                if (value := _parse_quantity(text, field)) is not None
            }
            values = values or self._fallback_number(field, context, title)
            return values if len(values) == 1 else set()

        labels = " ".join(value for key in _TEXT_PARAMS.get(field, ()) for value in params.get(key, ()))
        if field == "version":
            versions = set(re.findall(r"(?<![\w.])\d+(?:\.\d+){0,2}(?![\w.])", labels))
            versions.update(re.findall(r"\b(?:верси[яию]|version)\s+(\d+(?:\.\d+){0,2})(?![\w.])", title))
            versions.update(re.findall(r"\b(?:19|20)\d{2}\b", title))
            return versions
        if field == "installed_os":
            if re.search(r"\bбез ос\b|\bбез операционной системы\b", labels + " " + context):
                return {"none"}
            if re.search(r"\bwindows\b|\blinux\b|\bmacos\b", labels or context):
                return {"installed"}
            return set()
        if field == "connection":
            source = labels or context
            wireless = re.search(r"\bбеспроводн\w*|\bwireless\b|\bbluetooth\b", source)
            wired = re.search(r"\bпроводн\w*|\bwired\b", source)
            return {name for name, present in (("wireless", wireless), ("wired", wired)) if present}
        if field == "license_type":
            values = set()
            if re.search(r"бессрочн|perpetual", labels + " " + title):
                values.add("perpetual")
            if re.search(r"подписк|subscription", labels + " " + title):
                values.add("subscription")
            return values if len(values) == 1 else set()
        if field == "platform":
            found = re.search(r"(?:^|[•;])\s*платформа\s*:\s*([^•;]+)", description)
            source = labels + " " + (found.group(1) if found else "")
            return set(re.findall(r"windows(?:\s+(?:10|11))?|astra\s*linux|linux|macos|android|ios", source))
        if field == "memory_type":
            return set(MEMORY_TYPE_PATTERN.findall(labels or context))
        if field == "form_factor":
            return {
                value.replace(" ", "-").replace("sodimm", "so-dimm")
                for value in re.findall(r"\b(?:m\.2|so[- ]?dimm|dimm|atx|sfx)\b", labels or title)
            } - ({"dimm"} if re.search(r"so[- ]?dimm", labels or title) else set())
        if field == "memory_usage":
            return {"laptop" if "ноутбук" in labels else "desktop" if "десктоп" in labels else labels} if labels else set()
        if field == "connector":
            return {re.sub(r"[ -]", "", match) for match in re.findall(
                r"\busb[- ]?c\b|\bhdmi\b|\bdisplayport\b|\brj[- ]?45\b", labels + " " + context
            )}
        return set()

    @staticmethod
    def _fallback_number(field: str, context: str, title: str) -> set[Decimal]:
        patterns = {
            "power_w": rf"(?<!\w)({_NUMBER})\s*(?:вт|w)(?!\w)",
            "diagonal_in": rf"(?<![\w.])({_NUMBER})\s*(?:\"|″|in(?:ch)?\b|дюйм\w*)",
            "length_cm": rf"(?<!\w)({_NUMBER}\s*(?:м|m|см|cm))(?!\w)",
            "devices": r"\bна\s+(\d+)\s+(?:пк|устройств\w*|рабоч\w+ мест\w*)(?!\w)",
            "users": r"\bна\s+(\d+)\s+пользовател\w*(?!\w)",
            "term_months": rf"\bна\s+({_NUMBER}\s*(?:год\w*|лет|мес\.?|месяц\w*))(?!\w)",
        }
        if field in {"memory_gb", "storage_gb"}:
            # Prefer quantities tied to the requested specification rather than any GB in a device title.
            if field == "memory_gb" and re.search(r"ddr|озу|оперативн|видеокарт", context):
                source = _MEMORY_CHIP_LAYOUT.sub("", context)
                source = re.sub(r"(?:ssd|hdd)\s+\d+(?:[.,]\d+)?\s*(?:гб|gb|тб|tb)", "", source)
            elif field == "storage_gb" and re.search(r"ssd|hdd|диск|накопител|флеш|флэш", context):
                source = re.sub(r"\d+(?:[.,]\d+)?\s*(?:гб|gb)\s+(?:ddr\d|озу)", "", context)
            else:
                return set()
            return {number(value) * (1000 if unit in {"тб", "tb"} else 1) for value, unit in _GB.findall(source)}
        pattern = patterns.get(field)
        if not pattern:
            return set()
        source = title if field in {"devices", "users", "term_months"} else context
        return {value for match in re.finditer(pattern, source) if (value := _parse_quantity(match[1], field)) is not None}

    def _cached_values(self, offer: Offer, field: str) -> set[str | Decimal]:
        key = (offer.offer_id, field)
        values = self._cached.get(key)
        if values is None:
            values = self._values(offer, field)
            self._cached[key] = values
        return values

    def feature_values(self, offer: Offer, field: str) -> frozenset[str | Decimal]:
        return frozenset(self._cached_values(offer, field))

    def check(self, offer: Offer, condition: Condition) -> bool | None:
        values = self._cached_values(offer, condition.field)
        expected = condition.value
        if not values:
            return None
        if condition.field == "model" and isinstance(expected, str):
            if _MODEL.fullmatch(expected) and "gpu" in self._product_type(offer):
                expected = re.sub(r"^(?:rx|rtx|gtx)\s+", "", expected)
                models = {re.sub(r"^(?:rx|rtx|gtx)\s+", "", value)
                          for value in _MODEL.findall(query_key(offer.name))}
                return expected in models if models else None
            pattern = re.compile(r"(?<!\w)" + re.escape(expected) + r"(?!\w)")
            return any(pattern.search(str(value)) for value in values)
        if condition.field == "platform" and isinstance(expected, str):
            expected = expected.replace("astralinux", "astra linux")
            values = {str(value).replace("astralinux", "astra linux") for value in values}
            if expected == "windows":
                values = {value.split()[0] for value in values}
            elif expected.startswith("windows ") and "windows" in values and not any(value.startswith("windows ") for value in values):
                return None
        if condition.field == "connector" and isinstance(expected, str):
            expected = re.sub(r"[ -]", "", expected)
            if expected not in values and "cable" not in self._product_type(offer):
                return None
        if condition.operator == "eq":
            return expected in values
        if condition.operator == "ne":
            return expected not in values
        numbers = [value for value in values if isinstance(value, Decimal)]
        if not numbers or not isinstance(expected, Decimal):
            return None
        if condition.operator == "ge":
            return any(value >= expected for value in numbers)
        if condition.operator == "gt":
            return any(value > expected for value in numbers)
        if condition.operator == "le":
            return any(value <= expected for value in numbers)
        return any(value < expected for value in numbers)
