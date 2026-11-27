"""Universal cross-sell directions, interface normalization and reusable validators.

The LLM may propose any registered direction.  This module turns catalog facts into
normalized interface/port evidence and applies deterministic validators.  A missing
attribute never proves compatibility: the only successful outcome is CONFIRMED.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .recommendation_facts import Fact, Fields, Status, Validation, normalized


LEGACY_DIRECTION_IDS = ("ups_battery", "laptop_bag", "office_antivirus")


@dataclass(frozen=True, slots=True)
class Direction:
    id: str
    label: str
    candidate_kind: str
    category_ids: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    validator: str
    requirements: tuple[str, ...]
    query_terms: tuple[str, ...]
    description: str


DIRECTIONS: dict[str, Direction] = {
    "laptop_mouse": Direction(
        "laptop_mouse", "Мышь для ноутбука", "mouse", ("82833",), ("laptop",), "interface",
        ("Интерфейс мыши (USB-A, USB-C или Bluetooth)", "Свободный соответствующий порт ноутбука"),
        ("мыш", "mouse"), "Проводная или беспроводная мышь, подключаемая к ноутбуку.",
    ),
    "laptop_keyboard": Direction(
        "laptop_keyboard", "Клавиатура для ноутбука", "keyboard", ("82833",), ("laptop",), "interface",
        ("Интерфейс клавиатуры (USB-A, USB-C или Bluetooth)", "Свободный соответствующий порт ноутбука"),
        ("клавиатур", "keyboard"), "Внешняя клавиатура для ноутбука.",
    ),
    "laptop_keyboard_mouse": Direction(
        "laptop_keyboard_mouse", "Комплект клавиатура и мышь", "keyboard_mouse", ("82833",), ("laptop",), "interface",
        ("Интерфейс комплекта (USB-A, USB-C или Bluetooth)", "Свободный соответствующий порт ноутбука"),
        ("комплект клавиатур", "клавиатура+мышь", "клавиатура и мышь", "keyboard+mouse"),
        "Комплект из клавиатуры и мыши.",
    ),
    "laptop_headset": Direction(
        "laptop_headset", "Гарнитура для ноутбука", "headset", ("88831",), ("laptop",), "interface",
        ("Интерфейс гарнитуры (USB-A, USB-C, 3,5 мм или Bluetooth)", "Свободный соответствующий порт ноутбука"),
        ("гарнитур", "наушник", "headset", "headphone"), "Гарнитура с микрофоном для звонков и работы.",
    ),
    "laptop_monitor": Direction(
        "laptop_monitor", "Монитор для ноутбука", "monitor", ("80843",), ("laptop",), "interface",
        ("Видеовход монитора (HDMI, DisplayPort или VGA)", "Соответствующий видеовыход ноутбука"),
        ("монитор", "monitor", "дисплей"), "Внешний монитор с проверенным видеовходом.",
    ),
    "laptop_usb_hub": Direction(
        "laptop_usb_hub", "USB-концентратор для ноутбука", "usb_hub", ("84206",), ("laptop",), "interface",
        ("Тип входного порта концентратора (USB-A или USB-C)", "Свободный соответствующий порт ноутбука"),
        ("usb-хаб", "usb хаб", "концентратор", "хаб", "usb hub"), "USB-концентратор с подтверждённым входным портом.",
    ),
}
DIRECTION_IDS = (*LEGACY_DIRECTION_IDS, *DIRECTIONS)
UNIVERSAL_DIRECTION_IDS = tuple(DIRECTIONS)

_LEGACY_QUERY_TERMS = {
    "laptop_bag": ("сумк", "чехол", "рюкзак", "bag"),
    "ups_battery": ("батаре", "аккумулятор", "battery"),
    "office_antivirus": ("антивирус", "antivirus", "защит"),
}
_GENERIC_TERMS = ("аксессуар", "допродаж", "докупить", "перифери", "что еще", "что ещё", "оборудован")


def direction(direction_id: str) -> Direction | None:
    return DIRECTIONS.get(direction_id)


def requested_direction_ids(text: str) -> tuple[str, ...]:
    """Return every direction explicitly mentioned in a user query, in stable order."""
    value = normalized(text)
    found = []
    for identifier, terms in _LEGACY_QUERY_TERMS.items():
        if any(term in value for term in terms):
            found.append(identifier)
    for identifier, item in DIRECTIONS.items():
        if any(term in value for term in item.query_terms):
            found.append(identifier)
    return tuple(dict.fromkeys(found))


def generic_accessory_query(text: str) -> bool:
    value = normalized(text)
    return any(term in value for term in _GENERIC_TERMS)


def direction_applies(item: Direction, anchor_kind: str) -> bool:
    return anchor_kind in item.anchor_kinds


def candidate_kinds(item: Direction) -> tuple[str, ...]:
    # A combo set is a keyboard replacement, but it is not a mouse replacement.
    if item.id == "laptop_keyboard":
        return ("keyboard", "keyboard_mouse")
    return (item.candidate_kind,)


# ---------------------------------------------------------------------------
# Ports and interfaces


_PORT_KEYS = ("usb_a", "usb_c", "hdmi", "displayport", "vga", "rj45", "audio_35", "thunderbolt", "ps2")
_PORT_LABELS = {
    "usb_a": "USB Type-A", "usb_c": "USB Type-C", "hdmi": "HDMI", "displayport": "DisplayPort",
    "vga": "VGA", "rj45": "RJ45", "audio_35": "Audio 3,5 мм", "thunderbolt": "Thunderbolt",
    "ps2": "PS/2", "bluetooth": "Bluetooth",
}
_PORT_FIELDS = {
    "разъемы type-c": "usb_c", "разъемы usb type-c": "usb_c", "разъемы usb-c": "usb_c",
    "разъемы usb 2.0": "usb_a", "разъемы usb 3.0": "usb_a", "разъемы usb 3.1": "usb_a",
    "разъемы usb 3.2": "usb_a", "разъемы usb": "usb_a",
    "разъемы hdmi": "hdmi", "разъемы displayport": "displayport", "разъемы vga": "vga",
    "разъемы rj45": "rj45", "audio mini-jack": "audio_35", "разъемы thunderbolt": "thunderbolt",
    "разъемы ps/2": "ps2", "разъемы ps2": "ps2",
}
_NUMBER = re.compile(r"\d+")


def _count(value: str) -> int | None:
    text = normalized(value)
    if text in {"нет", "отсутствует", "0"}:
        return 0
    if match := _NUMBER.search(text):
        try:
            return int(match[0])
        except ValueError:
            return None
    if text in {"есть", "да"}:
        return 1
    return None


@dataclass(frozen=True, slots=True)
class PortFacts:
    ports: dict[str, int] = field(default_factory=dict)
    absent: frozenset[str] = frozenset()
    bluetooth: str | None = None
    usb_c_dp_alt: bool | None = None
    usb_c_pd: bool | None = None
    os: str | None = None
    sources: tuple[str, ...] = ()

    def has(self, key: str) -> bool | None:
        if key == "bluetooth":
            return None if self.bluetooth is None else True
        if self.ports.get(key, 0) > 0:
            return True
        if key in self.absent or key in self.ports:
            return False
        return None

    def count(self, key: str) -> int | None:
        state = self.has(key)
        if state is None:
            return None
        return self.ports.get(key, 0) if state else 0

    def missing(self) -> tuple[str, ...]:
        missing = [key for key in (*_PORT_KEYS, "bluetooth") if self.has(key) is None]
        if self.usb_c_dp_alt is None:
            missing.append("usb_c_dp_alt")
        if self.usb_c_pd is None:
            missing.append("usb_c_pd")
        return tuple(missing)

    def public(self) -> dict:
        return {
            "ports": {key: self.ports[key] for key in _PORT_KEYS if self.ports.get(key, 0) > 0},
            "absent": sorted(self.absent),
            "bluetooth": self.bluetooth,
            "usb_c_dp_alt": self.usb_c_dp_alt,
            "usb_c_pd": self.usb_c_pd,
            "os": self.os,
        }


def _bluetooth_version(value: str) -> str | None:
    text = normalized(value)
    if "bluetooth" not in text:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    return match[1].replace(",", ".") if match else "yes"


def _os_name(value: str) -> str | None:
    text = normalized(value)
    if not text or "без ос" in text or "нет ос" in text:
        return None
    for name in ("windows", "linux", "macos", "android", "ios"):
        if name in text:
            return name
    return None


def read_port_facts(fields: Fields, specs: list[dict] | tuple[dict, ...] = ()) -> PortFacts:
    ports: dict[str, int] = {}
    absent: set[str] = set()
    sources: list[str] = []
    bluetooth = None
    usb_c_dp_alt = None
    usb_c_pd = None
    os_name = None

    for mapping, source in ((fields.parameters, "XML param"), (fields.description, "XML описание")):
        for label, values in mapping.items():
            key = _PORT_FIELDS.get(label)
            if key is None and label.startswith(("разъемы usb", "разъемы type", "разъемы usb type")):
                key = "usb_c" if "type-c" in label or "type c" in label else "usb_a"
            if key:
                for value in values:
                    count = _count(value)
                    if count is None:
                        continue
                    if count <= 0:
                        absent.add(key)
                    else:
                        ports[key] = max(ports.get(key, 0), count)
                    sources.append(f"{source}: {label}")
            if label in {"беспроводное соединение", "интегрированные устройства", "wireless"}:
                for value in values:
                    if version := _bluetooth_version(value):
                        bluetooth = version
                        sources.append(f"{source}: {label}")
            if label in {"операционная система", "os"}:
                for value in values:
                    if parsed := _os_name(value):
                        os_name = parsed
                        sources.append(f"{source}: {label}")

    for spec in specs:
        official = spec.get("ports")
        if isinstance(official, dict):
            for key, value in official.items():
                if key == "bluetooth":
                    bluetooth = str(value) if value not in (None, "", False) else None
                    continue
                if key == "usb_c_dp_alt":
                    usb_c_dp_alt = bool(value)
                    continue
                if key == "usb_c_pd":
                    usb_c_pd = bool(value)
                    continue
                if key == "os":
                    os_name = _os_name(str(value)) if value else None
                    continue
                if key not in _PORT_KEYS:
                    continue
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    continue
                if count <= 0:
                    ports.pop(key, None)
                    absent.add(key)
                else:
                    ports[key] = count
                    absent.discard(key)
            sources.append(str(spec["source"]))
        if "bluetooth" in spec:
            bluetooth = str(spec["bluetooth"])
            sources.append(str(spec["source"]))
        if "usb_c_dp_alt" in spec:
            usb_c_dp_alt = bool(spec["usb_c_dp_alt"])
            sources.append(str(spec["source"]))
        if "usb_c_pd" in spec:
            usb_c_pd = bool(spec["usb_c_pd"])
            sources.append(str(spec["source"]))
        if "os" in spec:
            os_name = _os_name(str(spec["os"]))
            sources.append(str(spec["source"]))

    return PortFacts(ports, frozenset(absent), bluetooth, usb_c_dp_alt, usb_c_pd, os_name,
                     tuple(dict.fromkeys(sources)))


@dataclass(frozen=True, slots=True)
class InterfaceOption:
    kind: str
    label: str
    source: str


@dataclass(frozen=True, slots=True)
class CandidateInterfaces:
    kind: str
    options: tuple[InterfaceOption, ...] = ()
    requires: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(option.kind for option in self.options))


def _option_kind(text: str) -> tuple[str, ...]:
    value = normalized(text)
    found = []
    if any(token in value for token in ("type-c", "type c", "usb-c", "usb c", "usb type-c", "usb type c")):
        found.append("usb_c")
    if any(token in value for token in ("type-a", "type a", "usb-a", "usb a", "usb type-a", "usb type a")):
        found.append("usb_a")
    if "thunderbolt" in value:
        found.append("usb_c")
    if any(token in value for token in ("micro usb", "micro-usb", "usb 2.0 micro", "usb micro")):
        found.append("usb_micro")
    if "lightning" in value:
        found.append("lightning")
    if "rj9" in value:
        found.append("rj9")
    if "bluetooth" in value:
        found.append("bluetooth")
    if any(token in value for token in ("радиоканал", "2,4 ггц", "2.4 ghz", "2.4ггц", "wireless 2.4")):
        found.append("radio_24")
    if "ps/2" in value or "ps2" in value:
        found.append("ps2")
    if any(token in value for token in ("mini-jack", "mini jack", "3.5", "3,5", "aux", "audio")):
        found.append("audio_35")
    if "hdmi" in value:
        found.append("hdmi")
    if "displayport" in value or "display port" in value or re.search(r"\bdp\b", value):
        found.append("displayport")
    if "vga" in value or "d-sub" in value or "dsub" in value:
        found.append("vga")
    if "dvi" in value:
        found.append("dvi")
    if not found and re.search(r"\busb\b", value):
        # In this catalog bare USB 2.0/3.x on input devices denotes the usual
        # Type-A cable; Type-C is always labelled explicitly.
        found.append("usb_a")
    return tuple(dict.fromkeys(found))


def _options(values, source: str) -> list[InterfaceOption]:
    options = []
    for value in values:
        for kind in _option_kind(value):
            options.append(InterfaceOption(kind, value[:120], source))
    return options


def _candidate_platforms(fields: Fields) -> tuple[str, ...]:
    values = fields.values("поддерживаемые операционные системы", "поддерживаемые ос", "совместимость с ос",
                           "операционная система", "platform")
    text = normalized(" ".join(values))
    return tuple(name for name in ("windows", "linux", "macos", "android", "ios") if name in text)


def _upstream_options(offer, fields: Fields) -> list[InterfaceOption]:
    text = " ".join((offer.name, fields.text, " ".join(fields.values("описание"))))
    pattern = re.compile(
        r"(входящие|выходящие|вход\w*|выход\w*)\s*(\d+)?\s*[×x]?\s*usb\s*([\d.]+)?\s*(type-[ac])?",
        re.I,
    )
    sides = []
    for match in pattern.finditer(normalized(text)):
        count = int(match[2]) if match[2] else None
        sides.append((match[1], count, match[4] or ""))
    upstream = None
    if sides:
        ones = [side for side in sides if side[1] == 1]
        upstream = ones[0] if len(ones) == 1 else None
    if upstream and upstream[2]:
        kind = "usb_c" if "c" in upstream[2] else "usb_a"
        return [InterfaceOption(kind, f"upstream {upstream[2]}", "XML описание: входной порт")]
    # A model name such as "4USB-C" describes downstream ports, not necessarily
    # the upstream connector.  Without an explicit upstream type the hub stays UNKNOWN.
    return []


def parse_candidate_interfaces(
    kind: str, fields: Fields, offer, specs: list[dict] | tuple[dict, ...] = (),
) -> CandidateInterfaces:
    options: list[InterfaceOption] = []
    requires: list[str] = []
    sources: list[str] = []
    notes: list[str] = []
    platforms = _candidate_platforms(fields)

    if kind in {"mouse", "keyboard", "keyboard_mouse"}:
        values = fields.values("интерфейс", "тип подключения", "подключение")
        options = _options(values, "XML param: Интерфейс")
        if not options:
            options = _options((offer.name, fields.text), "XML описание")
    elif kind == "headset":
        headset_type = normalized(" ".join(fields.values("тип")))
        wireless = normalized(" ".join(fields.values("беспроводное соединение")))
        connection = normalized(" ".join(fields.values("подключение")))
        is_bluetooth = "bluetooth" in headset_type or "bluetooth" in wireless
        if is_bluetooth:
            options.append(InterfaceOption("bluetooth", wireless or headset_type, "XML param: Беспроводное соединение"))
            wired = "смешан" in connection or ("провод" in connection and "беспровод" not in connection)
            if wired:
                options.extend(_options(fields.values("интерфейс подключения"), "XML param: Интерфейс подключения"))
            else:
                notes.append("Type-C указан как порт зарядки, аудиоканал Bluetooth")
        else:
            options = _options(fields.values("интерфейс подключения", "подключение"), "XML param: Интерфейс подключения")
            if not options:
                options = _options((offer.name, fields.text), "XML описание")
    elif kind == "monitor":
        values = fields.values("интерфейсы", "разъемы", "видеовходы", "подключение")
        options = _options(values, "XML param: Интерфейсы")
        if not any(option.kind in {"hdmi", "displayport", "vga", "dvi"} for option in options):
            options = _options((offer.name, fields.text), "XML описание")
        if not any(option.kind in {"hdmi", "displayport", "vga", "dvi"} for option in options):
            if any(option.kind == "usb_c" for option in options):
                # A USB-C video input is not proof of DisplayPort Alt Mode.
                requires.append("usb_c_video")
                notes.append("USB-C видеовход не подтверждает совместимость без явного DP Alt Mode")
    elif kind == "usb_hub":
        options = _upstream_options(offer, fields)
        if not options:
            values = fields.values("входной интерфейс", "интерфейс", "подключение")
            options = _options(values, "XML param: Интерфейс")
        if any("power delivery" in normalized(value) or "pd " in normalized(value)
               for value in fields.values("интерфейс", "подключение", "описание")):
            requires.append("usb_c_pd")

    for spec in specs:
        upstream = spec.get("upstream")
        if isinstance(upstream, dict):
            for key, value in upstream.items():
                if key in {"usb_a", "usb_c"} and value:
                    options.append(InterfaceOption(key, f"upstream {key}", str(spec["source"])))
        for value in spec.get("interfaces", ()):
            for option_kind in _option_kind(str(value)):
                options.append(InterfaceOption(option_kind, str(value)[:120], str(spec["source"])))
        if spec.get("requires_usb_c_pd"):
            requires.append("usb_c_pd")
        if spec.get("requires_usb_c_dp_alt"):
            requires.append("usb_c_dp_alt")
        if spec.get("platforms"):
            platforms = tuple(dict.fromkeys((*platforms, *(normalized(value) for value in spec["platforms"]))))

    unique = tuple(dict.fromkeys((option.kind, option.label, option.source) for option in options))
    ordered = tuple(InterfaceOption(*value) for value in unique)
    sources = tuple(dict.fromkeys((*(option.source for option in ordered), *sources)))
    return CandidateInterfaces(kind, ordered, tuple(dict.fromkeys(requires)), platforms, sources, tuple(notes))


# ---------------------------------------------------------------------------
# Validators


_OPTION_PORT = {
    "usb_a": "usb_a", "usb_c": "usb_c", "bluetooth": "bluetooth", "radio_24": "usb_a",
    "audio_35": "audio_35", "ps2": "ps2", "hdmi": "hdmi", "displayport": "displayport", "vga": "vga",
}
_INCOMPATIBLE_OPTIONS = {"usb_micro", "rj9", "lightning", "dvi"}


def _state(anchor: PortFacts, option_kind: str) -> bool | None:
    if option_kind in _INCOMPATIBLE_OPTIONS:
        return False
    port = _OPTION_PORT.get(option_kind)
    return anchor.has(port) if port else False


def _port_text(anchor: PortFacts, key: str) -> str:
    count = anchor.count(key)
    label = _PORT_LABELS.get(key, key)
    if count is None:
        return f"{label}: нет подтверждённых данных"
    if count <= 0:
        return f"{label}: нет"
    return f"{label}: {count}"


class InterfaceCompatibility:
    """Match a candidate's physical interface with a confirmed anchor port."""

    @staticmethod
    def validate(anchor: PortFacts, candidate: CandidateInterfaces, item: Direction | None = None) -> Validation:
        if not candidate.options:
            return Validation(Status.UNKNOWN, "missing_candidate_interface", "Не указан интерфейс подключения.",
                              missing=("interface",), sources=candidate.sources)
        supported = []
        unknown = []
        rejected = []
        for option in candidate.options:
            state = _state(anchor, option.kind)
            if state is True:
                supported.append(option)
            elif state is None:
                unknown.append(option)
            else:
                rejected.append(option)
        if supported:
            option = supported[0]
            port = _OPTION_PORT.get(option.kind, option.kind)
            reason = (f"Подтверждено: {_port_text(anchor, port)}; "
                      f"кандидат подключается через {_PORT_LABELS.get(port, option.kind)}.")
            return Validation(Status.CONFIRMED, "interface_match", reason,
                              facts={"matched_interface": option.kind, "candidate_label": option.label},
                              sources=tuple(dict.fromkeys((*anchor.sources, *candidate.sources))))
        if unknown:
            ports = tuple(dict.fromkeys(_OPTION_PORT.get(option.kind, option.kind) for option in unknown))
            return Validation(
                Status.UNKNOWN, "missing_anchor_interface",
                "Недостаточно данных о портах ноутбука: " + ", ".join(_PORT_LABELS.get(port, port) for port in ports) + ".",
                missing=ports, sources=tuple(dict.fromkeys((*anchor.sources, *candidate.sources))),
            )
        return Validation(
            Status.REJECTED, "incompatible_interface",
            "Ни один интерфейс кандидата не подтверждён у ноутбука.",
            facts={"candidate_interfaces": [option.kind for option in rejected]}, sources=candidate.sources,
        )


class ProtocolCompatibility:
    """Check protocol-level requirements that a physical port alone cannot prove."""

    @staticmethod
    def validate(anchor: PortFacts, candidate: CandidateInterfaces) -> Validation:
        checks = {
            "usb_c_dp_alt": (anchor.usb_c_dp_alt, "DisplayPort Alt Mode"),
            "usb_c_pd": (anchor.usb_c_pd, "Power Delivery"),
        }
        for requirement in candidate.requires:
            if requirement not in checks:
                continue
            state, label = checks[requirement]
            if state is None:
                return Validation(Status.UNKNOWN, f"missing_{requirement}",
                                  f"Не подтверждена поддержка {label} у ноутбука.",
                                  missing=(requirement,), sources=anchor.sources)
            if not state:
                return Validation(Status.REJECTED, f"unsupported_{requirement}",
                                  f"Ноутбук не поддерживает {label}.", sources=anchor.sources)
        return Validation(Status.CONFIRMED, "protocol_compatible", "Протокольных ограничений не выявлено.",
                          sources=tuple(dict.fromkeys((*anchor.sources, *candidate.sources))))


class DimensionCompatibility:
    """Compare a candidate limit with an anchor size; missing data stays UNKNOWN."""

    @staticmethod
    def validate(anchor_value, candidate_value, *, anchor_label: str = "объект",
                 candidate_label: str = "кандидат") -> Validation:
        if anchor_value is None:
            return Validation(Status.UNKNOWN, "missing_anchor_dimensions",
                              f"Не указаны габариты: {anchor_label}.", missing=("anchor_dimensions",))
        if candidate_value is None:
            return Validation(Status.UNKNOWN, "missing_candidate_dimensions",
                              f"Не указаны габариты: {candidate_label}.", missing=("candidate_dimensions",))
        try:
            if any(Decimal(str(axis)) > Decimal(str(limit)) for axis, limit in zip(anchor_value, candidate_value)):
                return Validation(Status.REJECTED, "incompatible_dimensions",
                                  f"Габариты не подходят: {anchor_label} больше {candidate_label}.")
        except (InvalidOperation, TypeError):
            return Validation(Status.UNKNOWN, "conflicting_facts", "Не удалось сравнить габариты.")
        return Validation(Status.CONFIRMED, "dimensions_match", "Габариты совместимы.")


class PlatformCompatibility:
    """Apply explicit OS restrictions only when the candidate states them."""

    @staticmethod
    def validate(anchor_os: str | None, candidate_platforms: tuple[str, ...]) -> Validation:
        if not candidate_platforms:
            return Validation(Status.CONFIRMED, "platform_agnostic", "Ограничения по ОС не заявлены.")
        if not anchor_os:
            return Validation(Status.UNKNOWN, "missing_platform", "Не подтверждена операционная система ноутбука.",
                              missing=("platform",))
        if any(platform in anchor_os for platform in candidate_platforms):
            return Validation(Status.CONFIRMED, "platform_match", f"Поддерживается {anchor_os}.")
        return Validation(Status.REJECTED, "unsupported_platform",
                          "Кандидат не поддерживает операционную систему ноутбука.")


class ArticleCompatibility:
    @staticmethod
    def validate(required: str | None, candidate: str | None) -> Validation:
        if not required:
            return Validation(Status.UNKNOWN, "missing_required_article", "Не указан требуемый артикул.",
                              missing=("required_article",))
        if not candidate:
            return Validation(Status.UNKNOWN, "missing_candidate_article", "Не указан артикул кандидата.",
                              missing=("candidate_article",))
        if normalized(required) == normalized(candidate):
            return Validation(Status.CONFIRMED, "article_match", f"Артикул {candidate} совпадает.")
        return Validation(Status.REJECTED, "wrong_article", f"Артикул {candidate} не совпадает с {required}.")


class CapacityCompatibility:
    @staticmethod
    def validate(required: int | None, available: int | None) -> Validation:
        if required is None:
            return Validation(Status.UNKNOWN, "missing_required_capacity", "Не указано требуемое количество.",
                              missing=("required_capacity",))
        if available is None:
            return Validation(Status.UNKNOWN, "missing_capacity", "Не указана ёмкость лицензии.",
                              missing=("capacity",))
        if available >= required:
            return Validation(Status.CONFIRMED, "capacity_match",
                              f"Ёмкость {available} покрывает требуемые {required}.")
        return Validation(Status.REJECTED, "insufficient_capacity",
                          f"Ёмкости {available} недостаточно для {required}.")


class PowerCompatibility:
    @staticmethod
    def validate(anchor: PortFacts, candidate: CandidateInterfaces) -> Validation:
        if "usb_c_pd" not in candidate.requires:
            return Validation(Status.CONFIRMED, "power_not_required", "Дополнительное питание не требуется.")
        if anchor.usb_c_pd is None:
            return Validation(Status.UNKNOWN, "missing_usb_c_pd", "Не подтверждена поддержка Power Delivery.",
                              missing=("usb_c_pd",), sources=anchor.sources)
        if not anchor.usb_c_pd:
            return Validation(Status.REJECTED, "unsupported_usb_c_pd", "Ноутбук не поддерживает Power Delivery.",
                              sources=anchor.sources)
        return Validation(Status.CONFIRMED, "power_match", "Power Delivery подтверждён.", sources=anchor.sources)


class LicenseCompatibility:
    """Reusable platform + capacity check for license-like candidates."""

    @staticmethod
    def validate(anchor_os: str | None, candidate_platforms: tuple[str, ...],
                 required_devices: int | None, covered_devices: int | None) -> Validation:
        platform = PlatformCompatibility.validate(anchor_os, candidate_platforms)
        if platform.status != Status.CONFIRMED:
            return platform
        return CapacityCompatibility.validate(required_devices, covered_devices)


def validate_universal(
    anchor: PortFacts, candidate: CandidateInterfaces, item: Direction,
) -> Validation:
    """Run the reusable validators in order and keep the strongest failure."""
    if "usb_c_video" in candidate.requires:
        return Validation(Status.UNKNOWN, "usb_c_video_unproven",
                          "USB-C видеовход монитора не подтверждает DisplayPort Alt Mode.",
                          missing=("usb_c_video",), sources=candidate.sources)
    results = [InterfaceCompatibility.validate(anchor, candidate, item)]
    if results[0].status == Status.CONFIRMED:
        results.append(ProtocolCompatibility.validate(anchor, candidate))
        results.append(PowerCompatibility.validate(anchor, candidate))
        results.append(PlatformCompatibility.validate(anchor.os, candidate.platforms))
    for status in (Status.REJECTED, Status.UNKNOWN):
        selected = next((result for result in results if result.status == status), None)
        if selected:
            return selected
    return results[0]


def requirements(item: Direction) -> dict:
    return {
        "direction_id": item.id, "label": item.label, "validator": item.validator,
        "requirements": list(item.requirements), "candidate_kind": item.candidate_kind,
        "category_ids": list(item.category_ids),
    }


def classify_candidate(category_kind: str, fields: Fields) -> str | None:
    """Map a catalog category/type to a universal candidate kind."""
    if category_kind == "input_device":
        value = normalized(" ".join(fields.values("тип устройства")))
        if "клавиатура+мышь" in value or "клавиатура + мышь" in value or "комплект" in value:
            return "keyboard_mouse"
        if "мышь" in value:
            return "mouse"
        if "клавиатура" in value:
            return "keyboard"
        return None
    if category_kind == "headset":
        value = normalized(" ".join(fields.values("тип")))
        if "микрофон" in value or value == "наушники":
            return None
        if "гарнитура" in value or "наушники с микрофоном" in value:
            return "headset"
        return None
    if category_kind == "monitor":
        return "monitor"
    if category_kind == "usb_hub":
        return "usb_hub"
    return None
