import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from operator import eq, ge, gt, le, lt, ne

from .query import Condition, ParsedQuery, number
from .recommendation_facts import Fact, Fields, Status, Validation, normalized, resolve, spec_values


_PLATFORM_FIELDS = ("платформа", "поддерживаемая ос", "поддерживаемые ос", "операционная система",
                    "поддерживаемые операционные системы", "совместимые ос", "ос",
                    "поддерживаемые платформы", "совместимая архитектура/платформа")
_UNITS = {
    "к-во узлов": "node", "количество узлов": "node",
    "к-во устройств": "device", "количество устройств": "device", "число устройств": "device",
    "число рабочих мест": "device", "количество рабочих мест": "device", "количество пк": "device",
    "к-во пользователей": "user", "количество пользователей": "user",
    "к-во серверов": "server", "количество серверов": "server",
}
_UNIT_SUFFIX = r"пк|устройств\w*|рабоч(?:ее|их|ие)\s+мест\w*|узел|узл\w*|пользовател\w*|сервер\w*"
_COUNT = re.compile(rf"(?:на\s+)?([1-9]\d*)(?:\s+(?:шт\.?|{_UNIT_SUFFIX}))?")
_RANGE = re.compile(rf"(?:от\s+)?([1-9]\d*)\s*(?:[-–—]|до)\s*([1-9]\d*)(?:\s+(?:шт\.?|{_UNIT_SUFFIX}))?")
_TITLE_COUNT = re.compile(rf"\b(?:на|и)\s+([1-9]\d*)\s+({_UNIT_SUFFIX})(?!\w)")
_MINIMUM = re.compile(r"\b(?:минимум\s+|минимальная покупка\s*:?\s*)(?:от\s+)?([1-9]\d*)")
_PER_UNIT = re.compile(r"\b(?:цена|стоимость) за (?:1|одн\w+) (?:узел|устройство|пк|лицензию)\b")
_OS = re.compile(
    r"(?<!\w)(?:windows(?:\s+(?:server(?:\s+\d+)?|\d+(?:\.\d+)?|xp|vista))?|"
    r"astra linux|alt linux|ред ос|ubuntu|debian|linux|macos(?:\s+\d+(?:\.\d+)*)?|"
    r"android(?:\s+\d+(?:\.\d+)*)?|ios(?:\s+\d+(?:\.\d+)*)?)(?!\w)"
)
_NEGATION = re.compile(r"\b(?:без|кроме|нет поддержки|не поддерж\w*|не для|не)\b")
_PLATFORM_CLAUSE = re.compile(r"[,;•/]|\b(?:но|однако)\b|[.]")
_OPTIONAL = re.compile(r"\b(?:желательно|предпочтительно|по возможности)\b")
_CONSUMER_ONLY = re.compile(
    r"\b(?:для дома(?!\s+и\s+бизнеса)|для семьи|семейн\w*|для личн\w+\s+пользован\w*|"
    r"персональн\w*\s+(?:лиценз\w*|верси\w*|подпис\w*)|microsoft 365\s+(?:персональн\w*|personal)|"
    r"физлиц\w*|физическ\w+\s+лиц\w*|family|personal|home\s+(?:and|&)\s+student)\b"
)
_OTHER_PURCHASE = re.compile(r"\b(?:продл\w+|миграц\w*|переход\w*|upgrade|renewal|migration)\b")
_RESTRICTED = re.compile(r"\b(?:академическ\w*|образовательн\w*|учебн\w*|некоммерческ\w*)\b")
_WORKSTATIONS = re.compile(r"\b(?:endpoint\s+(?:security|protection)|рабочи[ех]\s+станци\w*|desktop security|workstations)\b")
_WORKSTATION_PURPOSE = re.compile(r"\b(?:рабочи[ех]\s+станци\w*|workstations|клиентски\w+\s+пк)\b")
_SERVER_PURPOSE = re.compile(r"\b(?:почтов\w*\s+сервер\w*|для\s+сервер\w*|windows\s+server|шлюз\w*|mail\s+server)\b")
_MULTIPLATFORM = re.compile(r"\b(?:cross[- ]?platform|кроссплатформ\w*)\b")
_OFFICE_SUITE = re.compile(r"\b(?:microsoft office|microsoft 365|mobioffice|officesuite|onlyoffice|libreoffice|"
                         r"wps office|мой\s*офис|р7[- ]офис|офисн\w+\s+пакет\w*)\b")
_OFFICE_OTHER = re.compile(r"\b(?:mdm|виртуальн\w+\s+офис|add[- ]?ins?|надстройк\w*|управление инфраструктурой)\b")
_PROTECTED_SUITE = re.compile(r"(?:офис|office)\s*[+]\s*антивирус|\b(?:включает|в составе)\s+(?:\w+\s+){0,3}антивирус")
_FULL = {"полная версия", "новая продажа", "full", "base", "новая лицензия"}
_BUSINESS = {"юрлицо", "юридическое лицо", "юридические лица", "для юридических лиц", "для организаций"}
_BUSINESS_BUYER = re.compile(r"\b(?:юрлиц\w*|юридическ\w+\s+лиц\w*|b2b|для организации)\b")
_PRIVATE_BUYER = re.compile(r"\b(?:b2c|частн\w+\s+покупател\w*)\b")
_PRO32_ENDPOINT = re.compile(r"\bpro32 endpoint security (standard|advanced)\b")
_OPERATORS = {"eq": eq, "ne": ne, "ge": ge, "gt": gt, "le": le, "lt": lt}


@dataclass(frozen=True, slots=True)
class Capacity:
    unit: str
    package: int | None
    lower: int
    upper: int
    minimum_purchase: int | None
    per_unit: bool
    extra_units: bool

    def allocation(self, devices: int) -> tuple[int, int] | None:
        if self.unit not in {"device", "node"} or self.minimum_purchase is None:
            return None
        if self.per_unit:
            quantity = max(devices, self.lower, self.minimum_purchase)
            return (quantity, quantity) if quantity <= self.upper else None
        if self.package is None:
            return None
        quantity = max((devices + self.package - 1) // self.package, self.minimum_purchase)
        if quantity > 1 and not self.extra_units:
            return None
        return self.package * quantity, quantity


@dataclass(frozen=True, slots=True)
class LicenseFacts:
    platforms: Fact[frozenset[str]]
    unsupported_platforms: frozenset[str]
    platforms_complete: bool
    capacity: Fact[Capacity]
    desktop_limit: Fact[int]
    eligibility: Validation
    purchase_types: tuple[str, ...]
    customer_types: tuple[str, ...]
    purpose: tuple[str, ...]
    term_months: int | None
    edition_level: int | None = None

    @property
    def devices(self) -> int | None:
        capacity = self.capacity.value
        return capacity.package if capacity and capacity.unit in {"device", "node"} else None


@dataclass(frozen=True, slots=True)
class EndpointContext:
    platforms: tuple[tuple[str, ...], ...]
    devices: int
    buyer: str = ""
    term_conditions: tuple[Condition, ...] = ()

    def supports_term(self, months: int) -> bool:
        return all(_OPERATORS[condition.operator](months, condition.value) for condition in self.term_conditions)


def _platform_text(value: str) -> str:
    value = re.sub(r"\b(?:apple\s+)?(?:mac\s*os(?:\s+x)?|os x)\b", "macos", normalized(value))
    value = re.sub(r"\b(?:win|windows)\s*(\d{1,2}|server)\b", r"windows \1", value)
    value = re.sub(r"\b(?:альт|астра)\s+(?:linux|линукс)\b",
                   lambda match: "alt linux" if match[0].startswith("альт") else "astra linux", value)
    return re.sub(r"\b(windows\s+\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", r"\1/windows \2", value)


def platform_family(platform: str) -> str:
    if platform in {"astra linux", "alt linux", "ред ос", "ubuntu", "debian"}:
        return "linux"
    if platform.startswith("windows server"):
        return "windows server"
    return platform.partition(" ")[0]


def supports_platform(known: str, required: str) -> bool:
    return known == required or platform_family(known) == required


def _platform_mentions(value: str):
    text = _platform_text(value)
    previous_end = 0
    negative = False
    for match in _OS.finditer(text):
        prefix = text[previous_end:match.start()]
        clauses = _PLATFORM_CLAUSE.split(prefix)
        if len(clauses) > 1:
            negative = False
        if _NEGATION.search(clauses[-1]):
            negative = True
        is_unsupported = re.match(r"\s+не поддерж\w*\b", text[match.end():])
        yield match.group(), prefix, negative or bool(is_unsupported)
        previous_end = match.end()


def _platform_set(value: str) -> frozenset[str]:
    positives = set()
    exceptions = set()
    for platform, _, negative in _platform_mentions(value):
        (exceptions if negative else positives).add(platform)
    return frozenset(platform for platform in positives
                     if not any(supports_platform(platform, excluded) for excluded in exceptions))


def _platform_fact(fields: Fields, specs: list[dict]) -> tuple[Fact[frozenset[str]], frozenset[str]]:
    evidence = defaultdict(set)
    exclusions = set()
    sources = []
    declarations = fields.evidence(*_PLATFORM_FIELDS)
    if not declarations:
        for value, source in ((fields.title, "XML название"), (fields.model, "XML model")):
            if re.search(r"\b(?:для|под)\s+(?:windows|win\s*\d|mac\s*os|linux|astra linux)\b", value):
                declarations.append((value, source))
    for raw, source in declarations:
        mentions = list(_platform_mentions(raw))
        if mentions:
            sources.append(source)
        for platform, _, negative in mentions:
            if negative:
                exclusions.add(platform)
            else:
                evidence[source.partition(":")[0]].add(platform)
    possible = set().union(*evidence.values())
    contradictory = {platform for platform in possible
                     if any(supports_platform(platform, excluded) for excluded in exclusions)}
    common = frozenset(platform for platform in possible
                       if platform not in contradictory
                       and all(any(supports_platform(platform, known) for known in clause)
                               for clause in evidence.values()))
    conflict = bool(contradictory) or len({frozenset(clause) for clause in evidence.values()}) > 1
    official = spec_values(specs, "platforms", lambda values: frozenset(normalized(value) for value in values))
    if official:
        verified = resolve([(frozenset(clause), source) for source, clause in evidence.items()], official)
        if verified.value:
            denied = frozenset(platform for platform in exclusions
                               if not any(supports_platform(supported, platform) for supported in verified.value))
            return Fact(verified.value, verified.sources, verified.conflict or conflict or denied != exclusions), denied
        return verified, frozenset()
    return Fact(common or None, tuple(dict.fromkeys(sources)), conflict), frozenset(exclusions - contradictory)


def _unit(text: str) -> str:
    if text.startswith("польз"):
        return "user"
    if text.startswith("сервер"):
        return "server"
    return "node" if text.startswith(("узл", "узел")) else "device"


def _limits(value: str) -> tuple[int, int] | None:
    if match := _COUNT.fullmatch(value):
        return int(match[1]), int(match[1])
    if match := _RANGE.fullmatch(value):
        lower, upper = int(match[1]), int(match[2])
        return (lower, upper) if lower <= upper else None
    return None


def _capacity(fields: Fields, specs: list[dict]) -> Fact[Capacity]:
    evidence = []
    for label, unit in _UNITS.items():
        for value, source in fields.evidence(label):
            if limits := _limits(value):
                evidence.append(((unit, *limits), source))
                suffix = re.search(rf"(?<!\w)({_UNIT_SUFFIX})$", value)
                if suffix:
                    evidence.append(((_unit(suffix[1]), *limits), source))
    for text, source in ((fields.title, "XML название"), (fields.model, "XML model")):
        evidence.extend(((_unit(match[2]), int(match[1]), int(match[1])), source)
                        for match in _TITLE_COUNT.finditer(text))
    # Workstation licenses can count endpoints as nodes or as devices.
    units = {value[0] for value, _ in evidence}
    if units == {"device", "node"}:
        evidence = [(("node", *value[1:]), source) for value, source in evidence]
    fact = resolve(evidence, spec_values(specs, "fixed_device_count", lambda value: ("device", int(value), int(value))))
    if fact.value is None:
        return Fact(sources=fact.sources, conflict=fact.conflict)
    unit, lower, upper = fact.value
    minimum_words = fields.values("минимальная покупка", "минимальный объем покупки", "минимальный объем заказа")
    minima = {int(match[1]) for value in minimum_words if (match := re.match(r"(?:(?:от|минимум)\s+)?([1-9]\d*)", value))}
    minima.update(int(match[1]) for match in _MINIMUM.finditer(fields.title))
    official_minima = {int(spec["minimum_purchase"]) for spec in specs if spec.get("minimum_purchase")}
    if official_minima:
        minima = official_minima
    if len(minima) > 1:
        return Fact(sources=(*fact.sources, "XML минимальная покупка"), conflict=True)
    price_terms = fields.values("единица цены", "условия покупки", "цена указана")
    per_unit = any(value in {"за устройство", "за узел", "за 1 устройство", "за 1 узел"} or _PER_UNIT.search(value)
                   for value in (*price_terms, fields.title))
    needs_proof = per_unit or any(spec.get("minimum_purchase_required") is True for spec in specs)
    minimum_purchase = next(iter(minima), None if needs_proof else 1)
    if per_unit and minimum_purchase is not None and minimum_purchase > upper:
        return Fact(sources=(*fact.sources, "XML минимальная покупка"), conflict=True)
    extra = any(value in {"да", "разрешено"} for value in fields.values("возможность дозакупки", "дополнительные единицы"))
    capacity = Capacity(unit, lower if lower == upper and not per_unit else None, lower, upper, minimum_purchase,
                        per_unit, extra)
    sources = (*fact.sources, "XML единица цены") if per_unit else fact.sources
    return Fact(capacity, sources, fact.conflict)


def office_scope(fields: Fields, known_taxonomy: bool) -> tuple[bool, str]:
    purpose = " ".join(fields.values("назначение", "тип программного обеспечения", "состав комплекта"))
    identity = " ".join((fields.title, fields.model, purpose))
    if _PROTECTED_SUITE.search(identity):
        return False, "protection_included"
    if _CONSUMER_ONLY.search(identity):
        return False, "restricted_customer"
    if _OFFICE_OTHER.search(identity) or _OTHER_PURCHASE.search(fields.title):
        return False, "not_office_suite"
    # Require an identifiable package and either catalog categorization or a declared purpose.
    if (known_taxonomy or _OFFICE_SUITE.search(purpose)) and _OFFICE_SUITE.search(identity):
        return True, "office_suite_and_category"
    return False, "not_office_suite"


def is_new_endpoint_license(fields: Fields, specs: tuple[dict, ...] | list[dict] = ()) -> Validation:
    purposes = fields.values("назначение", "тип защиты", "тип программного обеспечения")
    purpose = " ".join(purposes)
    role = " ".join((fields.title, fields.model, purpose))
    workstation_sources = tuple(spec["source"] for spec in specs if spec.get("workstation_protection") is True)
    if not (_WORKSTATIONS.search(role) or workstation_sources) or (
        _SERVER_PURPOSE.search(role) and not _WORKSTATION_PURPOSE.search(role)
    ):
        return Validation(Status.REJECTED, "not_workstation_protection")
    customers = fields.values("пользователь", "тип клиента")
    organizations = fields.values("тип организации")
    if _RESTRICTED.search(" ".join((fields.title, *customers, *organizations))) or _CONSUMER_ONLY.search(fields.title):
        return Validation(Status.REJECTED, "restricted_customer")
    if not (_BUSINESS.intersection(customers) or "коммерческая" in organizations):
        return Validation(Status.UNKNOWN, "restricted_customer", missing=("customer_type",))
    kinds = fields.values("тип лицензии", "тип покупки")
    if _OTHER_PURCHASE.search(fields.title) or (kinds and any(kind not in _FULL for kind in kinds)):
        return Validation(Status.REJECTED, "incompatible_license_type")
    if not kinds:
        return Validation(Status.UNKNOWN, "incompatible_license_type", missing=("license_type",))
    return Validation(Status.CONFIRMED, "eligible_license", sources=workstation_sources)


def read_license_facts(fields: Fields, specs: list[dict], *, office: bool = False) -> LicenseFacts:
    term = resolve([
        (int(match[1]) * (12 if match[2] in {"год", "года", "лет"} else 1), source)
        for value, source in fields.evidence("срок действия", "срок лицензии")
        if (match := re.fullmatch(r"([1-9]\d*)\s*(мес\.?|месяц\w*|год|года|лет)", value))
    ])
    platforms, unsupported = _platform_fact(fields, specs)
    complete = (any("platforms" in spec for spec in specs)
                or not _MULTIPLATFORM.search(fields.title)
                or len({platform_family(value) for value in platforms.value or ()}) > 1)
    return LicenseFacts(
        platforms, unsupported, complete, _capacity(fields, specs),
        resolve([], spec_values(specs, "desktop_device_limit", int)),
        Validation(Status.CONFIRMED, "office_package") if office else is_new_endpoint_license(fields, specs),
        tuple(dict.fromkeys(fields.values("тип лицензии", "тип покупки"))),
        tuple(dict.fromkeys(fields.values("тип клиента", "пользователь", "тип организации"))),
        tuple(dict.fromkeys(fields.values("назначение", "тип защиты"))), term.value,
        {"standard": 1, "advanced": 2}[edition[1]] if (edition := _PRO32_ENDPOINT.search(fields.title)) else None,
    )


def _requested_platforms(
    query: ParsedQuery | None, filters: tuple[Condition, ...], known: LicenseFacts,
):
    groups = []
    declared = known.platforms.value or frozenset()
    exclusions = set(known.unsupported_platforms)
    for platform, prefix, negative in _platform_mentions(query.normalized if query else ""):
        if _OPTIONAL.search(prefix):
            continue
        if negative:
            exclusions.add(platform)
        elif groups and re.search(r"\b(?:или|or)\b|/", prefix):
            groups[-1].add(platform)
        else:
            groups.append({platform})
    for condition in query.conditions if query else ():
        if condition.field != "platform" or not condition.required or not isinstance(condition.value, str):
            continue
        values = _platform_set(condition.value)
        if condition.operator == "ne":
            exclusions.update(values)
        elif (condition.operator == "eq" and values
              and not all(any(supports_platform(seen, value) for group in groups for seen in group) for value in values)):
            groups.append(set(values))
    explicit = {value for cond in filters if cond.field == "platform" and cond.operator == "eq"
                and isinstance(cond.value, str) for value in _platform_set(cond.value)}
    if explicit:
        if groups and not any(supports_platform(a, b) or supports_platform(b, a)
                              for group in groups for a in group for b in explicit):
            return Validation(Status.REJECTED, "conflicting_context")
        for group in groups:
            if len(group) > 1 and any(supports_platform(a, b) or supports_platform(b, a) for a in group for b in explicit):
                group.intersection_update({a for a in group
                                           if any(supports_platform(a, b) or supports_platform(b, a) for b in explicit)})
        groups.extend({value} for value in sorted(explicit))
    if not groups and known.platforms_complete:
        groups = [{value} for value in sorted(declared)]
    if not groups:
        return Validation(Status.UNKNOWN, "missing_platform", missing=("platform",))
    checked = []
    for group in groups:
        viable = tuple(sorted(value for value in group
                             if not any(supports_platform(value, excl) or supports_platform(excl, value) for excl in exclusions)
                             and (not declared or any(supports_platform(value, platform) or supports_platform(platform, value)
                                                      for platform in declared))))
        if not viable:
            if not known.platforms_complete and not all(
                any(supports_platform(value, excl) for excl in exclusions) for value in group
            ):
                return Validation(Status.UNKNOWN, "missing_platform", missing=("platform",))
            return Validation(Status.REJECTED, "conflicting_context")
        if viable not in checked:
            checked.append(viable)
    return tuple(checked)


def _requested_devices(conditions: tuple[Condition, ...], known: LicenseFacts):
    lower = 1
    upper = None
    exact = set()
    excluded = set()
    specified = False
    for condition in conditions:
        if condition.field != "devices":
            continue
        try:
            value = condition.value if isinstance(condition.value, Decimal) else number(condition.value)
        except InvalidOperation:
            return Validation(Status.REJECTED, "conflicting_context")
        if not value.is_finite() or value <= 0 or value != value.to_integral_value():
            return Validation(Status.REJECTED, "conflicting_context")
        count = int(value)
        if condition.operator in {"ge", "gt"}:
            lower = max(lower, count + (condition.operator == "gt"))
            specified = True
        elif condition.operator in {"le", "lt"}:
            limit = count - (condition.operator == "lt")
            upper = limit if upper is None else min(upper, limit)
        elif condition.operator == "eq":
            exact.add(count)
        elif condition.operator == "ne":
            excluded.add(count)
    if len(exact) > 1 or (upper is not None and lower > upper):
        return Validation(Status.REJECTED, "conflicting_context")
    needed = next(iter(exact), lower if specified else known.devices)
    if needed is None:
        return Validation(Status.UNKNOWN, "unknown_license_capacity", missing=("devices",))
    capacity = known.capacity.value
    has_proven_limit = capacity and capacity.unit in {"device", "node"} and (capacity.package or capacity.per_unit)
    if (needed < lower or needed in excluded or (upper is not None and needed > upper)
            or (has_proven_limit and capacity.allocation(needed) is None)):
        return Validation(Status.REJECTED, "conflicting_context")
    return needed


def endpoint_requirements(known: LicenseFacts, query: ParsedQuery | None, filters: tuple[Condition, ...]):
    text = query.normalized if query else ""
    if _OTHER_PURCHASE.search(text):
        return Validation(Status.REJECTED, "incompatible_license_type")
    unresolved = tuple(name for name, fact in (("platform", known.platforms), ("devices", known.capacity))
                       if fact.conflict and fact.value is None)
    if unresolved:
        return Validation(Status.UNKNOWN, "conflicting_facts",
                          sources=(*known.platforms.sources, *known.capacity.sources))
    required = tuple(condition for condition in (query.conditions if query else ()) if condition.required) + filters
    buyers = {normalized(cond.value) for cond in required
              if cond.field == "customer_type" and cond.operator == "eq" and isinstance(cond.value, str)}
    private = bool(buyers & {"b2c", "физлицо"} or _CONSUMER_ONLY.search(text) or _PRIVATE_BUYER.search(text))
    business = bool(buyers & {"b2b", *_BUSINESS} or _BUSINESS_BUYER.search(text))
    if private:
        return Validation(Status.REJECTED, "conflicting_context" if business else "restricted_customer")
    if (not any(cond.field == "devices" for cond in required)
            and any(cond.field == "users" for cond in required)):
        return Validation(Status.UNKNOWN, "unknown_license_capacity", missing=("devices",))
    platforms = _requested_platforms(query, filters, known)
    devices = _requested_devices(required, known)
    failures = [value for value in (platforms, devices) if isinstance(value, Validation)]
    if failures:
        decision = next((value for value in failures if value.status == Status.REJECTED), failures[0])
        return Validation(decision.status, decision.reason_code,
                          missing=tuple(field for value in failures for field in value.missing))
    desktop_limit = known.desktop_limit.value
    if desktop_limit is not None and devices > desktop_limit and all(
        platform_family(platform) not in {"android", "ios"} for options in platforms for platform in options
    ):
        return Validation(Status.REJECTED, "conflicting_context",
                          facts={"desktop_limit": desktop_limit, "required_devices": devices},
                          sources=known.desktop_limit.sources)
    return EndpointContext(platforms, devices, "business" if business else "",
                           tuple(cond for cond in required if cond.field == "term_months"
                                 and isinstance(cond.value, Decimal) and cond.value.is_finite()))


def endpoint_preference_count(facts: LicenseFacts, preferences: tuple[Condition, ...]) -> int:
    score = 0
    for condition in preferences:
        if condition.field == "platform" and isinstance(condition.value, str):
            requested = _platform_set(condition.value)
            supported = any(supports_platform(known, want) for known in (facts.platforms.value or ()) for want in requested)
            if requested and condition.operator in {"eq", "ne"}:
                score += supported == (condition.operator == "eq")
        elif condition.field == "devices" and facts.devices is not None and isinstance(condition.value, Decimal):
            if condition.value.is_finite():
                score += _OPERATORS[condition.operator](facts.devices, condition.value)
        elif condition.field == "term_months" and facts.term_months is not None and isinstance(condition.value, Decimal):
            if condition.value.is_finite():
                score += _OPERATORS[condition.operator](facts.term_months, condition.value)
    return score
