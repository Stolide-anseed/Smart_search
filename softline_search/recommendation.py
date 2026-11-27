import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from itertools import chain
from pathlib import Path
from time import perf_counter
from xml.etree.ElementTree import Element

from .catalog import Offer, catalog_elements, read_offer
from .cross_sell import (
    DIRECTIONS, UNIVERSAL_DIRECTION_IDS, CandidateInterfaces, Direction, PortFacts,
    candidate_kinds, classify_candidate, direction_applies, generic_accessory_query,
    parse_candidate_interfaces, read_port_facts, requested_direction_ids, requirements,
    validate_universal,
)
from .index import _catalog_families
from .query import Condition, ParsedQuery, TYPE_PATTERNS, number, product_type
from .query_catalog import _KNOWN_CATEGORY_TYPES
from .recommendation_facts import (
    BAG_LIMITS, LAPTOP_DIMENSIONS, Dimensions, Fact, Fields, Status, Validation,
    diagonal, dimensions, load_specs, models, normalized, product_key, read_fields as _fields, resolve, spec_values,
)
from .recommendation_licenses import (
    EndpointContext, LicenseFacts, endpoint_preference_count, endpoint_requirements,
    office_scope, read_license_facts, supports_platform, _requested_platforms,
)
from .vendor_evidence import ProductIdentity, VendorEvidenceService, VendorFact


_CATEGORY_TYPES = {
    "82080": "ups", "82014": "ups_battery", "70824": "laptop", "83844": "notebook_bag",
    "1145": "office", "872": "office", "870": "office", "222323": "office", "650": "antivirus_business",
    "82833": "input_device", "88831": "headset", "80843": "monitor", "84206": "usb_hub",
}
_CATEGORY_NAMES = {**_KNOWN_CATEGORY_TYPES, "антивирусы для организаций": "antivirus_business"}
_SCENARIOS = {"ups": "ups_battery", "laptop": "laptop_bag", "office": "office_antivirus"}
_RELEVANT_TYPES = set(_SCENARIOS) | {"ups_battery", "notebook_bag", "antivirus_business"}
_UNIVERSAL_CATEGORY_KINDS = {"82833": "input_device", "88831": "headset", "80843": "monitor", "84206": "usb_hub"}
_UNIVERSAL_LIMIT = 8
_VENDOR_LIMIT = 2
_BATTERY_FIELDS = ("совместимая батарея", "совместимые батареи", "сменная батарея", "артикул сменной батареи")
_BATTERY_ARTICLE = re.compile(r"(?:APC)?(RBC[0-9]+)", re.IGNORECASE | re.ASCII)
_TRAIT_FIELDS = ("форм-фактор", "тип устройства", "тип программного обеспечения", "назначение")
_MODEL_FIELDS = ("совместимые модели", "совместимые модели ноутбуков", "совместимые артикулы ноутбуков")
_WRONG_LAPTOP = re.compile(r"^(?:планшет|монитор|сумка|чехол|аккумулятор|подставка|блок питания)\b")
_LAPTOP_ACCESSORIES = {"monitor", "notebook_bag", "ups_battery", "tablet", "ram", "psu", "charger"}
_LEGACY_BATTERY_LIMIT = 2
_CANDIDATE_KINDS = {"ups_battery": "ups_battery", "laptop_bag": "notebook_bag",
                    "office_antivirus": "antivirus_business"}
_LENOVO_MODEL = re.compile(
    r"\b(?:thinkbook \d{2}[a-z]? g\d{1,2} [a-z]{3}|thinkpad [a-z]\d{2} gen \d{1,2} (?:intel|amd)"
    r"|v\d{2} g\d{1,2} [a-z]{3})\b", re.I,
)
_LENOVO_IDEAPAD_ALIAS = re.compile(r"\bideapad ip3 g7 (\d{2}[a-z]{3}\d{1,2})\b", re.I)
_ASUS_MODEL = re.compile(
    r"(?<![A-Z0-9-])([A-Z]{1,4}\d{3,4}[A-Z][A-Z0-9]{0,6})(?:-[A-Z0-9]{2,12})?(?![A-Z0-9-])",
    re.I | re.ASCII,
)


@dataclass(frozen=True, slots=True)
class BatteryRecommendation:
    offer: Offer
    battery_reference: str
    available: bool
    has_price: bool


@dataclass(frozen=True, slots=True)
class Recommendation:
    offer: Offer
    available: bool
    has_price: bool
    reason: str
    sources: tuple[str, ...] = ()
    purchase_quantity: int = 1
    total_price: Decimal | None = None
    direction_id: str = ""
    direction_label: str = ""


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    scenario_id: str | None
    items: list[Recommendation]
    status: Status = Status.CONFIRMED
    reason_code: str = ""
    missing_context: tuple[str, ...] = ()
    diagnostics: dict | None = None
    directions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CatalogOffer:
    offer: Offer
    available: bool
    has_price: bool


@dataclass(frozen=True, slots=True)
class _Storage:
    dimensions: Fact[Dimensions]
    diagonal: Fact[Decimal]
    models: Fact[frozenset[str]]
    strict_diagonal: bool = False


@dataclass(frozen=True, slots=True)
class _Match:
    product: _CatalogOffer
    validation: Validation
    rank: tuple[int, ...] = ()
    battery_reference: str = ""
    purchase_quantity: int = 1
    direction_id: str = ""


@dataclass(frozen=True, slots=True)
class _UniversalCandidate:
    record: _CatalogOffer
    kind: str
    interfaces: CandidateInterfaces


def _battery_article(value: str) -> str:
    match = _BATTERY_ARTICLE.fullmatch(value.strip())
    return match[1].upper() if match else ""


def supports_battery_recommendation(offer: Offer) -> bool:
    return offer.category_id == "82080" and normalized(offer.vendor) == "apc"


def _has_price(offer: Offer) -> bool:
    try:
        price = number(offer.price)
    except InvalidOperation:
        return False
    return price.is_finite() and price > 0


def _battery_references(fields: Fields, specs: list[dict]) -> Fact[tuple[str, ...]]:
    references = defaultdict(list)
    for value, source in fields.evidence(*_BATTERY_FIELDS):
        references[source.partition(":")[0]].extend(
            token.upper() for token in re.split(r"[,;/]|\s+и\s+", value) if _battery_article(token)
        )
    declared = [(tuple(dict.fromkeys(values)), source) for source, values in references.items() if values]
    official = spec_values(specs, "battery_articles",
                           lambda values: tuple(value.upper() for value in values if _battery_article(value)))
    trusted = official or declared
    articles = {frozenset(_battery_article(value) for value in values) for values, _ in trusted}
    sources = tuple(source for _, source in trusted)
    if len(articles) != 1:
        return Fact(sources=sources, conflict=len(articles) > 1)
    chosen = trusted[0][0]
    conflicts = any(frozenset(map(_battery_article, values)) != next(iter(articles)) for values, _ in declared)
    return Fact(chosen, sources, conflicts)


def _size_text(size: Dimensions) -> str:
    return " × ".join(format(value, "f").replace(".", ",") for value in size) + " мм"


def _platform_label(platform: str) -> str:
    name, _, version = platform.partition(" ")
    label = {"windows": "Windows", "linux": "Linux", "macos": "macOS", "android": "Android",
             "ios": "iOS", "astra": "Astra", "alt": "ALT"}.get(name, name.capitalize())
    return (label + " " + version).strip()


def _to_json(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (tuple, list)):
        return [_to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json(item) for key, item in value.items()}
    return _to_json(asdict(value)) if is_dataclass(value) else value


class RecommendationAgent:
    def __init__(self, catalog_path: Path) -> None:
        started = perf_counter()
        self._offers: dict[str, _CatalogOffer] = {}
        self._category_types = _CATEGORY_TYPES.copy()
        self._category_names: dict[str, str] = {}
        self._category_offers: dict[str, list[_CatalogOffer]] = defaultdict(list)
        self._identities: dict[str, ProductIdentity] = {}
        self._kinds: dict[str, tuple[str, ...]] = {}
        self._route_notes: dict[str, str] = {}
        self._references: dict[str, Fact[tuple[str, ...]]] = {}
        self._batteries: dict[str, list[_CatalogOffer]] = defaultdict(list)
        self._laptops: dict[str, _Storage] = {}
        self._bags: list[tuple[_CatalogOffer, _Storage]] = []
        self._bag_facts: dict[str, _Storage] = {}
        self._offices: dict[str, LicenseFacts] = {}
        self._endpoints: list[tuple[_CatalogOffer, LicenseFacts]] = []
        self._endpoint_facts: dict[str, LicenseFacts] = {}
        self._endpoints_missing_platform: list[_Match] = []
        self._endpoint_exclusions: Counter[tuple[Status, str]] = Counter()
        self._endpoint_examples: dict[str, _Match] = {}
        self._ports: dict[str, PortFacts] = {}
        self._universal: dict[str, list[_UniversalCandidate]] = defaultdict(list)
        self._universal_by_id: dict[str, _UniversalCandidate] = {}
        specs = load_specs(Path(__file__).with_name("compatibility_specs.json"))
        counts = Counter()
        available_counts = Counter()
        enriched = set()
        conflicts = Counter()

        for element in catalog_elements(catalog_path):
            if element.tag == "category":
                category_id = element.attrib["id"]
                self._category_names[category_id] = " ".join((element.text or category_id).split())
                if category := self._category_types.get(category_id) or _CATEGORY_NAMES.get(normalized(element.text or "")):
                    self._category_types[category_id] = category
                continue

            offer = read_offer(element)
            record = _CatalogOffer(offer, element.get("available", "").lower() == "true", _has_price(offer))
            self._offers[offer.offer_id] = record
            self._category_offers[offer.category_id].append(record)
            counts[offer.category_id] += 1
            if record.available:
                available_counts[offer.category_id] += 1
            category = self._category_types.get(offer.category_id, "")
            if offer.category_id in _UNIVERSAL_CATEGORY_KINDS:
                fields = _fields(element)
                self._index_universal(record, _UNIVERSAL_CATEGORY_KINDS[offer.category_id], fields, specs)
                continue
            if category and category not in _RELEVANT_TYPES:
                continue
            title = normalized(offer.name)
            if category == "laptop" and (_WRONG_LAPTOP.search(title) or product_type(title)[0] in _LAPTOP_ACCESSORIES):
                self._route_notes[offer.offer_id] = "not_a_laptop"
                continue
            title_kind = ""
            if not category and any(TYPE_PATTERNS[kind].search(title) for kind in _SCENARIOS):
                title_kind = product_type(title)[0]
            related_params = (normalized(param.get("name", "")) for param in element.findall("param"))
            if not (category or title_kind in _SCENARIOS or normalized(offer.vendor) == "apc"
                    or any(label in _TRAIT_FIELDS for label in related_params)):
                continue

            fields = _fields(element)
            traits = {product_type(value)[0] for value in fields.values(*_TRAIT_FIELDS)}
            if not category and normalized(offer.vendor) == "apc" and fields.values(*_BATTERY_FIELDS):
                traits.add("ups")
            kinds = {category} if category else (traits & set(_SCENARIOS) or {title_kind})
            if "office" in kinds:
                applicable, reason = office_scope(fields, category == "office")
                self._route_notes[offer.offer_id] = reason
                if not applicable:
                    kinds.remove("office")
            self._kinds[offer.offer_id] = tuple(kind for kind in _SCENARIOS if kind in kinds)
            vendor, article = product_key(offer.vendor, offer.vendor_code)
            for kind in kinds:
                official = specs.get((kind, vendor, article), [])
                if official:
                    enriched.add((vendor, article))
                self._prepare_offer(kind, record, fields, official, conflicts)
        self._families = _catalog_families(tuple(record.offer for record in self._offers.values()))
        for category in self._category_offers.values():
            category.sort(key=lambda record: (not record.available, not record.has_price))
        self._ranked_bags = sorted(self._bags, key=lambda row: (
            not bool(row[1].dimensions.value or row[1].models.value),
            not row[0].available, not row[0].has_price, row[0].offer.offer_id,
        ))
        for rows in self._universal.values():
            rows.sort(key=lambda candidate: (not candidate.record.available, not candidate.record.has_price,
                                             candidate.record.offer.offer_id))
        self._ranked_endpoints = sorted((*[record for record, _ in self._endpoints],
                                         *[row.product for row in self._endpoints_missing_platform]),
                                        key=lambda record: (not record.available, not record.has_price,
                                                            record.offer.offer_id))
        self.index_stats = {
            "offers_by_category": dict(counts), "available_by_category": dict(available_counts),
            "enriched_skus": len(enriched), "conflicting_facts": dict(conflicts),
            "laptops_with_dimensions": sum(storage.dimensions.value is not None for storage in self._laptops.values()),
            "bags_with_dimensions": sum(storage.dimensions.value is not None for _, storage in self._bags),
            "bags_with_compatible_models": sum(bool(storage.models.value) for _, storage in self._bags),
            "ups_with_reference": sum(bool(fact.value) for fact in self._references.values()),
            "office_with_platform_and_devices": sum(bool(facts.platforms.value and facts.devices) for facts in self._offices.values()),
            "endpoints_with_platform_and_capacity": sum(bool(facts.platforms.value and facts.capacity.value)
                                                       for _, facts in self._endpoints),
            "candidates_by_scenario": {
                "ups_battery": sum(map(len, self._batteries.values())), "laptop_bag": len(self._bags),
                "office_antivirus": len(self._endpoints) + self._endpoint_exclusions.total(),
            },
            "universal_candidates": {kind: len(rows) for kind, rows in self._universal.items()},
            "laptops_with_port_facts": sum(bool(ports.ports or ports.absent or ports.bluetooth is not None)
                                           for ports in self._ports.values()),
            "endpoints_ready_for_context_check": len(self._endpoints),
            "build_ms": (perf_counter() - started) * 1000,
        }

    def _index_universal(
        self, record: _CatalogOffer, category_kind: str, fields: Fields, specs: dict,
    ) -> None:
        kind = classify_candidate(category_kind, fields)
        if not kind:
            return
        vendor, article = product_key(record.offer.vendor, record.offer.vendor_code)
        official = specs.get((kind, vendor, article), [])
        interfaces = parse_candidate_interfaces(kind, fields, record.offer, official)
        candidate = _UniversalCandidate(record, kind, interfaces)
        self._universal[kind].append(candidate)
        self._universal_by_id[record.offer.offer_id] = candidate

    def _prepare_offer(
        self, kind: str, record: _CatalogOffer, fields: Fields, specs: list[dict], conflicts: Counter,
    ) -> None:
        offer = record.offer
        if kind in _RELEVANT_TYPES:
            curated = next((str(spec["model"]) for spec in specs if spec.get("model")), "")
            # Lenovo's machine names encode chassis variants. Do not strip the variant suffix.
            title = offer.name if normalized(offer.vendor) == "lenovo" and kind == "laptop" else ""
            lenovo_model = _LENOVO_MODEL.search(title)
            alias = _LENOVO_IDEAPAD_ALIAS.search(title)
            model = curated or (
                lenovo_model[0] if lenovo_model else f"IdeaPad 3 {alias[1]}" if alias else fields.model[:79]
            )
            chassis_model = chassis_source = spec_source = ""
            if normalized(offer.vendor) == "asus" and kind == "laptop":
                # A model code parsed from the feed guides lookup but cannot certify a chassis.
                asus_model = _ASUS_MODEL.search(model) or _ASUS_MODEL.search(offer.name)
                if asus_model:
                    model = asus_model[0].upper()
                    chassis_model = asus_model[1].upper()
                binding = next((spec for spec in specs if spec.get("identity_source")
                                and normalized(spec.get("chassis_model", "")) == normalized(chassis_model)), None)
                if binding:
                    chassis_source = binding["identity_source"]
                    spec_source = binding["source"]
            self._identities.setdefault(offer.offer_id, ProductIdentity(
                offer.vendor, model, offer.vendor_code, kind, exact_model=bool(curated or lenovo_model),
                offer_id=offer.offer_id, chassis_model=chassis_model, chassis_source=chassis_source,
                spec_source=spec_source,
            ))
        if kind == "ups" and normalized(offer.vendor) == "apc":
            refs = _battery_references(fields, specs)
            self._references[offer.offer_id] = refs
            conflicts["battery_articles"] += refs.conflict
        elif kind == "ups_battery" and normalized(offer.vendor) == "apc":
            if article := _battery_article(offer.vendor_code):
                self._batteries[article].append(record)
        elif kind in {"laptop", "notebook_bag"}:
            storage = self._prepare_storage(kind, offer, fields, specs)
            conflicts["dimensions"] += storage.dimensions.conflict
            conflicts["diagonal"] += storage.diagonal.conflict
            conflicts["compatible_models"] += storage.models.conflict
            if kind == "laptop":
                self._laptops[offer.offer_id] = storage
                self._ports[offer.offer_id] = read_port_facts(fields, specs)
            else:
                self._bags.append((record, storage))
                self._bag_facts[offer.offer_id] = storage
        elif kind in {"office", "antivirus_business"}:
            license_facts = read_license_facts(fields, specs, office=kind == "office")
            conflicts["platform"] += license_facts.platforms.conflict
            conflicts["license_capacity"] += license_facts.capacity.conflict
            if kind == "office":
                self._offices[offer.offer_id] = license_facts
            else:
                self._endpoint_facts[offer.offer_id] = license_facts
                if failure := self._endpoint_data_failure(license_facts):
                    match = _Match(record, failure)
                    self._endpoint_exclusions[failure.status, failure.reason_code] += 1
                    self._endpoint_examples.setdefault(failure.reason_code, match)
                    if failure.reason_code == "missing_platform" and license_facts.devices:
                        self._endpoints_missing_platform.append(match)
                else:
                    self._endpoints.append((record, license_facts))

    def _prepare_storage(self, kind: str, offer: Offer, fields: Fields, specs: list[dict]) -> _Storage:
        if kind == "laptop":
            known_models = frozenset(value for value in (
                normalized(offer.vendor_code), fields.model, *(normalized(spec.get("model", "")) for spec in specs),
            ) if value)
            return _Storage(dimensions(fields, LAPTOP_DIMENSIONS, specs),
                            diagonal(fields, ("диагональ экрана", "диагональ дисплея"), specs), Fact(known_models))
        return _Storage(
            dimensions(fields, BAG_LIMITS, specs),
            diagonal(fields, ("максимальная диагональ", "максимальная диагональ ноутбука"), specs),
            models(fields, _MODEL_FIELDS, specs), any(spec.get("diagonal_is_limit") is True for spec in specs),
        )

    def anchor_ids(self, scenario: str) -> tuple[str, ...]:
        if scenario in DIRECTIONS:
            return tuple(oid for oid, record in self._offers.items()
                         if record.available and self._anchor_kind(oid) == "laptop")
        kind = next((kind for kind, name in _SCENARIOS.items() if name == scenario), "")
        return tuple(oid for oid, record in self._offers.items()
                     if record.available and kind in self._kinds.get(oid, ())
                     and (kind != "ups" or normalized(record.offer.vendor) == "apc"))

    def get_offer(self, offer_id: str) -> Offer | None:
        row = self._offers.get(offer_id)
        return row.offer if row else None

    def product_identity(self, offer_id: str) -> ProductIdentity:
        offer = self._offers[offer_id].offer
        identity = self._identities.get(offer_id)
        return identity or ProductIdentity(offer.vendor, "", offer.vendor_code, "", offer_id=offer_id)

    def catalog_categories(self) -> dict[str, str]:
        return {category_id: self._category_names.get(category_id, category_id)
                for category_id in self._category_offers if category_id}

    def category_products(self, category_id: str, *, limit: int = 10) -> tuple[Offer, ...]:
        return tuple(row.offer for row in self._category_offers.get(category_id, ())[:max(0, min(limit, 20))])

    def applicable_scenarios(
        self, offer_id: str, query: ParsedQuery | None = None, *, filters: tuple[Condition, ...] = (),
    ) -> tuple[str, ...]:
        offer = self.get_offer(offer_id)
        kind, _ = self._route(offer, query, filters) if offer else (None, "")
        return (_SCENARIOS[kind],) if kind else ()

    def _anchor_kind(self, offer_id: str) -> str:
        kinds = self._kinds.get(offer_id, ())
        return "laptop" if "laptop" in kinds else kinds[0] if kinds else ""

    def applicable_directions(self, offer_id: str) -> tuple[str, ...]:
        """All universal directions structurally available for an anchor."""
        anchor_kind = self._anchor_kind(offer_id)
        if not anchor_kind:
            return ()
        available = []
        for identifier, item in DIRECTIONS.items():
            if not direction_applies(item, anchor_kind):
                continue
            if any(self._category_offers.get(category_id) for category_id in item.category_ids):
                available.append(identifier)
        return tuple(available)

    def requested_directions(
        self, offer_id: str, query: ParsedQuery | None = None, *, filters: tuple[Condition, ...] = (),
    ) -> tuple[str, ...]:
        """Directions explicitly named in the query, or all for a generic accessory request."""
        applicable = set(self.applicable_directions(offer_id))
        if not applicable:
            return ()
        text = query.normalized if query else ""
        return tuple(identifier for identifier in requested_direction_ids(text) if identifier in applicable)

    def cross_sell_directions(self, offer_id: str) -> tuple[dict, ...]:
        """Describe available directions, their requirements and confirmed coverage."""
        ports = self._ports.get(offer_id, PortFacts())
        result = []
        for identifier in self.applicable_directions(offer_id):
            item = DIRECTIONS[identifier]
            candidates = self._universal_candidates(identifier)
            confirmed = 0
            for candidate in candidates:
                validation = validate_universal(ports, candidate.interfaces, item)
                confirmed += validation.status == Status.CONFIRMED
            result.append({
                **requirements(item), "candidate_count": len(candidates), "confirmed_count": confirmed,
                "missing_anchor_attributes": list(ports.missing()),
            })
        return tuple(result)

    def compatibility_requirements(self, offer_id: str, direction_id: str) -> dict:
        """Return the deterministic contract for one direction without exposing catalog internals."""
        if direction_id not in DIRECTIONS:
            if direction_id in _SCENARIOS.values():
                return {"direction_id": direction_id, "legacy": True, "requirements": [],
                        "candidate_count": len(self.scenario_candidates(offer_id, direction_id))}
            return {"direction_id": direction_id, "error": "unsupported_direction"}
        item = DIRECTIONS[direction_id]
        if direction_id not in self.applicable_directions(offer_id):
            return {**requirements(item), "error": "unsupported_direction"}
        ports = self._ports.get(offer_id, PortFacts())
        candidates = self._universal_candidates(direction_id)
        confirmed = 0
        examples = []
        for candidate in candidates:
            validation = validate_universal(ports, candidate.interfaces, item)
            if validation.status == Status.CONFIRMED:
                confirmed += 1
            if len(examples) < 3:
                examples.append({
                    "offer_id": candidate.record.offer.offer_id,
                    "name": candidate.record.offer.name[:120],
                    "status": validation.status,
                    "reason_code": validation.reason_code,
                    "interfaces": list(candidate.interfaces.kinds()),
                })
        return {
            **requirements(item), "anchor_ports": ports.public(),
            "missing_anchor_attributes": list(ports.missing()),
            "candidate_count": len(candidates),
            "available_count": sum(candidate.record.available for candidate in candidates),
            "confirmed_count": confirmed, "examples": examples,
        }

    def check_compatibility_batch(
        self, offer_id: str, direction_id: str, candidate_ids: tuple[str, ...],
        query: ParsedQuery | None = None, *, filters: tuple[Condition, ...] = (),
        facts: tuple[VendorFact, ...] = (),
    ) -> tuple[dict, ...]:
        results = []
        for candidate_id in dict.fromkeys(candidate_ids):
            validation = self.check_compatibility(offer_id, candidate_id, direction_id, query,
                                                  filters=filters, facts=facts)
            results.append({
                "candidate_id": candidate_id, "status": validation.status,
                "reason_code": validation.reason_code, "reason": validation.reason,
                "missing": list(validation.missing), "sources": list(validation.sources),
            })
        return tuple(results)

    def needs_direction_planning(
        self, offer_id: str, query: ParsedQuery | None, local: RecommendationResult,
    ) -> bool:
        """Whether an LLM should choose directions beyond the local fast path."""
        applicable = self.applicable_directions(offer_id)
        if not applicable:
            return False
        if local.items and any(direction in DIRECTIONS for direction in local.directions):
            return False
        text = query.normalized if query else ""
        if not text:
            return False
        named = requested_direction_ids(text)
        if generic_accessory_query(text):
            return True
        if any(direction in DIRECTIONS for direction in named):
            return not local.items
        if any(direction in _SCENARIOS.values() for direction in named):
            return False
        return not local.items

    def is_candidate(self, offer_id: str, direction_id: str) -> bool:
        item = DIRECTIONS.get(direction_id)
        candidate = self._universal_by_id.get(offer_id)
        return bool(item and candidate and candidate.kind in candidate_kinds(item))

    def _universal_candidates(self, direction_id: str) -> list[_UniversalCandidate]:
        item = DIRECTIONS.get(direction_id)
        if item is None:
            return []
        rows = [candidate for kind in candidate_kinds(item) for candidate in self._universal.get(kind, ())]
        return sorted(rows, key=lambda candidate: (
            not candidate.record.available, not candidate.record.has_price,
            candidate.record.offer.offer_id,
        ))

    def _universal_match(
        self, offer_id: str, candidate_id: str, direction_id: str,
    ) -> _Match | Validation:
        if offer_id not in self._offers or candidate_id not in self._offers:
            return Validation(Status.REJECTED, "unknown_offer")
        if direction_id not in self.applicable_directions(offer_id):
            return Validation(Status.REJECTED, "unsupported_direction")
        if self._families[offer_id] == self._families[candidate_id]:
            return Validation(Status.REJECTED, "same_product_family")
        candidate = self._universal_by_id.get(candidate_id)
        item = DIRECTIONS[direction_id]
        if candidate is None or candidate.kind not in candidate_kinds(item):
            return Validation(Status.REJECTED, "wrong_candidate_type")
        ports = self._ports.get(offer_id)
        if ports is None:
            return Validation(Status.UNKNOWN, "missing_anchor_ports",
                              "Не удалось прочитать порты ноутбука.", missing=("ports",))
        validation = validate_universal(ports, candidate.interfaces, item)
        return _Match(candidate.record, validation, direction_id=direction_id)

    def _validate_universal(self, offer_id: str, candidate: _UniversalCandidate, item: Direction) -> _Match:
        ports = self._ports.get(offer_id, PortFacts())
        validation = validate_universal(ports, candidate.interfaces, item)
        return _Match(candidate.record, validation, direction_id=item.id)

    def _vendor_fact(self, offer_id: str, attribute: str, facts: tuple[VendorFact, ...]) -> Fact:
        identity = self.product_identity(offer_id)
        return resolve([
            (fact.value, url)
            for fact in facts if fact.attribute == attribute and fact.applies_to(identity)
            for url in (fact.source_url, fact.identity_source_url) if url
        ])

    def _references_with_evidence(self, offer_id: str, facts: tuple[VendorFact, ...]) -> Fact[tuple[str, ...]]:
        local = self._references[offer_id]
        return self._vendor_fact(offer_id, "battery_articles", facts) if facts and not (local.value or local.conflict) else local

    def _storage_with_evidence(self, offer_id: str, storage: _Storage, facts: tuple[VendorFact, ...]) -> _Storage:
        if not facts or storage.dimensions.value or storage.dimensions.conflict:
            return storage
        fact = self._vendor_fact(offer_id, "dimensions_mm", facts)
        verified = tuple(Decimal(value) for value in fact.value) if fact.value else None
        return replace(storage, dimensions=Fact(verified, fact.sources, fact.conflict))

    def _license_with_evidence(self, offer_id: str, license: LicenseFacts, facts: tuple[VendorFact, ...]) -> LicenseFacts:
        if not facts or license.platforms.conflict:
            return license
        fact = self._vendor_fact(offer_id, "platforms", facts)
        if not fact.value:
            return license
        known = license.platforms.value or frozenset()
        verified = frozenset(platform for platform in fact.value
                             if (not known or any(supports_platform(platform, base) for base in known))
                             and not any(supports_platform(platform, denied) for denied in license.unsupported_platforms))
        if not verified:
            return license
        # More precise versions refine a generic platform. They cannot broaden a previously verified scope.
        return replace(license, platforms=Fact(known | verified, (*license.platforms.sources, *fact.sources)),
                       platforms_complete=license.platforms_complete or not known)

    def product_details(
        self, offer_id: str, *, facts: tuple[VendorFact, ...] = (),
        query: ParsedQuery | None = None, filters: tuple[Condition, ...] = (),
    ) -> dict:
        record = self._offers[offer_id]
        offer = record.offer
        identity = self.product_identity(offer_id)
        details = {}
        missing = []
        if identity.kind == "ups" and offer_id in self._references:
            refs = self._references_with_evidence(offer_id, facts)
            details["battery_articles"] = refs
            if not (refs.value or refs.conflict):
                missing.append("battery_articles")
        storage = self._laptops.get(offer_id) or self._bag_facts.get(offer_id)
        if storage:
            storage = self._storage_with_evidence(offer_id, storage, facts)
            details.update(dimensions_mm=storage.dimensions, diagonal_inches=storage.diagonal, compatible_models=storage.models)
            if not (storage.dimensions.value or storage.dimensions.conflict):
                missing.append("dimensions_mm")
        license = self._offices.get(offer_id) or self._endpoint_facts.get(offer_id)
        if license:
            license = self._license_with_evidence(offer_id, license, facts)
            details.update(platforms=license.platforms, device_count=license.devices)
            known = license.platforms.value or frozenset()
            requested = _requested_platforms(query, filters, license)
            version_gap = not isinstance(requested, Validation) and any(
                any(supports_platform(platform, base) for platform in group for base in known)
                and not any(supports_platform(base, platform) for platform in group for base in known)
                for group in requested
            )
            if not license.platforms.conflict and (not known or version_gap):
                missing.append("platforms")
        if offer_id in self._ports:
            ports = self._ports[offer_id]
            details["ports"] = Fact(ports.public(), ports.sources)
        if candidate := self._universal_by_id.get(offer_id):
            details["interfaces"] = Fact({
                "kind": candidate.kind,
                "options": [option.kind for option in candidate.interfaces.options],
                "requires": list(candidate.interfaces.requires),
                "platforms": list(candidate.interfaces.platforms),
            }, candidate.interfaces.sources)
        return _to_json({
            "offer_id": offer_id, "name": offer.name[:200], "vendor": offer.vendor[:80],
            "article": offer.vendor_code[:80], "model": identity.product_model[:100], "kind": identity.kind,
            "chassis_model": identity.chassis_model, "chassis_source": identity.chassis_source,
            "category_id": offer.category_id, "available": record.available, "has_price": record.has_price,
            "price": offer.price if record.has_price else "", "currency": offer.currency, "facts": details,
            "missing_attributes": missing if identity.domains and identity.subject else [],
            "missing_port_attributes": list(self._ports[offer_id].missing()) if offer_id in self._ports else [],
        })

    def scenario_candidates(
        self, offer_id: str, scenario_id: str, *, limit: int = 10, facts: tuple[VendorFact, ...] = (),
        category_id: str = "", query: ParsedQuery | None = None, filters: tuple[Condition, ...] = (),
    ) -> tuple[Offer, ...]:
        count = max(0, min(limit, 20))
        if not count:
            return ()
        if scenario_id in DIRECTIONS:
            if scenario_id not in self.applicable_directions(offer_id):
                return ()
            item = DIRECTIONS[scenario_id]
            ports = self._ports.get(offer_id, PortFacts())
            preferred, fallback = [], []
            seen = {self._families[offer_id]}
            for candidate in self._universal_candidates(scenario_id):
                offer = candidate.record.offer
                family = self._families[offer.offer_id]
                if family in seen or (category_id and offer.category_id != category_id):
                    continue
                validation = validate_universal(ports, candidate.interfaces, item)
                if validation.status == Status.REJECTED:
                    continue
                seen.add(family)
                (preferred if validation.status == Status.CONFIRMED else fallback).append(offer)
            return tuple((preferred + fallback)[:count])
        if scenario_id not in self.applicable_scenarios(offer_id, query, filters=filters):
            return ()
        if scenario_id == "ups_battery":
            refs = self._references_with_evidence(offer_id, facts)
            records = [row for article in refs.value or () for row in self._batteries.get(_battery_article(article), ())]
            records.sort(key=lambda row: (not row.available, not row.has_price, row.offer.offer_id))
        elif scenario_id == "laptop_bag":
            records = [row for row, _ in self._ranked_bags]
        else:
            records = self._ranked_endpoints
        preferred = self._recommend_legacy(offer_id, query, filters=filters, facts=facts).items
        seen = {self._families[offer_id]}
        candidates = []
        for row in chain(preferred, records):
            offer = row.offer
            family = self._families[offer.offer_id]
            if family in seen or (category_id and offer.category_id != category_id):
                continue
            seen.add(family)
            candidates.append(offer)
            if len(candidates) == count:
                break
        return tuple(candidates)

    def _candidate_match(
        self, offer_id: str, candidate_id: str, scenario_id: str, query: ParsedQuery | None,
        filters: tuple[Condition, ...], facts: tuple[VendorFact, ...],
    ) -> _Match | Validation:
        if offer_id not in self._offers or candidate_id not in self._offers:
            return Validation(Status.REJECTED, "unknown_offer")
        if scenario_id in DIRECTIONS:
            return self._universal_match(offer_id, candidate_id, scenario_id)
        if scenario_id not in self.applicable_scenarios(offer_id, query, filters=filters):
            return Validation(Status.REJECTED, "unsupported_scenario")
        if self._families[offer_id] == self._families[candidate_id]:
            return Validation(Status.REJECTED, "same_product_family")
        record = self._offers[candidate_id]
        candidate = record.offer
        identity = self._identities.get(candidate_id)
        if not identity or identity.kind != _CANDIDATE_KINDS.get(scenario_id):
            return Validation(Status.REJECTED, "wrong_candidate_type")
        if scenario_id == "ups_battery":
            if normalized(candidate.vendor) != "apc" or not _battery_article(candidate.vendor_code):
                return Validation(Status.REJECTED, "wrong_battery_article")
            refs = self._references_with_evidence(offer_id, facts)
            if not refs.value:
                return Validation(Status.UNKNOWN, "conflicting_facts" if refs.conflict else "missing_battery_reference",
                                  missing=("battery_articles",), sources=refs.sources)
            reference = next((part for part in refs.value if _battery_article(part) == _battery_article(candidate.vendor_code)), "")
            return self._validate_battery(record, reference, refs) if reference else Validation(
                Status.REJECTED, "wrong_battery_article", sources=refs.sources)
        if scenario_id == "laptop_bag":
            laptop = self._storage_with_evidence(offer_id, self._laptops[offer_id], facts)
            bag = self._storage_with_evidence(candidate_id, self._bag_facts[candidate_id], facts)
            return self._validate_bag(laptop, record, bag, self._bag_preferences(query))
        office = self._license_with_evidence(offer_id, self._offices[offer_id], facts)
        context = endpoint_requirements(office, query, filters)
        if isinstance(context, Validation):
            return context
        license = self._license_with_evidence(candidate_id, self._endpoint_facts[candidate_id], facts)
        return self._endpoint_data_failure(license) or self._validate_endpoint(
            record, license, context, tuple(cond for cond in query.conditions if not cond.required) if query else (),
        )

    def check_compatibility(
        self, offer_id: str, candidate_id: str, scenario_id: str, query: ParsedQuery | None = None, *,
        filters: tuple[Condition, ...] = (), facts: tuple[VendorFact, ...] = (),
    ) -> Validation:
        result = self._candidate_match(offer_id, candidate_id, scenario_id, query, filters, facts)
        return result.validation if isinstance(result, _Match) else result

    def select_confirmed(
        self, offer_id: str, pairs: list[tuple[str, str]], query: ParsedQuery | None = None, *,
        filters: tuple[Condition, ...] = (), facts: tuple[VendorFact, ...] = (),
    ) -> list[Recommendation]:
        source = self._offers[offer_id].offer
        confirmed = []
        for scenario, candidate_id in dict.fromkeys(pairs):
            match = self._candidate_match(offer_id, candidate_id, scenario, query, filters, facts)
            if isinstance(match, _Match) and match.validation.status == Status.CONFIRMED:
                confirmed.append(match)
        return [self._recommendation(match) for match in self._select(source, confirmed)]

    def recommend_with_evidence(
        self, offer_id: str, evidence: VendorEvidenceService, query: ParsedQuery | None = None, *,
        filters: tuple[Condition, ...] = (), debug: bool = False,
    ) -> RecommendationResult:
        return self.recommend(offer_id, query, filters=filters, debug=debug, facts=evidence.active_facts())

    def enrichment_targets(
        self, offer_id: str, result: RecommendationResult, query: ParsedQuery | None = None, *,
        filters: tuple[Condition, ...] = (), facts: tuple[VendorFact, ...] = (),
    ) -> tuple[tuple[str, str], ...]:
        if any(direction in DIRECTIONS for direction in result.directions):
            return ()
        if (not result.scenario_id or result.missing_context
                or any(item.available and item.has_price for item in result.items)
                or result.reason_code in {"battery_not_in_catalog", "conflicting_context", "conflicting_facts",
                                          "restricted_customer", "incompatible_license_type"}):
            return ()

        def missing(identifier: str) -> list[tuple[str, str]]:
            details = self.product_details(identifier, facts=facts, query=query, filters=filters)
            return [(identifier, attribute) for attribute in details["missing_attributes"]]

        if result.scenario_id == "ups_battery":
            return tuple(missing(offer_id)) if self._batteries else ()
        if result.scenario_id == "laptop_bag":
            records = [record for record, _ in self._ranked_bags if record.available and record.has_price]
            if not records:
                return ()
            if anchor := missing(offer_id):
                return tuple(anchor)
        else:
            records = [record for record in self._ranked_endpoints if record.available and record.has_price]
            if anchor := missing(offer_id):
                return tuple(anchor)

        targets = []
        for record in records:
            candidate_id = record.offer.offer_id
            if not (attributes := missing(candidate_id)):
                continue
            decision = self.check_compatibility(offer_id, candidate_id, result.scenario_id, query,
                                                filters=filters, facts=facts)
            if decision.status == Status.UNKNOWN and decision.reason_code in {"missing_bag_dimensions", "missing_platform"}:
                targets.extend(attributes)
                if len(targets) >= 3:
                    break
        return tuple(targets[:3])

    def _route(self, source: Offer, query: ParsedQuery | None, filters: tuple[Condition, ...]) -> tuple[str | None, str]:
        applicable = [kind for kind in self._kinds.get(source.offer_id, ())
                      if kind != "ups" or normalized(source.vendor) == "apc"]
        if not applicable:
            return None, self._route_notes.get(source.offer_id, "unsupported_scenario")
        category = self._category_types.get(source.category_id)
        if category in applicable:
            return category, self._route_notes.get(source.offer_id, "catalog_category")
        conditions = (query.conditions if query else ()) + filters
        context = {cond.value for cond in conditions if cond.required and cond.field == "product_type" and cond.operator == "eq"}
        if query:
            context.add(product_type(normalized(query.normalized))[0])
        return next((kind for kind in applicable if kind in context), applicable[0]), "product_attributes"

    def recommend(
        self, offer_id: str, query: ParsedQuery | None = None, *, filters: tuple[Condition, ...] = (),
        debug: bool = False, facts: tuple[VendorFact, ...] = (),
        directions: tuple[str, ...] | None = None,
    ) -> RecommendationResult:
        """Recommend products for legacy scenarios and universal directions.

        Without an explicit ``directions`` argument only query-named directions are used;
        a plain laptop query keeps the legacy bag fast path.
        """
        if directions is None:
            requested = self.requested_directions(offer_id, query, filters=filters)
        else:
            requested = tuple(dict.fromkeys(directions))
        universal = tuple(identifier for identifier in requested if identifier in DIRECTIONS)
        legacy = tuple(identifier for identifier in requested if identifier in _SCENARIOS.values())
        if not universal:
            return self._recommend_legacy(offer_id, query, filters=filters, debug=debug, facts=facts)
        universal_result = self._recommend_universal(offer_id, universal, query, filters=filters,
                                                     debug=debug, facts=facts)
        if not legacy:
            return universal_result
        legacy_result = self._recommend_legacy(offer_id, query, filters=filters, debug=debug, facts=facts)
        return self._merge_results(legacy_result, universal_result, requested)

    def _merge_results(
        self, legacy: RecommendationResult, universal: RecommendationResult, requested: tuple[str, ...],
    ) -> RecommendationResult:
        items = [*legacy.items, *universal.items]
        seen = set()
        unique = []
        for item in items:
            family = self._families[item.offer.offer_id]
            if family not in seen:
                seen.add(family)
                unique.append(item)
        legacy_directions = (legacy.scenario_id,) if legacy.items and legacy.scenario_id else ()
        directions = tuple(dict.fromkeys((*legacy_directions, *legacy.directions, *universal.directions)))
        scenario_id = next((direction for direction in requested if direction in directions), None)
        diagnostics = None
        if legacy.diagnostics is not None or universal.diagnostics is not None:
            detail = universal.diagnostics or legacy.diagnostics or {}
            diagnostics = {
                "legacy": legacy.diagnostics, "universal": universal.diagnostics,
                "directions": list(directions),
                "timings_ms": detail.get("timings_ms", {}),
                "live_validation": detail.get("live_validation", {}),
            }
        if unique:
            return RecommendationResult(scenario_id, unique, Status.CONFIRMED, "compatible", (), diagnostics,
                                        directions)
        failure = universal if universal.status == Status.UNKNOWN else legacy
        return RecommendationResult(scenario_id, [], failure.status, failure.reason_code,
                                    failure.missing_context, diagnostics, directions)

    def _recommend_universal(
        self, offer_id: str, direction_ids: tuple[str, ...], query: ParsedQuery | None, *,
        filters: tuple[Condition, ...], debug: bool, facts: tuple[VendorFact, ...],
    ) -> RecommendationResult:
        started = perf_counter()
        source = self._offers[offer_id].offer
        applicable = set(self.applicable_directions(offer_id))
        selected_directions = tuple(identifier for identifier in direction_ids if identifier in applicable)
        ports = self._ports.get(offer_id, PortFacts())
        routed = perf_counter()
        preferences = self._universal_preferences(query)
        matches = []
        for identifier in selected_directions:
            item = DIRECTIONS[identifier]
            for candidate in self._universal_candidates(identifier):
                match = self._validate_universal(offer_id, candidate, item)
                name = normalized(candidate.record.offer.name)
                preference_count = sum(term in name for term in preferences)
                matches.append(replace(match, rank=(not candidate.record.available, not candidate.record.has_price,
                                                    -preference_count)))
        retrieved = perf_counter()
        confirmed = [match for match in matches if match.validation.status == Status.CONFIRMED]
        selected = self._select_universal(source, confirmed)
        validated = perf_counter()
        items = [self._recommendation(match) for match in selected]
        generated = perf_counter()

        counts = Counter(match.validation.status for match in matches)
        reasons = Counter(match.validation.reason_code for match in matches
                          if match.validation.status != Status.CONFIRMED)
        direction_stats = []
        for identifier in selected_directions:
            direction_matches = [match for match in matches if match.direction_id == identifier]
            direction_counts = Counter(match.validation.status for match in direction_matches)
            direction_stats.append({
                "direction_id": identifier, "label": DIRECTIONS[identifier].label,
                "candidates": len(direction_matches), "confirmed": direction_counts[Status.CONFIRMED],
                "unknown": direction_counts[Status.UNKNOWN], "rejected": direction_counts[Status.REJECTED],
                "requirements": list(DIRECTIONS[identifier].requirements),
            })
        outcome = self._universal_outcome(items, matches, reasons)
        confirmed_ids = {item.direction_id for item in items}
        confirmed_directions = tuple(identifier for identifier in selected_directions if identifier in confirmed_ids)
        scenario_id = confirmed_directions[0] if confirmed_directions else (
            selected_directions[0] if selected_directions else None)
        missing = tuple(dict.fromkeys(field for match in matches if match.validation.status == Status.UNKNOWN
                                      for field in match.validation.missing)) or ports.missing()
        diagnostics = None
        if debug:
            examples = {}
            for match in matches:
                examples.setdefault(match.validation.reason_code, {
                    "id": match.product.offer.offer_id, "direction_id": match.direction_id,
                    "validation": match.validation,
                })
            diagnostics = _to_json({
                "offer_id": offer_id, "name": source.name, "article": source.vendor_code,
                "scenario_id": scenario_id, "directions": direction_stats,
                "anchor_ports": ports.public(), "missing_anchor_attributes": list(ports.missing()),
                "requirements": {identifier: list(DIRECTIONS[identifier].requirements)
                                 for identifier in selected_directions},
                "routing": {"applicable": bool(selected_directions), "reason_code": "direction_request"},
                "context": {"query": query.original if query else "", "filters": filters,
                            "requirements": [requirements(DIRECTIONS[identifier])
                                             for identifier in selected_directions]},
                "live_validation": {
                    "checked": len(matches), "confirmed": counts[Status.CONFIRMED],
                    "unknown": counts[Status.UNKNOWN], "rejected": counts[Status.REJECTED], "errors": 0,
                },
                "candidates": len(matches), "confirmed": counts[Status.CONFIRMED],
                "unknown": counts[Status.UNKNOWN], "rejected": counts[Status.REJECTED],
                "reason_counts": dict(reasons), "after_deduplication": len(selected),
                "recommendations": len(items), "decision": outcome,
                "candidate_examples": examples, "index": self.index_stats,
                "timings_ms": {name: (right - left) * 1000 for name, left, right in (
                    ("routing", started, routed), ("context", routed, routed),
                    ("retrieval", routed, retrieved), ("validation", retrieved, retrieved),
                    ("ranking", retrieved, validated), ("response", validated, generated),
                    ("total", started, generated),
                )},
            })
        return RecommendationResult(scenario_id, items, outcome.status, outcome.reason_code,
                                    missing if not items else (), diagnostics,
                                    confirmed_directions or selected_directions)

    def _universal_preferences(self, query: ParsedQuery | None) -> tuple[str, ...]:
        if query is None:
            return ()
        terms = re.findall(r"\w+", normalized(query.normalized))
        stop = {"для", "подбери", "нужен", "нужна", "нужно", "ноутбука", "ноутбук", "какой", "какая",
                "посоветуй", "аксессуар", "аксессуары", "допродажа", "докупить"}
        return tuple(term for term in terms if len(term) > 3 and term not in stop)

    def _select_universal(self, source: Offer, matches: list[_Match]) -> list[_Match]:
        source_family = self._families[source.offer_id]
        candidates = [match for match in matches if self._families[match.product.offer.offer_id] != source_family]
        ready = [match for match in candidates if match.product.available and match.product.has_price]
        eligible = ready or candidates
        currencies = {match.product.offer.currency.replace("RUR", "RUB") for match in eligible}
        compare_price = bool(ready) and len(currencies) == 1
        eligible.sort(key=lambda item: (
            item.rank, number(item.product.offer.price) if compare_price else 0,
            item.product.offer.offer_id,
        ))
        selected = []
        families = set()
        vendors = Counter()
        per_direction = Counter()
        # First pass: one confirmed product per direction for visible diversity.
        for match in eligible:
            family = self._families[match.product.offer.offer_id]
            vendor = normalized(match.product.offer.vendor)
            if family in families or vendors[vendor] >= _VENDOR_LIMIT or per_direction[match.direction_id] >= 1:
                continue
            selected.append(match)
            families.add(family)
            vendors[vendor] += 1
            per_direction[match.direction_id] += 1
            if len(selected) >= _UNIVERSAL_LIMIT:
                return selected
        # Second pass: fill remaining slots with the commercial ranking.
        for match in eligible:
            if len(selected) >= _UNIVERSAL_LIMIT:
                break
            family = self._families[match.product.offer.offer_id]
            vendor = normalized(match.product.offer.vendor)
            if family in families or vendors[vendor] >= _VENDOR_LIMIT:
                continue
            selected.append(match)
            families.add(family)
            vendors[vendor] += 1
        return selected

    def _universal_outcome(self, items, matches, reasons) -> Validation:
        if items:
            return Validation(Status.CONFIRMED, "compatible")
        unknown = Counter(match.validation.reason_code for match in matches
                          if match.validation.status == Status.UNKNOWN)
        if unknown:
            reason = unknown.most_common(1)[0][0]
            return next(match.validation for match in matches
                        if match.validation.status == Status.UNKNOWN and match.validation.reason_code == reason)
        return Validation(Status.REJECTED, reasons.most_common(1)[0][0] if reasons else "no_candidates")

    def _recommend_legacy(
        self, offer_id: str, query: ParsedQuery | None = None, *, filters: tuple[Condition, ...] = (),
        debug: bool = False, facts: tuple[VendorFact, ...] = (),
    ) -> RecommendationResult:
        started = perf_counter()
        source = self._offers[offer_id].offer
        kind, route_reason = self._route(source, query, filters)
        routed = perf_counter()
        anchor_facts = (self._references.get(offer_id) if kind == "ups" else self._laptops.get(offer_id)
                        if kind == "laptop" else self._offices.get(offer_id))
        if facts:
            if kind == "ups":
                anchor_facts = self._references_with_evidence(offer_id, facts)
            elif kind == "laptop":
                anchor_facts = self._storage_with_evidence(offer_id, anchor_facts, facts)
            elif kind == "office":
                facts = tuple(fact for fact in facts if fact.attribute == "platforms")
                anchor_facts = self._license_with_evidence(offer_id, anchor_facts, facts)
        user_context = None
        context_failure = None
        if kind is None:
            context_failure = Validation(Status.REJECTED, "unsupported_scenario")
        elif kind == "office":
            context = endpoint_requirements(anchor_facts, query, filters)
            if isinstance(context, Validation):
                context_failure = context
            else:
                user_context = context
        extracted = perf_counter()

        candidates = []
        missing_batteries = []
        exclusions = {}
        if context_failure is None:
            if kind == "ups":
                refs = anchor_facts
                if not refs.value:
                    reason = "conflicting_facts" if refs.conflict else "missing_battery_reference"
                    context_failure = Validation(Status.UNKNOWN, reason,
                                                 missing=("battery_articles",), sources=refs.sources)
                else:
                    missing_batteries = [article for article in refs.value if _battery_article(article) not in self._batteries]
                    candidates = [(battery, article) for article in refs.value
                                  for battery in self._batteries.get(_battery_article(article), ())]
                    if not candidates:
                        context_failure = Validation(Status.REJECTED, "battery_not_in_catalog", sources=refs.sources)
            elif kind == "laptop":
                candidates = self._bags
            else:
                candidates = self._endpoints
                exclusions = self._endpoint_exclusions
                if facts:
                    candidates = [(record, self._license_with_evidence(record.offer.offer_id, license, facts))
                                  for record, license in candidates]
                    exclusions = exclusions.copy()
                    for pending in self._endpoints_missing_platform:
                        record = pending.product
                        oid = record.offer.offer_id
                        license = self._license_with_evidence(oid, self._endpoint_facts[oid], facts)
                        if self._endpoint_data_failure(license) is None:
                            candidates.append((record, license))
                            key = (pending.validation.status, pending.validation.reason_code)
                            exclusions[key] -= 1
                            if not exclusions[key]:
                                del exclusions[key]
        retrieved = perf_counter()

        preferences = tuple(cond for cond in query.conditions if not cond.required) if query else ()
        matches = []
        if kind == "ups":
            matches = [self._validate_battery(candidate, reference, anchor_facts) for candidate, reference in candidates]
        elif kind == "laptop":
            terms = self._bag_preferences(query)
            matches = [self._validate_bag(anchor_facts, candidate,
                                         self._storage_with_evidence(candidate.offer.offer_id, storage, facts), terms)
                       for candidate, storage in candidates]
        elif candidates and user_context:
            matches = [self._validate_endpoint(candidate, facts, user_context, preferences)
                       for candidate, facts in candidates]
        validated = perf_counter()
        selected = self._select(source, [match for match in matches if match.validation.status == Status.CONFIRMED])
        ranked = perf_counter()
        items = [self._recommendation(match) for match in selected]
        generated = perf_counter()

        reasons = Counter()
        if debug or not items:
            reasons.update(match.validation.reason_code for match in matches if match.validation.status != Status.CONFIRMED)
            for (_, reason), count in exclusions.items():
                reasons[reason] += count
            if context_failure:
                reasons[context_failure.reason_code] += 1
        outcome = self._outcome(items, context_failure, matches, reasons, exclusions)
        scenario_id = _SCENARIOS.get(kind)
        diagnostics = None
        if debug:
            counts = Counter(match.validation.status for match in matches)
            validation_counts = {
                "checked": len(matches), "confirmed": counts[Status.CONFIRMED],
                "unknown": counts[Status.UNKNOWN], "rejected": counts[Status.REJECTED], "errors": 0,
            }
            for (status, _), count in exclusions.items():
                counts[status] += count
            evidence = (*self._endpoint_examples.values(), *matches) if exclusions else matches
            examples = {}
            for match in evidence:
                examples.setdefault(match.validation.reason_code, {
                    "id": match.product.offer.offer_id, "validation": match.validation,
                })
            diagnostics = _to_json({
                "offer_id": offer_id, "name": source.name, "article": source.vendor_code, "scenario_id": scenario_id,
                "routing": {"applicable": kind is not None, "reason_code": route_reason},
                "context": {"query": query.original if query else "", "filters": filters, "requirements": user_context},
                "anchor_facts": anchor_facts,
                "live_validation": validation_counts, "index_exclusions": sum(exclusions.values()),
                "candidates": sum(counts.values()), "confirmed": counts[Status.CONFIRMED],
                "unknown": counts[Status.UNKNOWN], "rejected": counts[Status.REJECTED],
                "reason_counts": dict(reasons), "after_deduplication": len(selected), "recommendations": len(items),
                "decision": outcome, "candidate_examples": examples, "missing_battery_articles": missing_batteries,
                "index": self.index_stats,
                "timings_ms": {name: (right - left) * 1000 for name, left, right in (
                    ("routing", started, routed), ("context", routed, extracted), ("retrieval", extracted, retrieved),
                    ("validation", retrieved, validated), ("ranking", validated, ranked), ("response", ranked, generated),
                    ("total", started, generated),
                )},
            })
        missing_context = tuple(field for field in outcome.missing if field in {"platform", "devices"}) if context_failure else ()
        return RecommendationResult(scenario_id, items, outcome.status, outcome.reason_code, missing_context, diagnostics)

    def _outcome(self, items, failure, matches, reasons, exclusions) -> Validation:
        if items:
            return Validation(Status.CONFIRMED, "compatible")
        if failure:
            return failure
        unknown = Counter(match.validation.reason_code for match in matches if match.validation.status == Status.UNKNOWN)
        unknown.update({reason: count for (status, reason), count in exclusions.items() if status == Status.UNKNOWN})
        if unknown:
            reason = unknown.most_common(1)[0][0]
            if (Status.UNKNOWN, reason) in exclusions:
                return self._endpoint_examples[reason].validation
            return next(match.validation for match in matches if match.validation.reason_code == reason
                        and match.validation.status == Status.UNKNOWN)
        return Validation(Status.REJECTED, reasons.most_common(1)[0][0] if reasons else "no_candidates")

    def _validate_battery(self, candidate: _CatalogOffer, reference: str, refs: Fact) -> _Match:
        reason = f"Для ИБП указана сменная батарея {reference}. Ей соответствует артикул {candidate.offer.vendor_code}."
        return _Match(candidate, Validation(Status.CONFIRMED, "battery_article_match", reason,
                                            facts={"battery_article": reference}, sources=refs.sources),
                      battery_reference=reference)

    def _bag_preferences(self, query: ParsedQuery | None) -> tuple[str, ...]:
        text = normalized(query.normalized) if query else ""
        match = re.search(r"\b(?:сумк\w*|чехол\w*)\s+(.+)", text)
        if not match:
            return ()
        return tuple(term for term in re.findall(r"\w+", match[1]) if len(term) > 2 and term not in {"для", "ноутбука"})

    def _validate_bag(self, laptop: _Storage, candidate: _CatalogOffer, bag: _Storage, terms: tuple[str, ...]) -> _Match:
        exact = (laptop.models.value or frozenset()) & (bag.models.value or frozenset())
        size, limit = laptop.dimensions.value, bag.dimensions.value
        sources = tuple(dict.fromkeys((*laptop.dimensions.sources, *bag.dimensions.sources, *bag.models.sources)))
        facts = {"laptop_mm": size, "maximum_mm": limit, "exact_model": min(exact) if exact else None}
        if size and limit and any(axis > bound for axis, bound in zip(size, limit, strict=True)):
            return _Match(candidate, Validation(Status.REJECTED, "incompatible_dimensions", facts=facts, sources=sources))
        if (bag.strict_diagonal and laptop.diagonal.value is not None and bag.diagonal.value is not None
                and laptop.diagonal.sources and all(not source.startswith("XML") for source in laptop.diagonal.sources)
                and laptop.diagonal.value > bag.diagonal.value):
            return _Match(candidate, Validation(Status.REJECTED, "incompatible_diagonal",
                                                facts={"laptop_inches": laptop.diagonal.value, "limit_inches": bag.diagonal.value},
                                                sources=(*laptop.diagonal.sources, *bag.diagonal.sources)))
        if ((laptop.dimensions.conflict and size is None) or (bag.dimensions.conflict and limit is None)
                or (bag.models.conflict and bag.models.value is None)):
            return _Match(candidate, Validation(Status.UNKNOWN, "conflicting_facts", facts=facts, sources=sources))
        if exact:
            reason = f"Модель {min(exact)} указана в списке совместимых."
        elif size and limit:
            reason = f"Габариты ноутбука {_size_text(size)}. Предельные габариты для сумки {_size_text(limit)}."
        else:
            code = "missing_laptop_dimensions" if not size else "missing_bag_dimensions"
            missing = (*(("laptop_dimensions",) if not size else ()), *(("bag_dimensions",) if not limit else ()))
            return _Match(candidate, Validation(Status.UNKNOWN, code, facts=facts, missing=missing, sources=sources))
        name = normalized(f"{candidate.offer.name} {candidate.offer.vendor}")
        preference_count = sum(term in name for term in terms)
        return _Match(candidate, Validation(Status.CONFIRMED, "model_match" if exact else "dimensions_match",
                                            reason, facts, sources=sources), (not bool(exact), -preference_count))

    def _endpoint_data_failure(self, facts: LicenseFacts) -> Validation | None:
        if facts.eligibility.status != Status.CONFIRMED:
            return facts.eligibility
        if not facts.platforms.value:
            code = "conflicting_facts" if facts.platforms.conflict else "missing_platform"
            return Validation(Status.UNKNOWN, code, missing=("platform",), sources=facts.platforms.sources)
        capacity = facts.capacity.value
        if not capacity or (capacity.package is None and not capacity.per_unit):
            code = "conflicting_facts" if facts.capacity.conflict else "unknown_license_capacity"
            return Validation(Status.UNKNOWN, code, missing=("capacity",), sources=facts.capacity.sources)
        if capacity.unit not in {"node", "device"}:
            return Validation(Status.REJECTED, "incompatible_license_unit", sources=facts.capacity.sources)
        if capacity.minimum_purchase is None:
            return Validation(Status.UNKNOWN, "unknown_minimum_purchase", missing=("minimum_purchase",),
                              sources=tuple(dict.fromkeys((*facts.capacity.sources, *facts.eligibility.sources))))
        return None

    def _validate_endpoint(
        self, candidate: _CatalogOffer, facts: LicenseFacts, context: EndpointContext, preferences: tuple[Condition, ...],
    ) -> _Match:
        if any(all(any(supports_platform(request, excluded) for excluded in facts.unsupported_platforms)
                   for request in alternatives) for alternatives in context.platforms):
            return _Match(candidate, Validation(Status.REJECTED, "unsupported_platform", sources=facts.platforms.sources))
        known = facts.platforms.value
        capacity = facts.capacity.value
        assert known and capacity
        if context.term_conditions:
            if facts.term_months is None:
                return _Match(candidate, Validation(Status.UNKNOWN, "unknown_license_term", missing=("term_months",)))
            if not context.supports_term(facts.term_months):
                return _Match(candidate, Validation(Status.REJECTED, "incompatible_license_term"))
        matched = []
        for alternatives in context.platforms:
            possible = [request for request in alternatives if not any(supports_platform(request, excluded)
                                                                      for excluded in facts.unsupported_platforms)]
            proof = sorted(platform for platform in known if any(supports_platform(platform, request) for request in possible))
            if not proof:
                if any(supports_platform(request, platform) for request in possible for platform in known):
                    return _Match(candidate, Validation(Status.UNKNOWN, "missing_platform", missing=("platform_version",),
                                                         sources=facts.platforms.sources))
                return _Match(candidate, Validation(Status.REJECTED, "unsupported_platform", sources=facts.platforms.sources))
            matched.append(proof[0])
        if not (allocation := capacity.allocation(context.devices)):
            return _Match(candidate, Validation(Status.REJECTED, "insufficient_devices", sources=facts.capacity.sources))
        covered, quantity = allocation
        platforms = ", ".join(map(_platform_label, dict.fromkeys(matched)))
        reason = f"Поддерживается {platforms}. Корпоративная лицензия покрывает {covered} ПК, требуется {context.devices}."
        if capacity.per_unit or quantity > 1:
            reason += f" Количество приобретаемых лицензий — {quantity}. Цена в карточке указана за одну."
        if not context.buyer:
            reason += " Условие продажи по каталогу — юридическое лицо."
        excess = covered - context.devices
        proof = {"platforms": matched, "devices": covered, "required_devices": context.devices, "quantity": quantity,
                 "unit": capacity.unit, "term_months": facts.term_months, "excess_devices": excess,
                 "edition_level": facts.edition_level}
        return _Match(candidate, Validation(Status.CONFIRMED, "platform_and_license_match", reason, proof, sources=tuple(
            dict.fromkeys((*facts.platforms.sources, *facts.capacity.sources, *facts.eligibility.sources,
                           "XML тип лицензии", "XML тип клиента")))),
            (-endpoint_preference_count(facts, preferences), excess), purchase_quantity=quantity)

    @staticmethod
    def _dominated_license(match: _Match, preferred: _Match) -> bool:
        if (match is preferred or match.validation.reason_code != "platform_and_license_match"
                or preferred.validation.reason_code != "platform_and_license_match"):
            return False
        offer, alternative = match.product.offer, preferred.product.offer
        facts, better = match.validation.facts, preferred.validation.facts
        # PRO32 documents Advanced as a superset of Standard. Preserve other editions
        # when the catalog provides no comparable feature information.
        return (
            normalized(offer.vendor) == normalized(alternative.vendor) == "pro32"
            and offer.currency.replace("RUR", "RUB") == alternative.currency.replace("RUR", "RUB")
            and facts["term_months"] is not None and facts["term_months"] == better["term_months"]
            and facts["edition_level"] is not None and better["edition_level"] is not None
            and better["edition_level"] >= facts["edition_level"] and preferred.rank[0] <= match.rank[0]
            and better["excess_devices"] < facts["excess_devices"]
            and number(alternative.price) * preferred.purchase_quantity < number(offer.price) * match.purchase_quantity
        )

    def _select(self, source: Offer, matches: list[_Match], *, preserve_order: bool = False) -> list[_Match]:
        source_family = self._families[source.offer_id]
        candidates = [match for match in matches if self._families[match.product.offer.offer_id] != source_family]
        ready = [match for match in candidates if match.product.available and match.product.has_price]
        eligible = ready or candidates
        if not preserve_order:
            currencies = {match.product.offer.currency.replace("RUR", "RUB") for match in eligible}
            compare_price = bool(ready) and len(currencies) == 1
            eligible.sort(key=lambda item: (item.rank, number(item.product.offer.price) * item.purchase_quantity
                                            if compare_price else 0, item.purchase_quantity, item.product.offer.offer_id))
        selected = []
        families = set()
        battery_articles = set()
        for match in eligible:
            article = _battery_article(match.battery_reference)
            family = self._families[match.product.offer.offer_id]
            if family not in families and (not article or article not in battery_articles):
                selected.append(match)
                families.add(family)
                if article:
                    battery_articles.add(article)
        if ready:
            selected = [match for match in selected
                        if not any(self._dominated_license(match, other) for other in selected)]
        return selected

    def _recommendation(self, match: _Match) -> Recommendation:
        record = match.product
        total = number(record.offer.price) * match.purchase_quantity if record.has_price else None
        item = DIRECTIONS.get(match.direction_id)
        return Recommendation(record.offer, record.available, record.has_price, match.validation.reason,
                              match.validation.sources, match.purchase_quantity, total,
                              match.direction_id, item.label if item else "")


def recommend_batteries(catalog_path: Path, selected: str) -> tuple[Offer, list[BatteryRecommendation]]:
    selected = selected.strip()
    if not selected:
        raise ValueError("Укажите ID предложения или его URL из фида.")
    agent = RecommendationAgent(catalog_path)
    offers = [record.offer for record in agent._offers.values() if selected in (record.offer.offer_id, record.offer.url)]
    if not offers:
        raise ValueError(f"Предложение {selected} отсутствует в XML-каталоге.")
    if len(offers) > 1:
        raise ValueError("Ссылке соответствуют несколько предложений. Укажите ID нужного предложения.")
    source = offers[0]
    refs = agent._references.get(source.offer_id, Fact())
    matches = [agent._validate_battery(battery, reference, refs) for reference in refs.value or ()
               for battery in agent._batteries.get(_battery_article(reference), ())]
    return source, [
        BatteryRecommendation(match.product.offer, match.battery_reference, match.product.available, match.product.has_price)
        for match in agent._select(source, matches, preserve_order=True)[:_LEGACY_BATTERY_LIMIT]
    ]
