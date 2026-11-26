import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from xml.etree.ElementTree import Element

from .catalog import Offer, catalog_elements, read_offer
from .index import _catalog_families
from .query import Condition, ParsedQuery, TYPE_PATTERNS, number, product_type
from .query_catalog import _KNOWN_CATEGORY_TYPES
from .recommendation_facts import (
    BAG_LIMITS, LAPTOP_DIMENSIONS, Dimensions, Fact, Fields, Status, Validation,
    diagonal, dimensions, load_specs, models, normalized, product_key, read_fields as _fields, spec_values,
)
from .recommendation_licenses import (
    EndpointContext, LicenseFacts, endpoint_preference_count, endpoint_requirements,
    office_scope, read_license_facts, supports_platform,
)


_CATEGORY_TYPES = {
    "82080": "ups", "82014": "ups_battery", "70824": "laptop", "83844": "notebook_bag",
    "1145": "office", "872": "office", "870": "office", "222323": "office", "650": "antivirus_business",
}
_CATEGORY_NAMES = {**_KNOWN_CATEGORY_TYPES, "антивирусы для организаций": "antivirus_business"}
_SCENARIOS = {"ups": "ups_battery", "laptop": "laptop_bag", "office": "office_antivirus"}
_RELEVANT_TYPES = set(_SCENARIOS) | {"ups_battery", "notebook_bag", "antivirus_business"}
_BATTERY_FIELDS = ("совместимая батарея", "совместимые батареи", "сменная батарея", "артикул сменной батареи")
_BATTERY_ARTICLE = re.compile(r"(?:APC)?(RBC[0-9]+)", re.IGNORECASE | re.ASCII)
_TRAIT_FIELDS = ("форм-фактор", "тип устройства", "тип программного обеспечения", "назначение")
_MODEL_FIELDS = ("совместимые модели", "совместимые модели ноутбуков", "совместимые артикулы ноутбуков")
_WRONG_LAPTOP = re.compile(r"^(?:планшет|монитор|сумка|чехол|аккумулятор|подставка|блок питания)\b")
_LAPTOP_ACCESSORIES = {"monitor", "notebook_bag", "ups_battery", "tablet", "ram", "psu", "charger"}
_LEGACY_BATTERY_LIMIT = 2


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


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    scenario_id: str | None
    items: list[Recommendation]
    status: Status = Status.CONFIRMED
    reason_code: str = ""
    missing_context: tuple[str, ...] = ()
    diagnostics: dict | None = None


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
        self._kinds: dict[str, tuple[str, ...]] = {}
        self._route_notes: dict[str, str] = {}
        self._references: dict[str, Fact[tuple[str, ...]]] = {}
        self._batteries: dict[str, list[_CatalogOffer]] = defaultdict(list)
        self._laptops: dict[str, _Storage] = {}
        self._bags: list[tuple[_CatalogOffer, _Storage]] = []
        self._offices: dict[str, LicenseFacts] = {}
        self._endpoints: list[tuple[_CatalogOffer, LicenseFacts]] = []
        self._endpoint_exclusions: Counter[tuple[Status, str]] = Counter()
        self._endpoint_examples: dict[str, _Match] = {}
        specs = load_specs(Path(__file__).with_name("compatibility_specs.json"))
        counts = Counter()
        available_counts = Counter()
        enriched = set()
        conflicts = Counter()

        for element in catalog_elements(catalog_path):
            if element.tag == "category":
                category_id = element.attrib["id"]
                if category := self._category_types.get(category_id) or _CATEGORY_NAMES.get(normalized(element.text or "")):
                    self._category_types[category_id] = category
                continue

            offer = read_offer(element)
            record = _CatalogOffer(offer, element.get("available", "").lower() == "true", _has_price(offer))
            self._offers[offer.offer_id] = record
            counts[offer.category_id] += 1
            if record.available:
                available_counts[offer.category_id] += 1
            category = self._category_types.get(offer.category_id, "")
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
            "endpoints_ready_for_context_check": len(self._endpoints),
            "build_ms": (perf_counter() - started) * 1000,
        }

    def _prepare_offer(
        self, kind: str, record: _CatalogOffer, fields: Fields, specs: list[dict], conflicts: Counter,
    ) -> None:
        offer = record.offer
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
            else:
                self._bags.append((record, storage))
        elif kind in {"office", "antivirus_business"}:
            license_facts = read_license_facts(fields, specs, office=kind == "office")
            conflicts["platform"] += license_facts.platforms.conflict
            conflicts["license_capacity"] += license_facts.capacity.conflict
            if kind == "office":
                self._offices[offer.offer_id] = license_facts
            elif failure := self._endpoint_data_failure(license_facts):
                self._endpoint_exclusions[failure.status, failure.reason_code] += 1
                self._endpoint_examples.setdefault(failure.reason_code, _Match(record, failure))
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
        kind = next((kind for kind, name in _SCENARIOS.items() if name == scenario), "")
        return tuple(oid for oid, record in self._offers.items()
                     if record.available and kind in self._kinds.get(oid, ())
                     and (kind != "ups" or normalized(record.offer.vendor) == "apc"))

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
        debug: bool = False,
    ) -> RecommendationResult:
        started = perf_counter()
        source = self._offers[offer_id].offer
        kind, route_reason = self._route(source, query, filters)
        routed = perf_counter()
        anchor_facts = (self._references.get(offer_id) if kind == "ups" else self._laptops.get(offer_id)
                        if kind == "laptop" else self._offices.get(offer_id))
        user_context = None
        context_failure = None
        if kind is None:
            context_failure = Validation(Status.REJECTED, "unsupported_scenario")
        elif kind == "office":
            context = endpoint_requirements(self._offices[offer_id], query, filters)
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
                refs = self._references[offer_id]
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
        retrieved = perf_counter()

        preferences = tuple(cond for cond in query.conditions if not cond.required) if query else ()
        matches = []
        if kind == "ups":
            matches = [self._validate_battery(candidate, reference, self._references[offer_id])
                       for candidate, reference in candidates]
        elif kind == "laptop":
            terms = self._bag_preferences(query)
            matches = [self._validate_bag(anchor_facts, candidate, storage, terms) for candidate, storage in candidates]
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
        proof = {"platforms": matched, "devices": covered, "required_devices": context.devices, "quantity": quantity,
                 "unit": capacity.unit, "term_months": facts.term_months}
        return _Match(candidate, Validation(Status.CONFIRMED, "platform_and_license_match", reason, proof, sources=tuple(
            dict.fromkeys((*facts.platforms.sources, *facts.capacity.sources, "XML тип лицензии", "XML тип клиента")))),
            (-endpoint_preference_count(facts, preferences), covered - context.devices, quantity), purchase_quantity=quantity)

    def _select(self, source: Offer, matches: list[_Match], *, preserve_order: bool = False) -> list[_Match]:
        source_family = self._families[source.offer_id]
        candidates = [match for match in matches if self._families[match.product.offer.offer_id] != source_family]
        ready = [match for match in candidates if match.product.available and match.product.has_price]
        eligible = ready or candidates
        if not preserve_order:
            currencies = {match.product.offer.currency.replace("RUR", "RUB") for match in eligible}
            compare_price = bool(ready) and len(currencies) == 1
            eligible.sort(key=lambda item: (item.rank, number(item.product.offer.price) * item.purchase_quantity
                                            if compare_price else 0, item.product.offer.offer_id))
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
        return selected

    def _recommendation(self, match: _Match) -> Recommendation:
        record = match.product
        total = number(record.offer.price) * match.purchase_quantity if record.has_price else None
        return Recommendation(record.offer, record.available, record.has_price, match.validation.reason,
                              match.validation.sources, match.purchase_quantity, total)


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
