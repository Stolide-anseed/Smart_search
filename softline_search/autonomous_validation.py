"""Check catalogue cross-sells without requiring an exhaustive compatibility proof."""

import re
from dataclasses import replace

from .catalog import Offer
from .query import Condition, ParsedQuery
from .query_catalog import QueryCatalog
from .recommendation import _CANDIDATE_KINDS, RecommendationAgent
from .recommendation_facts import Status, Validation, normalized, product_key
from .vendor_evidence import VendorFact


_COMPLEMENT_ROLE = re.compile(
    r"^(?:(?P<consumable>(?:(?:лазерн\w*|струйн\w*) )?картридж\w*"
    r"|тонер\w*|чернил\w*|расходн\w* материал\w*)"
    r"|(?P<accessory>аксессуар\w*|чехол\w*|сумк\w*|рюкзак\w*|подставк\w*"
    r"|кронштейн\w*|кабел\w*|адаптер\w*|переходник\w*)"
    r"|(?P<replacement>сменн\w* (?:батаре\w*|аккумулятор\w*)|запчаст\w*|ремкомплект\w*)"
    r"|(?P<extension>дополнени\w*|расширени\w*|плагин\w*|дополнительн\w* модул\w*))\b"
)
_PRINTING_CONTEXT = re.compile(r"\b(?:принтер\w*|мфу|плоттер\w*|тонер\w*|печат\w*|струйн\w*|лазерн\w*)\b")
_GENERIC_SOFTWARE = {"софт", "по", "программное обеспечение", "приложение"}
_PARTIAL_REASON = "Перед покупкой уточните совместимость по характеристикам производителя."


def _complement_role(offer: Offer, catalog: QueryCatalog) -> str:
    detail = catalog.details.get(offer.offer_id)
    attributes = detail.attributes if detail else {}
    role_fields = ("тип устройства", "тип продукта", "назначение", "тип программного обеспечения")
    labels = (
        offer.name,
        *(value for field in role_fields for value in attributes.get(field, ())),
        catalog.categories.get(offer.category_id, ""),
    )
    for label in labels:
        if match := _COMPLEMENT_ROLE.match(normalized(label)):
            return match.lastgroup
    return ""


def _software_types(offer: Offer, catalog: QueryCatalog) -> set[str]:
    detail = catalog.details.get(offer.offer_id)
    values = detail.attributes.get("тип программного обеспечения", ()) if detail else ()
    return {normalized(value) for value in values} - _GENERIC_SOFTWARE - {""}


def _known_validation(
    source: Offer, candidate: Offer, rules: RecommendationAgent,
    query: ParsedQuery | None, filters: tuple[Condition, ...], facts: tuple[VendorFact, ...],
) -> Validation | None:
    identity = rules.product_identity(candidate.offer_id)
    scenarios = (
        identifier for identifier in rules.applicable_scenarios(source.offer_id, query, filters=filters)
        if _CANDIDATE_KINDS.get(identifier) == identity.kind
    )
    directions = (
        identifier for identifier in rules.applicable_directions(source.offer_id)
        if rules.is_candidate(candidate.offer_id, identifier)
    )
    validations = [
        rules.check_compatibility(source.offer_id, candidate.offer_id, identifier, query,
                                  filters=filters, facts=facts)
        for identifier in (*scenarios, *directions)
    ]
    # A proven conflict can coexist with a successful check of another characteristic.
    for status in (Status.REJECTED, Status.CONFIRMED, Status.UNKNOWN):
        if result := next((item for item in validations if item.status == status), None):
            return result
    return None


def validate_cross_sell(
    source: Offer, candidate: Offer, catalog: QueryCatalog, rules: RecommendationAgent, *,
    query: ParsedQuery | None = None, filters: tuple[Condition, ...] = (), facts: tuple[VendorFact, ...] = (),
) -> Validation:
    """Keep open catalogue directions provisional and apply available checks to known pairs."""
    anchor = rules.get_offer(source.offer_id)
    offered = rules.get_offer(candidate.offer_id)
    if anchor is None or offered is None:
        return Validation(Status.REJECTED, "unknown_offer")
    if anchor.offer_id == offered.offer_id:
        return Validation(Status.REJECTED, "same_offer")

    vendor, article = product_key(anchor.vendor, anchor.vendor_code)
    same_category = anchor.category_id == offered.category_id
    same_group = anchor.group_id and anchor.group_id == offered.group_id
    same_article = same_category and vendor and article and (vendor, article) == product_key(
        offered.vendor, offered.vendor_code,
    )
    if same_group or same_article:
        return Validation(Status.REJECTED, "same_product_family")

    known = _known_validation(anchor, offered, rules, query, filters, facts)
    if known and known.status != Status.UNKNOWN:
        return known

    candidate_role = _complement_role(offered, catalog)
    adds_role = candidate_role and candidate_role != _complement_role(anchor, catalog)
    anchor_types = catalog.product_types(anchor)
    offered_types = catalog.product_types(offered)
    if candidate_role == "consumable" and anchor_types & {"laptop", "ups", "monitor"}:
        application = (offered.name, catalog.categories.get(offered.category_id, ""), *catalog.purposes(offered))
        if any(_PRINTING_CONTEXT.search(normalized(label)) for label in application):
            return Validation(Status.REJECTED, "different_target_product",
                              "По каталогу расходный материал предназначен для печати.")

    shared_types = (anchor_types & offered_types) - {"software"}
    same_model = same_category and normalized(anchor.name) == normalized(offered.name)
    same_software_type = (same_category and "software" in anchor_types & offered_types
                          and bool(_software_types(anchor, catalog) & _software_types(offered, catalog)))
    if not adds_role and (shared_types or same_model or same_software_type):
        return Validation(Status.REJECTED, "substitute_product", "Товар выполняет ту же основную функцию.")

    if known:
        return replace(known, reason=known.reason or _PARTIAL_REASON, missing=known.missing or ("compatibility",))
    return Validation(Status.UNKNOWN, "partial_catalog_check", _PARTIAL_REASON, missing=("compatibility",))
