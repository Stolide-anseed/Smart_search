import json
import re
import sqlite3
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree.ElementTree import ParseError

from .catalog import Offer
from .index import SearchIndex, SearchMode
from .query import Condition, number
from .query_catalog import SOFTWARE_CATEGORY_ID
from .query_normalization import query_key


_ASSET_ROOT = Path(__file__).with_name("demo_web")
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_PLATFORMS = {"windows", "windows 10", "windows 11", "macos", "linux", "astra linux", "android", "ios"}


class _BadRequest(ValueError):
    pass


def _parameter(parameters: dict[str, list[str]], key: str) -> str:
    values = parameters.get(key)
    if values is None:
        return ""
    if len(values) != 1:
        raise _BadRequest(f"Укажите параметр {key} один раз.")
    return values[0].strip()


def _positive_number(raw: str, message: str, *, integer: bool = False) -> Decimal:
    pattern = r"[0-9]+" if integer else r"[0-9]+(?:[.,][0-9]+)?"
    if re.fullmatch(pattern, raw):
        value = number(raw)
        if value > 0:
            return value
    raise _BadRequest(message)


def _filters(
    parameters: dict[str, list[str]], categories: dict[str, str], vendors: set[str]
) -> tuple[Condition, ...]:
    result = []
    if category_id := _parameter(parameters, "category_id"):
        if category_id not in categories:
            raise _BadRequest("Выберите категорию из каталога.")
        field = "software_category" if category_id == SOFTWARE_CATEGORY_ID else "category_id"
        result.append(Condition(field, "eq", category_id, f"Категория — {categories[category_id]}"))
    if vendor := _parameter(parameters, "vendor"):
        if vendor not in vendors:
            raise _BadRequest("Выберите производителя из каталога.")
        result.append(Condition("brand", "eq", query_key(vendor).strip(" «»\""), f"Производитель — {vendor}"))
    if price := _parameter(parameters, "price_to"):
        maximum = _positive_number(price, "Укажите положительную максимальную цену в рублях.")
        result.append(Condition("price_rub", "le", maximum, f"До {price} ₽"))
    if platform := _parameter(parameters, "platform"):
        normalized = " ".join(query_key(platform).split())
        if normalized not in _PLATFORMS:
            raise _BadRequest("Выберите поддерживаемый фильтр ОС.")
        result.append(Condition("platform", "eq", normalized, f"ОС — {platform}"))
    if devices := _parameter(parameters, "devices"):
        minimum = _positive_number(devices, "Укажите положительное целое число рабочих мест.", integer=True)
        result.append(Condition("devices", "ge", minimum, f"Рабочих мест от {devices}"))
    return tuple(result)


def _product(offer: Offer) -> dict:
    return {
        "id": offer.offer_id,
        "name": offer.name or offer.offer_id,
        "article": offer.vendor_code,
        "price": offer.price,
        "currency": offer.currency,
        "url": offer.url,
    }


def _search_products(
    index: SearchIndex, catalog: Path, mode: SearchMode, query: str, filters: tuple[Condition, ...]
) -> dict:
    result = index.search_checked(query, catalog, mode=mode, filters=filters)
    products = []
    for hit in result.hits + result.reserve:
        missing = result.unconfirmed.get(hit.offer.offer_id, ())
        products.append({
            **_product(hit.offer),
            "unconfirmed": missing,
            "can_recommend": not missing,
        })
    return {
        "normalized": result.query.normalized,
        "conditions": list(dict.fromkeys(condition.source for condition in (*result.query.conditions, *filters))),
        "ambiguities": result.query.ambiguities,
        "products": products,
    }


def serve_demo(index: SearchIndex, catalog: Path, mode: SearchMode, port: int) -> None:
    from .recommendation import RecommendationAgent

    if not catalog.is_file():
        raise FileNotFoundError(f"Каталог {catalog} не найден.")
    if not 0 <= port <= 65535:
        raise ValueError("Номер порта должен быть от 0 до 65535.")

    offer_ids = {offer.offer_id for offer in index.offers}

    class DemoHandler(BaseHTTPRequestHandler):
        def _send(self, data: bytes, content_type: str, status: HTTPStatus) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            try:
                url = urlsplit(self.path)
                if url.path in _ASSETS:
                    name, content_type = _ASSETS[url.path]
                    self._send((_ASSET_ROOT / name).read_bytes(), content_type, HTTPStatus.OK)
                elif url.path == "/api/filters":
                    self._json(filter_options)
                elif url.path in {"/api/search", "/api/recommend"}:
                    parameters = parse_qs(url.query, keep_blank_values=True)
                    query = _parameter(parameters, "q")
                    filters = _filters(parameters, categories, vendors)
                    if url.path == "/api/search":
                        if not query:
                            raise _BadRequest("Введите поисковый запрос.")
                        self._json(_search_products(index, catalog, mode, query, filters))
                        return
                    offer_id = _parameter(parameters, "offer_id")
                    if not offer_id:
                        raise _BadRequest("Укажите товар для подбора.")
                    if offer_id not in offer_ids:
                        self._json({"error": "Товар не найден в каталоге."}, HTTPStatus.NOT_FOUND)
                        return
                    debug = _parameter(parameters, "debug")
                    if debug not in {"", "0", "1"}:
                        raise _BadRequest("Допустимые значения debug — 0 и 1.")
                    parsed = index.parse_query(query, catalog) if query else None
                    try:
                        result = agent.recommend(offer_id, parsed, filters=filters, debug=debug == "1")
                    except KeyError:
                        self._json({"error": "Товар не найден в каталоге."}, HTTPStatus.NOT_FOUND)
                        return
                    response = {
                        "scenario_id": result.scenario_id,
                        "status": result.status,
                        "reason_code": result.reason_code,
                        "missing_context": result.missing_context,
                        "products": [
                            {
                                **_product(item.offer),
                                "price": item.offer.price if item.has_price else "",
                                "available": item.available,
                                "reason": item.reason,
                                "sources": item.sources,
                                "purchase_quantity": item.purchase_quantity,
                                "total_price": str(item.total_price) if item.total_price is not None else None,
                            }
                            for item in result.items
                        ],
                    }
                    if result.diagnostics is not None:
                        response["diagnostics"] = result.diagnostics
                    self._json(response)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except _BadRequest as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except (OSError, ValueError, RuntimeError, sqlite3.Error, ParseError) as error:
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    with HTTPServer(("127.0.0.1", port), DemoHandler) as server:
        print("Подготавливаем каталог для поиска и допродаж.", flush=True)
        query_catalog = index.prepare_query_catalog(catalog)
        categories = {
            offer.category_id: query_catalog.categories.get(offer.category_id) or offer.category_id
            for offer in index.offers
            if offer.category_id
        }
        categories[SOFTWARE_CATEGORY_ID] = "ПО"
        vendors = {offer.vendor for offer in index.offers if offer.vendor}
        filter_options = {
            "categories": [
                {"id": category_id, "name": name, "is_software": query_catalog.is_software_category(category_id)}
                for category_id, name in sorted(
                    categories.items(),
                    key=lambda item: (item[0] != SOFTWARE_CATEGORY_ID, query_key(item[1]), item[0]),
                )
            ],
            "vendors": sorted(vendors, key=lambda vendor: (query_key(vendor), vendor)),
        }
        agent = RecommendationAgent(catalog)
        print(f"Демо открыто на http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
