import argparse
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from xml.etree.ElementTree import ParseError

from .catalog import Offer
from .index import DEMO_RESULT_COUNT, MODES, RESULT_LIMIT, SearchHit, SearchIndex, build_index
from .recommendation import recommend_batteries


_DEFAULT_CATALOG = Path("Mindbox.yml.gz.xml")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Гибридный поиск по каталогу Softline.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index", type=Path, default=Path(".search-index.sqlite3"), help="Файл индекса")
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Устройство E5")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-index", parents=[common], help="Построить индекс XML-каталога")
    build.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="Исходный XML")
    build.add_argument("--batch-size", type=int, default=64, help="Размер пакета при кодировании E5")

    search = commands.add_parser("search", parents=[common], help="Вывести до четырёх товарных семейств")
    search.add_argument("query", help="Текст запроса")
    search.add_argument("--mode", choices=MODES, default="hybrid", help="Способ поиска")
    search.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="XML для условий запроса")
    search.add_argument("--raw", action="store_true", help="Исходный поиск без разбора запроса")

    recommend = commands.add_parser("recommend", help="Подобрать подтверждённую сменную батарею для ИБП APC")
    recommend.add_argument("offer", help="ID предложения или точный URL из XML-каталога")
    recommend.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="Исходный XML")

    demo = commands.add_parser("demo", parents=[common], help="Запустить локальную страницу поиска и допродажи")
    demo.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="Исходный XML")
    demo.add_argument("--mode", choices=MODES, default="hybrid", help="Способ поиска")
    demo.add_argument("--port", type=int, default=8765, help="Порт локальной страницы")

    parse = commands.add_parser("parse-query", parents=[common], help="Показать нормализацию и условия запроса")
    parse.add_argument("query", help="Текст запроса")
    parse.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="XML для словаря запроса")

    evaluation = commands.add_parser("evaluate", parents=[common], help="Сравнить методы на Excel-выборке")
    evaluation.add_argument(
        "--queries", type=Path, default=Path("Выборка поисковых запросов.xlsx"), help="Файл Excel"
    )
    evaluation.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG, help="XML для условий запроса")
    evaluation.add_argument("--raw", action="store_true", help="Оценить исходный поиск без разбора запроса")
    return parser


def _print_offer(offer: Offer, number: int, *, has_price: bool) -> None:
    print(f"{number:>2}. {offer.name}")
    if has_price:
        currency = "₽" if offer.currency in {"RUR", "RUB"} else offer.currency
        print(f"    Цена {offer.price} {currency}".rstrip())
    else:
        print("    Цена не указана в фиде")
    if offer.url:
        print(f"    {offer.url}")


def _print_hits(
    hits: list[SearchHit], unconfirmed: dict[str, tuple[str, ...]] | None = None, *, start: int = 1
) -> None:
    if not hits:
        print("По запросу нет совпадений.")
    for number, hit in enumerate(hits, start=start):
        offer = hit.offer
        _print_offer(offer, number, has_price=bool(offer.price) and offer.price not in {"0", "0.0", "0.00"})
        missing = (unconfirmed or {}).get(offer.offer_id, ())
        if missing:
            print(f"    Не подтверждены условия — {', '.join(missing)}")


def _print_recommendations(catalog_path: Path, selected: str) -> None:
    source, recommendations = recommend_batteries(catalog_path, selected)
    print(f"Выбранный товар — {source.name or source.offer_id}")
    if not recommendations:
        print("Подтверждённых сопутствующих товаров по текущему правилу нет.")
        return
    print("\nСопутствующие товары")
    for number, item in enumerate(recommendations, start=1):
        _print_offer(item.offer, number, has_price=item.has_price)
        if not item.available:
            print("    Недоступен по данным фида")
        print(
            f"    Совместимость — в фиде для ИБП указана батарея {item.battery_reference}, "
            f"артикул товара {item.offer.vendor_code}."
        )


def _print_evaluation(index: SearchIndex, workbook: Path, catalog_path: Path | None) -> None:
    from .evaluation import evaluate, metrics

    rows = evaluate(index, workbook, catalog_path=catalog_path)
    summary = metrics(rows)
    cutoff = RESULT_LIMIT if catalog_path is None else DEMO_RESULT_COUNT
    multiple_labels = f"Recall@{cutoff}" in summary.get("hybrid", {})
    print(f"Сопоставлено запросов с эталонами {sum(row.mapped for row in rows)} из {len(rows)}.")
    for mode, scores in summary.items():
        print(f"{mode:>6}  " + "  ".join(f"{name}={score:.3f}" for name, score in scores.items()))
    positions = "Позиции эталонов" if multiple_labels else "Позиция первого эталона"
    print(f"\n{positions} в топ-{cutoff}")
    for row in rows:
        if not row.mapped:
            print(f"{row.query}  |  эталон не сопоставлен ({row.target_url})")
            continue
        ranks = "  ".join(
            f"{mode}={','.join(map(str, positions if multiple_labels else positions[:1])) or '—'}"
            for mode, positions in row.hit_positions.items()
        )
        print(f"{row.query}  |  {ranks}")
        if row.first_hybrid_result:
            print(f"  Первый результат гибрида — {row.first_hybrid_result}")


def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    try:
        if args.command == "build-index":
            count = build_index(
                args.catalog,
                args.index,
                device=args.device,
                batch_size=args.batch_size,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            print(f"Индекс {args.index} готов. Предложений {count:,}.")
            return
        if args.command == "recommend":
            _print_recommendations(args.catalog, args.offer)
            return
        with closing(SearchIndex(args.index, args.device)) as index:
            if args.command == "search":
                if args.raw:
                    _print_hits(index.search(args.query, mode=args.mode))
                else:
                    result = index.search_checked(args.query, args.catalog, mode=args.mode)
                    if result.query.normalized != args.query:
                        print(f"Нормализованный запрос — {result.query.normalized}")
                    for ambiguity in result.query.ambiguities:
                        print(f"Уточнение — {ambiguity}")
                    _print_hits(result.hits)
                    if result.reserve:
                        print("\nРезервные предложения с неподтверждёнными условиями")
                        _print_hits(result.reserve, result.unconfirmed, start=len(result.hits) + 1)
                    if result.analogues:
                        print("\nАналоги с другим артикулом")
                        for number, item in enumerate(result.analogues, start=1):
                            price = item.offer.price
                            _print_offer(item.offer, number, has_price=bool(price) and price not in {"0", "0.0", "0.00"})
                            print(f"    {item.reason}")
            elif args.command == "demo":
                from .demo import serve_demo

                serve_demo(index, args.catalog, args.mode, args.port)
            elif args.command == "parse-query":
                parsed = index.parse_query(args.query, args.catalog)
                print(json.dumps(asdict(parsed), ensure_ascii=False, indent=2, default=str))
            else:
                _print_evaluation(index, args.queries, None if args.raw else args.catalog)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, ParseError) as error:
        parser.exit(1, f"Ошибка. {error}\n")


if __name__ == "__main__":
    main()
