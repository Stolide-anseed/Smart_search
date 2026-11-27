# PROGRESS — Universal Cross-Sell Agent

Ветка: `OpenCode/Cross-SellAgent`.

## Baseline

- Дата старта: 2026-11-27.
- Все 77 существующих тестов проходили до изменений.
- Реальный каталог: `Mindbox.yml.gz.xml` (292 МБ, 9 311 категорий, 136 040 предложений, 49 ноутбуков).
- Baseline-метрики сохраняются в `outputs/softline-fix/baseline-review.json` и `outputs/softline-fix/catalog-coverage.json`.

## Контракты (согласованы до реализации)

### Направления

Единый реестр `softline_search/cross_sell.py`:

- legacy: `ups_battery`, `laptop_bag`, `office_antivirus`;
- P0: `laptop_mouse`, `laptop_keyboard`, `laptop_keyboard_mouse`, `laptop_headset`,
  `laptop_monitor`, `laptop_usb_hub`.

`Direction` содержит: `id`, `label`, `candidate_kind`, `category_ids`, `anchor_kinds`,
`validator`, `requirements`, `query_terms`.

### Валидаторы

`InterfaceCompatibility`, `ProtocolCompatibility`, `DimensionCompatibility`,
`PlatformCompatibility`, `ArticleCompatibility`, `CapacityCompatibility`,
`PowerCompatibility`, `LicenseCompatibility`.

Каждый возвращает только `Validation(Status.CONFIRMED | UNKNOWN | REJECTED, reason_code, ...)`.
Отсутствующая характеристика никогда не даёт `CONFIRMED`.

### `Recommendation`

Добавлены поля с defaults: `direction_id: str = ""`, `direction_label: str = ""`.
`RecommendationResult` добавляет `directions: tuple[str, ...] = ()`.
`scenario_id` остаётся для обратной совместимости и равен первому направлению.

### `RecommendationAgent`

- `recommend(..., directions: tuple[str, ...] | None = None)`.
- `directions=None` — автоопределение по запросу; без периферийных слов ноутбук
  по-прежнему ведёт в `laptop_bag`.
- `applicable_directions(offer_id, query=None, filters=())`.
- `scenario_candidates` и `check_compatibility` принимают как legacy scenario ID,
  так и universal direction ID.
- `compatibility_requirements(offer_id, direction_id)` — требования, порты якоря,
  недостающие атрибуты, число кандидатов.
- `check_compatibility_batch(offer_id, direction_id, candidate_ids)`.
- `cross_sell_directions(offer_id)` — доступные направления и их полнота.

### `LLMRecommendationAgent`

- `_SCENARIOS` расширен до всех direction ID; поле финального JSON остаётся
  `scenario_id` для обратной совместимости.
- Новые tools: `get_cross_sell_directions`, `get_compatibility_requirements`,
  `check_compatibility_batch`.
- LLM запускается при наличии enrichment targets, явном периферийном запросе,
  generic-запросе про аксессуары или пустой локальной выдаче при доступных направлениях.
- Сервер независимо перепроверяет все выбранные direction/candidate пары.
- При падении LLM/API подтверждённые локальные результаты сохраняются.

### API/UI

`/api/recommend` дополнительно отдаёт:

- `directions`: `[{id, label, count}]`;
- у товара: `direction_id`, `direction_label`, `reason`, `sources`.

«Подходит к этому товару» и «Похожие товары» остаются раздельными.
Журнал действий показывает реальные tool calls, включая batch-проверку.

## Этапы

- [x] Baseline: 77 тестов.
- [x] Анализ XML и официальных источников.
- [ ] `cross_sell.py`: реестр направлений, нормализация, валидаторы.
- [ ] Интеграция в `RecommendationAgent`.
- [ ] Интеграция в `LLMRecommendationAgent`.
- [ ] API/UI.
- [ ] Тесты.
- [ ] Реальные проверки, метрики, отчёт.

## Сравнение веток

- `main` и `OpenCode/Cross-SellAgent` указывают на `109a7d9`. После переключения на `main` доработки Cross-Sell находятся в единственной рабочей копии репозитория. Отдельного снимка незакоммиченных изменений ветки `OpenCode/Cross-SellAgent` в Git нет.

## Подготовка коммита

- Локальный запуск `unittest discover -s tests` перед коммитом прошёл все 107 тестов.
- Для коммита выбрана ветка `OpenCode/Cross-SellAgent`.


## Codex/Autonomus-agent

- Создана отдельная рабочая копия от b40a801. Исходная ветка OpenCode/Cross-SellAgent служит контрольной.
- Подтверждён исходный результат. Пройдены все 107 тестов.
- Задача первой версии — открытый выбор направлений допродажи моделью. Модель вызывает поиск каталога и уточняет результаты. Перед выдачей сервер выполняет частичную проверку каждого предложения.
- Реализация разделена между субагентами. Проверку изменений, итоговую интеграцию и реальные прогоны выполняет основной агент.
- Выполнено fast-forward слияние OpenCode/Cross-SellAgent в main. Обе ветки указывают на b40a801.
- Интегрированы открытые направления, частичная проверка и API top-k. В интерфейсе различаются подтверждённые и частично проверенные позиции.
- В ревью закрыты случаи чужого расходника, конфликта проверок и нескольких приобретаемых лицензий. Исходные и новые тесты прошли вместе. Всего 122.
- Живой HTTP-прогон с GPT-6 Luna через ProxyAPI. Для HP LaserJet M111w (423674) модель выбрала «Картриджи для печати», вызвала get_product_details, search_catalog и check_compatibility_batch. Сервер повторно проверил результат и вернул NVP NV-W1500X (530966) со статусом UNKNOWN. Подбор занял 7,11 с, потребовал четыре обращения к GPT и завершился без fallback.
- Демо автономной ветки доступно на локальном порту 8880. Пример запроса для проверки — 7MD68A.

## Single Agent V3.1

- Baseline автономной ветки зафиксирован на 11c4581. Исходный набор из 122 тестов прошёл.
- За основу взят существующий цикл вызова инструментов. Контекст, правила выбора и контракты V3.1 выделяются отдельно.
- Реализованы открытые связи, повторный поиск и проверка кандидатов каталога без прежней маршрутизации. Демо переключается между V3.1 и baseline. Целевые проверки V3.1 прошли.
- Первый живой прогон подтвердил 6/7 покрытых товаров у обеих версий. V3.1 выполнила пять повторных поисков и вернула дополнение для чекового принтера. Для Deli ET500 модель завершила подбор пустым результатом после поиска. Добавлен контекст категории и назначения, запущена повторная проверка V3.1.

### Итоги V3.1

В финальном живом прогоне сравнили обе версии на семи товарах из одного XML-снимка, с одним индексом и GPT-6 Luna. В выборке ноутбук, денежный ящик, DDR4, лазерный и чековый принтеры, ламинатор и ИБП. Результаты и разметка сохранены в [comparison.json](outputs/softline-v3/comparison.json).

| Показатель | Baseline 11c4581 | V3.1 |
|---|---:|---:|
| Товары с допродажами | 6/7 | 6/7 |
| Позиций в выдаче | 13 | 10 |
| Релевантные среди оценённых вручную | 12/12 | 9/10 |
| Задержка p50 | 7,09 с | 5,83 с |
| Задержка p95 | 9,23 с | 7,40 с |
| Вызовы GPT | 35 | 30 |
| Повторные поиски по альтернативному запросу | 0 | 2 |

- Для Deli ET500 V3.1 нашла два чековых принтера как оборудование кассового рабочего места. Для MERTECH G58 обе версии завершили подбор без допродаж. В V3.1 модель вернула пустой список после трёх поисков, код причины `model_selected_none`.
- Для ИБП V3.1 предложила PDU с неподходящим входом питания со статусом `UNKNOWN`. Baseline вернула батарею APCRBC141. Этот случай показывает ограничение частичной проверки при неизвестной модели связи.
- Для одного предложения baseline по ламинатору в проверенных полях не подтверждена толщина плёнки. Пара исключена из ручной оценки. Результаты отражают один прогон на небольшой выборке и не измеряют техническую совместимость.
- Финальные проверки прошли. 129 Python-тестов и проверка синтаксиса JavaScript.

### Изменённые файлы

- `softline_search/llm_agent_v3.py` реализует открытый выбор отношений и выдачу V3.1.
- `softline_search/llm_agent.py` и `softline_search/autonomous_validation.py` предоставляют общий цикл и частичные проверки.
- `softline_search/__main__.py`, `softline_search/demo.py` и `softline_search/demo_web/app.js` подключают новую версию к CLI, HTTP и интерфейсу.
- `tests/test_llm_agent_v3.py` и `tests/test_demo_agent_api.py` проверяют поиск, итоговую выдачу и обе версии HTTP API.
- `evaluate_single_agent_v3.py` и `outputs/softline-v3/comparison.json` содержат воспроизводимый замер и его результаты.
- `README.md` и `PROGRESS.md` описывают запуск и фиксируют состояние работы.

### Prompt update

- Заменены инструкции агента V3.1 точным текстом, предоставленным пользователем. Перед переносом ветки проверяются целевые тесты агента и API.

- Целевые тесты V3.1 после замены промта: 8 тестов и 15 subtests пройдены. Для запуска добавлен pytest только в общий .venv.

