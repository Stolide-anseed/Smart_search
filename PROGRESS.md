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
