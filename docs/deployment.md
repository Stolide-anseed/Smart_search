# Развёртывание на Linux с Docker Compose

Dockerfile и конфигурация Nginx находятся в ветке `feature/deployment`. Конфигурация проверялась для Ubuntu x86_64 с 8 CPU и 16 ГБ RAM. Для запуска нужны Docker Engine, Docker Compose и свободный порт 80. При сборке и первой загрузке модели нужен исходящий HTTPS.

## Подготовка

В отдельном каталоге клонируйте ветку развёртывания.

```bash
git clone --branch feature/deployment https://github.com/Stolide-anseed/Smart_search.git
cd Smart_search
install -m 600 .env.example .env
mkdir -p deployment-data/snapshot deployment-data/state
```

Поместите файлы из одного снимка каталога в указанные каталоги.

| Путь | Содержимое |
| --- | --- |
| `deployment-data/snapshot/Mindbox.yml.gz.xml` | Распакованный XML-каталог |
| `deployment-data/snapshot/.search-index.sqlite3` | Индекс того же XML, совместимый со схемой и закреплённой ревизией E5 |
| `deployment-data/state/.vendor-facts.sqlite3` | Необязательный кеш проверенных характеристик |

```bash
chmod 755 deployment-data/snapshot
chmod 644 deployment-data/snapshot/Mindbox.yml.gz.xml deployment-data/snapshot/.search-index.sqlite3
sudo chown -R 10001:10001 deployment-data/state
sudo chmod 750 deployment-data/state
```

Процесс приложения работает под UID/GID `10001:10001`. Каталог состояния должен быть доступен ему на запись. Снимок монтируется только для чтения. Другие пути задаются через `SMART_SEARCH_SNAPSHOT_DIR` и `SMART_SEARCH_STATE_DIR`.

Для допродаж укажите в `.env` значения `OPENAI_API_KEY`, `OPENAI_BASE_URL` и `LLM_MODEL` для провайдера с поддержкой Responses API. В шаблоне указано `AGENT_MAX_CONCURRENT_RUNS=4`. Для стенда задайте `2`, чтобы согласовать предел приложения с Nginx. `LLM_ENABLED=false` оставляет доступным поиск.

## Запуск и проверка

```bash
docker compose config --quiet
docker compose up -d --build --wait --wait-timeout 1200
docker compose ps
curl -fsS http://127.0.0.1/healthz
curl -fsSG --data-urlencode 'q=7MD68A' http://127.0.0.1/api/search
```

Приложение использует Python 3.13, CPU-сборку PyTorch и V3.2. Первый запуск загружает около 2,2 ГБ весов E5-large. Именованный том `hf-cache` сохраняет их между запусками. После первого прогрева можно включить `HF_HUB_OFFLINE=1`.

`/healthz` проверяет готовность приложения после прогрева E5. Для проверки подключения LLM откройте сайт и запросите допродажу с карточки. При ошибке запуска выполните `docker compose logs --tail=50 app nginx`.

## Эксплуатация

- Лимиты приложения составляют 7 CPU и 12 ГБ памяти. У Nginx — 0,5 CPU и 128 МБ. При замене контейнера поиск недоступен до повторного прогрева.
- Nginx ограничивает обычное API до 4 запросов в секунду с IP и 24 на стенд. Допродажи ограничены до 6 запросов в минуту с IP, 20 на стенд и двух одновременных запросов. При превышении лимита возвращается HTTP 429. Допускаются короткие всплески.
- Публичный режим отключает подробную диагностику через `debug=1`. Сайт работает по HTTP без аутентификации.
- Перед обновлением сохраните версию приложения, пару XML и индекса, а также согласованную копию базы характеристик. Остановите приложение перед заменой открытых файлов SQLite. Для остановки с сохранением модели используйте `docker compose down`.

[Ветка с конфигурацией](https://github.com/Stolide-anseed/Smart_search/tree/feature/deployment)
