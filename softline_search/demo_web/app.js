const byId = (id) => document.getElementById(id);
const form = byId("search-form");
const input = byId("query");
const submit = byId("search-submit");
const filters = byId("search-filters");
const filterFields = filters.querySelectorAll("input[name], select[name]");
const filterStatus = byId("filter-status");
const categoryValue = byId("category-value");
const softwareFilters = byId("software-filters");
const results = byId("results");
const searchStatus = byId("search-status");
const workspace = byId("workspace");
const queryDetails = byId("query-details");
const recommendationPanel = byId("recommendation-panel");
const recommendationTitle = byId("recommendation-title");
const recommendationSource = byId("recommendation-source");
const recommendationMode = byId("recommendation-mode");
const recommendationStatus = byId("recommendation-status");
const recommendationContext = byId("recommendation-context");
const recommendations = byId("recommendations");
const agentTrace = byId("agent-trace");
const agentTools = byId("agent-tools");
const suggestionList = byId("search-suggestions");
const formatNumber = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 6 });
const pluralRules = new Intl.PluralRules("ru-RU");

let searchRequest;
let recommendationRequest;
let suggestionTimer;
let suggestionRequest;
let selectedSuggestion = -1;
let filterOptionsRequested = false;
let directionGroupCounter = 0;
const softwareCategoryIds = new Set(["software"]);
const productCountLabels = { one: "товар", few: "товара", many: "товаров" };

function element(tag, className, text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function showStatus(node, text, error = false) {
  node.textContent = text;
  node.hidden = !text;
  node.dataset.error = String(error);
}

async function getJson(path, params = {}, signal) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(query ? `${path}?${query}` : path, { signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload) {
    throw new Error(payload?.error || "Попробуйте ещё раз.");
  }
  return payload;
}

function filterKey(text) {
  return text.normalize("NFKC").toLocaleLowerCase("ru-RU").replaceAll("ё", "е").trim();
}

class FilterPicker {
  constructor(input, menu, selection, options = []) {
    this.input = input;
    this.menu = menu;
    this.selection = selection;
    this.options = new Map();
    this.addOptions([{ value: "", label: input.placeholder }, ...options]);

    input.addEventListener("focus", () => this.open());
    input.addEventListener("click", () => {
      if (menu.hidden) this.open();
    });
    input.addEventListener("input", () => {
      this.setValue("");
      this.open(input.value);
    });
    input.addEventListener("keydown", (event) => this.handleKey(event));
    input.addEventListener("blur", () => {
      const typed = filterKey(input.value);
      const exact = typed && [...this.options.values()].filter(({ key }) => key === typed);
      this.choose(exact?.length === 1 ? exact[0].value : selection.value);
    });
    menu.addEventListener("mousedown", (event) => {
      if (event.target.closest("[data-index]")) event.preventDefault();
    });
    menu.addEventListener("click", (event) => {
      const option = event.target.closest("[data-index]");
      if (option) this.choose(this.matches[Number(option.dataset.index)].value);
    });
  }

  addOptions(options) {
    for (const { value, label } of options) {
      if (!this.options.has(value)) this.options.set(value, { value, label, key: filterKey(label) });
    }
    if (!this.menu.hidden) this.open(this.selection.value ? "" : this.input.value);
  }

  setValue(value) {
    if (this.selection.value === value) return;
    this.selection.value = value;
    this.selection.dispatchEvent(new Event("change"));
  }

  choose(value) {
    this.input.value = value ? this.options.get(value).label : "";
    this.setValue(value);
    this.close();
  }

  open(text = this.selection.value ? "" : this.input.value) {
    const query = filterKey(text);
    this.matches = [...this.options.values()].filter(({ key }) => key.includes(query));
    if (query) this.matches.sort((left, right) => Number(right.key.startsWith(query)) - Number(left.key.startsWith(query)));

    const options = this.matches.map(({ label }, index) => {
      const option = element("li", "", label);
      option.id = `${this.menu.id}-${index}`;
      option.dataset.index = index;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      return option;
    });
    if (!options.length) {
      const empty = element("li", "", "Нет вариантов");
      empty.setAttribute("role", "option");
      empty.setAttribute("aria-disabled", "true");
      options.push(empty);
    }
    this.menu.replaceChildren(...options);
    this.menu.hidden = false;
    this.menu.scrollTop = 0;
    this.input.setAttribute("aria-expanded", "true");
    this.highlight(this.matches.length ? Math.max(0, this.matches.findIndex(({ value }) => value === this.selection.value)) : -1);
  }

  highlight(index) {
    this.menu.querySelector('[aria-selected="true"]')?.setAttribute("aria-selected", "false");
    this.active = index;
    if (index < 0) {
      this.input.removeAttribute("aria-activedescendant");
      return;
    }
    const option = this.menu.children[index];
    option.setAttribute("aria-selected", "true");
    this.input.setAttribute("aria-activedescendant", option.id);
    const offset = option.offsetTop;
    if (offset < this.menu.scrollTop) this.menu.scrollTop = offset;
    const bottom = offset + option.offsetHeight;
    if (bottom > this.menu.scrollTop + this.menu.clientHeight) this.menu.scrollTop = bottom - this.menu.clientHeight;
  }

  close() {
    this.menu.hidden = true;
    this.input.setAttribute("aria-expanded", "false");
    this.input.removeAttribute("aria-activedescendant");
  }

  handleKey(event) {
    if (event.isComposing) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (this.menu.hidden) this.open();
      else if (this.matches.length) {
        const direction = event.key === "ArrowDown" ? 1 : -1;
        this.highlight(Math.max(0, Math.min(this.active + direction, this.matches.length - 1)));
      }
    } else if (!this.menu.hidden && event.key === "Enter") {
      event.preventDefault();
      if (this.active >= 0) this.choose(this.matches[this.active].value);
    } else if (!this.menu.hidden && event.key === "Escape") {
      event.preventDefault();
      this.choose(this.selection.value);
    }
  }
}

const categoryPicker = new FilterPicker(byId("category-filter"), byId("category-options"), categoryValue,
  [{ value: "software", label: "ПО" }]);
const vendorPicker = new FilterPicker(byId("vendor-filter"), byId("vendor-options"), byId("vendor-value"));

async function loadFilterOptions() {
  if (filterOptionsRequested) return;
  filterOptionsRequested = true;
  showStatus(filterStatus, "Загружаем категории и производителей…");
  try {
    const { categories, vendors } = await getJson("/api/filters");
    const nameCounts = new Map();
    for (const { id, name, is_software } of categories) {
      nameCounts.set(name, (nameCounts.get(name) ?? 0) + 1);
      if (is_software) softwareCategoryIds.add(id);
    }
    categoryPicker.addOptions(categories.map(({ id, name }) => ({
      value: id, label: nameCounts.get(name) > 1 ? `${name} (${id})` : name,
    })));
    vendorPicker.addOptions(vendors.map((vendor) => ({ value: vendor, label: vendor })));
    updateSoftwareFilters();
    showStatus(filterStatus, "");
  } catch (error) {
    filterOptionsRequested = false;
    showStatus(filterStatus, `Не удалось загрузить категории и производителей. ${error.message}`, true);
  }
}

function updateSoftwareFilters() {
  const show = !categoryValue.value || softwareCategoryIds.has(categoryValue.value);
  softwareFilters.hidden = !show;
  softwareFilters.disabled = !show;
  if (!show) {
    for (const field of softwareFilters.querySelectorAll("input, select")) field.value = "";
  }
}

function catalogLink(source) {
  let url;
  try {
    url = new URL(source);
  } catch {
    return null;
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") return null;
  return externalLink(url, "Открыть в каталоге ↗");
}

function externalLink(url, text) {
  const link = element("a", "offer-link", text);
  link.href = url.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function documentationLinks(sources) {
  const list = element("ul", "spec-sources");
  for (const source of new Set(sources ?? [])) {
    let url;
    try {
      url = new URL(source);
    } catch {
      continue;
    }
    if (url.protocol !== "https:" || url.username || url.password) continue;
    const item = element("li", "");
    item.append(externalLink(url, "Документация · " + url.hostname));
    list.append(item);
  }
  return list;
}

function directionLabel(product) {
  const label = product?.relationship ?? product?.direction_label;
  return typeof label === "string" ? label.trim() : "";
}

function directionGroups(products, directions = []) {
  const items = Array.isArray(products) ? products : [];
  const labels = items.map(directionLabel);
  if (items.length < 2 || labels.some((label) => !label)) return [];

  const groups = new Map();
  for (const [index, product] of items.entries()) {
    const label = labels[index];
    const key = product.direction_id || label;
    let group = groups.get(key);
    if (!group) {
      group = { label, products: [] };
      groups.set(key, group);
    }
    group.products.push(product);
  }
  if (groups.size < 2) return [];

  const ordered = [];
  const seen = new Set();
  for (const direction of Array.isArray(directions) ? directions : []) {
    const label = typeof direction?.label === "string" ? direction.label.trim() : "";
    const key = groups.has(direction?.id) ? direction.id : label;
    if (seen.has(key) || !groups.has(key)) continue;
    ordered.push(groups.get(key));
    seen.add(key);
  }
  for (const [key, group] of groups) {
    if (!seen.has(key)) ordered.push(group);
  }
  return ordered;
}

function productCountLabel(count) {
  return count + " " + productCountLabels[pluralRules.select(count)];
}

const criterionLabels = {
  anchor_dimensions: "габариты исходного товара",
  candidate_dimensions: "габариты дополнения",
  laptop_dimensions: "габариты ноутбука",
  bag_dimensions: "внутренние габариты сумки",
  battery_articles: "артикул батареи",
  required_article: "артикул комплектующей",
  candidate_article: "артикул предложения",
  interface: "интерфейс подключения",
  ports: "порты подключения",
  platform: "операционная система",
  platform_version: "версия операционной системы",
  capacity: "ёмкость",
  required_capacity: "требуемая ёмкость",
  minimum_purchase: "минимальный объём покупки",
  term_months: "срок лицензии",
  usb_c_pd: "поддержка USB-C Power Delivery",
  usb_c_video: "поддержка передачи видео по USB-C",
};

function productCard(product, searchSnapshot = null, { showDirection = false, titleTag = "h3" } = {}) {
  const card = element("article", "product-card");
  const badge = showDirection ? directionLabel(product) : "";
  if (badge) {
    const header = element("div", "product-header");
    header.append(element(titleTag, "product-title", product.name));
    header.append(element("span", "direction-badge", badge));
    card.append(header);
  } else {
    card.append(element(titleTag, "product-title", product.name));
  }
  if (product.article) {
    card.append(element("p", "product-meta", `Артикул ${product.article}`));
  }

  const price = Number(String(product.price ?? "").replace(/\s/g, "").replace(",", "."));
  const currency = ["RUR", "RUB"].includes(product.currency) ? "₽" : product.currency;
  const knownPrice = Number.isFinite(price) && price > 0;
  card.append(element(
    "p", knownPrice ? "product-price" : "product-price price-unknown",
    knownPrice ? `${formatNumber.format(price)} ${currency}`.trim() : "Цена не указана в фиде",
  ));
  if (product.purchase_quantity > 1 && product.total_price !== null) {
    card.append(element("p", "product-meta",
      "Итого за " + product.purchase_quantity + " — " + formatNumber.format(Number(product.total_price)) + " " + currency,
    ));
  }

  if (product.unconfirmed?.length) {
    card.append(element("p", "warning", `Не подтверждены условия — ${product.unconfirmed.join(", ")}`));
  }
  if (product.available === false) {
    card.append(element("p", "warning", "Недоступен по данным фида"));
  } else if (product.available === true) {
    card.append(element("p", "product-meta", "В наличии по данным фида"));
  }
  if (product.alternative) {
    card.append(element("p", "status", product.alternative));
  }
  if (product.verification_status === "CONFIRMED") {
    card.append(element("p", "product-meta", "Совместимость подтверждена"));
  } else if (product.verification_status === "UNKNOWN") {
    const state = product.relationship ? "Совместимость требует сверки" : "Проверено частично";
    card.append(element("p", "warning", state));
    const criteria = (product.missing_criteria ?? [])
      .filter((field) => field !== "compatibility")
      .map((field) => criterionLabels[field]).filter(Boolean);
    if (criteria.length) card.append(element("p", "product-meta", "Нужно сверить — " + criteria.join(", ")));
  }
  if (product.reason) {
    card.append(element("p", "status compatibility", product.reason));
  }
  const sources = documentationLinks(product.sources);
  if (sources.childElementCount) card.append(sources);

  const actions = element("div", "product-actions");
  const link = catalogLink(product.url);
  if (link) actions.append(link);
  for (const [enabled, action, label] of [
    [product.can_recommend, "recommend", "Подходит к этому товару"],
    [product.can_compare, "similar", "Похожие товары"],
  ]) {
    if (!enabled || !searchSnapshot) continue;
    const button = element("button", "recommend-button", label);
    button.type = "button";
    button.setAttribute("aria-controls", "recommendation-panel");
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => loadProductOptions(product, button, action, searchSnapshot));
    actions.append(button);
  }
  if (actions.childElementCount) card.append(actions);
  return card;
}

function renderRecommendations(products, directions) {
  const items = Array.isArray(products) ? products : [];
  const groups = directionGroups(items, directions);
  if (!groups.length) {
    recommendations.replaceChildren(...items.map((item) => productCard(item, null, { showDirection: true })));
    return;
  }

  recommendations.replaceChildren(...groups.map((group) => {
    const headingId = "direction-group-" + (++directionGroupCounter);
    const section = element("section", "direction-group");
    section.setAttribute("aria-labelledby", headingId);
    const heading = element("div", "direction-heading");
    const title = element("h3", "direction-title", group.label);
    title.id = headingId;
    heading.append(title, element("span", "direction-count", productCountLabel(group.products.length)));
    const list = element("div", "product-list");
    list.append(...group.products.map((item) => productCard(item, null, { titleTag: "h4" })));
    section.append(heading, list);
    return section;
  }));
}

function renderQueryDetails(data, query) {
  const changed = data.normalized !== query;
  byId("normalized-line").hidden = !changed;
  byId("normalized-query").textContent = data.normalized;
  byId("conditions-line").hidden = data.conditions.length === 0;
  byId("conditions").replaceChildren(...data.conditions.map((text) => element("li", "", text)));
  showStatus(byId("ambiguities"), data.ambiguities.join(" · "));
  queryDetails.hidden = !changed && !data.conditions.length && !data.ambiguities.length;
}

function renderResults(data, searchSnapshot) {
  const count = data.products.length;
  if (!count) {
    showStatus(searchStatus, "Подходящих товаров по этому запросу нет.");
    return;
  }

  const familyLabels = { one: "семейство", few: "семейства", many: "семейств" };
  const confirmed = data.confirmed_count ?? count;
  const reserve = count - confirmed;
  const analogues = data.analogues ?? [];
  byId("result-count").textContent = confirmed + " " + familyLabels[pluralRules.select(confirmed)]
    + (reserve ? " · резерв " + reserve : "");
  results.replaceChildren(...data.products.map((product) => productCard(product, searchSnapshot)));
  byId("analogue-results").replaceChildren(...analogues.map((product) => productCard(product)));
  byId("search-analogues").hidden = analogues.length === 0;
  recommendationPanel.hidden = true;
  recommendationTitle.textContent = "Сопутствующие товары";
  recommendationSource.hidden = true;
  recommendations.replaceChildren();
  showStatus(recommendationStatus, "");
  showStatus(recommendationContext, "");
  showStatus(recommendationMode, "");
  renderAgentTrace([]);
  workspace.hidden = false;
  showStatus(searchStatus, "");
}

function currentSearchParams(query) {
  const params = { q: query };
  for (const field of filterFields) {
    if (!field.matches(":disabled") && field.value) params[field.name] = field.value;
  }
  return params;
}

async function loadSearch(value) {
  const query = value.trim();
  if (!query) {
    input.focus();
    showStatus(searchStatus, "Введите поисковый запрос.", true);
    return;
  }

  searchRequest?.abort();
  recommendationRequest?.abort();
  closeSuggestions();
  const request = new AbortController();
  searchRequest = request;
  input.value = query;
  const searchParams = Object.freeze(currentSearchParams(query));
  form.setAttribute("aria-busy", "true");
  submit.textContent = "Ищем…";
  workspace.hidden = true;
  queryDetails.hidden = true;
  showStatus(searchStatus, "Ищем товары в каталоге. Первый поиск может занять больше времени из-за загрузки модели.");

  try {
    const data = await getJson("/api/search", searchParams, request.signal);
    if (request.signal.aborted || searchRequest !== request) return;
    renderQueryDetails(data, query);
    renderResults(data, searchParams);
  } catch (error) {
    if (!request.signal.aborted) showStatus(searchStatus, `Ошибка поиска. ${error.message}`, true);
  } finally {
    if (searchRequest === request) {
      searchRequest = null;
      form.removeAttribute("aria-busy");
      submit.textContent = "Найти";
    }
  }
}

const toolLabels = {
  get_product_details: "Характеристики товара",
  get_catalog_categories: "Категории каталога",
  search_catalog: "Поиск по каталогу",
  catalog_search_retry: "Поиск по другому запросу",
  check_candidates: "Проверка найденных кандидатов",
  validate_v3_candidates: "Проверка итоговых кандидатов",
  get_category_products: "Поиск в категории",
  get_verified_specs: "Проверка сохранённых характеристик",
  search_vendor_specs: "Поиск документации производителя",
  check_compatibility: "Проверка совместимости",
  get_cross_sell_directions: "Направления сопутствующих товаров",
  get_compatibility_requirements: "Требования совместимости",
  check_compatibility_batch: "Пакетная проверка совместимости",
  compare_products: "Сравнение предложений",
  prepare_context: "Подготовка данных товара",
  agent_selection: "Предложения агента для проверки",
  validate_autonomous_candidates: "Проверка предложений агента",
  validate_final: "Проверка итоговой выдачи",
  web_search: "Поиск официального источника",
  rule_based_recommendation: "Подбор по правилам",
  recommendation_cache: "Чтение кеша результата",
  shared_agent_run: "Использование результата параллельного запроса",
  agent_execution: "Выполнение агентного подбора",
};

const outcomeLabels = {
  llm_disabled: "Агент отключён в настройках.",
  missing_api_key: "Ключ провайдера не настроен.",
  no_catalog_search: "Агент не нашёл направление для поиска.",
  no_catalog_matches: "По поисковым запросам нет товаров в каталоге.",
  no_relevant_candidates: "Найденные товары не прошли проверку назначения.",
  model_selected_none: "Агент не отобрал полезных дополнений из найденных позиций.",
  all_candidates_rejected: "Предложенные позиции не прошли проверку.",
  no_compatible_products: "Сопутствующих товаров по этим условиям не найдено.",
  missing_product_facts: "Для проверки совместимости не хватает характеристик товара.",
  search_no_results: "Поиск не нашёл документацию с нужной характеристикой.",
  source_unavailable: "Документ производителя не удалось загрузить.",
  document_extraction_failed: "Из документа не удалось извлечь подтверждённую характеристику.",
  identity_mismatch: "Документ не удалось связать с выбранной моделью или артикулом.",
  conflicting_facts: "Источники дают противоречивые характеристики.",
  provider_timeout: "Истекло время онлайн-подбора.",
  provider_error: "Онлайн-подбор прервался.",
  iteration_limit: "Достигнут лимит шагов агента.",
};

function renderAgentTrace(events) {
  agentTrace.hidden = !events.length;
  agentTrace.open = false;
  agentTools.replaceChildren(...events.map((event) => {
    const name = event.name ?? event.tool;
    const entry = element("li", "");
    entry.append(element("strong", "", toolLabels[name] ?? name ?? "Подбор"));
    const status = ({
      ok: "Выполнено", error: "Ошибка", skipped: "Пропущено", cached: "Из кеша",
      confirmed: "Подтверждено", unknown: "Недостаточно данных",
      rejected: "Отклонено", timeout: "Истекло время", empty: "Результатов нет",
    })[event.status] ?? event.status;
    const elapsed = Number.isFinite(event.elapsed_ms) ? formatNumber.format(Math.round(event.elapsed_ms)) + " мс" : "";
    const count = event.count > 0 ? "Результатов " + formatNumber.format(event.count) : "";
    const detail = ({
      source_unavailable: "Источник недоступен", vendor_source_unavailable: "Источник недоступен",
      no_verifiable_fact: "Подтверждения в документации нет", missing_verified_spec: "Характеристика пока неизвестна",
      provider_timeout: "Время ожидания истекло", web_search_limit: "Достигнут лимит внешнего поиска",
    })[event.error_code] ?? outcomeLabels[event.error_code];
    const description = [status, count, detail, elapsed].filter(Boolean).join(" · ");
    if (description) entry.append(element("p", "product-meta", description));
    if (event.compatibility) {
      const { checked, confirmed, rejected, unknown, errors } = event.compatibility;
      entry.append(element("p", "product-meta", [
        "Проверено " + checked, "подтверждено " + confirmed, "отклонено " + rejected,
        "недостаточно данных " + unknown, "ошибок " + errors,
      ].join(" · ")));
    }
    return entry;
  }));
}

async function loadProductOptions(product, button, action, searchSnapshot) {
  if (searchRequest || !results.contains(button)) return;
  recommendationRequest?.abort();
  const request = new AbortController();
  recommendationRequest = request;
  for (const choice of results.querySelectorAll(".recommend-button")) {
    choice.setAttribute("aria-pressed", String(choice === button));
  }
  const similar = action === "similar";
  recommendationPanel.hidden = false;
  recommendationTitle.textContent = similar ? "Похожие товары" : "Сопутствующие товары";
  recommendationSource.textContent = product.name;
  recommendationSource.hidden = false;
  recommendations.replaceChildren();
  showStatus(recommendationStatus, similar ? "Ищем альтернативы в каталоге…" : "Ищем сопутствующие товары…");
  showStatus(recommendationContext, "");
  showStatus(recommendationMode, "");
  renderAgentTrace([]);
  if (matchMedia("(max-width: 49rem)").matches) {
    recommendationPanel.scrollIntoView({ block: "start" });
  }

  try {
    const data = await getJson("/api/" + action, {
      ...searchSnapshot, offer_id: product.id,
    }, request.signal);
    if (request.signal.aborted || recommendationRequest !== request) return;
    const products = Array.isArray(data.products) ? data.products : [];
    if (similar) {
      recommendations.replaceChildren(...products.map((item) => productCard(item)));
    } else {
      renderRecommendations(products, data.directions ?? []);
    }
    showStatus(recommendationStatus, products.length ? "" : "Не найдено");
    if (similar) return;
    const mode = ({
      autonomous_agent: "Направления выбрал агент",
      single_agent_v3_1: "Single Agent V3.1",
      local: "Локальный подбор по правилам",
      llm_agent: "Агентный подбор",
      rule_based_fallback: "Резервный подбор по правилам",
      cached_result: data.origins?.includes("autonomous_selection")
        ? "Направления выбрал агент" : "Подбор из кеша",
    })[data.mode];
    if (mode) showStatus(recommendationMode, mode + " · " + formatNumber.format(Math.round(data.elapsed_ms)) + " мс");
    renderAgentTrace(data.tools ?? []);
    const missing = (data.missing_context ?? []).map((field) => ({
      platform: "ОС", devices: "число рабочих мест",
    })[field]).filter(Boolean);
    if (missing.length) showStatus(recommendationContext, "Для проверки укажите " + missing.join(" и ") + ".");
    else {
      const note = outcomeLabels[data.outcome_code] ?? outcomeLabels[data.execution_status];
      const source = data.origins?.includes("external_lookup") ? "Выполнены обращения к внешним источникам." : "";
      showStatus(recommendationContext, [note, source].filter(Boolean).join(" "));
    }
  } catch (error) {
    if (!request.signal.aborted) showStatus(recommendationStatus, `Ошибка подбора. ${error.message}`, true);
  } finally {
    if (recommendationRequest === request) recommendationRequest = null;
  }
}

function closeSuggestions() {
  clearTimeout(suggestionTimer);
  suggestionRequest?.abort();
  selectedSuggestion = -1;
  suggestionList.hidden = true;
  suggestionList.replaceChildren();
  input.setAttribute("aria-expanded", "false");
  input.removeAttribute("aria-activedescendant");
}

function chooseSuggestion(item) {
  input.value = item.textContent;
  form.requestSubmit();
}

input.addEventListener("input", () => {
  closeSuggestions();
  const query = input.value.trim();
  if (query.length < 2) return;
  suggestionTimer = setTimeout(async () => {
    const request = new AbortController();
    suggestionRequest = request;
    try {
      const data = await getJson("/api/suggest", { q: query }, request.signal);
      if (request.signal.aborted || input.value.trim() !== query || !data.suggestions.length) return;
      suggestionList.replaceChildren(...data.suggestions.map((label, index) => {
        const item = element("li", "", label);
        item.id = "search-suggestion-" + index;
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", "false");
        return item;
      }));
      suggestionList.hidden = false;
      input.setAttribute("aria-expanded", "true");
    } catch {
      if (!request.signal.aborted) closeSuggestions();
    }
  }, 250);
});

input.addEventListener("keydown", (event) => {
  if (suggestionList.hidden) return;
  if (event.key === "Escape") {
    event.preventDefault();
    closeSuggestions();
  } else if (event.key === "Enter" && selectedSuggestion >= 0) {
    event.preventDefault();
    chooseSuggestion(suggestionList.children[selectedSuggestion]);
  } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    suggestionList.children[selectedSuggestion]?.setAttribute("aria-selected", "false");
    const count = suggestionList.childElementCount;
    const direction = event.key === "ArrowDown" ? 1 : -1;
    selectedSuggestion = selectedSuggestion === -1 ? (direction === 1 ? 0 : count - 1)
      : (selectedSuggestion + direction + count) % count;
    const item = suggestionList.children[selectedSuggestion];
    item.setAttribute("aria-selected", "true");
    input.setAttribute("aria-activedescendant", item.id);
    item.scrollIntoView({ block: "nearest" });
  }
});
input.addEventListener("blur", closeSuggestions);
suggestionList.addEventListener("mousedown", (event) => event.preventDefault());
suggestionList.addEventListener("click", (event) => {
  const item = event.target.closest('[role="option"]');
  if (item) chooseSuggestion(item);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  loadSearch(input.value);
});

form.addEventListener("invalid", (event) => {
  if (filters.contains(event.target)) filters.open = true;
}, true);

filters.addEventListener("toggle", () => {
  if (filters.open) loadFilterOptions();
});

categoryValue.addEventListener("change", updateSoftwareFilters);

byId("reset-filters").addEventListener("click", () => {
  categoryPicker.choose("");
  vendorPicker.choose("");
  for (const field of filterFields) field.value = "";
  updateSoftwareFilters();
});

for (const button of document.querySelectorAll("[data-query]")) {
  button.addEventListener("click", () => {
    input.value = button.dataset.query;
    form.requestSubmit();
  });
}

updateSoftwareFilters();
loadFilterOptions();
