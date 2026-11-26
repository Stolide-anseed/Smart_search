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
const recommendationSource = byId("recommendation-source");
const recommendationStatus = byId("recommendation-status");
const recommendationContext = byId("recommendation-context");
const recommendations = byId("recommendations");
const formatNumber = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 6 });
const pluralRules = new Intl.PluralRules("ru-RU");

let searchRequest;
let recommendationRequest;
let filterOptionsRequested = false;
const softwareCategoryIds = new Set(["software"]);

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
  const link = element("a", "offer-link", "Открыть в каталоге ↗");
  link.href = url.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function productCard(product) {
  const card = element("article", "product-card");
  card.append(element("h3", "product-title", product.name));
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
  if (product.reason) {
    card.append(element("p", "status compatibility", product.reason));
  }

  const actions = element("div", "product-actions");
  const link = catalogLink(product.url);
  if (link) actions.append(link);
  if (product.can_recommend) {
    const button = element("button", "recommend-button", "Подходит к этому товару");
    button.type = "button";
    button.setAttribute("aria-controls", "recommendation-panel");
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => loadRecommendations(product, button));
    actions.append(button);
  }
  if (actions.childElementCount) card.append(actions);
  return card;
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

function renderResults(data) {
  const count = data.products.length;
  if (!count) {
    showStatus(searchStatus, "Подходящих товаров по этому запросу нет.");
    return;
  }

  const familyLabels = { one: "семейство", few: "семейства", many: "семейств" };
  byId("result-count").textContent = `${count} ${familyLabels[pluralRules.select(count)]}`;
  results.replaceChildren(...data.products.map(productCard));
  recommendationPanel.hidden = !data.products.some((product) => product.can_recommend);
  recommendationSource.hidden = true;
  recommendations.replaceChildren();
  showStatus(recommendationStatus, "Нажмите «Подходит к этому товару» на карточке.");
  showStatus(recommendationContext, "");
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
  const request = new AbortController();
  searchRequest = request;
  input.value = query;
  const searchParams = currentSearchParams(query);
  form.setAttribute("aria-busy", "true");
  submit.textContent = "Ищем…";
  workspace.hidden = true;
  queryDetails.hidden = true;
  showStatus(searchStatus, "Ищем товары в каталоге. Первый поиск может занять больше времени из-за загрузки модели.");

  try {
    const data = await getJson("/api/search", searchParams, request.signal);
    if (request.signal.aborted) return;
    renderQueryDetails(data, query);
    renderResults(data);
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

async function loadRecommendations(product, button) {
  recommendationRequest?.abort();
  const request = new AbortController();
  recommendationRequest = request;
  for (const choice of results.querySelectorAll(".recommend-button")) {
    choice.setAttribute("aria-pressed", String(choice === button));
  }
  recommendationSource.textContent = product.name;
  recommendationSource.hidden = false;
  recommendations.replaceChildren();
  showStatus(recommendationStatus, "Проверяем совместимость по каталогу…");
  showStatus(recommendationContext, "");
  if (matchMedia("(max-width: 49rem)").matches) {
    recommendationPanel.scrollIntoView({ block: "start" });
  }

  try {
    const data = await getJson("/api/recommend", {
      ...currentSearchParams(input.value.trim()), offer_id: product.id,
    }, request.signal);
    if (request.signal.aborted) return;
    recommendations.replaceChildren(...data.products.map((item) => productCard(item)));
    showStatus(recommendationStatus, data.products.length ? "" : "Не найдено");
    const missing = (data.missing_context ?? []).map((field) => ({
      platform: "ОС", devices: "число рабочих мест",
    })[field]).filter(Boolean);
    if (missing.length) showStatus(recommendationContext, "Для проверки укажите " + missing.join(" и ") + ".");
  } catch (error) {
    if (!request.signal.aborted) showStatus(recommendationStatus, `Ошибка подбора. ${error.message}`, true);
  } finally {
    if (recommendationRequest === request) recommendationRequest = null;
  }
}

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
