const MODEL_COLORS = ["#1a5fff", "#02967d", "#de8a00", "#cf4f36", "#5f4be8"];
const LANGUAGE_ORDER = ["ar", "en", "fr", "es"];
const LANGUAGE_LABELS = {
  ar: "Arabic",
  en: "English",
  fr: "French",
  es: "Spanish",
};
const LANGUAGE_COLORS = {
  ar: "#d84b4b",
  en: "#1a5fff",
  fr: "#11a579",
  es: "#7a56f3",
};
const DIFFICULTY_ORDER = ["easy", "normal", "hard"];
const DIFFICULTY_LABELS = {
  easy: "Easy",
  normal: "Normal",
  hard: "Hard",
};
const SORT_OPTIONS = [
  { value: "ranking", label: "Board Rank" },
  { value: "macro_language_accuracy_pct", label: "Macro Language Accuracy" },
  { value: "macro_intent_accuracy_pct", label: "Macro Intent Accuracy" },
  { value: "overall_accuracy_pct", label: "Overall Accuracy" },
  { value: "avg_latency_ms", label: "Average Latency" },
  { value: "runtime_cost_seconds", label: "Runtime Cost Proxy" },
];

const state = {
  raw: null,
  prepared: null,
  board: "all_models_comparison",
  textMode: "original",
  category: "all",
  focusModel: "all",
  sortMetric: "ranking",
  showFailuresOnly: false,
  languages: new Set(),
  difficulties: new Set(),
  sourceLabel: "Waiting For Data",
};

const elements = {
  heroWinner: document.getElementById("heroWinner"),
  dataSourcePill: document.getElementById("dataSourcePill"),
  loadJsonButton: document.getElementById("loadJsonButton"),
  jsonFileInput: document.getElementById("jsonFileInput"),
  toolbarSummary: document.getElementById("toolbarSummary"),
  boardFilter: document.getElementById("boardFilter"),
  textModeFilter: document.getElementById("textModeFilter"),
  categoryFilter: document.getElementById("categoryFilter"),
  focusModel: document.getElementById("focusModel"),
  sortMetric: document.getElementById("sortMetric"),
  showFailuresOnly: document.getElementById("showFailuresOnly"),
  languageFilters: document.getElementById("languageFilters"),
  difficultyFilters: document.getElementById("difficultyFilters"),
  kpiSection: document.getElementById("kpiSection"),
  findingsPanel: document.getElementById("findingsPanel"),
  leaderboard: document.getElementById("leaderboard"),
  languageChart: document.getElementById("languageChart"),
  scatterChart: document.getElementById("scatterChart"),
  costChart: document.getElementById("costChart"),
  heatmap: document.getElementById("heatmap"),
  testExplorer: document.getElementById("testExplorer"),
  failurePanel: document.getElementById("failurePanel"),
  footerMeta: document.getElementById("footerMeta"),
  tooltip: document.getElementById("tooltip"),
};

bootstrap();

async function bootstrap() {
  attachEvents();
  seedStaticControls();

  try {
    const data = await fetchBenchmarkJson("./arabic_encoder_benchmark_results.json");
    setData(data, "Auto Loaded");
  } catch (error) {
    renderNoData(
      "The dashboard could not load a compatible benchmark JSON automatically. Re-run the benchmark or load a fresh results file manually."
    );
  }
}

function attachEvents() {
  elements.loadJsonButton.addEventListener("click", () => {
    elements.jsonFileInput.click();
  });

  elements.jsonFileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const raw = JSON.parse(await file.text());
      setData(raw, `Loaded: ${file.name}`);
    } catch (error) {
      renderNoData("The selected file is not a valid benchmark JSON document.");
    }
  });

  elements.boardFilter.addEventListener("change", (event) => {
    state.board = event.target.value;
    const boardDefaults = state.prepared?.boards?.[state.board]?.default_languages;
    state.languages = new Set(boardDefaults?.length ? boardDefaults : state.prepared.languages);
    state.focusModel = "all";
    syncDynamicControls();
    renderAll();
  });

  elements.textModeFilter.addEventListener("change", (event) => {
    state.textMode = event.target.value;
    renderAll();
  });

  elements.categoryFilter.addEventListener("change", (event) => {
    state.category = event.target.value;
    renderAll();
  });

  elements.focusModel.addEventListener("change", (event) => {
    state.focusModel = event.target.value;
    renderAll();
  });

  elements.sortMetric.addEventListener("change", (event) => {
    state.sortMetric = event.target.value;
    renderAll();
  });

  elements.showFailuresOnly.addEventListener("change", (event) => {
    state.showFailuresOnly = event.target.checked;
    renderAll();
  });

  document.addEventListener("click", (event) => {
    const focusTarget = event.target.closest("[data-focus-model]");
    if (focusTarget) {
      state.focusModel = focusTarget.dataset.focusModel || "all";
      elements.focusModel.value = state.focusModel;
      renderAll();
      return;
    }

    const categoryTarget = event.target.closest("[data-filter-category]");
    if (categoryTarget) {
      state.category = categoryTarget.dataset.filterCategory || "all";
      elements.categoryFilter.value = state.category;
      renderAll();
      return;
    }

    const languageToggle = event.target.closest("[data-language-toggle]");
    if (languageToggle) {
      toggleSetValue(state.languages, languageToggle.dataset.languageToggle);
      renderFilterChips();
      renderAll();
      return;
    }

    const difficultyToggle = event.target.closest("[data-difficulty-toggle]");
    if (difficultyToggle) {
      toggleSetValue(state.difficulties, difficultyToggle.dataset.difficultyToggle);
      renderFilterChips();
      renderAll();
      return;
    }
  });

  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest("[data-tooltip]");
    if (!target) {
      return;
    }

    elements.tooltip.hidden = false;
    elements.tooltip.innerHTML = target.dataset.tooltip;
  });

  document.addEventListener("mousemove", (event) => {
    if (elements.tooltip.hidden) {
      return;
    }
    elements.tooltip.style.left = `${event.clientX + 16}px`;
    elements.tooltip.style.top = `${event.clientY + 16}px`;
  });

  document.addEventListener("mouseout", (event) => {
    if (event.target.closest("[data-tooltip]")) {
      elements.tooltip.hidden = true;
    }
  });
}

async function fetchBenchmarkJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return response.json();
}

function setData(raw, sourceLabel) {
  try {
    state.prepared = prepareBenchmarkData(raw);
  } catch (error) {
    renderNoData(error.message || "The JSON file does not match the expected benchmark schema.");
    return;
  }

  state.raw = raw;
  state.sourceLabel = sourceLabel;

  const defaults = state.prepared.defaultView;
  state.board = defaults.board;
  state.textMode = defaults.textMode;
  state.languages = new Set(defaults.languages);
  state.difficulties = new Set(defaults.difficulties);
  state.category = "all";
  state.focusModel = "all";
  state.sortMetric = "ranking";
  state.showFailuresOnly = false;

  syncDynamicControls();
  renderAll();
}

function prepareBenchmarkData(raw) {
  if (!raw?.models?.length || !raw?.boards || !raw?.metadata?.text_modes) {
    throw new Error("This dashboard expects the new multilingual benchmark JSON format. Re-run the benchmark script first.");
  }

  const metadata = raw.metadata ?? {};
  const tests = raw.tests ?? [];
  const boards = raw.boards ?? {};
  const categories = [...new Set(tests.map((test) => test.category).filter(Boolean))].sort();
  const languages = getOrderedLanguages(tests.map((test) => test.language).filter(Boolean));
  const difficulties = getOrderedDifficulties(tests.map((test) => test.difficulty).filter(Boolean));
  const rankingLookup = buildRankingLookup(boards);

  const models = (raw.models ?? []).map((model, index) => {
    const config = model.config ?? {};
    const summariesByTextMode = model.summaries_by_text_mode ?? {};
    const name = config.name ?? summariesByTextMode[metadata.primary_text_mode]?.model_name ?? `Model ${index + 1}`;

    const resultsByTextMode = Object.fromEntries(
      Object.entries(model.results_by_text_mode ?? {}).map(([textMode, results]) => [
        textMode,
        (results ?? []).map((result) => ({
          ...result,
          model_name: name,
          hf_id: config.hf_id ?? summariesByTextMode[textMode]?.hf_id ?? "",
          color: MODEL_COLORS[index % MODEL_COLORS.length],
        })),
      ])
    );

    return {
      model_name: name,
      hf_id: config.hf_id ?? "",
      family: model.family ?? config.family ?? "unknown",
      board_memberships: model.board_memberships ?? ["all_models_comparison"],
      load_time_s: Number(model.load_time_s ?? 0),
      summariesByTextMode,
      resultsByTextMode,
      color: MODEL_COLORS[index % MODEL_COLORS.length],
      initials: getInitials(name),
      boardRanks: rankingLookup,
    };
  });

  const defaultView = {
    board: metadata.default_dashboard_view?.board ?? metadata.primary_board ?? Object.keys(boards)[0],
    textMode: metadata.default_dashboard_view?.text_mode ?? metadata.primary_text_mode ?? metadata.text_modes[0],
    languages: metadata.default_dashboard_view?.languages ?? languages,
    difficulties: metadata.default_dashboard_view?.difficulties ?? difficulties,
  };

  return {
    metadata,
    tests,
    boards,
    models,
    categories,
    languages,
    difficulties,
    defaultView,
  };
}

function buildRankingLookup(boards) {
  const lookup = new Map();
  Object.entries(boards ?? {}).forEach(([boardKey, board]) => {
    Object.entries(board.rankings_by_text_mode ?? {}).forEach(([textMode, entries]) => {
      (entries ?? []).forEach((entry) => {
        lookup.set(`${boardKey}|${textMode}|${entry.model_name}`, entry.rank);
      });
    });
  });
  return lookup;
}

function seedStaticControls() {
  elements.sortMetric.innerHTML = SORT_OPTIONS.map(
    (option) => `<option value="${option.value}">${escapeHtml(option.label)}</option>`
  ).join("");
}

function syncDynamicControls() {
  if (!state.prepared) {
    return;
  }

  const boardEntries = Object.values(state.prepared.boards);
  elements.boardFilter.innerHTML = boardEntries
    .map((board) => `<option value="${board.key}">${escapeHtml(board.label)}</option>`)
    .join("");

  elements.textModeFilter.innerHTML = (state.prepared.metadata.text_modes ?? []).map(
    (textMode) => `<option value="${textMode}">${escapeHtml(capitalize(textMode))}</option>`
  ).join("");

  elements.categoryFilter.innerHTML = [
    `<option value="all">All Categories</option>`,
    ...state.prepared.categories.map((category) => `<option value="${category}">${escapeHtml(category)}</option>`),
  ].join("");

  const boardModels = getBoardModels();
  elements.focusModel.innerHTML = [
    `<option value="all">All Models</option>`,
    ...boardModels.map((model) => `<option value="${escapeAttribute(model.model_name)}">${escapeHtml(model.model_name)}</option>`),
  ].join("");

  if (!boardModels.some((model) => model.model_name === state.focusModel)) {
    state.focusModel = "all";
  }

  elements.boardFilter.value = state.board;
  elements.textModeFilter.value = state.textMode;
  elements.categoryFilter.value = state.category;
  elements.focusModel.value = state.focusModel;
  elements.sortMetric.value = state.sortMetric;
  elements.showFailuresOnly.checked = state.showFailuresOnly;
  renderFilterChips();
}

function renderFilterChips() {
  if (!state.prepared) {
    return;
  }

  const activeLanguages = getSelectedLanguages();
  elements.languageFilters.innerHTML = state.prepared.languages.map((language) => {
    const active = activeLanguages.includes(language);
    return `
      <button class="chip-toggle ${active ? "active" : ""}" type="button" data-language-toggle="${escapeHtml(language)}">
        ${escapeHtml(getLanguageLabel(language))}
      </button>
    `;
  }).join("");

  const activeDifficulties = getSelectedDifficulties();
  elements.difficultyFilters.innerHTML = state.prepared.difficulties.map((difficulty) => {
    const active = activeDifficulties.includes(difficulty);
    return `
      <button class="chip-toggle ${active ? "active" : ""}" type="button" data-difficulty-toggle="${escapeHtml(difficulty)}">
        ${escapeHtml(getDifficultyLabel(difficulty))}
      </button>
    `;
  }).join("");
}

function renderAll() {
  if (!state.prepared) {
    return;
  }

  renderHero();
  renderToolbarSummary();
  renderKpis();
  renderFindings();
  renderLeaderboard();
  renderLanguageComparisonChart();
  renderScatterPlot();
  renderCostChart();
  renderHeatmap();
  renderTestExplorer();
  renderFailurePanel();
  renderFooter();
}

function renderNoData(message) {
  const emptyMarkup = `
    <div class="empty-state">
      <h3>Benchmark data unavailable</h3>
      <p class="muted">${escapeHtml(message)}</p>
    </div>
  `;

  elements.dataSourcePill.textContent = "No Data";
  elements.heroWinner.innerHTML = emptyMarkup;
  elements.toolbarSummary.innerHTML = emptyMarkup;
  [
    elements.kpiSection,
    elements.findingsPanel,
    elements.leaderboard,
    elements.languageChart,
    elements.scatterChart,
    elements.costChart,
    elements.heatmap,
    elements.testExplorer,
    elements.failurePanel,
    elements.footerMeta,
  ].forEach((element) => {
    element.innerHTML = emptyMarkup;
  });
}

function getBoardModels() {
  return state.prepared.models.filter((model) => model.board_memberships.includes(state.board));
}

function getSelectedLanguages() {
  const selected = [...state.languages].filter((language) => state.prepared.languages.includes(language));
  return selected.length ? selected : [...state.prepared.languages];
}

function getSelectedDifficulties() {
  const selected = [...state.difficulties].filter((difficulty) => state.prepared.difficulties.includes(difficulty));
  return selected.length ? selected : [...state.prepared.difficulties];
}

function resultMatchesFilters(result) {
  const languageMatch = getSelectedLanguages().includes(result.language);
  const difficultyMatch = getSelectedDifficulties().includes(result.difficulty);
  const categoryMatch = state.category === "all" || result.category === state.category;
  return languageMatch && difficultyMatch && categoryMatch;
}

function getVisibleModelViews() {
  const boardModels = getBoardModels();
  const views = boardModels.map((model) => buildModelView(model)).filter(Boolean);
  return sortModelViews(views, state.sortMetric);
}

function buildModelView(model) {
  const rawResults = model.resultsByTextMode[state.textMode] ?? [];
  const results = rawResults.filter(resultMatchesFilters);
  if (!results.length) {
    return null;
  }

  const perLanguage = computeAccuracyByField(results, "language", getSelectedLanguages());
  const perDifficulty = computeAccuracyByField(results, "difficulty", state.prepared.difficulties);
  const perCategory = computeAccuracyByField(results, "category", state.prepared.categories);
  const perIntent = computeAccuracyByField(results, "expected_key");
  const overall = computeAccuracy(results);
  const macroLanguage = computeMacroAccuracy(perLanguage);
  const macroIntent = computeMacroAccuracy(perIntent);
  const macroDifficulty = computeMacroAccuracy(perDifficulty);
  const confident = computeConfidentAccuracy(results);
  const avgLatency = average(results.map((result) => Number(result.latency_ms ?? 0)));
  const avgMargin = average(results.map((result) => Number(result.margin ?? 0)));
  const avgScore = average(results.map((result) => Number(result.score ?? 0)));
  const wrongCount = results.filter((result) => !result.correct).length;
  const runtimeCostSeconds = Number(model.load_time_s ?? 0) + (avgLatency * results.length) / 1000;
  const avgSpeedTestsPerSec = avgLatency > 0 ? 1000 / avgLatency : 0;
  const rank = model.boardRanks.get(`${state.board}|${state.textMode}|${model.model_name}`) ?? Number.MAX_SAFE_INTEGER;

  return {
    ...model,
    results,
    rank,
    overall_accuracy_pct: overall,
    macro_language_accuracy_pct: macroLanguage,
    macro_intent_accuracy_pct: macroIntent,
    macro_difficulty_accuracy_pct: macroDifficulty,
    confident_accuracy_pct: confident.accuracy,
    coverage_above_threshold_pct: confident.coverage,
    avg_latency_ms: avgLatency,
    avg_speed_tests_per_sec: avgSpeedTestsPerSec,
    avg_margin: avgMargin,
    avg_score: avgScore,
    runtime_cost_seconds: runtimeCostSeconds,
    correctCount: results.length - wrongCount,
    wrongCount,
    per_language_accuracy_pct: perLanguage,
    per_difficulty_accuracy_pct: perDifficulty,
    per_category_accuracy_pct: perCategory,
    per_intent_accuracy_pct: perIntent,
    bestCategory: pickEdge(perCategory, true),
    weakestCategory: pickEdge(perCategory, false),
  };
}

function renderHero() {
  const models = getVisibleModelViews();
  const winner = models[0];
  elements.dataSourcePill.textContent = state.sourceLabel;

  if (!winner) {
    elements.heroWinner.innerHTML = renderEmpty("No models have results under the current filters.");
    return;
  }

  elements.heroWinner.innerHTML = `
    <p class="tiny-label">Current winner</p>
    <h2 class="winner-name">${escapeHtml(winner.model_name)}</h2>
    <p class="muted">${escapeHtml(winner.hf_id)}</p>
    <div class="winner-metrics">
      ${renderWinnerStat("Macro Language", formatPct(winner.macro_language_accuracy_pct), `${winner.results.length} filtered tests`) }
      ${renderWinnerStat("Macro Intent", formatPct(winner.macro_intent_accuracy_pct), `${escapeHtml(state.textMode)} text mode`) }
      ${renderWinnerStat("Latency", formatMs(winner.avg_latency_ms), `${escapeHtml(getBoardLabel(state.board))}`) }
    </div>
  `;
}

function renderToolbarSummary() {
  const languages = getSelectedLanguages().map(getLanguageLabel).join(", ");
  const difficulties = getSelectedDifficulties().map(getDifficultyLabel).join(", ");
  const categoryText = state.category === "all" ? "all intent families" : state.category;

  elements.toolbarSummary.innerHTML = `
    <p class="section-kicker">Current View</p>
    <h2>${escapeHtml(getBoardLabel(state.board))} on ${escapeHtml(state.textMode)} text</h2>
    <p class="muted">
      Languages: <strong>${escapeHtml(languages)}</strong>.
      Difficulties: <strong>${escapeHtml(difficulties)}</strong>.
      Category filter: <strong>${escapeHtml(categoryText)}</strong>.
    </p>
  `;
}

function renderKpis() {
  const models = getVisibleModelViews();
  if (!models.length) {
    elements.kpiSection.innerHTML = renderEmpty("No KPI data for the current filters.");
    return;
  }

  const winner = models[0];
  const fastest = [...models].sort((a, b) => a.avg_latency_ms - b.avg_latency_ms)[0];
  const bestArabic = [...models].sort((a, b) => (b.per_language_accuracy_pct.ar ?? -1) - (a.per_language_accuracy_pct.ar ?? -1))[0];
  const bestHard = [...models].sort((a, b) => (b.per_difficulty_accuracy_pct.hard ?? -1) - (a.per_difficulty_accuracy_pct.hard ?? -1))[0];

  const cards = [
    {
      accent: "#2d63d6",
      label: "Current Leader",
      value: formatPct(winner.macro_language_accuracy_pct),
      title: winner.model_name,
      note: "Primary view follows board ranking and current filters.",
    },
    {
      accent: "#1c9c7d",
      label: "Fastest",
      value: formatMs(fastest.avg_latency_ms),
      title: fastest.model_name,
      note: "Average latency over the filtered result set.",
    },
    {
      accent: "#df9a27",
      label: "Best Arabic",
      value: formatPct(bestArabic.per_language_accuracy_pct.ar),
      title: bestArabic.model_name,
      note: "Arabic accuracy stays visible even inside multilingual views.",
    },
    {
      accent: "#c85a43",
      label: "Best Hard Cases",
      value: formatPct(bestHard.per_difficulty_accuracy_pct.hard),
      title: bestHard.model_name,
      note: "Hard difficulty remains a secondary analysis slice.",
    },
  ];

  elements.kpiSection.innerHTML = cards.map((card) => `
    <article class="kpi-card" style="--accent:${card.accent}">
      <span class="kpi-label">${escapeHtml(card.label)}</span>
      <strong class="kpi-value">${escapeHtml(card.value)}</strong>
      <h3>${escapeHtml(card.title)}</h3>
      <p>${escapeHtml(card.note)}</p>
    </article>
  `).join("");
}

function renderFindings() {
  const models = getVisibleModelViews();
  if (!models.length) {
    elements.findingsPanel.innerHTML = renderEmpty("No findings available for the current filters.");
    return;
  }

  const winner = models[0];
  const weakestCategory = pickMostCommonWeakness(models);
  const widestGap = pickLargestLanguageGap(models);

  const findings = [
    {
      title: "Top line result",
      text: `${winner.model_name} is leading ${getBoardLabel(state.board)} on ${state.textMode} text under the current filters.`,
    },
    {
      title: "Largest language gap",
      text: widestGap,
    },
    {
      title: "Weakest intent family",
      text: weakestCategory,
    },
  ];

  elements.findingsPanel.innerHTML = findings.map((finding) => `
    <article class="findings-card">
      <p class="section-kicker">Key Finding</p>
      <h3>${escapeHtml(finding.title)}</h3>
      <p>${escapeHtml(finding.text)}</p>
    </article>
  `).join("");
}

function renderLeaderboard() {
  const models = getVisibleModelViews();
  if (!models.length) {
    elements.leaderboard.innerHTML = renderEmpty("No models have results in the current view.");
    return;
  }

  elements.leaderboard.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Decision Table</p>
        <h3 class="card-title">Board ranking plus per-language accuracy</h3>
      </div>
      <p class="metric-note">${models.length} models in view.</p>
    </div>
    <div class="leaderboard-grid">
      ${models.map((model) => renderLeaderboardRow(model)).join("")}
    </div>
  `;
}

function renderLeaderboardRow(model) {
  const focused = state.focusModel === model.model_name;
  const languagesMarkup = getSelectedLanguages().map((language) => renderLanguagePill(language, model.per_language_accuracy_pct[language])).join("");
  const tooltip = `
    <strong>${escapeHtml(model.model_name)}</strong><br />
    Macro language accuracy: ${formatPct(model.macro_language_accuracy_pct)}<br />
    Macro intent accuracy: ${formatPct(model.macro_intent_accuracy_pct)}<br />
    Overall accuracy: ${formatPct(model.overall_accuracy_pct)}<br />
    Avg latency: ${formatMs(model.avg_latency_ms)}<br />
    Avg speed: ${formatSpeed(model.avg_speed_tests_per_sec)}
  `;

  return `
    <article
      class="leaderboard-row clickable ${focused ? "focused" : ""}"
      data-focus-model="${escapeHtml(model.model_name)}"
      data-tooltip="${escapeAttribute(tooltip)}"
      style="--avatar:${model.color}"
    >
      <div class="leaderboard-main">
        <div class="leaderboard-identity">
          <div class="avatar-badge">${escapeHtml(model.initials)}</div>
          <div class="identity-copy">
            <div class="leaderboard-tags">
              <span class="rank-badge">#${model.rank === Number.MAX_SAFE_INTEGER ? "-" : model.rank}</span>
              <span class="trait-badge">${escapeHtml(model.family.replaceAll("_", " "))}</span>
              <span class="trait-badge">${escapeHtml(model.bestCategory.name)} strongest</span>
            </div>
            <h3>${escapeHtml(model.model_name)}</h3>
            <p>${escapeHtml(model.hf_id)}</p>
            <div class="leaderboard-tags">${languagesMarkup}</div>
          </div>
        </div>
        <div class="leaderboard-metrics">
          ${renderMetricBlock("Macro Language", formatPct(model.macro_language_accuracy_pct), model.macro_language_accuracy_pct)}
          ${renderMetricBlock("Macro Intent", formatPct(model.macro_intent_accuracy_pct), model.macro_intent_accuracy_pct)}
          ${renderMetricBlock("Overall", formatPct(model.overall_accuracy_pct), model.overall_accuracy_pct)}
          ${renderMetricBlock("Latency", formatMs(model.avg_latency_ms))}
          ${renderMetricBlock("Filtered Tests", String(model.results.length))}
        </div>
      </div>
    </article>
  `;
}

function renderLanguageComparisonChart() {
  const models = getVisibleModelViews();
  const languages = getSelectedLanguages();
  if (!models.length || !languages.length) {
    elements.languageChart.innerHTML = renderEmpty("No language comparison data is available.");
    return;
  }

  elements.languageChart.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Language Accuracy</p>
        <h3 class="card-title">Per-model performance in the selected languages</h3>
      </div>
      <p class="metric-note">Current text mode: ${escapeHtml(state.textMode)}</p>
    </div>
    <div class="cost-grid">
      ${models.map((model) => `
        <article class="cost-row">
          <div class="cost-row-head">
            <div class="cost-row-title">
              <div class="leaderboard-tags">
                <span class="rank-badge">#${model.rank === Number.MAX_SAFE_INTEGER ? "-" : model.rank}</span>
                <span class="trait-badge">${escapeHtml(model.model_name)}</span>
              </div>
              <h3>${escapeHtml(model.model_name)}</h3>
            </div>
            <div class="cost-total">
              <strong>${escapeHtml(formatPct(model.macro_language_accuracy_pct))}</strong>
              <span class="muted">macro language</span>
            </div>
          </div>
          <div class="mini-bar-list">
            ${languages.map((language) => renderMiniBar(language, model.per_language_accuracy_pct[language], LANGUAGE_COLORS[language] ?? model.color)).join("")}
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function renderScatterPlot() {
  const models = getVisibleModelViews();
  if (!models.length) {
    elements.scatterChart.innerHTML = renderEmpty("No scatter data is available.");
    return;
  }

  const width = 920;
  const height = 360;
  const padding = 46;
  const maxSpeed = Math.max(...models.map((model) => model.avg_speed_tests_per_sec ?? 0), 0);
  const speedRange = expandRange([0, maxSpeed || 1]);
  const qualityRange = [0, 100];

  const points = models.map((model) => {
    const x = scale(model.avg_speed_tests_per_sec ?? 0, speedRange, padding, width - padding);
    const y = scale(model.macro_language_accuracy_pct, qualityRange, height - padding, padding);
    return `
      <g class="scatter-point" data-focus-model="${escapeHtml(model.model_name)}">
        <circle cx="${x}" cy="${y}" r="11" fill="${model.color}"></circle>
        <text x="${x + 14}" y="${y + 4}">${escapeHtml(compactName(model.model_name))}</text>
      </g>
    `;
  }).join("");

  elements.scatterChart.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Speed vs Quality</p>
        <h3 class="card-title">Macro language accuracy against average speed</h3>
      </div>
      <p class="metric-note">Higher speed and higher accuracy move up and right.</p>
    </div>
    <div class="chart-frame">
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Speed versus quality scatter plot">
        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#b8c8df" stroke-width="2"></line>
        <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#b8c8df" stroke-width="2"></line>
        <text class="scatter-axis-label" x="${width / 2}" y="${height - 10}" text-anchor="middle">Average speed (tests/sec)</text>
        <text class="scatter-axis-label" x="18" y="${height / 2}" transform="rotate(-90 18 ${height / 2})" text-anchor="middle">Macro language accuracy (%)</text>
        ${points}
      </svg>
      ${renderLegend(models)}
    </div>
  `;
}

function renderCostChart() {
  const models = getVisibleModelViews();
  if (!models.length) {
    elements.costChart.innerHTML = renderEmpty("No runtime cost data is available.");
    return;
  }

  const maxCost = Math.max(...models.map((model) => model.runtime_cost_seconds), 1);
  elements.costChart.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Runtime Cost Proxy</p>
        <h3 class="card-title">Load time plus filtered inference time</h3>
      </div>
      <p class="metric-note">This is a relative comparison on the current machine only.</p>
    </div>
    <div class="cost-grid">
      ${models.map((model) => {
        const width = (model.runtime_cost_seconds / maxCost) * 100;
        const inference = Math.max(model.runtime_cost_seconds - model.load_time_s, 0);
        const loadShare = model.runtime_cost_seconds > 0 ? (model.load_time_s / model.runtime_cost_seconds) * 100 : 0;
        const inferenceShare = 100 - loadShare;
        return `
          <article class="cost-row ${state.focusModel === model.model_name ? "focused" : ""}">
            <div class="cost-row-head">
              <div class="cost-row-title">
                <div class="leaderboard-tags">
                  <span class="rank-badge">#${model.rank === Number.MAX_SAFE_INTEGER ? "-" : model.rank}</span>
                  <span class="trait-badge">${escapeHtml(compactName(model.model_name))}</span>
                </div>
                <h3>${escapeHtml(model.model_name)}</h3>
              </div>
              <div class="cost-total">
                <strong>${escapeHtml(formatSeconds(model.runtime_cost_seconds))}</strong>
                <span class="muted">${model.results.length} filtered tests</span>
              </div>
            </div>
            <div class="cost-track">
              <div class="cost-stack" style="width:${width}%">
                <span class="cost-segment load" style="width:${loadShare}%"></span>
                <span class="cost-segment inference" style="width:${inferenceShare}%"></span>
              </div>
            </div>
            <div class="cost-meta">
              <span class="mini-pill">Load ${escapeHtml(formatSeconds(model.load_time_s))}</span>
              <span class="mini-pill">Inference ${escapeHtml(formatSeconds(inference))}</span>
              <span class="mini-pill">Latency ${escapeHtml(formatMs(model.avg_latency_ms))}</span>
            </div>
          </article>
        `;
      }).join("")}
    </div>
  `;
}

function renderHeatmap() {
  const models = getVisibleModelViews();
  const categories = state.category === "all" ? state.prepared.categories : [state.category];
  if (!models.length || !categories.length) {
    elements.heatmap.innerHTML = renderEmpty("No category data is available.");
    return;
  }

  const header = `
    <div class="heatmap-row" style="--cols:${models.length}">
      <div class="heatmap-header">Category</div>
      ${models.map((model) => `
        <button class="heatmap-header clickable" type="button" data-focus-model="${escapeHtml(model.model_name)}">
          ${escapeHtml(compactName(model.model_name))}
        </button>
      `).join("")}
    </div>
  `;

  const rows = categories.map((category) => `
    <div class="heatmap-row" style="--cols:${models.length}">
      <button class="heatmap-label clickable" type="button" data-filter-category="${escapeHtml(category)}">
        ${escapeHtml(category)}
      </button>
      ${models.map((model) => {
        const value = model.per_category_accuracy_pct[category];
        return `
          <div class="heatmap-cell" style="background:${heatColor(value ?? 0)}">
            ${formatPct(value)}
          </div>
        `;
      }).join("")}
    </div>
  `).join("");

  elements.heatmap.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Intent Families</p>
        <h3 class="card-title">Semantic category accuracy by model</h3>
      </div>
      <p class="metric-note">Difficulty is filtered separately and never mixed into category.</p>
    </div>
    <div class="heatmap-grid">${header}${rows}</div>
  `;
}

function renderTestExplorer() {
  const tests = getVisibleTests();
  if (!tests.length) {
    elements.testExplorer.innerHTML = renderEmpty("No tests match the current filters.");
    return;
  }

  elements.testExplorer.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Evidence</p>
        <h3 class="card-title">${tests.length} sentence${tests.length === 1 ? "" : "s"} in view</h3>
      </div>
      <p class="metric-note">Showing ${escapeHtml(state.textMode)} text-mode results.</p>
    </div>
    <div class="test-accordion">
      ${tests.map((test) => renderTestCase(test)).join("")}
    </div>
  `;
}

function renderTestCase(test) {
  const results = state.focusModel === "all" ? test.modelResults : test.modelResults.filter((result) => result.model_name === state.focusModel);
  return `
    <details class="test-case">
      <summary class="test-summary">
        <div class="test-summary-head">
          <span class="rank-badge">${escapeHtml(test.id)}</span>
          <span class="status-chip info">${escapeHtml(getLanguageLabel(test.language))}</span>
          <span class="status-chip info">${escapeHtml(test.category)}</span>
          <span class="status-chip info">${escapeHtml(getDifficultyLabel(test.difficulty))}</span>
        </div>
        <div class="sentence-block ${test.language === "ar" ? "rtl" : ""}">${escapeHtml(test.text)}</div>
        ${test.evaluated_text !== test.text ? `<p class="muted">Expanded text: ${escapeHtml(test.evaluated_text)}</p>` : ""}
      </summary>
      <div class="test-expanded">
        <div class="test-model-grid">
          ${results.map((result) => renderTestModelCard(result)).join("")}
        </div>
      </div>
    </details>
  `;
}

function renderTestModelCard(result) {
  const outcomeClass = result.correct ? (result.confident ? "good" : "info") : result.confident ? "bad" : "warn";
  return `
    <article class="test-model-card ${state.focusModel === result.model_name ? "focused" : ""}">
      <div class="leaderboard-tags">
        <span class="rank-badge">#${result.rank === Number.MAX_SAFE_INTEGER ? "-" : result.rank}</span>
        <span class="status-chip ${outcomeClass}">${result.correct ? "Correct" : "Wrong"}</span>
      </div>
      <h4>${escapeHtml(result.model_name)}</h4>
      <div class="test-model-metrics">
        ${renderMetricBlock("Score", Number(result.score).toFixed(3))}
        ${renderMetricBlock("Margin", Number(result.margin).toFixed(3))}
        ${renderMetricBlock("Latency", formatMs(result.latency_ms))}
      </div>
      <div class="failure-badges">
        <span class="status-chip info">Predicted: ${escapeHtml(shortLabel(result.pred_label))}</span>
        <span class="status-chip warn">Expected: ${escapeHtml(shortLabel(result.correct_label))}</span>
      </div>
    </article>
  `;
}

function renderFailurePanel() {
  const failures = getVisibleFailures();
  if (!failures.length) {
    elements.failurePanel.innerHTML = renderEmpty("No failures in the current view.");
    return;
  }

  elements.failurePanel.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Failure Analysis</p>
        <h3 class="card-title">${failures.length} failed predictions in view</h3>
      </div>
      <p class="metric-note">Sorted by confident errors first.</p>
    </div>
    <div class="failure-grid">
      ${failures.map((failure) => `
        <article class="failure-card">
          <div class="leaderboard-tags">
            <span class="rank-badge">#${failure.rank === Number.MAX_SAFE_INTEGER ? "-" : failure.rank}</span>
            <span class="status-chip ${failure.confident ? "bad" : "warn"}">${failure.confident ? "High confidence wrong" : "Wrong"}</span>
            <span class="status-chip info">${escapeHtml(getLanguageLabel(failure.language))}</span>
          </div>
          <h4>${escapeHtml(failure.model_name)}</h4>
          <div class="sentence-block ${failure.language === "ar" ? "rtl" : ""}">${escapeHtml(failure.text)}</div>
          <div class="test-model-metrics">
            ${renderMetricBlock("Score", Number(failure.score).toFixed(3))}
            ${renderMetricBlock("Margin", Number(failure.margin).toFixed(3))}
            ${renderMetricBlock("Latency", formatMs(failure.latency_ms))}
          </div>
          <div class="failure-badges">
            <span class="status-chip bad">Predicted: ${escapeHtml(shortLabel(failure.pred_label))}</span>
            <span class="status-chip info">Expected: ${escapeHtml(shortLabel(failure.correct_label))}</span>
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function renderFooter() {
  const metadata = state.prepared.metadata;
  elements.footerMeta.innerHTML = `
    <div>
      <p class="section-kicker">Benchmark Metadata</p>
      <h3 class="card-title">One multilingual benchmark, multiple comparison boards</h3>
      <p>Default view is ${escapeHtml(getBoardLabel(metadata.primary_board))} on ${escapeHtml(metadata.primary_text_mode)} text.</p>
    </div>
    <div class="meta-chip-list">
      <span class="mini-pill">Device ${escapeHtml(String(metadata.device ?? "unknown"))}</span>
      <span class="mini-pill">${metadata.num_models ?? 0} models</span>
      <span class="mini-pill">${metadata.num_tests ?? 0} tests</span>
      <span class="mini-pill">${metadata.repeats ?? 1} repeats</span>
      <span class="mini-pill">Score ${metadata.thresholds?.score ?? 0}</span>
      <span class="mini-pill">Margin ${metadata.thresholds?.margin ?? 0}</span>
    </div>
  `;
}

function getVisibleTests() {
  const models = getVisibleModelViews();
  const modelLookup = new Map(models.map((model) => [model.model_name, model]));
  return state.prepared.tests
    .filter((test) => getSelectedLanguages().includes(test.language))
    .filter((test) => getSelectedDifficulties().includes(test.difficulty))
    .filter((test) => state.category === "all" || test.category === state.category)
    .map((test) => {
      const modelResults = models.map((model) => {
        const result = (model.resultsByTextMode?.[state.textMode] ?? []).find((entry) => entry.test_id === test.id);
        if (!result) {
          return null;
        }
        return {
          ...result,
          rank: modelLookup.get(model.model_name)?.rank ?? Number.MAX_SAFE_INTEGER,
        };
      }).filter(Boolean).sort((a, b) => a.rank - b.rank);

      return {
        ...test,
        evaluated_text: modelResults[0]?.evaluated_text ?? test.text,
        modelResults,
      };
    })
    .filter((test) => test.modelResults.length > 0)
    .filter((test) => !state.showFailuresOnly || test.modelResults.some((result) => !result.correct));
}

function getVisibleFailures() {
  return getVisibleTests()
    .flatMap((test) => test.modelResults.filter((result) => !result.correct).map((result) => ({ ...result, text: test.text })))
    .filter((failure) => state.focusModel === "all" || failure.model_name === state.focusModel)
    .sort((a, b) => Number(b.confident) - Number(a.confident) || a.margin - b.margin || (b.score ?? 0) - (a.score ?? 0));
}

function sortModelViews(models, metric) {
  const sorted = [...models];
  sorted.sort((a, b) => {
    if (metric === "ranking") {
      return a.rank - b.rank || (b.macro_language_accuracy_pct ?? 0) - (a.macro_language_accuracy_pct ?? 0);
    }
    if (metric === "avg_latency_ms" || metric === "runtime_cost_seconds") {
      return (a[metric] ?? 0) - (b[metric] ?? 0);
    }
    return (b[metric] ?? 0) - (a[metric] ?? 0);
  });
  return state.focusModel === "all" ? sorted : sorted.filter((model) => model.model_name === state.focusModel);
}

function computeAccuracy(results) {
  if (!results.length) {
    return 0;
  }
  return (results.filter((result) => result.correct).length / results.length) * 100;
}

function computeAccuracyByField(results, field, orderedValues = []) {
  const values = orderedValues.length ? orderedValues : [...new Set(results.map((result) => result[field]).filter(Boolean))];
  return Object.fromEntries(values.map((value) => {
    const group = results.filter((result) => result[field] === value);
    return [value, group.length ? round((group.filter((result) => result.correct).length / group.length) * 100, 6) : null];
  }));
}

function computeMacroAccuracy(entries) {
  const values = Object.values(entries).filter((value) => value != null);
  return values.length ? average(values) : 0;
}

function computeConfidentAccuracy(results) {
  if (!results.length) {
    return { accuracy: 0, coverage: 0 };
  }
  const confident = results.filter((result) => result.confident);
  if (!confident.length) {
    return { accuracy: 0, coverage: 0 };
  }
  return {
    accuracy: (confident.filter((result) => result.correct).length / confident.length) * 100,
    coverage: (confident.length / results.length) * 100,
  };
}

function pickEdge(entries, highest) {
  const filtered = Object.entries(entries).filter(([, value]) => value != null);
  if (!filtered.length) {
    return { name: "n/a", value: 0 };
  }
  const target = filtered.reduce((best, current) => highest ? (current[1] > best[1] ? current : best) : (current[1] < best[1] ? current : best));
  return { name: target[0], value: target[1] ?? 0 };
}

function pickMostCommonWeakness(models) {
  const items = models.map((model) => model.weakestCategory.name).filter((value) => value && value !== "n/a");
  if (!items.length) {
    return "No category weakness is visible in the current slice.";
  }
  const counts = countValues(items);
  const [name] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  return `${name} appears most often as the weakest intent family across the visible models.`;
}

function pickLargestLanguageGap(models) {
  const selected = getSelectedLanguages();
  if (selected.length < 2) {
    return "Select at least two languages to compare language-specific gaps.";
  }
  let bestMessage = "Language gaps are minimal in the current slice.";
  let bestGap = -1;
  models.forEach((model) => {
    selected.forEach((left) => {
      selected.forEach((right) => {
        if (left >= right) {
          return;
        }
        const gap = Math.abs((model.per_language_accuracy_pct[left] ?? 0) - (model.per_language_accuracy_pct[right] ?? 0));
        if (gap > bestGap) {
          bestGap = gap;
          bestMessage = `${model.model_name} shows the largest gap between ${getLanguageLabel(left)} and ${getLanguageLabel(right)} at ${gap.toFixed(1)} points.`;
        }
      });
    });
  });
  return bestMessage;
}

function renderMetricBlock(label, value, progressValue = null) {
  return `
    <div class="metric-block">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong class="metric-value">${escapeHtml(value)}</strong>
      ${progressValue == null ? "" : `<div class="progress"><span style="--value:${clamp(progressValue, 0, 100)}%"></span></div>`}
    </div>
  `;
}

function renderWinnerStat(label, value, note) {
  return `
    <div class="winner-stat">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <span class="muted">${escapeHtml(note)}</span>
    </div>
  `;
}

function renderLanguagePill(language, value) {
  return `<span class="mini-pill">${escapeHtml(getLanguageLabel(language))}: ${escapeHtml(formatPct(value))}</span>`;
}

function renderMiniBar(language, value, color) {
  return `
    <div class="mini-bar-row">
      <span class="mini-bar-label">${escapeHtml(getLanguageLabel(language))}</span>
      <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${clamp(value ?? 0, 0, 100)}%;background:${color}"></div></div>
      <span class="mini-bar-value">${escapeHtml(formatPct(value))}</span>
    </div>
  `;
}

function renderLegend(models) {
  return `
    <div class="chart-legend">
      ${models.map((model) => `
        <button class="legend-item clickable" type="button" data-focus-model="${escapeHtml(model.model_name)}">
          <span class="legend-swatch" style="background:${model.color}"></span>
          <span>${escapeHtml(compactName(model.model_name))}</span>
        </button>
      `).join("")}
    </div>
  `;
}

function renderEmpty(message) {
  return `<div class="empty-state"><p class="muted">${escapeHtml(message)}</p></div>`;
}

function toggleSetValue(setObject, value) {
  if (!value) {
    return;
  }
  if (setObject.has(value)) {
    setObject.delete(value);
  } else {
    setObject.add(value);
  }
}

function countValues(values) {
  const map = new Map();
  values.forEach((value) => map.set(value, (map.get(value) ?? 0) + 1));
  return map;
}

function getRange(values) {
  return values.length ? [Math.min(...values), Math.max(...values)] : [0, 1];
}

function expandRange(range) {
  const [min, max] = range;
  if (min === max) {
    return [min - 1, max + 1];
  }
  const padding = (max - min) * 0.1;
  return [min - padding, max + padding];
}

function scale(value, range, outMin, outMax) {
  const [min, max] = range;
  if (max === min) {
    return (outMin + outMax) / 2;
  }
  const ratio = (value - min) / (max - min);
  return outMin + ratio * (outMax - outMin);
}

function heatColor(value) {
  const clamped = clamp(value, 0, 100);
  const hue = (clamped / 100) * 120;
  return `linear-gradient(135deg, hsla(${hue}, 65%, 42%, 0.96), hsla(${Math.max(hue - 16, 0)}, 70%, 34%, 0.94))`;
}

function getOrderedLanguages(languages) {
  const unique = [...new Set(languages)];
  return unique.sort((left, right) => {
    const leftIndex = LANGUAGE_ORDER.indexOf(left);
    const rightIndex = LANGUAGE_ORDER.indexOf(right);
    return (leftIndex === -1 ? 99 : leftIndex) - (rightIndex === -1 ? 99 : rightIndex) || left.localeCompare(right);
  });
}

function getOrderedDifficulties(difficulties) {
  const unique = [...new Set(difficulties)];
  return unique.sort((left, right) => DIFFICULTY_ORDER.indexOf(left) - DIFFICULTY_ORDER.indexOf(right));
}

function getLanguageLabel(language) {
  return LANGUAGE_LABELS[language] ?? String(language ?? "").toUpperCase();
}

function getDifficultyLabel(difficulty) {
  return DIFFICULTY_LABELS[difficulty] ?? capitalize(String(difficulty ?? ""));
}

function getBoardLabel(boardKey) {
  return state.prepared?.boards?.[boardKey]?.label ?? boardKey;
}

function compactName(name = "") {
  return name.replace("multilingual-", "").replace("MiniLM-L12-", "MiniLM ");
}

function getInitials(name = "") {
  return name.split(/[\s/-]+/).filter(Boolean).slice(0, 2).map((part) => part[0]?.toUpperCase() ?? "").join("");
}

function shortLabel(label = "") {
  return label.length > 34 ? `${label.slice(0, 34)}...` : label;
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + Number(value || 0), 0) / values.length : 0;
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatPct(value) {
  return value == null || Number.isNaN(value) ? "--" : `${Number(value).toFixed(1)}%`;
}

function formatMs(value) {
  return value == null || Number.isNaN(value) ? "--" : `${Number(value).toFixed(1)} ms`;
}

function formatSpeed(value) {
  return value == null || Number.isNaN(value) ? "--" : `${Number(value).toFixed(value >= 10 ? 1 : 2)} tests/s`;
}

function formatSeconds(value) {
  return value == null || Number.isNaN(value) ? "--" : `${Number(value).toFixed(value >= 10 ? 1 : 2)} s`;
}

function capitalize(value = "") {
  return value ? value[0].toUpperCase() + value.slice(1) : value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("\n", "&#10;");
}
