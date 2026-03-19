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

const SORT_OPTIONS = [
  { value: "ranking", label: "Benchmark Rank" },
  { value: "macro_intent_accuracy_pct", label: "Macro Intent Accuracy" },
  { value: "overall_accuracy_pct", label: "Overall Accuracy" },
  { value: "avg_latency_ms", label: "Average Latency" },
  { value: "benchmark_total_runtime_s", label: "Runtime Cost Proxy" },
  { value: "confident_accuracy_pct", label: "Confident Accuracy" },
  { value: "riskScore", label: "Risk Score" },
];

const state = {
  raw: null,
  prepared: null,
  category: "all",
  focusModel: "all",
  sortMetric: "ranking",
  showFailuresOnly: false,
  sourceLabel: "Waiting For Data",
};

const elements = {
  heroWinner: document.getElementById("heroWinner"),
  dataSourcePill: document.getElementById("dataSourcePill"),
  loadJsonButton: document.getElementById("loadJsonButton"),
  jsonFileInput: document.getElementById("jsonFileInput"),
  toolbarSummary: document.getElementById("toolbarSummary"),
  categoryFilter: document.getElementById("categoryFilter"),
  focusModel: document.getElementById("focusModel"),
  sortMetric: document.getElementById("sortMetric"),
  showFailuresOnly: document.getElementById("showFailuresOnly"),
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
      "The dashboard could not load the benchmark JSON automatically. Use the load button or run a small local server."
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
  state.raw = raw;
  state.prepared = prepareBenchmarkData(raw);
  state.sourceLabel = sourceLabel;
  syncDynamicControls();
  renderAll();
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
  elements.toolbarSummary.innerHTML = `
    <p class="section-kicker">Current View</p>
    <h2>Awaiting benchmark data</h2>
    <p class="muted">${escapeHtml(message)}</p>
  `;

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

function prepareBenchmarkData(raw) {
  const metadata = raw.metadata ?? {};
  const thresholds = metadata.thresholds ?? { score: 0.6, margin: 0.04 };
  const rawTests = raw.tests ?? [];
  const testLookup = new Map(rawTests.map((test) => [test.id, test]));
  const rankingLookup = new Map((raw.ranking ?? []).map((entry) => [entry.model_name, entry.rank]));
  const categories = [...new Set(rawTests.map((test) => test.category))];
  const languages = getOrderedLanguages(rawTests.map((test) => test.language).filter(Boolean));
  const benchmarkTestCount = Number(metadata.num_tests ?? rawTests.length ?? 0);

  const models = (raw.models ?? []).map((model, index) => {
    const summary = model.summary ?? {};
    const results = (model.results ?? []).map((result) => ({
      ...result,
      language: result.language ?? testLookup.get(result.test_id)?.language ?? "unknown",
      model_name: summary.model_name,
      statusClass: getStatusClass(result),
    }));
    const categoryAccuracy = summary.per_category_accuracy_pct ?? {};
    const languageAccuracy =
      summary.per_language_accuracy_pct ?? computeAccuracyByField(results, "language", languages);
    const categoryEntries = Object.entries(categoryAccuracy).filter(([, value]) => value != null);
    const wrongCount = results.filter((result) => !result.correct).length;
    const highConfidenceWrongCount = results.filter((result) => !result.correct && result.confident).length;
    const lowMarginCount = results.filter((result) => result.margin < thresholds.margin).length;
    const bestCategory = pickCategoryEdge(categoryEntries, true);
    const weakestCategory = pickCategoryEdge(categoryEntries, false);

    return {
      ...summary,
      results,
      rank: rankingLookup.get(summary.model_name) ?? index + 1,
      color: MODEL_COLORS[index % MODEL_COLORS.length],
      initials: getInitials(summary.model_name ?? `M${index + 1}`),
      categoryAccuracy,
      languageAccuracy,
      bestCategory,
      weakestCategory,
      hardAccuracy: categoryAccuracy.hard ?? 0,
      correctCount: results.length - wrongCount,
      wrongCount,
      highConfidenceWrongCount,
      lowMarginCount,
    };
  });

  const latencyRange = getRange(models.map((model) => model.avg_latency_ms ?? 0));
  const marginRange = getRange(models.map((model) => model.avg_margin ?? 0));
  const macroRange = getRange(models.map((model) => model.macro_intent_accuracy_pct ?? 0));

  const scoredModels = models.map((model) => {
    const speedScore = normalizeInverse(model.avg_latency_ms ?? 0, latencyRange);
    const marginScore = normalize(model.avg_margin ?? 0, marginRange);
    const macroScore = normalize(model.macro_intent_accuracy_pct ?? 0, macroRange);
    const confidenceScore = (model.confident_accuracy_pct ?? 0) / 100;
    const coverageScore = (model.coverage_above_threshold_pct ?? 0) / 100;
    const avgSpeedTestsPerSec = latencyToThroughput(model.avg_latency_ms);
    const benchmarkInferenceRuntimeS = ((model.avg_latency_ms ?? 0) * benchmarkTestCount) / 1000;
    const benchmarkTotalRuntimeS = (model.load_time_s ?? 0) + benchmarkInferenceRuntimeS;
    const balancedScore =
      macroScore * 40 +
      ((model.overall_accuracy_pct ?? 0) / 100) * 20 +
      speedScore * 15 +
      marginScore * 15 +
      confidenceScore * 10;
    const riskScore = clamp(
      ((model.wrongCount / Math.max(1, model.results.length)) * 55 +
        (model.highConfidenceWrongCount / Math.max(1, model.results.length)) * 25 +
        (model.lowMarginCount / Math.max(1, model.results.length)) * 10 +
        (1 - coverageScore) * 10) *
        100,
      0,
      100
    );

    return {
      ...model,
      avg_speed_tests_per_sec: round(avgSpeedTestsPerSec, 2),
      benchmark_inference_runtime_s: round(benchmarkInferenceRuntimeS, 3),
      benchmark_total_runtime_s: round(benchmarkTotalRuntimeS, 3),
      speedScore: round(speedScore * 100, 1),
      balancedScore: round(balancedScore, 1),
      riskScore: round(riskScore, 1),
    };
  });

  const cheapestRuntime = Math.min(...scoredModels.map((model) => model.benchmark_total_runtime_s ?? Number.POSITIVE_INFINITY));
  const enrichedModels = scoredModels.map((model) => ({
    ...model,
    runtime_cost_multiple:
      Number.isFinite(cheapestRuntime) && cheapestRuntime > 0
        ? round((model.benchmark_total_runtime_s ?? 0) / cheapestRuntime, 2)
        : 1,
  }));

  const tests = (raw.tests ?? []).map((test) => {
    const modelResults = enrichedModels
      .map((model) => {
        const result = model.results.find((entry) => entry.test_id === test.id);
        return result
          ? {
              ...result,
              color: model.color,
              rank: model.rank,
            }
          : null;
      })
      .filter(Boolean)
      .sort((left, right) => left.rank - right.rank);

    return {
      ...test,
      modelResults,
      correctCount: modelResults.filter((entry) => entry.correct).length,
      wrongCount: modelResults.filter((entry) => !entry.correct).length,
      highConfidenceWrong: modelResults.filter((entry) => !entry.correct && entry.confident).length,
    };
  });

  const categoryAverages = categories
    .map((category) => {
      const values = enrichedModels
        .map((model) => model.categoryAccuracy[category])
        .filter((value) => value != null);
      return {
        category,
        average: values.length ? average(values) : 0,
      };
    })
    .sort((left, right) => left.average - right.average);

  const winners = {
    bestOverall: [...enrichedModels].sort((a, b) => a.rank - b.rank)[0] ?? null,
    bestBalanced:
      [...enrichedModels].sort((a, b) => (b.balancedScore ?? 0) - (a.balancedScore ?? 0))[0] ?? null,
    fastest:
      [...enrichedModels].sort((a, b) => (a.avg_latency_ms ?? 0) - (b.avg_latency_ms ?? 0))[0] ?? null,
    cheapest:
      [...enrichedModels].sort((a, b) => (a.benchmark_total_runtime_s ?? 0) - (b.benchmark_total_runtime_s ?? 0))[0] ??
      null,
    safest: [...enrichedModels].sort((a, b) => (a.riskScore ?? 0) - (b.riskScore ?? 0))[0] ?? null,
    bestConfidence:
      [...enrichedModels].sort(
        (a, b) =>
          (b.confident_accuracy_pct ?? 0) - (a.confident_accuracy_pct ?? 0) ||
          (b.coverage_above_threshold_pct ?? 0) - (a.coverage_above_threshold_pct ?? 0)
      )[0] ?? null,
    hardestCategory: categoryAverages[0] ?? null,
  };

  return {
    metadata,
    thresholds,
    models: enrichedModels,
    tests,
    categories,
    languages,
    winners,
    categoryAverages,
  };
}

function renderHero() {
  const { winners, metadata, models } = state.prepared;
  const winner = winners.bestOverall;
  elements.dataSourcePill.textContent = state.sourceLabel;

  if (!winner) {
    elements.heroWinner.innerHTML = `<div class="empty-state"><p class="muted">No ranking data available.</p></div>`;
    return;
  }

  const second = [...models]
    .sort((a, b) => a.rank - b.rank)
    .find((model) => model.model_name !== winner.model_name);
  const lead = second
    ? `${formatSigned((winner.macro_intent_accuracy_pct ?? 0) - (second.macro_intent_accuracy_pct ?? 0))} macro lead`
    : "Only one model available";

  elements.heroWinner.innerHTML = `
    <p class="tiny-label">Top model right now</p>
    <h2 class="winner-name">${escapeHtml(winner.model_name)}</h2>
    <p class="muted">${escapeHtml(winner.hf_id)}</p>
    <div class="winner-metrics">
      ${renderWinnerStat("Macro Intent", formatPct(winner.macro_intent_accuracy_pct), lead)}
      ${renderWinnerStat("Overall Accuracy", formatPct(winner.overall_accuracy_pct), `${winner.correctCount}/${winner.results.length} correct`)}
      ${renderWinnerStat("Average Latency", formatMs(winner.avg_latency_ms), `Device: ${escapeHtml(String(metadata.device ?? "unknown"))}`)}
    </div>
  `;
}

function renderToolbarSummary() {
  const { metadata, winners } = state.prepared;
  const categoryText = state.category === "all" ? "all categories" : state.category;
  const focusText = state.focusModel === "all" ? "all models" : state.focusModel;

  elements.toolbarSummary.innerHTML = `
    <p class="section-kicker">Current View</p>
    <h2>${escapeHtml(winners.bestBalanced?.model_name ?? "No Data")} is the cleanest balance of quality and speed</h2>
    <p class="muted">
      ${metadata.num_models ?? 0} models, ${metadata.num_tests ?? 0} tests, currently viewing
      <strong>${escapeHtml(categoryText)}</strong> with focus on
      <strong>${escapeHtml(focusText)}</strong>.
    </p>
  `;
}

function renderKpis() {
  const { winners } = state.prepared;
  const cards = [
    {
      accent: "#2d63d6",
      label: "Best Overall",
      value: formatPct(winners.bestOverall?.macro_intent_accuracy_pct),
      title: winners.bestOverall?.model_name ?? "n/a",
      note: "Primary ranking now uses macro intent accuracy.",
    },
    {
      accent: "#1c9c7d",
      label: "Fastest",
      value: formatMs(winners.fastest?.avg_latency_ms),
      title: winners.fastest?.model_name ?? "n/a",
      note: "Lowest average inference latency.",
    },
    {
      accent: "#df9a27",
      label: "Most Trustworthy",
      value: `${winners.safest?.riskScore?.toFixed(1) ?? "--"}`,
      title: winners.safest?.model_name ?? "n/a",
      note: "Lower risk means fewer weak or dangerous errors.",
    },
    {
      accent: "#c85a43",
      label: "Hardest Category",
      value: formatPct(winners.hardestCategory?.average),
      title: winners.hardestCategory?.category ?? "n/a",
      note: "Lowest average performance across models.",
    },
  ];

  elements.kpiSection.innerHTML = cards
    .map(
      (card) => `
        <article class="kpi-card" style="--accent:${card.accent}">
          <span class="kpi-label">${escapeHtml(card.label)}</span>
          <strong class="kpi-value">${escapeHtml(card.value)}</strong>
          <h3>${escapeHtml(card.title)}</h3>
          <p>${escapeHtml(card.note)}</p>
        </article>
      `
    )
    .join("");
}

function renderFindings() {
  const { winners, models } = state.prepared;
  const hardestLeader = [...models].sort((a, b) => (b.hardAccuracy ?? 0) - (a.hardAccuracy ?? 0))[0];

  const findings = [
    {
      title: "Top line result",
      text: `${winners.bestOverall?.model_name ?? "N/A"} leads the benchmark and should be the first model you compare everything against.`,
    },
    {
      title: "Best speed-quality tradeoff",
      text: `${winners.bestBalanced?.model_name ?? "N/A"} is the cleanest choice when you care about both accuracy and runtime.`,
    },
    {
      title: "Main weakness",
      text: `${winners.hardestCategory?.category ?? "N/A"} is the weakest area overall, while ${hardestLeader?.model_name ?? "N/A"} handles hard cases best.`,
    },
  ];

  elements.findingsPanel.innerHTML = findings
    .map(
      (finding) => `
        <article class="findings-card">
          <p class="section-kicker">Key Finding</p>
          <h3>${escapeHtml(finding.title)}</h3>
          <p>${escapeHtml(finding.text)}</p>
        </article>
      `
    )
    .join("");
}

function renderLeaderboard() {
  const models = sortModels(state.prepared.models, state.sortMetric);
  elements.leaderboard.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Decision Table</p>
        <h3 class="card-title">Only the metrics needed to compare models quickly</h3>
      </div>
      <p class="metric-note">${models.length} models in the benchmark.</p>
    </div>
    <div class="leaderboard-grid">
      ${models.map((model) => renderLeaderboardRow(model)).join("")}
    </div>
  `;
}

function renderLeaderboardRow(model) {
  const focused = state.focusModel === model.model_name;
  const accuracyProgress = clamp(model.macro_intent_accuracy_pct ?? 0, 0, 100);
  const tooltip = `
    <strong>${escapeHtml(model.model_name)}</strong><br />
    Macro intent accuracy: ${formatPct(model.macro_intent_accuracy_pct)}<br />
    Overall accuracy: ${formatPct(model.overall_accuracy_pct)}<br />
    Avg latency: ${formatMs(model.avg_latency_ms)}<br />
    Risk score: ${model.riskScore.toFixed(1)}
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
              <span class="rank-badge">#${model.rank}</span>
              <span class="trait-badge">${escapeHtml(model.bestCategory.name)} strongest</span>
            </div>
            <h3>${escapeHtml(model.model_name)}</h3>
            <p>${escapeHtml(model.hf_id)}</p>
          </div>
        </div>
        <div class="leaderboard-metrics">
          ${renderMetricBlock("Macro Intent", formatPct(model.macro_intent_accuracy_pct), accuracyProgress)}
          ${renderMetricBlock("Overall", formatPct(model.overall_accuracy_pct))}
          ${renderMetricBlock("Latency", formatMs(model.avg_latency_ms))}
          ${renderMetricBlock("Coverage", formatPct(model.coverage_above_threshold_pct))}
          ${renderMetricBlock("Risk", `${model.riskScore.toFixed(1)}`)}
        </div>
      </div>
    </article>
  `;
}

function renderLanguageComparisonChart() {
  const models = state.prepared.models;
  const languages = getOrderedLanguages(state.prepared.languages);

  if (!models.length || !languages.length) {
    elements.languageChart.innerHTML = renderEmpty("No language comparison data available.");
    return;
  }

  const shownLanguages = languages.slice(0, 4);
  const width = 940;
  const height = 430;
  const padding = { top: 28, right: 24, bottom: 82, left: 58 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const yTicks = [0, 25, 50, 75, 100];
  const groupWidth = innerWidth / Math.max(models.length, 1);
  const barGap = 8;
  const barWidth = Math.min(26, (groupWidth - 28 - (shownLanguages.length - 1) * barGap) / shownLanguages.length);
  const bestBilingual =
    [...models].sort((left, right) => bilingualAverage(right) - bilingualAverage(left))[0] ?? null;

  const grid = yTicks
    .map((tick) => {
      const y = padding.top + innerHeight - (tick / 100) * innerHeight;
      return `
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="rgba(97, 112, 134, 0.16)" />
        <text class="language-tick-label" x="${padding.left - 10}" y="${y + 4}" text-anchor="end">${tick}</text>
      `;
    })
    .join("");

  const bars = models
    .map((model, modelIndex) => {
      const totalBarWidth = shownLanguages.length * barWidth + (shownLanguages.length - 1) * barGap;
      const groupX = padding.left + modelIndex * groupWidth + (groupWidth - totalBarWidth) / 2;
      const labelX = padding.left + modelIndex * groupWidth + groupWidth / 2;
      const dimmed = state.focusModel !== "all" && state.focusModel !== model.model_name;
      const opacity = dimmed ? 0.35 : 0.95;

      const modelBars = shownLanguages
        .map((language, languageIndex) => {
          const value = model.languageAccuracy?.[language] ?? 0;
          const barHeight = (value / 100) * innerHeight;
          const x = groupX + languageIndex * (barWidth + barGap);
          const y = padding.top + innerHeight - barHeight;
          const tooltip = `
            <strong>${escapeHtml(model.model_name)}</strong><br />
            Language: ${escapeHtml(getLanguageLabel(language))}<br />
            Accuracy: ${formatPct(value)}
          `;

          return `
            <rect
              class="language-bar clickable"
              x="${x}"
              y="${y}"
              width="${barWidth}"
              height="${Math.max(barHeight, 4)}"
              rx="8"
              fill="${LANGUAGE_COLORS[language] ?? model.color}"
              fill-opacity="${opacity}"
              data-focus-model="${escapeHtml(model.model_name)}"
              data-tooltip="${escapeAttribute(tooltip)}"
            ></rect>
          `;
        })
        .join("");

      return `
        <g>
          ${modelBars}
          <text class="language-x-label" x="${labelX}" y="${height - 28}" text-anchor="middle">${escapeHtml(
            compactName(model.model_name)
          )}</text>
        </g>
      `;
    })
    .join("");

  elements.languageChart.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Cross-Lingual View</p>
        <h3 class="card-title">${escapeHtml(
          bestBilingual?.model_name ?? "N/A"
        )} is strongest across Arabic and English</h3>
      </div>
      <p class="metric-note">Arabic is red, English is blue, with French and Spanish beside them.</p>
    </div>
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Per-model accuracy comparison across benchmark languages">
      ${grid}
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="rgba(20, 36, 58, 0.28)" />
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(20, 36, 58, 0.28)" />
      <text class="language-axis-label" x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">Accuracy (%)</text>
      ${bars}
    </svg>
    ${renderLanguageLegend(shownLanguages)}
  `;
}

function renderScatterPlot() {
  const models = state.prepared.models;
  if (!models.length) {
    elements.scatterChart.innerHTML = renderEmpty("No model data available.");
    return;
  }

  const width = 940;
  const height = 420;
  const padding = { top: 36, right: 52, bottom: 54, left: 62 };
  const xRange = expandRange(getRange(models.map((model) => model.avg_speed_tests_per_sec ?? 0)), 0.1);
  const yRange = expandRange(getRange(models.map((model) => model.macro_intent_accuracy_pct ?? 0)), 0.1);
  const maxRadius = Math.max(...models.map((model) => getScatterRadius(model)));
  const pointInset = maxRadius + 8;
  const plotLeft = padding.left + pointInset;
  const plotRight = width - padding.right - pointInset;
  const plotTop = padding.top + pointInset;
  const plotBottom = height - padding.bottom - pointInset;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;
  const xMid = plotLeft + plotWidth / 2;
  const yMid = plotTop + plotHeight / 2;
  const xTicks = buildAxisTicks(xRange);
  const yTicks = buildAxisTicks(yRange);

  const grid = [
    ...yTicks.map(({ ratio }) => {
      const y = plotTop + (1 - ratio) * plotHeight;
      return `<line x1="${plotLeft}" y1="${y}" x2="${plotRight}" y2="${y}" stroke="rgba(97, 112, 134, 0.18)" />`;
    }),
    ...xTicks.map(({ ratio }) => {
      const x = plotLeft + ratio * plotWidth;
      return `<line x1="${x}" y1="${plotTop}" x2="${x}" y2="${plotBottom}" stroke="rgba(97, 112, 134, 0.12)" />`;
    }),
  ].join("");

  const tickLabels = [
    ...xTicks.map(({ value, ratio }) => {
      const x = plotLeft + ratio * plotWidth;
      return `<text class="scatter-tick-label" x="${x}" y="${plotBottom + 24}" text-anchor="middle">${escapeHtml(
        formatAxisValue(value, "speed")
      )}</text>`;
    }),
    ...yTicks.map(({ value, ratio }) => {
      const y = plotTop + (1 - ratio) * plotHeight;
      return `<text class="scatter-tick-label" x="${plotLeft - 10}" y="${y + 4}" text-anchor="end">${escapeHtml(
        formatAxisValue(value, "pct")
      )}</text>`;
    }),
  ].join("");

  const points = models
    .map((model) => {
      const x = plotLeft + normalize(model.avg_speed_tests_per_sec ?? 0, xRange) * plotWidth;
      const y = plotTop + (1 - normalize(model.macro_intent_accuracy_pct ?? 0, yRange)) * plotHeight;
      const radius = getScatterRadius(model);
      const dimmed = state.focusModel !== "all" && state.focusModel !== model.model_name;
      const opacity = dimmed ? 0.35 : 0.88;
      const tooltip = `
        <strong>${escapeHtml(model.model_name)}</strong><br />
        Macro intent accuracy: ${formatPct(model.macro_intent_accuracy_pct)}<br />
        Speed: ${formatSpeed(model.avg_speed_tests_per_sec)}<br />
        Avg latency: ${formatMs(model.avg_latency_ms)}<br />
        Coverage: ${formatPct(model.coverage_above_threshold_pct)}
      `;

      return `
        <g class="scatter-point clickable" data-focus-model="${escapeHtml(model.model_name)}" data-tooltip="${escapeAttribute(
          tooltip
        )}" style="opacity:${opacity}">
          <circle cx="${x}" cy="${y}" r="${radius}" fill="${model.color}" fill-opacity="0.92" stroke="#ffffff" stroke-width="3"></circle>
          <text x="${x}" y="${y + radius + 18}" text-anchor="middle">${escapeHtml(compactName(model.model_name))}</text>
        </g>
      `;
    })
    .join("");

  elements.scatterChart.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Tradeoff View</p>
        <h3 class="card-title">Higher is better, further right is faster</h3>
      </div>
      <p class="metric-note">Bubble size shows confident coverage.</p>
    </div>
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Macro intent accuracy versus classification speed">
      ${grid}
      <line x1="${plotLeft}" y1="${plotBottom}" x2="${plotRight}" y2="${plotBottom}" stroke="rgba(20, 36, 58, 0.28)" />
      <line x1="${plotLeft}" y1="${plotTop}" x2="${plotLeft}" y2="${plotBottom}" stroke="rgba(20, 36, 58, 0.28)" />
      <line x1="${xMid}" y1="${plotTop}" x2="${xMid}" y2="${plotBottom}" stroke="rgba(97, 112, 134, 0.18)" stroke-dasharray="6 6" />
      <line x1="${plotLeft}" y1="${yMid}" x2="${plotRight}" y2="${yMid}" stroke="rgba(97, 112, 134, 0.18)" stroke-dasharray="6 6" />
      ${tickLabels}
      <text class="scatter-axis-label" x="${width / 2}" y="${height - 10}" text-anchor="middle">Speed (tests/sec)</text>
      <text class="scatter-axis-label" x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">Macro intent accuracy (%)</text>
      <text class="quadrant-label" x="${plotLeft + 16}" y="${plotTop + 18}">Slow + Strong</text>
      <text class="quadrant-label" x="${xMid + 16}" y="${plotTop + 18}">Fast + Strong</text>
      <text class="quadrant-label" x="${plotLeft + 16}" y="${plotBottom - 14}">Slow + Weak</text>
      <text class="quadrant-label" x="${xMid + 16}" y="${plotBottom - 14}">Fast + Weak</text>
      ${points}
    </svg>
    ${renderLegend(models)}
  `;
}

function renderCostChart() {
  const { metadata, winners } = state.prepared;
  const models = [...state.prepared.models].sort(
    (a, b) => (a.benchmark_total_runtime_s ?? 0) - (b.benchmark_total_runtime_s ?? 0)
  );

  if (!models.length) {
    elements.costChart.innerHTML = renderEmpty("No runtime cost data available.");
    return;
  }

  const maxTotal = Math.max(...models.map((model) => model.benchmark_total_runtime_s ?? 0), 1);
  const axisTicks = buildAxisTicks([0, maxTotal]);

  const axis = `
    <div class="cost-axis">
      ${axisTicks
        .map(
          (tick) => `
            <span>${escapeHtml(formatAxisValue(tick.value, "seconds"))}</span>
          `
        )
        .join("")}
    </div>
  `;

  elements.costChart.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Operational View</p>
        <h3 class="card-title">${escapeHtml(
          winners.cheapest?.model_name ?? "N/A"
        )} is the lowest-cost model on this device</h3>
      </div>
      <p class="metric-note">
        Proxy only: load time + average inference time across ${metadata.num_tests ?? 0} tests on
        ${escapeHtml(String(metadata.device ?? "unknown"))}, averaged over ${metadata.repeats ?? 1}
        run${metadata.repeats === 1 ? "" : "s"} per test.
      </p>
    </div>
    ${axis}
    <div class="cost-grid">
      ${models.map((model) => renderCostRow(model, maxTotal)).join("")}
    </div>
    <div class="chart-legend">
      <span class="legend-item">
        <span class="legend-swatch cost-load-swatch"></span>
        <span>Load time</span>
      </span>
      <span class="legend-item">
        <span class="legend-swatch cost-inference-swatch"></span>
        <span>Inference time</span>
      </span>
    </div>
  `;
}

function renderCostRow(model, maxTotal) {
  const focused = state.focusModel === model.model_name;
  const dimmed = state.focusModel !== "all" && state.focusModel !== model.model_name;
  const totalRuntime = model.benchmark_total_runtime_s ?? 0;
  const loadRuntime = model.load_time_s ?? 0;
  const inferenceRuntime = model.benchmark_inference_runtime_s ?? 0;
  const barWidth = maxTotal > 0 ? (totalRuntime / maxTotal) * 100 : 0;
  const loadShare = totalRuntime > 0 ? (loadRuntime / totalRuntime) * 100 : 0;
  const inferenceShare = totalRuntime > 0 ? (inferenceRuntime / totalRuntime) * 100 : 0;
  const tooltip = `
    <strong>${escapeHtml(model.model_name)}</strong><br />
    Runtime cost proxy: ${formatSeconds(totalRuntime)}<br />
    Load time: ${formatSeconds(loadRuntime)}<br />
    Inference time: ${formatSeconds(inferenceRuntime)}<br />
    Relative cost: ${model.runtime_cost_multiple.toFixed(2)}x cheapest
  `;

  return `
    <article
      class="cost-row clickable ${focused ? "focused" : ""} ${dimmed ? "dimmed" : ""}"
      data-focus-model="${escapeHtml(model.model_name)}"
      data-tooltip="${escapeAttribute(tooltip)}"
    >
      <div class="cost-row-head">
        <div class="cost-row-title">
          <div class="leaderboard-tags">
            <span class="rank-badge">#${model.rank}</span>
            <span class="trait-badge">${escapeHtml(compactName(model.model_name))}</span>
          </div>
          <h3>${escapeHtml(model.model_name)}</h3>
        </div>
        <div class="cost-total">
          <strong>${escapeHtml(formatSeconds(totalRuntime))}</strong>
          <span class="muted">${escapeHtml(`${model.runtime_cost_multiple.toFixed(2)}x cheapest`)}</span>
        </div>
      </div>
      <div class="cost-track">
        <div class="cost-stack" style="width:${barWidth}%">
          <span class="cost-segment load" style="width:${loadShare}%"></span>
          <span class="cost-segment inference" style="width:${inferenceShare}%"></span>
        </div>
      </div>
      <div class="cost-meta">
        <span class="mini-pill">Load ${escapeHtml(formatSeconds(loadRuntime))}</span>
        <span class="mini-pill">Inference ${escapeHtml(formatSeconds(inferenceRuntime))}</span>
        <span class="mini-pill">Speed ${escapeHtml(formatSpeed(model.avg_speed_tests_per_sec))}</span>
      </div>
    </article>
  `;
}

function renderHeatmap() {
  const models = state.prepared.models;
  const categories = state.prepared.categories;

  if (!models.length || !categories.length) {
    elements.heatmap.innerHTML = renderEmpty("No category data available.");
    return;
  }

  const header = `
    <div class="heatmap-row" style="--cols:${models.length}">
      <div class="heatmap-header">Category</div>
      ${models
        .map(
          (model) => `
            <button class="heatmap-header clickable" type="button" data-focus-model="${escapeHtml(model.model_name)}">
              ${escapeHtml(compactName(model.model_name))}
            </button>
          `
        )
        .join("")}
    </div>
  `;

  const rows = categories
    .filter((category) => state.category === "all" || category === state.category)
    .map((category) => {
      return `
        <div class="heatmap-row" style="--cols:${models.length}">
          <button class="heatmap-label clickable" type="button" data-filter-category="${escapeHtml(category)}">
            ${escapeHtml(category)}
          </button>
          ${models
            .map((model) => {
              const value = model.categoryAccuracy[category] ?? 0;
              const tooltip = `
                <strong>${escapeHtml(model.model_name)}</strong><br />
                Category: ${escapeHtml(category)}<br />
                Accuracy: ${formatPct(value)}
              `;
              return `
                <div
                  class="heatmap-cell"
                  style="background:${heatColor(value)}"
                  data-tooltip="${escapeAttribute(tooltip)}"
                  data-focus-model="${escapeHtml(model.model_name)}"
                  data-filter-category="${escapeHtml(category)}"
                >
                  ${formatPct(value)}
                </div>
              `;
            })
            .join("")}
        </div>
      `;
    })
    .join("");

  elements.heatmap.innerHTML = `
    <div class="leaderboard-toolbar">
      <div>
        <p class="section-kicker">Diagnostic View</p>
        <h3 class="card-title">Where each model is strong and where it breaks</h3>
      </div>
      <p class="metric-note">Green means better category accuracy.</p>
    </div>
    <div class="heatmap-grid">
      ${header}
      ${rows}
    </div>
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
      <p class="metric-note">${state.showFailuresOnly ? "Showing only failed examples." : "Showing all matching examples."}</p>
    </div>
    <div class="test-accordion">
      ${tests.map((test) => renderTestCase(test)).join("")}
    </div>
  `;
}

function renderTestCase(test) {
  const results = getFocusedResults(test.modelResults);
  const tone = test.highConfidenceWrong > 0 ? "bad" : test.wrongCount > 0 ? "warn" : "good";
  return `
    <details class="test-case">
      <summary class="test-summary">
        <div class="test-summary-head">
          <span class="rank-badge">${escapeHtml(test.id)}</span>
          <span class="status-chip ${tone}">${test.correctCount}/${test.modelResults.length} correct</span>
          <span class="status-chip info">${escapeHtml(getLanguageLabel(test.language))}</span>
          <span class="status-chip info">${escapeHtml(test.category)}</span>
        </div>
        <div class="sentence-block rtl">${escapeHtml(test.text)}</div>
        <div class="test-summary-meta">
          <span>Expected: ${escapeHtml(test.expected)}</span>
          <span>High-confidence wrong: ${test.highConfidenceWrong}</span>
        </div>
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
  const focused = state.focusModel === result.model_name;
  const outcomeClass = result.correct ? (result.confident ? "good" : "info") : result.confident ? "bad" : "warn";
  return `
    <article class="test-model-card ${focused ? "focused" : ""}">
      <div class="leaderboard-tags">
        <span class="rank-badge">#${result.rank}</span>
        <span class="status-chip ${outcomeClass}">${result.correct ? "Correct" : "Wrong"}</span>
      </div>
      <h4>${escapeHtml(result.model_name)}</h4>
      <div class="test-model-metrics">
        ${renderMetricBlock("Score", result.score.toFixed(3))}
        ${renderMetricBlock("Margin", result.margin.toFixed(3))}
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
        <p class="section-kicker">Risk First</p>
        <h3 class="card-title">${failures.length} failure${failures.length === 1 ? "" : "s"} in view</h3>
      </div>
      <p class="metric-note">Sorted by high-confidence errors, then by low-margin misses.</p>
    </div>
    <div class="failure-grid">
      ${failures.map((failure) => renderFailureCard(failure)).join("")}
    </div>
  `;
}

function renderFailureCard(failure) {
  const severityClass = failure.confident ? "bad" : "warn";
  return `
    <article class="failure-card">
      <div class="leaderboard-tags">
        <span class="rank-badge">#${failure.rank}</span>
        <span class="status-chip ${severityClass}">${failure.confident ? "High risk" : "Low margin"}</span>
        <span class="status-chip info">${escapeHtml(getLanguageLabel(failure.language))}</span>
        <span class="status-chip info">${escapeHtml(failure.category)}</span>
      </div>
      <h4>${escapeHtml(failure.model_name)}</h4>
      <div class="sentence-block rtl">${escapeHtml(failure.text)}</div>
      <div class="test-model-metrics">
        ${renderMetricBlock("Score", failure.score.toFixed(3))}
        ${renderMetricBlock("Margin", failure.margin.toFixed(3))}
        ${renderMetricBlock("Latency", formatMs(failure.latency_ms))}
      </div>
      <div class="failure-badges">
        <span class="status-chip bad">Predicted: ${escapeHtml(shortLabel(failure.pred_label))}</span>
        <span class="status-chip info">Expected: ${escapeHtml(shortLabel(failure.correct_label))}</span>
      </div>
    </article>
  `;
}

function renderFooter() {
  const { metadata, thresholds } = state.prepared;
  elements.footerMeta.innerHTML = `
    <div>
      <p class="section-kicker">Benchmark Metadata</p>
      <h3 class="card-title">Minimal report, full dataset underneath</h3>
      <p>
        This page now keeps only the essential views up front and pushes sentence-level evidence and failure details to the bottom.
      </p>
    </div>
    <div class="meta-chip-list">
      <span class="mini-pill">Device ${escapeHtml(String(metadata.device ?? "unknown"))}</span>
      <span class="mini-pill">${metadata.num_models ?? 0} models</span>
      <span class="mini-pill">${metadata.num_tests ?? 0} tests</span>
      <span class="mini-pill">${metadata.repeats ?? 1} repeats</span>
      <span class="mini-pill">Score ${thresholds.score ?? 0}</span>
      <span class="mini-pill">Margin ${thresholds.margin ?? 0}</span>
    </div>
  `;
}

function seedStaticControls() {
  elements.sortMetric.innerHTML = SORT_OPTIONS.map(
    (option) => `<option value="${option.value}">${escapeHtml(option.label)}</option>`
  ).join("");
}

function syncDynamicControls() {
  const { categories, models } = state.prepared;
  elements.categoryFilter.innerHTML = [
    `<option value="all">All Categories</option>`,
    ...categories.map((category) => `<option value="${category}">${escapeHtml(category)}</option>`),
  ].join("");
  elements.focusModel.innerHTML = [
    `<option value="all">All Models</option>`,
    ...models.map((model) => `<option value="${escapeAttribute(model.model_name)}">${escapeHtml(model.model_name)}</option>`),
  ].join("");

  elements.categoryFilter.value = state.category;
  elements.focusModel.value = state.focusModel;
  elements.sortMetric.value = state.sortMetric;
  elements.showFailuresOnly.checked = state.showFailuresOnly;
}

function getVisibleTests() {
  return (state.prepared?.tests ?? []).filter((test) => {
    const categoryMatch = state.category === "all" || test.category === state.category;
    const failureMatch = !state.showFailuresOnly || test.wrongCount > 0;
    return categoryMatch && failureMatch;
  });
}

function getFocusedResults(results) {
  if (state.focusModel === "all") {
    return results;
  }
  return results.filter((result) => result.model_name === state.focusModel);
}

function getVisibleFailures() {
  const failures = getVisibleTests().flatMap((test) =>
    test.modelResults
      .filter((result) => !result.correct)
      .map((result) => ({
        ...result,
        text: test.text,
      }))
  );

  const filtered =
    state.focusModel === "all"
      ? failures
      : failures.filter((failure) => failure.model_name === state.focusModel);

  return filtered.sort((a, b) => {
    const confidenceDelta = Number(b.confident) - Number(a.confident);
    if (confidenceDelta !== 0) {
      return confidenceDelta;
    }
    return a.margin - b.margin || (b.score ?? 0) - (a.score ?? 0);
  });
}

function sortModels(models, metric) {
  const sorted = [...models];
  sorted.sort((a, b) => {
    if (metric === "ranking") {
      return a.rank - b.rank;
    }
    if (metric === "avg_latency_ms" || metric === "benchmark_total_runtime_s" || metric === "riskScore") {
      return (a[metric] ?? 0) - (b[metric] ?? 0);
    }
    return (b[metric] ?? 0) - (a[metric] ?? 0);
  });
  return sorted;
}

function renderMetricBlock(label, value, progressValue = null) {
  return `
    <div class="metric-block">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong class="metric-value">${escapeHtml(value)}</strong>
      ${
        progressValue == null
          ? ""
          : `<div class="progress"><span style="--value:${clamp(progressValue, 0, 100)}%"></span></div>`
      }
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

function renderLegend(models) {
  return `
    <div class="chart-legend">
      ${models
        .map(
          (model) => `
            <button class="legend-item clickable" type="button" data-focus-model="${escapeHtml(model.model_name)}">
              <span class="legend-swatch" style="background:${model.color}"></span>
              <span>${escapeHtml(compactName(model.model_name))}</span>
            </button>
          `
        )
        .join("")}
    </div>
  `;
}

function renderLanguageLegend(languages) {
  return `
    <div class="chart-legend">
      ${languages
        .map(
          (language) => `
            <span class="legend-item">
              <span class="legend-swatch" style="background:${LANGUAGE_COLORS[language] ?? "#617086"}"></span>
              <span>${escapeHtml(getLanguageLabel(language))}</span>
            </span>
          `
        )
        .join("")}
    </div>
  `;
}

function getScatterRadius(model) {
  return 14 + ((model.coverage_above_threshold_pct ?? 0) / 100) * 24;
}

function renderEmpty(message) {
  return `
    <div class="empty-state">
      <p class="muted">${escapeHtml(message)}</p>
    </div>
  `;
}

function getStatusClass(result) {
  if (result.correct && result.confident) {
    return "status-good-high";
  }
  if (result.correct) {
    return "status-good-low";
  }
  if (result.confident) {
    return "status-bad-high";
  }
  return "status-bad-low";
}

function pickCategoryEdge(entries, highest) {
  if (!entries.length) {
    return { name: "n/a", value: 0 };
  }
  const target = entries.reduce((best, current) =>
    highest ? (current[1] > best[1] ? current : best) : current[1] < best[1] ? current : best
  );
  return { name: target[0], value: target[1] ?? 0 };
}

function compactName(name = "") {
  return name
    .replace("multilingual-", "")
    .replace("-multilingual", "")
    .replace("MiniLM-L12-", "MiniLM ")
    .replace("sentence-transformers/", "")
    .replace("intfloat/", "")
    .replace("aubmindlab/", "")
    .replace("UBC-NLP/", "");
}

function shortLabel(label = "") {
  return label.length > 34 ? `${label.slice(0, 34)}...` : label;
}

function getInitials(name = "") {
  return name
    .split(/[\s/-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

function heatColor(value) {
  const clamped = clamp(value, 0, 100);
  const hue = (clamped / 100) * 120;
  return `linear-gradient(135deg, hsla(${hue}, 65%, 42%, 0.96), hsla(${Math.max(hue - 16, 0)}, 70%, 34%, 0.94))`;
}

function getRange(values) {
  if (!values.length) {
    return [0, 1];
  }
  return [Math.min(...values), Math.max(...values)];
}

function expandRange(range, paddingRatio = 0.1) {
  const [min, max] = range;
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (min === max) {
    const delta = Math.max(Math.abs(min) * paddingRatio, 1);
    return [min - delta, max + delta];
  }
  const delta = (max - min) * paddingRatio;
  return [min - delta, max + delta];
}

function buildAxisTicks(range, count = 5) {
  const [min, max] = range;
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [];
  }
  if (max === min) {
    return [{ value: min, ratio: 0.5 }];
  }
  return Array.from({ length: count }, (_, index) => {
    const ratio = index / (count - 1);
    return {
      value: min + (max - min) * ratio,
      ratio,
    };
  });
}

function computeAccuracyByField(results, field, orderedValues = []) {
  const values = orderedValues.length ? orderedValues : [...new Set(results.map((result) => result[field]).filter(Boolean))];
  return Object.fromEntries(
    values.map((value) => {
      const group = results.filter((result) => result[field] === value);
      const accuracy = group.length ? (group.filter((result) => result.correct).length / group.length) * 100 : null;
      return [value, accuracy == null ? null : round(accuracy, 6)];
    })
  );
}

function getOrderedLanguages(languages) {
  const unique = [...new Set((languages ?? []).filter(Boolean))];
  return unique.sort((left, right) => {
    const leftIndex = LANGUAGE_ORDER.indexOf(left);
    const rightIndex = LANGUAGE_ORDER.indexOf(right);
    const safeLeft = leftIndex === -1 ? LANGUAGE_ORDER.length : leftIndex;
    const safeRight = rightIndex === -1 ? LANGUAGE_ORDER.length : rightIndex;
    return safeLeft - safeRight || left.localeCompare(right);
  });
}

function getLanguageLabel(language) {
  return LANGUAGE_LABELS[language] ?? String(language ?? "").toUpperCase();
}

function bilingualAverage(model) {
  const arabic = model.languageAccuracy?.ar ?? 0;
  const english = model.languageAccuracy?.en ?? 0;
  if (model.languageAccuracy?.ar == null && model.languageAccuracy?.en == null) {
    return 0;
  }
  if (model.languageAccuracy?.ar == null) {
    return english;
  }
  if (model.languageAccuracy?.en == null) {
    return arabic;
  }
  return (arabic + english) / 2;
}

function normalize(value, range) {
  const [min, max] = range;
  if (max === min) {
    return 1;
  }
  return clamp((value - min) / (max - min), 0, 1);
}

function normalizeInverse(value, range) {
  return 1 - normalize(value, range);
}

function average(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatPct(value) {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  return `${Number(value).toFixed(1)}%`;
}

function formatMs(value) {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  return `${Number(value).toFixed(1)} ms`;
}

function formatSpeed(value) {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  return `${Number(value).toFixed(2)} tests/sec`;
}

function formatSeconds(value) {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  return `${Number(value).toFixed(value >= 10 ? 1 : 2)} s`;
}

function formatAxisValue(value, kind) {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  if (kind === "speed") {
    return Number(value).toFixed(value >= 10 ? 0 : 1);
  }
  if (kind === "seconds") {
    return Number(value).toFixed(value >= 10 ? 0 : 1);
  }
  return Number(value).toFixed(0);
}

function formatSigned(value) {
  const safe = Number(value ?? 0);
  const prefix = safe > 0 ? "+" : "";
  return `${prefix}${safe.toFixed(1)} pts`;
}

function latencyToThroughput(latencyMs) {
  const safeLatency = Number(latencyMs ?? 0);
  if (!Number.isFinite(safeLatency) || safeLatency <= 0) {
    return 0;
  }
  return 1000 / safeLatency;
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
