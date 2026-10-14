/* Red Wine MLR dashboard — loads results.json and renders charts with Chart.js */

const WINE = "#7b1e2b";
const WINE_LIGHT = "#a83246";
const ACCENT = "#c6803f";
const GOOD = "#4c9a6a";
const NAVY = "#2b3a67";

const FIGURES = [
  ["images/01_target_distribution.png", "品質分數分佈"],
  ["images/02_correlation_heatmap.png", "相關性熱力圖"],
  ["images/03_feature_selection.png", "特徵重要度 (SelectKBest)"],
  ["images/04_model_comparison.png", "模型比較 (RMSE / R²)"],
  ["images/05_residual_diagnostics.png", "殘差診斷"],
  ["images/06_prediction_intervals.png", "預測區間圖"],
  ["images/07_actual_vs_predicted.png", "實際 vs 預測"],
];

Chart.defaults.font.family =
  '"Segoe UI","PingFang TC","Microsoft JhengHei",system-ui,sans-serif';
Chart.defaults.color = "#5c5257";

async function main() {
  let data;
  try {
    const res = await fetch("./data/results.json", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    data = await res.json();
  } catch (e) {
    document.getElementById("app").insertAdjacentHTML(
      "afterbegin",
      `<div class="error-banner">無法載入 data/results.json（${e.message}）。
       請先執行主程式 <code>python 7114056032_hw2.py</code> 產生資料。</div>`
    );
    return;
  }

  renderSource(data);
  renderKpis(data);
  renderDistChart(data);
  renderCorrChart(data);
  renderFsChart(data);
  renderConsensus(data);
  renderModelCharts(data);
  renderMetricsTable(data);
  renderActualVsPredicted(data);
  renderPredictionInterval(data);
  renderCoefChart(data);
  renderDemo(data);
  renderGallery();
}

function renderSource(d) {
  const pill = document.getElementById("sourcePill");
  const label = d.data_source === "web" ? "🌐 自網路即時擷取" : "💾 本地檔案";
  pill.textContent = `${label} · ${d.n_rows} 筆 · ${d.n_features} 特徵`;
  pill.title = d.data_url || "";
}

function renderKpis(d) {
  const best = (d.metrics || []).find((m) => m.model === d.best_model) || {};
  const cards = [
    { label: "資料筆數", value: d.n_rows, sub: `缺失值 ${d.missing_total}` },
    { label: "特徵數", value: d.n_features, sub: "理化檢測" },
    { label: "最佳模型", value: d.best_model, sub: "依測試 R²" },
    { label: "最佳 R²", value: fmt(best.r2), sub: `RMSE ${fmt(best.rmse)}` },
  ];
  document.getElementById("kpiCards").innerHTML = cards
    .map(
      (c) => `<div class="card">
        <div class="label">${c.label}</div>
        <div class="value">${c.value}</div>
        <div class="sub">${c.sub}</div>
      </div>`
    )
    .join("");
}

function renderDistChart(d) {
  new Chart(document.getElementById("distChart"), {
    type: "bar",
    data: {
      labels: d.target_distribution.labels,
      datasets: [
        {
          label: "樣本數",
          data: d.target_distribution.counts,
          backgroundColor: WINE,
          borderRadius: 6,
        },
      ],
    },
    options: baseOpts("品質分數分佈", { x: "品質分數", y: "樣本數" }),
  });
}

function renderCorrChart(d) {
  const c = d.target_correlation;
  const colors = c.signed_r.map((v) => (v >= 0 ? GOOD : WINE_LIGHT));
  new Chart(document.getElementById("corrChart"), {
    type: "bar",
    data: {
      labels: c.features,
      datasets: [
        { label: "與品質的相關係數 r", data: c.signed_r, backgroundColor: colors, borderRadius: 5 },
      ],
    },
    options: {
      ...baseOpts("特徵與品質相關性", { x: "r", y: "" }),
      indexAxis: "y",
    },
  });
}

function renderFsChart(d) {
  const fs = d.feature_selection;
  // show ascending so the strongest is on top for a horizontal bar
  const pairs = fs.skb_features
    .map((f, i) => [f, fs.skb_scores[i]])
    .sort((a, b) => a[1] - b[1]);
  new Chart(document.getElementById("fsChart"), {
    type: "bar",
    data: {
      labels: pairs.map((p) => p[0]),
      datasets: [{ label: "F-score", data: pairs.map((p) => p[1]), backgroundColor: NAVY, borderRadius: 5 }],
    },
    options: { ...baseOpts("SelectKBest F-score", { x: "F-score", y: "" }), indexAxis: "y" },
  });
}

function renderConsensus(d) {
  const box = document.getElementById("consensusChips");
  const fs = d.feature_selection;
  box.innerHTML =
    `<span class="chip" style="background:#eef7f1;color:#2f6b48;border-color:#cfe6d8">共識特徵（≥2 法）</span>` +
    fs.consensus.map((f) => `<span class="chip">${f}</span>`).join("");
}

function renderModelCharts(d) {
  const labels = d.metrics.map((m) => m.model);
  const r2 = d.metrics.map((m) => m.r2);
  const rmse = d.metrics.map((m) => m.rmse);
  const hl = d.metrics.map((m) => (m.model === d.best_model ? GOOD : WINE_LIGHT));

  new Chart(document.getElementById("r2Chart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "R² (越高越好)", data: r2, backgroundColor: hl, borderRadius: 5 }] },
    options: wrapTicks(baseOpts("各模型測試 R²", { x: "", y: "R²" })),
  });
  new Chart(document.getElementById("rmseChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "RMSE (越低越好)", data: rmse, backgroundColor: WINE, borderRadius: 5 }] },
    options: wrapTicks(baseOpts("各模型測試 RMSE", { x: "", y: "RMSE" })),
  });
}

function renderMetricsTable(d) {
  const tb = document.querySelector("#metricsTable tbody");
  tb.innerHTML = d.metrics
    .map((m) => {
      const cls = m.model === d.best_model ? ' class="best"' : "";
      return `<tr${cls}>
        <td>${m.model}</td><td>${m.n_feat}</td><td>${fmt(m.mae)}</td>
        <td>${fmt(m.mse)}</td><td>${fmt(m.rmse)}</td><td>${fmt(m.r2)}</td>
        <td>${fmt(m.adj_r2)}</td><td>${fmt(m.cv_r2)}</td></tr>`;
    })
    .join("");
}

function renderActualVsPredicted(d) {
  const avp = d.actual_vs_predicted;
  const points = avp.actual.map((a, i) => ({ x: a, y: avp.predicted[i] }));
  const lo = Math.min(...avp.actual, ...avp.predicted) - 0.5;
  const hi = Math.max(...avp.actual, ...avp.predicted) + 0.5;
  new Chart(document.getElementById("avpChart"), {
    type: "scatter",
    data: {
      datasets: [
        { label: "測試樣本", data: points, backgroundColor: "rgba(123,30,43,0.55)", pointRadius: 4 },
        {
          label: "完美預測線",
          type: "line",
          data: [{ x: lo, y: lo }, { x: hi, y: hi }],
          borderColor: ACCENT,
          borderDash: [6, 6],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      ...baseOpts("Actual vs Predicted Quality", { x: "Actual Profit (Quality)", y: "Predicted Profit (Quality)" }),
      scales: {
        x: { title: { display: true, text: "Actual Profit (Quality)" }, min: lo, max: hi },
        y: { title: { display: true, text: "Predicted Profit (Quality)" }, min: lo, max: hi },
      },
    },
  });
}

function renderPredictionInterval(d) {
  const pi = d.prediction_interval;
  const sc = d.scatter_alcohol;
  const line = (arr) => pi.x.map((x, i) => ({ x, y: arr[i] }));
  new Chart(document.getElementById("piChart"), {
    type: "scatter",
    data: {
      datasets: [
        { label: "觀測值", data: sc.x.map((x, i) => ({ x, y: sc.y[i] })), backgroundColor: "rgba(123,30,43,0.18)", pointRadius: 3 },
        { label: "OLS 擬合", type: "line", data: line(pi.mean), borderColor: NAVY, borderWidth: 2, pointRadius: 0 },
        { label: "95% 預測區間上界", type: "line", data: line(pi.pi_upper), borderColor: "rgba(198,128,63,0.5)", backgroundColor: "rgba(198,128,63,0.15)", pointRadius: 0, fill: "+1" },
        { label: "95% 預測區間下界", type: "line", data: line(pi.pi_lower), borderColor: "rgba(198,128,63,0.5)", pointRadius: 0, fill: false },
        { label: "95% 信賴區間上界", type: "line", data: line(pi.ci_upper), borderColor: "rgba(43,58,103,0.6)", backgroundColor: "rgba(43,58,103,0.22)", pointRadius: 0, fill: "+1" },
        { label: "95% 信賴區間下界", type: "line", data: line(pi.ci_lower), borderColor: "rgba(43,58,103,0.6)", pointRadius: 0, fill: false },
      ],
    },
    options: {
      ...baseOpts("Quality vs Alcohol — 95% CI & PI", { x: "Alcohol (% vol)", y: "Quality" }),
      plugins: {
        legend: { labels: { filter: (i) => !i.text.includes("下界") } },
        title: { display: false },
      },
    },
  });
}

function renderCoefChart(d) {
  if (!d.coefficients) return;
  const c = d.coefficients;
  const pairs = c.features.map((f, i) => [f, c.values[i]]).sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]));
  const colors = pairs.map((p) => (p[1] >= 0 ? GOOD : WINE_LIGHT));
  new Chart(document.getElementById("coefChart"), {
    type: "bar",
    data: { labels: pairs.map((p) => p[0]), datasets: [{ label: "標準化係數", data: pairs.map((p) => p[1]), backgroundColor: colors, borderRadius: 5 }] },
    options: { ...baseOpts("標準化回歸係數", { x: "係數", y: "" }), indexAxis: "y" },
  });
}

function renderDemo(d) {
  if (!d.demo_predictions) return;
  document.getElementById("demoBox").innerHTML = d.demo_predictions
    .map(
      (p) => `<div class="d">
        <div class="a">實際 ${p.actual}</div>
        <div class="p">${p.predicted}</div>
        <div class="a">預測</div>
      </div>`
    )
    .join("");
}

function renderGallery() {
  document.getElementById("gallery").innerHTML = FIGURES.map(
    ([src, cap]) =>
      `<figure><img loading="lazy" src="./${src}" alt="${cap}"/><figcaption>${cap}</figcaption></figure>`
  ).join("");
}

/* helpers */
function fmt(v) {
  return typeof v === "number" ? v.toFixed(4).replace(/\.?0+$/, (m) => (m.includes(".") ? "" : m)) : v;
}
function baseOpts(title, axes) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { title: { display: true, text: title }, legend: { display: true } },
    scales: {
      x: { title: { display: !!axes.x, text: axes.x } },
      y: { title: { display: !!axes.y, text: axes.y } },
    },
  };
}
function wrapTicks(opts) {
  opts.scales.x.ticks = { maxRotation: 40, minRotation: 20, autoSkip: false, font: { size: 10 } };
  return opts;
}

main();
