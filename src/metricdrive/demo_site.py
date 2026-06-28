from __future__ import annotations

import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from metricdrive.hard_negatives import (
    hard_negative_payload,
    run_hard_negative_experiment,
)
from metricdrive.learning import run_learning_experiment
from metricdrive.metrics import ranked_scores
from metricdrive.preferences import generate_preferences
from metricdrive.rl_alignment import rl_alignment_payload, run_rl_alignment
from metricdrive.scenario import Scenario
from metricdrive.vlm_examples import generate_vlm_examples
from metricdrive.visualize import scenario_svg


DEFAULT_DEMO_OUTPUT = "docs/demo"
DEFAULT_DEMO_EPOCHS = 20


def generate_demo_site(
    scenarios: tuple[Scenario, ...],
    output_dir: str | Path = DEFAULT_DEMO_OUTPUT,
    epochs: int = DEFAULT_DEMO_EPOCHS,
) -> dict[str, object]:
    """Export a dependency-free static demo for GitHub Pages."""

    output = Path(output_dir)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    hard_negative = run_hard_negative_experiment(scenarios, epochs=epochs)
    hard_payload = hard_negative_payload(hard_negative)
    augmented = _augmented_scenarios_from_hard_negatives(scenarios)
    preferences = generate_preferences(augmented)
    learning = run_learning_experiment(augmented, epochs=epochs)
    rl_result = run_rl_alignment(scenarios, epochs=epochs)
    vlm_examples = generate_vlm_examples(scenarios)[:12]

    scenario_payloads = []
    learning_runs = {
        run.scenario_id: run for run in learning.heldout_selection_runs
    }
    preference_by_scenario = _first_preference_by_scenario(preferences)
    for scenario in augmented:
        asset_name = f"{scenario.scenario_id}.svg"
        asset_path = assets / asset_name
        asset_path.write_text(scenario_svg(scenario), encoding="utf-8")
        scenario_payloads.append(
            _scenario_payload(
                scenario=scenario,
                asset_path=f"assets/{asset_name}",
                learning_run=learning_runs[scenario.scenario_id],
                preference=preference_by_scenario[scenario.scenario_id],
            )
        )

    preview_path = assets / "metricdrive-explorer.svg"
    preview_path.write_text(
        _dashboard_preview_svg(hard_payload, scenario_payloads),
        encoding="utf-8",
    )

    payload = {
        "format": "metricdrive.demo.v1",
        "summary": {
            "scenario_count": len(scenarios),
            "candidate_count": sum(len(scenario.candidates) for scenario in augmented),
            "preference_pair_count": len(preferences),
            "generated_hard_negative_count": hard_negative.summary.generated_candidate_count,
            "learned_pairwise_accuracy": hard_negative.summary.learned_pairwise_accuracy,
            "learned_heldout_match_count": hard_negative.summary.learned_heldout_match_count,
            "learned_heldout_unsafe_count": hard_negative.summary.learned_heldout_unsafe_count,
        },
        "preview_asset": "assets/metricdrive-explorer.svg",
        "scenarios": scenario_payloads,
        "hard_negative": hard_payload,
        "rl_alignment": rl_alignment_payload(rl_result),
        "vlm_examples": [asdict(example) for example in vlm_examples],
    }

    (output / "scenarios.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output / "index.html").write_text(DEMO_HTML, encoding="utf-8")
    (output / "styles.css").write_text(DEMO_CSS, encoding="utf-8")
    (output / "app.js").write_text(DEMO_JS, encoding="utf-8")
    (output / "README.md").write_text(DEMO_README, encoding="utf-8")
    return payload


def _augmented_scenarios_from_hard_negatives(
    scenarios: tuple[Scenario, ...],
) -> tuple[Scenario, ...]:
    from metricdrive.hard_negatives import augment_scenarios_with_hard_negatives

    return augment_scenarios_with_hard_negatives(scenarios)


def _scenario_payload(
    scenario: Scenario,
    asset_path: str,
    learning_run,
    preference,
) -> dict[str, object]:
    scores = ranked_scores(scenario)
    return {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "title": scenario.title,
        "prompt": scenario.prompt,
        "tags": scenario.tags,
        "asset": asset_path,
        "metric_best_trajectory_id": scores[0].trajectory_id,
        "learned_trajectory_id": learning_run.selected_trajectory_id,
        "learned_utility": learning_run.learned_utility,
        "top_preference": asdict(preference),
        "scores": [
            {
                "rank": index,
                "trajectory_id": score.trajectory_id,
                "total": score.total,
                "progress_m": score.progress_m,
                "collision_clearance_m": score.collision_clearance_m,
                "vru_clearance_m": score.vru_clearance_m,
                "offroad_rate": score.offroad_rate,
                "comfort_cost": score.comfort_cost,
                "route_error_m": score.route_error_m,
                "imitation_error_m": score.imitation_error_m,
                "components": score.components,
            }
            for index, score in enumerate(scores, start=1)
        ],
    }


def _first_preference_by_scenario(preferences) -> dict[str, object]:
    selected = {}
    for preference in preferences:
        selected.setdefault(preference.scenario_id, preference)
    return selected


def _dashboard_preview_svg(
    hard_payload: dict[str, object],
    scenarios: list[dict[str, object]],
) -> str:
    summary = hard_payload["summary"]
    first = scenarios[0]
    bars = [
        ("Full objective", 1.0, "#0f766e"),
        ("No collision", 0.333, "#dc2626"),
        ("Progress only", 0.167, "#f59e0b"),
        ("Safety only", 0.333, "#2563eb"),
    ]
    bar_svg = []
    for index, (label, value, color) in enumerate(bars):
        y = 600 + index * 42
        width = 260 * value
        bar_svg.append(f'<text x="920" y="{y}" class="tiny">{escape(label)}</text>')
        bar_svg.append(f'<rect x="1040" y="{y - 14}" width="260" height="12" rx="6" fill="#e2e8f0"/>')
        bar_svg.append(f'<rect x="1040" y="{y - 14}" width="{width:.1f}" height="12" rx="6" fill="{color}"/>')

    return "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" viewBox="0 0 1440 900" role="img" aria-label="MetricDrive Explorer preview">',
            "<style>",
            "text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#0f172a}",
            ".title{font-size:54px;font-weight:800}.sub{font-size:24px;fill:#475569}.label{font-size:18px;font-weight:700}.tiny{font-size:15px;fill:#475569}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}",
            "</style>",
            '<rect width="1440" height="900" fill="#f8fafc"/>',
            '<rect x="44" y="40" width="1352" height="820" rx="24" fill="#ffffff" stroke="#cbd5e1"/>',
            '<text x="84" y="118" class="title">MetricDrive Explorer</text>',
            '<text x="84" y="158" class="sub">Metric-aligned planning loop for trajectory choices</text>',
            '<rect x="84" y="204" width="760" height="470" rx="18" fill="#f1f5f9" stroke="#cbd5e1"/>',
            '<rect x="130" y="260" width="660" height="318" rx="8" fill="#ffffff" stroke="#94a3b8"/>',
            '<path d="M160 520 C280 500, 430 390, 708 320" fill="none" stroke="#0f766e" stroke-width="10" stroke-linecap="round"/>',
            '<path d="M160 525 C320 475, 430 335, 735 250" fill="none" stroke="#dc2626" stroke-width="5" stroke-dasharray="12 12" stroke-linecap="round"/>',
            '<path d="M160 530 C300 560, 455 520, 730 500" fill="none" stroke="#2563eb" stroke-width="5" stroke-dasharray="8 12" stroke-linecap="round"/>',
            '<circle cx="520" cy="362" r="18" fill="#c026d3" opacity="0.82"/>',
            '<circle cx="662" cy="340" r="18" fill="#334155" opacity="0.75"/>',
            '<circle cx="740" cy="312" r="14" fill="#facc15" stroke="#854d0e" stroke-width="4"/>',
            '<text x="130" y="632" class="tiny">teal: metric best    red: unsafe hard negative    blue: logged/reference</text>',
            '<rect x="900" y="204" width="430" height="146" rx="18" fill="#ecfeff" stroke="#99f6e4"/>',
            f'<text x="928" y="248" class="label">Hard negatives</text>',
            f'<text x="928" y="294" class="title" style="font-size:42px">{summary["generated_candidate_count"]}</text>',
            f'<text x="1014" y="294" class="tiny">generated candidates, {summary["preference_pair_count"]} preference pairs</text>',
            '<rect x="900" y="376" width="430" height="148" rx="18" fill="#eff6ff" stroke="#bfdbfe"/>',
            '<text x="928" y="420" class="label">Learned reward recovery</text>',
            f'<text x="928" y="468" class="title" style="font-size:42px">{summary["learned_heldout_match_count"]}/6</text>',
            '<text x="1018" y="468" class="tiny">held-out metric-best matches, zero unsafe selections</text>',
            '<text x="900" y="570" class="label">Stress ablations</text>',
            *bar_svg,
            '<rect x="84" y="710" width="760" height="98" rx="18" fill="#0f172a"/>',
            f'<text x="116" y="756" fill="#ffffff" class="label">{escape(str(first["title"]))}</text>',
            '<text x="116" y="790" fill="#cbd5e1" class="tiny mono">prompt -> candidate trajectories -> metric preferences -> learned reward -> policy update</text>',
            "</svg>",
        )
    )


DEMO_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MetricDrive Explorer</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <header class="topbar">
      <a class="brand" href="./">MetricDrive</a>
      <nav aria-label="Primary navigation">
        <a href="#demo">Demo</a>
        <a href="#vlm">VLM examples</a>
        <a href="#rl">RL analogue</a>
        <a href="https://github.com/ethanvillalovoz/metricdrive">GitHub</a>
      </nav>
    </header>

    <main>
      <section class="intro" aria-labelledby="page-title">
        <div>
          <h1 id="page-title">Metric-aligned planning lab for autonomous-driving trajectory choices.</h1>
          <p>
            A public, laptop-scale analogue for planning representations, metric-derived preferences,
            learned reward selection, hard negatives, and tiny reward-optimization experiments.
          </p>
        </div>
        <div class="proof-grid" id="summary"></div>
      </section>

      <section class="workbench" id="demo" aria-label="Interactive planning demo">
        <aside class="scenario-rail">
          <h2>Scenarios</h2>
          <div id="scenarioButtons" class="scenario-buttons"></div>
        </aside>

        <section class="viewer">
          <div class="viewer-head">
            <div>
              <h2 id="scenarioTitle">Loading...</h2>
              <p id="scenarioPrompt"></p>
            </div>
            <div class="status-stack">
              <span id="metricBest"></span>
              <span id="learnedChoice"></span>
            </div>
          </div>
          <img id="scenarioImage" alt="Scenario trajectory visualization">
        </section>

        <section class="inspector">
          <h2>Metric-ranked candidates</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Candidate</th>
                  <th>Score</th>
                  <th>Clearance</th>
                  <th>Progress</th>
                </tr>
              </thead>
              <tbody id="candidateRows"></tbody>
            </table>
          </div>
          <div class="preference" id="preferenceBox"></div>
        </section>
      </section>

      <section class="evidence-grid">
        <article>
          <h2>Hard-negative stress test</h2>
          <div id="ablationBars" class="bars"></div>
        </article>
        <article id="vlm">
          <h2>VLM planning row</h2>
          <pre id="vlmExample"></pre>
        </article>
        <article id="rl">
          <h2>RL post-training analogue</h2>
          <div id="rlTable" class="table-wrap"></div>
        </article>
      </section>
    </main>

    <script src="app.js"></script>
  </body>
</html>
"""


DEMO_CSS = """* {
  box-sizing: border-box;
}

:root {
  color-scheme: light;
  --bg: #f8fafc;
  --surface: #ffffff;
  --surface-strong: #f1f5f9;
  --text: #0f172a;
  --muted: #475569;
  --border: #cbd5e1;
  --teal: #0f766e;
  --blue: #2563eb;
  --amber: #d97706;
  --red: #dc2626;
  --shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

a {
  color: inherit;
  text-decoration: none;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 4;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 24px;
  padding: 18px clamp(20px, 5vw, 68px);
  background: rgba(248, 250, 252, 0.9);
  border-bottom: 1px solid rgba(203, 213, 225, 0.7);
  backdrop-filter: blur(16px);
}

.brand {
  font-size: 18px;
  font-weight: 800;
  letter-spacing: 0;
}

nav {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 18px;
  color: var(--muted);
  font-size: 14px;
  font-weight: 650;
}

main {
  width: min(1460px, 100%);
  margin: 0 auto;
  padding: 34px clamp(18px, 4vw, 56px) 64px;
}

.intro {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.95fr);
  gap: 32px;
  align-items: end;
  margin-bottom: 34px;
}

h1,
h2,
p {
  margin-top: 0;
}

h1 {
  max-width: 980px;
  margin-bottom: 18px;
  font-size: clamp(42px, 7vw, 92px);
  line-height: 0.95;
  letter-spacing: 0;
}

h2 {
  margin-bottom: 16px;
  font-size: 18px;
  line-height: 1.2;
}

.intro p {
  max-width: 780px;
  margin-bottom: 0;
  color: var(--muted);
  font-size: 19px;
  line-height: 1.55;
}

.proof-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.proof {
  min-height: 112px;
  padding: 20px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.proof strong {
  display: block;
  margin-bottom: 8px;
  font-size: 32px;
  line-height: 1;
}

.proof span {
  color: var(--muted);
  font-size: 13px;
  font-weight: 650;
}

.workbench {
  display: grid;
  grid-template-columns: 230px minmax(430px, 1.2fr) minmax(390px, 0.95fr);
  gap: 18px;
  align-items: stretch;
}

.scenario-rail,
.viewer,
.inspector,
.evidence-grid article {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.scenario-rail,
.inspector,
.evidence-grid article {
  padding: 18px;
}

.scenario-buttons {
  display: grid;
  gap: 8px;
}

.scenario-buttons button {
  width: 100%;
  min-height: 48px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: #ffffff;
  color: var(--muted);
  font: inherit;
  font-size: 13px;
  font-weight: 700;
  text-align: left;
  cursor: pointer;
}

.scenario-buttons button[aria-pressed="true"] {
  border-color: #5eead4;
  background: #ecfeff;
  color: var(--teal);
}

.viewer {
  overflow: hidden;
}

.viewer-head {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 22px;
  border-bottom: 1px solid var(--border);
}

.viewer-head p {
  margin-bottom: 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.45;
}

.status-stack {
  display: grid;
  gap: 8px;
  min-width: 210px;
}

.status-stack span {
  display: block;
  padding: 10px 12px;
  border-radius: 7px;
  background: var(--surface-strong);
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
}

#scenarioImage {
  display: block;
  width: 100%;
  min-height: 360px;
  object-fit: contain;
  background: #ffffff;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 10px 8px;
  border-bottom: 1px solid #e2e8f0;
  font-size: 13px;
  text-align: left;
  white-space: nowrap;
}

th {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

td code,
pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.score-good {
  color: var(--teal);
  font-weight: 800;
}

.score-bad {
  color: var(--red);
  font-weight: 800;
}

.preference {
  margin-top: 18px;
  padding: 16px;
  border-left: 4px solid var(--teal);
  background: #f8fafc;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}

.evidence-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 18px;
  margin-top: 18px;
}

.bar-row {
  display: grid;
  grid-template-columns: 130px 1fr 44px;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.bar-track {
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: #e2e8f0;
}

.bar-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--teal);
}

pre {
  max-height: 340px;
  margin: 0;
  overflow: auto;
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #0f172a;
  color: #dbeafe;
  font-size: 12px;
  line-height: 1.55;
}

@media (max-width: 1180px) {
  .intro,
  .workbench,
  .evidence-grid {
    grid-template-columns: 1fr;
  }

  .status-stack {
    min-width: 0;
  }
}

@media (max-width: 720px) {
  .topbar,
  .viewer-head {
    align-items: flex-start;
    flex-direction: column;
  }

  .proof-grid {
    grid-template-columns: 1fr;
  }

  h1 {
    font-size: 42px;
  }
}
"""


DEMO_JS = """const state = {
  data: null,
  selectedScenarioId: null,
};

const els = {
  summary: document.querySelector("#summary"),
  scenarioButtons: document.querySelector("#scenarioButtons"),
  title: document.querySelector("#scenarioTitle"),
  prompt: document.querySelector("#scenarioPrompt"),
  image: document.querySelector("#scenarioImage"),
  metricBest: document.querySelector("#metricBest"),
  learnedChoice: document.querySelector("#learnedChoice"),
  rows: document.querySelector("#candidateRows"),
  preference: document.querySelector("#preferenceBox"),
  ablations: document.querySelector("#ablationBars"),
  vlm: document.querySelector("#vlmExample"),
  rl: document.querySelector("#rlTable"),
};

fetch("scenarios.json")
  .then((response) => response.json())
  .then((data) => {
    state.data = data;
    state.selectedScenarioId = data.scenarios[0].scenario_id;
    render();
  })
  .catch((error) => {
    document.body.innerHTML = `<main class="intro"><h1>MetricDrive Explorer failed to load.</h1><p>${escapeHtml(error.message)}</p></main>`;
  });

function render() {
  renderSummary();
  renderScenarioButtons();
  renderScenario();
  renderAblations();
  renderVlmExample();
  renderRlTable();
}

function renderSummary() {
  const summary = state.data.summary;
  const items = [
    [summary.scenario_count, "long-tail scenario families"],
    [summary.candidate_count, "candidate trajectories with hard negatives"],
    [summary.preference_pair_count, "metric-derived preference pairs"],
    [`${summary.learned_heldout_match_count}/6`, "held-out learned reward recovery"],
  ];
  els.summary.innerHTML = items
    .map(([value, label]) => `<div class="proof"><strong>${value}</strong><span>${label}</span></div>`)
    .join("");
}

function renderScenarioButtons() {
  els.scenarioButtons.innerHTML = state.data.scenarios
    .map((scenario) => {
      const active = scenario.scenario_id === state.selectedScenarioId;
      return `<button type="button" aria-pressed="${active}" data-scenario="${scenario.scenario_id}">${escapeHtml(scenario.category.replaceAll("_", " "))}</button>`;
    })
    .join("");

  els.scenarioButtons.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedScenarioId = button.dataset.scenario;
      render();
    });
  });
}

function renderScenario() {
  const scenario = currentScenario();
  els.title.textContent = scenario.title;
  els.prompt.textContent = scenario.prompt;
  els.image.src = scenario.asset;
  els.image.alt = `${scenario.title} trajectory visualization`;
  els.metricBest.textContent = `Metric best: ${scenario.metric_best_trajectory_id}`;
  els.learnedChoice.textContent = `Learned reward: ${scenario.learned_trajectory_id}`;
  els.rows.innerHTML = scenario.scores.map(candidateRow).join("");
  const pref = scenario.top_preference;
  els.preference.innerHTML = `<strong>Preference row:</strong> choose <code>${escapeHtml(pref.preferred_trajectory_id)}</code> over <code>${escapeHtml(pref.rejected_trajectory_id)}</code> because ${escapeHtml(pref.reasons.join("; "))}.`;
}

function candidateRow(score) {
  const scoreClass = score.collision_clearance_m < 0 ? "score-bad" : "score-good";
  return `<tr>
    <td>${score.rank}</td>
    <td><code>${escapeHtml(score.trajectory_id)}</code></td>
    <td class="${scoreClass}">${format(score.total)}</td>
    <td>${format(score.collision_clearance_m)}m</td>
    <td>${format(score.progress_m)}m</td>
  </tr>`;
}

function renderAblations() {
  const rows = state.data.hard_negative.ablation_summaries;
  els.ablations.innerHTML = rows
    .slice(0, 9)
    .map((row) => {
      const pct = Math.round(row.heldout_match_rate * 100);
      const color = row.heldout_unsafe_count > 0 ? "var(--red)" : row.heldout_match_rate === 1 ? "var(--teal)" : "var(--amber)";
      return `<div class="bar-row">
        <span>${escapeHtml(row.label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span>${row.heldout_match_count}/6</span>
      </div>`;
    })
    .join("");
}

function renderVlmExample() {
  const example = state.data.vlm_examples.find((item) => item.scenario_id === state.selectedScenarioId) || state.data.vlm_examples[0];
  const payload = {
    example_id: example.example_id,
    prompt: example.prompt,
    chosen: JSON.parse(example.chosen),
    rejected: JSON.parse(example.rejected),
  };
  els.vlm.textContent = JSON.stringify(payload, null, 2);
}

function renderRlTable() {
  const rows = state.data.rl_alignment.summaries;
  els.rl.innerHTML = `<table>
    <thead><tr><th>Method</th><th>Match</th><th>Unsafe</th><th>Gap</th></tr></thead>
    <tbody>
      ${rows.map((row) => `<tr>
        <td>${escapeHtml(row.label)}</td>
        <td>${row.metric_match_count}/${row.scenario_count}</td>
        <td>${row.unsafe_collision_count}</td>
        <td>${format(row.mean_metric_score_gap)}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;
}

function currentScenario() {
  return state.data.scenarios.find((scenario) => scenario.scenario_id === state.selectedScenarioId);
}

function format(value) {
  return Number(value).toFixed(3);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
"""


DEMO_README = """# MetricDrive Demo

This directory is generated by:

```bash
metricdrive export-demo --output docs/demo
```

It contains a dependency-free static explorer for GitHub Pages:

- `index.html`
- `styles.css`
- `app.js`
- `scenarios.json`
- `assets/*.svg`

Serve locally with:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000/demo/`.
"""
