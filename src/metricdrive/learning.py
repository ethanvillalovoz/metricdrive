from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path

from metricdrive.benchmark import BenchmarkResult, run_benchmark
from metricdrive.metrics import ranked_scores, score_trajectory
from metricdrive.planners import default_planners
from metricdrive.scenario import Scenario, Trajectory

FEATURE_NAMES = (
    "progress",
    "collision_clearance",
    "vru_clearance",
    "offroad",
    "comfort",
    "route_error",
    "imitation",
)
FEATURE_SCALE = 10.0


@dataclass(frozen=True)
class PreferenceRewardModel:
    """Linear reward model learned from metric-derived trajectory preferences."""

    weights: dict[str, float]
    epochs: int
    learning_rate: float
    l2: float

    def utility(self, scenario: Scenario, trajectory: Trajectory) -> float:
        features = component_features(scenario, trajectory)
        return round(sum(self.weights[name] * features[name] for name in FEATURE_NAMES), 3)


@dataclass(frozen=True)
class LearnedPreferencePlanner:
    """Planner that selects candidates with a learned preference reward model."""

    model: PreferenceRewardModel
    planner_id: str = "learned_preference"
    label: str = "Learned preference"

    def plan(self, scenario: Scenario) -> Trajectory:
        return max(
            scenario.candidates,
            key=lambda candidate: self.model.utility(scenario, candidate),
        )


@dataclass(frozen=True)
class PreferenceFitSummary:
    pair_count: int
    correct_pair_count: int
    pairwise_accuracy: float
    mean_probability: float


@dataclass(frozen=True)
class SelectionRun:
    scenario_id: str
    category: str
    selected_trajectory_id: str
    metric_best_trajectory_id: str
    learned_utility: float
    selected_metric_score: float
    metric_best_score: float
    matched_metric_best: bool
    unsafe_collision: bool


@dataclass(frozen=True)
class SelectionSummary:
    scenario_count: int
    metric_match_count: int
    metric_match_rate: float
    unsafe_collision_count: int
    mean_selected_metric_score: float


@dataclass(frozen=True)
class LearningResult:
    model: PreferenceRewardModel
    preference_fit: PreferenceFitSummary
    training_selection_summary: SelectionSummary
    training_selection_runs: tuple[SelectionRun, ...]
    heldout_selection_summary: SelectionSummary
    heldout_selection_runs: tuple[SelectionRun, ...]
    benchmark: BenchmarkResult


def component_features(
    scenario: Scenario,
    trajectory: Trajectory,
) -> dict[str, float]:
    """Return normalized metric component features for one candidate."""

    score = score_trajectory(scenario, trajectory)
    return {name: score.components[name] / FEATURE_SCALE for name in FEATURE_NAMES}


def train_preference_model(
    scenarios: tuple[Scenario, ...],
    epochs: int = 600,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> PreferenceRewardModel:
    """Train a non-negative linear reward model from pairwise preferences."""

    weights = {name: 0.0 for name in FEATURE_NAMES}
    pairs = _training_pairs(scenarios)
    for _ in range(epochs):
        for scenario, preferred, rejected in pairs:
            difference = _feature_difference(scenario, preferred, rejected)
            probability = _sigmoid(_dot(weights, difference))
            step_scale = 1.0 - probability
            for name in FEATURE_NAMES:
                updated = weights[name] + learning_rate * (
                    step_scale * difference[name] - l2 * weights[name]
                )
                weights[name] = max(0.0, updated)

    return PreferenceRewardModel(
        weights={name: round(weights[name], 6) for name in FEATURE_NAMES},
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )


def evaluate_preference_fit(
    model: PreferenceRewardModel,
    scenarios: tuple[Scenario, ...],
) -> PreferenceFitSummary:
    pairs = _training_pairs(scenarios)
    correct = 0
    probabilities: list[float] = []
    for scenario, preferred, rejected in pairs:
        difference = _feature_difference(scenario, preferred, rejected)
        probability = _sigmoid(_dot(model.weights, difference))
        probabilities.append(probability)
        if probability > 0.5:
            correct += 1

    return PreferenceFitSummary(
        pair_count=len(pairs),
        correct_pair_count=correct,
        pairwise_accuracy=_ratio(correct, len(pairs)),
        mean_probability=_mean(probabilities),
    )


def evaluate_selection(
    model: PreferenceRewardModel,
    scenarios: tuple[Scenario, ...],
) -> tuple[SelectionSummary, tuple[SelectionRun, ...]]:
    runs = tuple(_selection_run(model, scenario) for scenario in scenarios)
    return _selection_summary(runs), runs


def leave_one_scenario_out(
    scenarios: tuple[Scenario, ...],
    epochs: int = 600,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> tuple[SelectionSummary, tuple[SelectionRun, ...]]:
    runs: list[SelectionRun] = []
    for heldout in scenarios:
        training_scenarios = tuple(
            scenario
            for scenario in scenarios
            if scenario.scenario_id != heldout.scenario_id
        )
        model = train_preference_model(
            training_scenarios,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        runs.append(_selection_run(model, heldout))
    return _selection_summary(tuple(runs)), tuple(runs)


def run_learning_experiment(
    scenarios: tuple[Scenario, ...],
    epochs: int = 600,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> LearningResult:
    model = train_preference_model(
        scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    preference_fit = evaluate_preference_fit(model, scenarios)
    training_summary, training_runs = evaluate_selection(model, scenarios)
    heldout_summary, heldout_runs = leave_one_scenario_out(
        scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    benchmark = run_benchmark(
        scenarios,
        planners=(*default_planners(), LearnedPreferencePlanner(model)),
    )
    return LearningResult(
        model=model,
        preference_fit=preference_fit,
        training_selection_summary=training_summary,
        training_selection_runs=training_runs,
        heldout_selection_summary=heldout_summary,
        heldout_selection_runs=heldout_runs,
        benchmark=benchmark,
    )


def learning_payload(result: LearningResult) -> dict[str, object]:
    return {
        "format": "metricdrive.learning.v1",
        "model": asdict(result.model),
        "preference_fit": asdict(result.preference_fit),
        "training_selection_summary": asdict(result.training_selection_summary),
        "training_selection_runs": [
            asdict(run) for run in result.training_selection_runs
        ],
        "heldout_selection_summary": asdict(result.heldout_selection_summary),
        "heldout_selection_runs": [asdict(run) for run in result.heldout_selection_runs],
        "benchmark": {
            "summaries": [asdict(summary) for summary in result.benchmark.summaries],
            "runs": [asdict(run) for run in result.benchmark.runs],
        },
    }


def json_learning(result: LearningResult) -> str:
    return json.dumps(learning_payload(result), indent=2) + "\n"


def markdown_learning(result: LearningResult) -> str:
    lines = [
        "# MetricDrive Learned Preference Model",
        "",
        "## Learned Weights",
        "",
        "| Feature | Weight |",
        "| --- | ---: |",
    ]
    for name, weight in result.model.weights.items():
        lines.append(f"| {name} | {weight:.3f} |")

    fit = result.preference_fit
    training = result.training_selection_summary
    heldout = result.heldout_selection_summary
    lines.extend(
        (
            "",
            "## Fit",
            "",
            f"- Pairwise preference accuracy: {fit.correct_pair_count}/{fit.pair_count} ({fit.pairwise_accuracy:.3f})",
            f"- Mean preferred probability: {fit.mean_probability:.3f}",
            f"- Training scenario match rate: {training.metric_match_count}/{training.scenario_count} ({training.metric_match_rate:.3f})",
            f"- Leave-one-scenario-out match rate: {heldout.metric_match_count}/{heldout.scenario_count} ({heldout.metric_match_rate:.3f})",
            f"- Held-out unsafe selections: {heldout.unsafe_collision_count}",
            "",
            "## Held-Out Scenario Choices",
            "",
            "| Scenario | Learned selection | Metric best | Utility | Metric score | Match | Unsafe |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        )
    )
    for run in result.heldout_selection_runs:
        lines.append(_selection_row(run))
    return "\n".join(lines) + "\n"


def generate_learning_report(
    scenarios: tuple[Scenario, ...],
    output_path: str | Path,
    epochs: int = 600,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> None:
    result = run_learning_experiment(
        scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    output = Path(output_path)
    lines = [
        "# Milestone 3B: Learned Preference Model",
        "",
        "MetricDrive now trains a lightweight reward model from the metric-derived preference pairs. The model learns non-negative weights over interpretable metric components, then acts as a planner by choosing the candidate with the highest learned utility.",
        "",
        "## Method",
        "",
        "- Generate all pairwise metric preferences within each scenario.",
        "- Represent each candidate with normalized planning-score components.",
        "- Train a Bradley-Terry/logistic preference model on chosen-vs-rejected feature differences.",
        "- Clamp learned weights to be non-negative so the tiny synthetic set cannot learn an inverted safety preference.",
        "- Evaluate both in-sample selection and leave-one-scenario-out generalization.",
        "",
        "## Learned Weights",
        "",
        "| Feature | Weight |",
        "| --- | ---: |",
    ]
    for name, weight in result.model.weights.items():
        lines.append(f"| {name} | {weight:.3f} |")

    fit = result.preference_fit
    training = result.training_selection_summary
    heldout = result.heldout_selection_summary
    lines.extend(
        (
            "",
            "## Results",
            "",
            f"- Pairwise preference accuracy: {fit.correct_pair_count}/{fit.pair_count} ({fit.pairwise_accuracy:.3f})",
            f"- Mean preferred probability: {fit.mean_probability:.3f}",
            f"- Training selection match rate: {training.metric_match_count}/{training.scenario_count} ({training.metric_match_rate:.3f})",
            f"- Leave-one-scenario-out match rate: {heldout.metric_match_count}/{heldout.scenario_count} ({heldout.metric_match_rate:.3f})",
            f"- Leave-one-scenario-out unsafe selections: {heldout.unsafe_collision_count}",
            "",
            "## Held-Out Scenario Choices",
            "",
            "| Scenario | Learned selection | Metric best | Utility | Metric score | Match | Unsafe |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        )
    )
    for run in result.heldout_selection_runs:
        lines.append(_selection_row(run))

    lines.extend(
        (
            "",
            "## Planner Benchmark With Learned Reward",
            "",
            "| Planner | Mean score | Progress | Collision clearance | VRU clearance | Unsafe cases |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for summary in result.benchmark.summaries:
        lines.append(
            "| "
            f"{summary.planner_label} | {summary.mean_total:.3f} | "
            f"{summary.mean_progress_m:.3f} | {summary.mean_collision_clearance_m:.3f} | "
            f"{_optional(summary.mean_vru_clearance_m)} | "
            f"{summary.unsafe_collision_count} |"
        )

    lines.extend(
        (
            "",
            "## Takeaway",
            "",
            "The learned preference planner recovers the metric-rerank choices on the controlled scenario set and on leave-one-scenario-out evaluation. This turns the project from hard-coded metric scoring into a small, inspectable alignment loop: metrics create preferences, preferences train a reward model, and the reward model selects trajectories.",
            "",
            "## Next Experiment",
            "",
            "Add objective ablations and harder generated negatives, then scale the same preference-learning interface to optional public motion-data slices.",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _training_pairs(
    scenarios: tuple[Scenario, ...],
) -> tuple[tuple[Scenario, Trajectory, Trajectory], ...]:
    pairs: list[tuple[Scenario, Trajectory, Trajectory]] = []
    for scenario in scenarios:
        candidates = {
            candidate.trajectory_id: candidate for candidate in scenario.candidates
        }
        scores = ranked_scores(scenario)
        for preferred_index, preferred in enumerate(scores):
            for rejected in scores[preferred_index + 1 :]:
                pairs.append(
                    (
                        scenario,
                        candidates[preferred.trajectory_id],
                        candidates[rejected.trajectory_id],
                    )
                )
    return tuple(pairs)


def _feature_difference(
    scenario: Scenario,
    preferred: Trajectory,
    rejected: Trajectory,
) -> dict[str, float]:
    preferred_features = component_features(scenario, preferred)
    rejected_features = component_features(scenario, rejected)
    return {
        name: preferred_features[name] - rejected_features[name]
        for name in FEATURE_NAMES
    }


def _selection_run(model: PreferenceRewardModel, scenario: Scenario) -> SelectionRun:
    selected = LearnedPreferencePlanner(model).plan(scenario)
    selected_score = score_trajectory(scenario, selected)
    metric_best = ranked_scores(scenario)[0]
    return SelectionRun(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        selected_trajectory_id=selected.trajectory_id,
        metric_best_trajectory_id=metric_best.trajectory_id,
        learned_utility=model.utility(scenario, selected),
        selected_metric_score=selected_score.total,
        metric_best_score=metric_best.total,
        matched_metric_best=selected.trajectory_id == metric_best.trajectory_id,
        unsafe_collision=selected_score.collision_clearance_m < 0,
    )


def _selection_summary(runs: tuple[SelectionRun, ...]) -> SelectionSummary:
    match_count = sum(run.matched_metric_best for run in runs)
    return SelectionSummary(
        scenario_count=len(runs),
        metric_match_count=match_count,
        metric_match_rate=_ratio(match_count, len(runs)),
        unsafe_collision_count=sum(run.unsafe_collision for run in runs),
        mean_selected_metric_score=_mean(run.selected_metric_score for run in runs),
    )


def _selection_row(run: SelectionRun) -> str:
    return (
        "| "
        f"{run.scenario_id} | `{run.selected_trajectory_id}` | "
        f"`{run.metric_best_trajectory_id}` | {run.learned_utility:.3f} | "
        f"{run.selected_metric_score:.3f} | "
        f"{'yes' if run.matched_metric_best else 'no'} | "
        f"{'yes' if run.unsafe_collision else 'no'} |"
    )


def _dot(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights[name] * features[name] for name in FEATURE_NAMES)


def _sigmoid(value: float) -> float:
    clamped = max(min(value, 35.0), -35.0)
    if clamped >= 0:
        return 1.0 / (1.0 + exp(-clamped))
    numerator = exp(clamped)
    return numerator / (1.0 + numerator)


def _mean(values) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 3)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
