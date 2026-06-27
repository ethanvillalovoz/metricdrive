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
    mean_metric_best_score: float
    mean_metric_score_gap: float


@dataclass(frozen=True)
class LearningResult:
    model: PreferenceRewardModel
    preference_fit: PreferenceFitSummary
    training_selection_summary: SelectionSummary
    training_selection_runs: tuple[SelectionRun, ...]
    heldout_selection_summary: SelectionSummary
    heldout_selection_runs: tuple[SelectionRun, ...]
    benchmark: BenchmarkResult


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    label: str
    active_features: tuple[str, ...]


@dataclass(frozen=True)
class AblationRun:
    ablation_id: str
    label: str
    active_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    preference_fit: PreferenceFitSummary
    training_selection_summary: SelectionSummary
    heldout_selection_summary: SelectionSummary
    heldout_selection_runs: tuple[SelectionRun, ...]


@dataclass(frozen=True)
class AblationStudy:
    runs: tuple[AblationRun, ...]


def _without(*feature_names: str) -> tuple[str, ...]:
    removed = set(feature_names)
    return tuple(name for name in FEATURE_NAMES if name not in removed)


DEFAULT_ABLATION_SPECS = (
    AblationSpec("full_model", "Full objective", FEATURE_NAMES),
    AblationSpec(
        "no_collision",
        "No collision term",
        _without("collision_clearance"),
    ),
    AblationSpec(
        "no_vru_clearance",
        "No VRU clearance",
        _without("vru_clearance"),
    ),
    AblationSpec(
        "no_progress",
        "No progress",
        _without("progress"),
    ),
    AblationSpec(
        "no_route_error",
        "No route error",
        _without("route_error"),
    ),
    AblationSpec(
        "no_imitation",
        "No imitation",
        _without("imitation"),
    ),
    AblationSpec(
        "no_comfort",
        "No comfort",
        _without("comfort"),
    ),
    AblationSpec(
        "progress_only",
        "Progress only",
        ("progress",),
    ),
    AblationSpec(
        "safety_only",
        "Safety only",
        ("collision_clearance", "vru_clearance", "offroad"),
    ),
)


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
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> PreferenceRewardModel:
    """Train a non-negative linear reward model from pairwise preferences."""

    active_features = _validated_features(feature_names)
    weights = {name: 0.0 for name in FEATURE_NAMES}
    pairs = _training_pairs(scenarios)
    for _ in range(epochs):
        for scenario, preferred, rejected in pairs:
            difference = _feature_difference(scenario, preferred, rejected)
            probability = _sigmoid(_dot(weights, difference))
            step_scale = 1.0 - probability
            for name in active_features:
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
    feature_names: tuple[str, ...] = FEATURE_NAMES,
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
            feature_names=feature_names,
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


def run_ablation_study(
    scenarios: tuple[Scenario, ...],
    specs: tuple[AblationSpec, ...] = DEFAULT_ABLATION_SPECS,
    epochs: int = 80,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> AblationStudy:
    runs: list[AblationRun] = []
    for spec in specs:
        model = train_preference_model(
            scenarios,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            feature_names=spec.active_features,
        )
        preference_fit = evaluate_preference_fit(model, scenarios)
        training_summary, _ = evaluate_selection(model, scenarios)
        heldout_summary, heldout_runs = leave_one_scenario_out(
            scenarios,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            feature_names=spec.active_features,
        )
        runs.append(
            AblationRun(
                ablation_id=spec.ablation_id,
                label=spec.label,
                active_features=spec.active_features,
                removed_features=_removed_features(spec.active_features),
                preference_fit=preference_fit,
                training_selection_summary=training_summary,
                heldout_selection_summary=heldout_summary,
                heldout_selection_runs=heldout_runs,
            )
        )
    return AblationStudy(runs=tuple(runs))


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


def ablation_payload(study: AblationStudy) -> dict[str, object]:
    return {
        "format": "metricdrive.ablation_study.v1",
        "runs": [asdict(run) for run in study.runs],
    }


def json_ablation_study(study: AblationStudy) -> str:
    return json.dumps(ablation_payload(study), indent=2) + "\n"


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


def markdown_ablation_study(study: AblationStudy) -> str:
    lines = [
        "# MetricDrive Objective Ablations",
        "",
        "| Objective | Active features | Held-out match | Unsafe | Mean score | Score gap | Pairwise fit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in study.runs:
        lines.append(_ablation_summary_row(run))

    lines.extend(
        (
            "",
            "## Held-Out Failures",
            "",
            "| Objective | Scenario | Selected | Metric best | Score gap | Unsafe |",
            "| --- | --- | --- | --- | ---: | --- |",
        )
    )
    failures = _ablation_failures(study)
    if failures:
        lines.extend(failures)
    else:
        lines.append("| none | n/a | n/a | n/a | 0.000 | no |")
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
            "Use the objective ablation study to guide harder generated negatives that force richer tradeoffs between safety, progress, comfort, and route adherence.",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def generate_ablation_report(
    scenarios: tuple[Scenario, ...],
    output_path: str | Path,
    epochs: int = 80,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> None:
    study = run_ablation_study(
        scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    output = Path(output_path)
    lines = [
        "# Milestone 3C: Objective Ablation Study",
        "",
        "MetricDrive now tests which objective terms matter by retraining the learned preference model with individual metric components removed or isolated. The same leave-one-scenario-out protocol is used for every ablation.",
        "",
        "## Method",
        "",
        "- Start from the learned Bradley-Terry preference reward model.",
        "- Retrain with selected metric components removed or isolated.",
        "- Evaluate held-out scenario selections against the metric-rerank choice.",
        "- Track unsafe selections and metric-score gaps to expose failure modes.",
        "",
        "## Summary",
        "",
        "| Objective | Active features | Held-out match | Unsafe | Mean score | Score gap | Pairwise fit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in study.runs:
        lines.append(_ablation_summary_row(run))

    lines.extend(
        (
            "",
            "## Held-Out Failure Cases",
            "",
            "| Objective | Scenario | Selected | Metric best | Score gap | Unsafe |",
            "| --- | --- | --- | --- | ---: | --- |",
        )
    )
    failures = _ablation_failures(study)
    if failures:
        lines.extend(failures)
    else:
        lines.append("| none | n/a | n/a | n/a | 0.000 | no |")

    no_collision = _ablation_by_id(study, "no_collision").heldout_selection_summary
    progress_only = _ablation_by_id(study, "progress_only").heldout_selection_summary
    safety_only = _ablation_by_id(study, "safety_only").heldout_selection_summary
    lines.extend(
        (
            "",
            "## Takeaway",
            "",
            (
                "Collision clearance is the most brittle single term: removing it "
                f"leaves {no_collision.unsafe_collision_count} unsafe held-out "
                f"selection and drops match rate to {no_collision.metric_match_rate:.3f}. "
                "A progress-only objective recreates the dangerous baseline, "
                f"selecting unsafe trajectories in {progress_only.unsafe_collision_count} "
                f"of {progress_only.scenario_count} held-out scenarios. Safety-only "
                f"avoids collisions but matches only {safety_only.metric_match_count}/"
                f"{safety_only.scenario_count} metric-rerank choices, showing why "
                "the aligned objective needs both safety and progress terms."
            ),
            "",
            "## Next Experiment",
            "",
            "Generate harder negatives that force tradeoffs between progress, collision clearance, vulnerable-road-user clearance, comfort, and route adherence.",
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
    mean_selected_score = _mean(run.selected_metric_score for run in runs)
    mean_metric_best_score = _mean(run.metric_best_score for run in runs)
    return SelectionSummary(
        scenario_count=len(runs),
        metric_match_count=match_count,
        metric_match_rate=_ratio(match_count, len(runs)),
        unsafe_collision_count=sum(run.unsafe_collision for run in runs),
        mean_selected_metric_score=mean_selected_score,
        mean_metric_best_score=mean_metric_best_score,
        mean_metric_score_gap=round(mean_selected_score - mean_metric_best_score, 3),
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


def _validated_features(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    unknown = tuple(name for name in feature_names if name not in FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown feature name(s): {', '.join(unknown)}")
    if not feature_names:
        raise ValueError("At least one active feature is required.")
    return tuple(dict.fromkeys(feature_names))


def _removed_features(active_features: tuple[str, ...]) -> tuple[str, ...]:
    active = set(active_features)
    return tuple(name for name in FEATURE_NAMES if name not in active)


def _feature_list(feature_names: tuple[str, ...]) -> str:
    return ", ".join(feature_names)


def _ablation_summary_row(run: AblationRun) -> str:
    heldout = run.heldout_selection_summary
    fit = run.preference_fit
    return (
        "| "
        f"{run.label} | {_feature_list(run.active_features)} | "
        f"{heldout.metric_match_count}/{heldout.scenario_count} "
        f"({heldout.metric_match_rate:.3f}) | "
        f"{heldout.unsafe_collision_count} | "
        f"{heldout.mean_selected_metric_score:.3f} | "
        f"{heldout.mean_metric_score_gap:.3f} | "
        f"{fit.correct_pair_count}/{fit.pair_count} ({fit.pairwise_accuracy:.3f}) |"
    )


def _ablation_failures(study: AblationStudy) -> list[str]:
    rows: list[str] = []
    for run in study.runs:
        for selection in run.heldout_selection_runs:
            if selection.matched_metric_best and not selection.unsafe_collision:
                continue
            rows.append(
                "| "
                f"{run.label} | {selection.scenario_id} | "
                f"`{selection.selected_trajectory_id}` | "
                f"`{selection.metric_best_trajectory_id}` | "
                f"{selection.selected_metric_score - selection.metric_best_score:.3f} | "
                f"{'yes' if selection.unsafe_collision else 'no'} |"
            )
    return rows


def _ablation_by_id(study: AblationStudy, ablation_id: str) -> AblationRun:
    for run in study.runs:
        if run.ablation_id == ablation_id:
            return run
    raise ValueError(f"Missing ablation result: {ablation_id}")
