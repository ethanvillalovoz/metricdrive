from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import exp

from metricdrive.hard_negatives import augment_scenarios_with_hard_negatives
from metricdrive.meta_actions import infer_meta_action
from metricdrive.metrics import PlanningScore, ranked_scores, route_axis_progress
from metricdrive.scenario import Scenario, Trajectory


DEFAULT_RL_EPOCHS = 80
DEFAULT_RL_LEARNING_RATE = 0.35


@dataclass(frozen=True)
class PolicyRun:
    method_id: str
    label: str
    scenario_id: str
    category: str
    selected_trajectory_id: str
    selected_meta_action: str
    metric_best_trajectory_id: str
    selected_metric_score: float
    metric_best_score: float
    metric_score_gap: float
    unsafe_collision: bool


@dataclass(frozen=True)
class PolicySummary:
    method_id: str
    label: str
    scenario_count: int
    metric_match_count: int
    metric_match_rate: float
    unsafe_collision_count: int
    mean_metric_score: float
    mean_metric_score_gap: float


@dataclass(frozen=True)
class RlAlignmentResult:
    epochs: int
    learning_rate: float
    include_hard_negatives: bool
    summaries: tuple[PolicySummary, ...]
    runs: tuple[PolicyRun, ...]


def run_rl_alignment(
    scenarios: tuple[Scenario, ...],
    epochs: int = DEFAULT_RL_EPOCHS,
    learning_rate: float = DEFAULT_RL_LEARNING_RATE,
    include_hard_negatives: bool = True,
) -> RlAlignmentResult:
    """Run a tiny metric-reward post-training analogue over candidate policies."""

    source_scenarios = (
        augment_scenarios_with_hard_negatives(scenarios)
        if include_hard_negatives
        else scenarios
    )
    methods = (
        ("token_match", "Token-match imitation proxy", _token_match_choice),
        ("progress_reward", "Progress reward only", _progress_choice),
        ("metric_reward", "Metric reward rerank", _metric_choice),
        (
            "rl_aligned",
            "Metric-RL aligned policy",
            lambda scenario: _rl_choice(
                scenario,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
        ),
    )
    runs = tuple(
        _policy_run(method_id, label, scenario, chooser(scenario))
        for method_id, label, chooser in methods
        for scenario in source_scenarios
    )
    summaries = tuple(
        _policy_summary(method_id, label, runs)
        for method_id, label, _ in methods
    )
    return RlAlignmentResult(
        epochs=epochs,
        learning_rate=learning_rate,
        include_hard_negatives=include_hard_negatives,
        summaries=summaries,
        runs=runs,
    )


def rl_alignment_payload(result: RlAlignmentResult) -> dict[str, object]:
    return {
        "format": "metricdrive.rl_alignment.v1",
        "epochs": result.epochs,
        "learning_rate": result.learning_rate,
        "include_hard_negatives": result.include_hard_negatives,
        "summaries": [asdict(summary) for summary in result.summaries],
        "runs": [asdict(run) for run in result.runs],
    }


def json_rl_alignment(result: RlAlignmentResult) -> str:
    return json.dumps(rl_alignment_payload(result), indent=2) + "\n"


def markdown_rl_alignment(result: RlAlignmentResult) -> str:
    lines = [
        "# MetricDrive RL Alignment Analogue",
        "",
        f"- Epochs: {result.epochs}",
        f"- Learning rate: {result.learning_rate:.3f}",
        f"- Hard negatives included: {'yes' if result.include_hard_negatives else 'no'}",
        "",
        "| Method | Metric match | Unsafe | Mean score | Score gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in result.summaries:
        lines.append(
            "| "
            f"{summary.label} | "
            f"{summary.metric_match_count}/{summary.scenario_count} "
            f"({summary.metric_match_rate:.3f}) | "
            f"{summary.unsafe_collision_count} | "
            f"{summary.mean_metric_score:.3f} | "
            f"{summary.mean_metric_score_gap:.3f} |"
        )

    lines.extend(
        (
            "",
            "## Scenario Choices",
            "",
            "| Method | Scenario | Selected | Meta-action | Metric best | Gap | Unsafe |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        )
    )
    for run in result.runs:
        lines.append(
            "| "
            f"{run.label} | {run.scenario_id} | `{run.selected_trajectory_id}` | "
            f"`{run.selected_meta_action}` | `{run.metric_best_trajectory_id}` | "
            f"{run.metric_score_gap:.3f} | {'yes' if run.unsafe_collision else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _policy_run(
    method_id: str,
    label: str,
    scenario: Scenario,
    selected: Trajectory,
) -> PolicyRun:
    score_by_id = {score.trajectory_id: score for score in ranked_scores(scenario)}
    metric_best = ranked_scores(scenario)[0]
    selected_score = score_by_id[selected.trajectory_id]
    return PolicyRun(
        method_id=method_id,
        label=label,
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        selected_trajectory_id=selected.trajectory_id,
        selected_meta_action=infer_meta_action(scenario, selected).action_id,
        metric_best_trajectory_id=metric_best.trajectory_id,
        selected_metric_score=selected_score.total,
        metric_best_score=metric_best.total,
        metric_score_gap=round(selected_score.total - metric_best.total, 3),
        unsafe_collision=selected_score.collision_clearance_m < 0,
    )


def _policy_summary(
    method_id: str,
    label: str,
    runs: tuple[PolicyRun, ...],
) -> PolicySummary:
    method_runs = tuple(run for run in runs if run.method_id == method_id)
    match_count = sum(
        run.selected_trajectory_id == run.metric_best_trajectory_id
        for run in method_runs
    )
    return PolicySummary(
        method_id=method_id,
        label=label,
        scenario_count=len(method_runs),
        metric_match_count=match_count,
        metric_match_rate=_ratio(match_count, len(method_runs)),
        unsafe_collision_count=sum(run.unsafe_collision for run in method_runs),
        mean_metric_score=_mean(run.selected_metric_score for run in method_runs),
        mean_metric_score_gap=_mean(run.metric_score_gap for run in method_runs),
    )


def _token_match_choice(scenario: Scenario) -> Trajectory:
    imitation_candidates = tuple(
        candidate
        for candidate in scenario.candidates
        if candidate.trajectory_id.startswith("imitation_")
    )
    if imitation_candidates:
        return imitation_candidates[0]
    return min(
        scenario.candidates,
        key=lambda candidate: _score_by_candidate(scenario, candidate).imitation_error_m,
    )


def _progress_choice(scenario: Scenario) -> Trajectory:
    return max(
        scenario.candidates,
        key=lambda candidate: route_axis_progress(candidate, scenario.route_goal),
    )


def _metric_choice(scenario: Scenario) -> Trajectory:
    metric_best = ranked_scores(scenario)[0]
    return _candidate_by_id(scenario, metric_best.trajectory_id)


def _rl_choice(
    scenario: Scenario,
    epochs: int,
    learning_rate: float,
) -> Trajectory:
    candidates = scenario.candidates
    scores = tuple(_score_by_candidate(scenario, candidate) for candidate in candidates)
    rewards = tuple(score.total / 10.0 for score in scores)
    logits = [0.0 for _ in candidates]

    for _ in range(epochs):
        probabilities = _softmax(tuple(logits))
        baseline = sum(probability * reward for probability, reward in zip(probabilities, rewards))
        for index, reward in enumerate(rewards):
            logits[index] += learning_rate * probabilities[index] * (reward - baseline)

    best_index = max(range(len(candidates)), key=lambda index: logits[index])
    return candidates[best_index]


def _score_by_candidate(scenario: Scenario, candidate: Trajectory) -> PlanningScore:
    return next(
        score
        for score in ranked_scores(scenario)
        if score.trajectory_id == candidate.trajectory_id
    )


def _candidate_by_id(scenario: Scenario, trajectory_id: str) -> Trajectory:
    for candidate in scenario.candidates:
        if candidate.trajectory_id == trajectory_id:
            return candidate
    raise ValueError(f"Unknown candidate {trajectory_id} in {scenario.scenario_id}")


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    exponentials = tuple(exp(value - maximum) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _mean(values) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 3)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)
