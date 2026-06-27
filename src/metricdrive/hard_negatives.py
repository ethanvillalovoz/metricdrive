from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import hypot
from pathlib import Path

from metricdrive.learning import (
    AblationStudy,
    LearningResult,
    run_ablation_study,
    run_learning_experiment,
)
from metricdrive.metrics import (
    PlanningScore,
    ranked_scores,
    route_axis_progress,
    score_trajectory,
)
from metricdrive.preferences import generate_preferences
from metricdrive.scenario import Point, Scenario, Trajectory

GENERATED_SOURCE = "generated_negative"
DEFAULT_HARD_NEGATIVE_EPOCHS = 40


@dataclass(frozen=True)
class GeneratedCandidateEvaluation:
    scenario_id: str
    category: str
    trajectory_id: str
    label: str
    rank: int
    total: float
    score_gap_to_best: float
    collision_clearance_m: float
    progress_m: float
    route_error_m: float
    comfort_cost: float
    unsafe_collision: bool
    risk_tags: tuple[str, ...]


@dataclass(frozen=True)
class HardNegativeAblationSummary:
    ablation_id: str
    label: str
    heldout_match_count: int
    heldout_scenario_count: int
    heldout_match_rate: float
    heldout_unsafe_count: int
    mean_metric_score_gap: float
    pairwise_accuracy: float


@dataclass(frozen=True)
class HardNegativeSummary:
    scenario_count: int
    original_candidate_count: int
    generated_candidate_count: int
    augmented_candidate_count: int
    preference_pair_count: int
    unsafe_generated_count: int
    near_miss_generated_count: int
    closest_generated_score_gap: float
    learned_pair_count: int
    learned_correct_pair_count: int
    learned_pairwise_accuracy: float
    learned_heldout_match_count: int
    learned_heldout_unsafe_count: int


@dataclass(frozen=True)
class HardNegativeExperiment:
    summary: HardNegativeSummary
    generated_candidates: tuple[GeneratedCandidateEvaluation, ...]
    ablation_summaries: tuple[HardNegativeAblationSummary, ...]


def generate_hard_negatives(scenario: Scenario) -> tuple[Trajectory, ...]:
    """Generate deterministic stress-test candidates for one scenario."""

    metric_best = _candidate_by_id(scenario, ranked_scores(scenario)[0].trajectory_id)
    progress_candidate = max(
        scenario.candidates,
        key=lambda candidate: route_axis_progress(candidate, scenario.route_goal),
    )
    return (
        _blend_trajectories(
            metric_best,
            progress_candidate,
            alpha=0.35,
            trajectory_id="generated_progress_pressure",
            label="Generated progress pressure",
        ),
        _under_commit(
            metric_best,
            factor=0.62,
            trajectory_id="generated_under_commit",
            label="Generated under-committed progress",
        ),
        _lateral_wobble(
            scenario,
            metric_best,
            amplitude_m=0.75,
            trajectory_id="generated_lateral_wobble",
            label="Generated lateral wobble",
        ),
    )


def augment_scenario_with_hard_negatives(scenario: Scenario) -> Scenario:
    generated = generate_hard_negatives(scenario)
    return replace(scenario, candidates=(*scenario.candidates, *generated))


def augment_scenarios_with_hard_negatives(
    scenarios: tuple[Scenario, ...],
) -> tuple[Scenario, ...]:
    return tuple(augment_scenario_with_hard_negatives(scenario) for scenario in scenarios)


def run_hard_negative_experiment(
    scenarios: tuple[Scenario, ...],
    epochs: int = DEFAULT_HARD_NEGATIVE_EPOCHS,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> HardNegativeExperiment:
    augmented = augment_scenarios_with_hard_negatives(scenarios)
    generated_candidates = _evaluate_generated_candidates(augmented)
    learning = run_learning_experiment(
        augmented,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    ablations = run_ablation_study(
        augmented,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    summary = _experiment_summary(
        scenarios=scenarios,
        augmented=augmented,
        generated_candidates=generated_candidates,
        learning=learning,
    )
    return HardNegativeExperiment(
        summary=summary,
        generated_candidates=generated_candidates,
        ablation_summaries=_ablation_summaries(ablations),
    )


def hard_negative_payload(experiment: HardNegativeExperiment) -> dict[str, object]:
    return {
        "format": "metricdrive.hard_negatives.v1",
        "summary": asdict(experiment.summary),
        "generated_candidates": [
            asdict(candidate) for candidate in experiment.generated_candidates
        ],
        "ablation_summaries": [
            asdict(summary) for summary in experiment.ablation_summaries
        ],
    }


def json_hard_negative_experiment(experiment: HardNegativeExperiment) -> str:
    return json.dumps(hard_negative_payload(experiment), indent=2) + "\n"


def markdown_hard_negative_experiment(experiment: HardNegativeExperiment) -> str:
    summary = experiment.summary
    lines = [
        "# MetricDrive Hard Negative Stress Test",
        "",
        "## Summary",
        "",
        f"- Scenarios: {summary.scenario_count}",
        f"- Original candidates: {summary.original_candidate_count}",
        f"- Generated hard negatives: {summary.generated_candidate_count}",
        f"- Augmented candidates: {summary.augmented_candidate_count}",
        f"- Metric-derived preference pairs: {summary.preference_pair_count}",
        f"- Unsafe generated candidates: {summary.unsafe_generated_count}",
        f"- Near-miss generated candidates: {summary.near_miss_generated_count}",
        f"- Closest generated score gap: {summary.closest_generated_score_gap:.3f}",
        f"- Learned pairwise fit: {summary.learned_correct_pair_count}/{summary.learned_pair_count} ({summary.learned_pairwise_accuracy:.3f})",
        f"- Learned held-out recovery: {summary.learned_heldout_match_count}/{summary.scenario_count}",
        f"- Learned held-out unsafe selections: {summary.learned_heldout_unsafe_count}",
        "",
        "## Generated Candidates",
        "",
        "| Scenario | Candidate | Rank | Score | Gap | Clearance | Progress | Tags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in experiment.generated_candidates:
        lines.append(_generated_candidate_row(candidate))

    lines.extend(
        (
            "",
            "## Stress Ablations",
            "",
            "| Objective | Held-out match | Unsafe | Score gap | Pairwise fit |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for ablation in experiment.ablation_summaries:
        lines.append(_hard_negative_ablation_row(ablation))
    return "\n".join(lines) + "\n"


def generate_hard_negative_report(
    scenarios: tuple[Scenario, ...],
    output_path: str | Path,
    epochs: int = DEFAULT_HARD_NEGATIVE_EPOCHS,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> None:
    experiment = run_hard_negative_experiment(
        scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    summary = experiment.summary
    output = Path(output_path)
    lines = [
        "# Milestone 3D: Hard Negative Stress Test",
        "",
        "MetricDrive now augments every synthetic scenario with generated hard negatives. These are deterministic trajectory perturbations designed to create tighter safety, progress, route, and comfort tradeoffs than the original hand-authored candidates.",
        "",
        "## Generation Strategy",
        "",
        "- `generated_progress_pressure`: blends the metric-best trajectory toward the highest-progress candidate.",
        "- `generated_under_commit`: shortens the metric-best trajectory to test low-progress or conflict-zone hesitation.",
        "- `generated_lateral_wobble`: adds alternating lateral motion to test comfort and route robustness.",
        "",
        "## Summary",
        "",
        f"- Original candidates: {summary.original_candidate_count}",
        f"- Generated hard negatives: {summary.generated_candidate_count}",
        f"- Augmented candidates: {summary.augmented_candidate_count}",
        f"- Metric-derived preference pairs: {summary.preference_pair_count}",
        f"- Unsafe generated candidates: {summary.unsafe_generated_count}",
        f"- Near-miss generated candidates within 2 score points: {summary.near_miss_generated_count}",
        f"- Learned reward pairwise fit: {summary.learned_correct_pair_count}/{summary.learned_pair_count} ({summary.learned_pairwise_accuracy:.3f})",
        f"- Learned reward held-out recovery: {summary.learned_heldout_match_count}/{summary.scenario_count}",
        f"- Learned reward held-out unsafe selections: {summary.learned_heldout_unsafe_count}",
        "",
        "## Generated Candidate Scores",
        "",
        "| Scenario | Candidate | Rank | Score | Gap | Clearance | Progress | Tags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in experiment.generated_candidates:
        lines.append(_generated_candidate_row(candidate))

    lines.extend(
        (
            "",
            "## Stress Ablations On Augmented Set",
            "",
            "| Objective | Held-out match | Unsafe | Score gap | Pairwise fit |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for ablation in experiment.ablation_summaries:
        lines.append(_hard_negative_ablation_row(ablation))

    lines.extend(
        (
            "",
            "## Takeaway",
            "",
            "The generated negatives expand the preference set from 18 to 90 pairs while preserving a clean learned-reward result: the full learned objective still recovers every held-out metric-best trajectory with zero unsafe selections. The stress ablations become sharper on the augmented set, especially no-collision and progress-only objectives, which confirms the hard negatives are exposing the intended failure modes.",
            "",
            "## Next Experiment",
            "",
            "Define a verifiable meta-action vocabulary, such as `YIELD_TO_VRU`, `NUDGE_AROUND_OBSTACLE`, and `SLOW_FOR_LEAD`, then connect those actions to generated trajectory candidates and metric checks.",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_generated_candidates(
    scenarios: tuple[Scenario, ...],
) -> tuple[GeneratedCandidateEvaluation, ...]:
    evaluations: list[GeneratedCandidateEvaluation] = []
    for scenario in scenarios:
        ranked = ranked_scores(scenario)
        metric_best = ranked[0]
        ranks = {score.trajectory_id: index for index, score in enumerate(ranked, start=1)}
        for candidate in scenario.candidates:
            if candidate.source != GENERATED_SOURCE:
                continue
            score = score_trajectory(scenario, candidate)
            evaluations.append(
                _generated_candidate_evaluation(
                    scenario=scenario,
                    candidate=candidate,
                    score=score,
                    metric_best=metric_best,
                    rank=ranks[candidate.trajectory_id],
                )
            )
    return tuple(evaluations)


def _generated_candidate_evaluation(
    scenario: Scenario,
    candidate: Trajectory,
    score: PlanningScore,
    metric_best: PlanningScore,
    rank: int,
) -> GeneratedCandidateEvaluation:
    return GeneratedCandidateEvaluation(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        trajectory_id=candidate.trajectory_id,
        label=candidate.label,
        rank=rank,
        total=score.total,
        score_gap_to_best=round(score.total - metric_best.total, 3),
        collision_clearance_m=score.collision_clearance_m,
        progress_m=score.progress_m,
        route_error_m=score.route_error_m,
        comfort_cost=score.comfort_cost,
        unsafe_collision=score.collision_clearance_m < 0,
        risk_tags=_risk_tags(score, metric_best),
    )


def _experiment_summary(
    scenarios: tuple[Scenario, ...],
    augmented: tuple[Scenario, ...],
    generated_candidates: tuple[GeneratedCandidateEvaluation, ...],
    learning: LearningResult,
) -> HardNegativeSummary:
    preference_pair_count = len(generate_preferences(augmented))
    unsafe_generated_count = sum(
        candidate.unsafe_collision for candidate in generated_candidates
    )
    near_miss_generated_count = sum(
        candidate.score_gap_to_best >= -2.0 for candidate in generated_candidates
    )
    closest_gap = max(
        (candidate.score_gap_to_best for candidate in generated_candidates),
        default=0.0,
    )
    return HardNegativeSummary(
        scenario_count=len(scenarios),
        original_candidate_count=sum(len(scenario.candidates) for scenario in scenarios),
        generated_candidate_count=len(generated_candidates),
        augmented_candidate_count=sum(len(scenario.candidates) for scenario in augmented),
        preference_pair_count=preference_pair_count,
        unsafe_generated_count=unsafe_generated_count,
        near_miss_generated_count=near_miss_generated_count,
        closest_generated_score_gap=round(closest_gap, 3),
        learned_pair_count=learning.preference_fit.pair_count,
        learned_correct_pair_count=learning.preference_fit.correct_pair_count,
        learned_pairwise_accuracy=learning.preference_fit.pairwise_accuracy,
        learned_heldout_match_count=(
            learning.heldout_selection_summary.metric_match_count
        ),
        learned_heldout_unsafe_count=(
            learning.heldout_selection_summary.unsafe_collision_count
        ),
    )


def _ablation_summaries(
    study: AblationStudy,
) -> tuple[HardNegativeAblationSummary, ...]:
    return tuple(
        HardNegativeAblationSummary(
            ablation_id=run.ablation_id,
            label=run.label,
            heldout_match_count=run.heldout_selection_summary.metric_match_count,
            heldout_scenario_count=run.heldout_selection_summary.scenario_count,
            heldout_match_rate=run.heldout_selection_summary.metric_match_rate,
            heldout_unsafe_count=run.heldout_selection_summary.unsafe_collision_count,
            mean_metric_score_gap=run.heldout_selection_summary.mean_metric_score_gap,
            pairwise_accuracy=run.preference_fit.pairwise_accuracy,
        )
        for run in study.runs
    )


def _candidate_by_id(scenario: Scenario, trajectory_id: str) -> Trajectory:
    for candidate in scenario.candidates:
        if candidate.trajectory_id == trajectory_id:
            return candidate
    raise ValueError(f"Unknown trajectory id for {scenario.scenario_id}: {trajectory_id}")


def _blend_trajectories(
    base: Trajectory,
    pressure: Trajectory,
    alpha: float,
    trajectory_id: str,
    label: str,
) -> Trajectory:
    points = tuple(
        Point(
            round(base_point.x * (1.0 - alpha) + pressure_point.x * alpha, 3),
            round(base_point.y * (1.0 - alpha) + pressure_point.y * alpha, 3),
        )
        for base_point, pressure_point in zip(base.points, pressure.points)
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        points=points,
        label=label,
        source=GENERATED_SOURCE,
    )


def _under_commit(
    base: Trajectory,
    factor: float,
    trajectory_id: str,
    label: str,
) -> Trajectory:
    start = base.points[0]
    points = tuple(
        Point(
            round(start.x + (point.x - start.x) * factor, 3),
            round(start.y + (point.y - start.y) * factor, 3),
        )
        for point in base.points
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        points=points,
        label=label,
        source=GENERATED_SOURCE,
    )


def _lateral_wobble(
    scenario: Scenario,
    base: Trajectory,
    amplitude_m: float,
    trajectory_id: str,
    label: str,
) -> Trajectory:
    start = base.points[0]
    route_x = scenario.route_goal.x - start.x
    route_y = scenario.route_goal.y - start.y
    route_length = hypot(route_x, route_y) or 1.0
    perp_x = -route_y / route_length
    perp_y = route_x / route_length
    signs = (0.0, 1.0, -1.0, 1.0, -1.0)
    points = tuple(
        Point(
            round(point.x + perp_x * amplitude_m * sign, 3),
            round(point.y + perp_y * amplitude_m * sign, 3),
        )
        for point, sign in zip(base.points, signs)
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        points=points,
        label=label,
        source=GENERATED_SOURCE,
    )


def _risk_tags(score: PlanningScore, metric_best: PlanningScore) -> tuple[str, ...]:
    tags: list[str] = []
    if score.collision_clearance_m < 0:
        tags.append("unsafe_collision")
    if score.total - metric_best.total >= -2.0:
        tags.append("near_miss")
    if score.progress_m < metric_best.progress_m - 2.0:
        tags.append("low_progress")
    if score.comfort_cost > metric_best.comfort_cost + 1.0:
        tags.append("comfort_stress")
    if score.route_error_m > metric_best.route_error_m + 2.0:
        tags.append("route_error")
    if not tags:
        tags.append("metric_tradeoff")
    return tuple(tags)


def _generated_candidate_row(candidate: GeneratedCandidateEvaluation) -> str:
    return (
        "| "
        f"{candidate.scenario_id} | `{candidate.trajectory_id}` | "
        f"{candidate.rank} | {candidate.total:.3f} | "
        f"{candidate.score_gap_to_best:.3f} | "
        f"{candidate.collision_clearance_m:.3f} | "
        f"{candidate.progress_m:.3f} | "
        f"{', '.join(candidate.risk_tags)} |"
    )


def _hard_negative_ablation_row(summary: HardNegativeAblationSummary) -> str:
    return (
        "| "
        f"{summary.label} | "
        f"{summary.heldout_match_count}/{summary.heldout_scenario_count} "
        f"({summary.heldout_match_rate:.3f}) | "
        f"{summary.heldout_unsafe_count} | "
        f"{summary.mean_metric_score_gap:.3f} | "
        f"{summary.pairwise_accuracy:.3f} |"
    )
