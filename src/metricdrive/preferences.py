from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from metricdrive.metrics import PlanningScore, ranked_scores
from metricdrive.scenario import Scenario, Trajectory


@dataclass(frozen=True)
class PreferencePair:
    """Metric-derived preference between two trajectory candidates."""

    preference_id: str
    scenario_id: str
    category: str
    tags: tuple[str, ...]
    preferred_trajectory_id: str
    rejected_trajectory_id: str
    preferred_score: float
    rejected_score: float
    score_margin: float
    metric_deltas: dict[str, float]
    reason_categories: tuple[str, ...]
    reasons: tuple[str, ...]
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class PreferenceSummary:
    pair_count: int
    scenario_count: int
    mean_score_margin: float
    unsafe_rejection_count: int
    category_counts: dict[str, int]
    reason_category_counts: dict[str, int]


def generate_preferences(
    scenarios: tuple[Scenario, ...],
    min_score_margin: float = 0.0,
) -> tuple[PreferencePair, ...]:
    """Create pairwise preferences from metric-ranked candidate trajectories."""

    pairs: list[PreferencePair] = []
    for scenario in scenarios:
        scores = ranked_scores(scenario)
        for preferred_index, preferred in enumerate(scores):
            for rejected in scores[preferred_index + 1 :]:
                margin = round(preferred.total - rejected.total, 3)
                if margin <= min_score_margin:
                    continue
                pairs.append(_preference_pair(scenario, preferred, rejected, margin))
    return tuple(pairs)


def preference_summary(pairs: tuple[PreferencePair, ...]) -> PreferenceSummary:
    scenario_ids = {pair.scenario_id for pair in pairs}
    category_counts: dict[str, int] = {}
    reason_category_counts: dict[str, int] = {}
    for pair in pairs:
        category_counts[pair.category] = category_counts.get(pair.category, 0) + 1
        for category in pair.reason_categories:
            reason_category_counts[category] = reason_category_counts.get(category, 0) + 1

    return PreferenceSummary(
        pair_count=len(pairs),
        scenario_count=len(scenario_ids),
        mean_score_margin=_mean(pair.score_margin for pair in pairs),
        unsafe_rejection_count=sum(
            pair.metric_deltas["rejected_collision_clearance_m"] < 0 for pair in pairs
        ),
        category_counts=dict(sorted(category_counts.items())),
        reason_category_counts=dict(sorted(reason_category_counts.items())),
    )


def preferences_payload(pairs: tuple[PreferencePair, ...]) -> dict[str, object]:
    return {
        "format": "metricdrive.preferences.v1",
        "summary": asdict(preference_summary(pairs)),
        "pairs": [asdict(pair) for pair in pairs],
    }


def json_preferences(pairs: tuple[PreferencePair, ...]) -> str:
    return json.dumps(preferences_payload(pairs), indent=2) + "\n"


def markdown_preferences(pairs: tuple[PreferencePair, ...], limit: int | None = 12) -> str:
    summary = preference_summary(pairs)
    shown_pairs = pairs if limit is None else pairs[:limit]
    lines = [
        "# MetricDrive Preference Pairs",
        "",
        f"- Pair count: {summary.pair_count}",
        f"- Scenario count: {summary.scenario_count}",
        f"- Mean score margin: {summary.mean_score_margin:.3f}",
        f"- Unsafe rejected candidates: {summary.unsafe_rejection_count}",
        "",
        "## Reason Categories",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    for category, count in summary.reason_category_counts.items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        (
            "",
            "## Preference Examples",
            "",
            "| Scenario | Preferred | Rejected | Margin | Reasons |",
            "| --- | --- | --- | ---: | --- |",
        )
    )
    for pair in shown_pairs:
        lines.append(
            "| "
            f"{pair.scenario_id} | `{pair.preferred_trajectory_id}` | "
            f"`{pair.rejected_trajectory_id}` | {pair.score_margin:.3f} | "
            f"{'; '.join(pair.reasons)} |"
        )
    return "\n".join(lines) + "\n"


def save_preferences(path: str | Path, pairs: tuple[PreferencePair, ...]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json_preferences(pairs), encoding="utf-8")


def generate_preference_report(
    pairs: tuple[PreferencePair, ...],
    output_path: str | Path,
) -> None:
    summary = preference_summary(pairs)
    output = Path(output_path)
    lines = [
        "# Milestone 3: Metric-Derived Preferences",
        "",
        "MetricDrive now converts trajectory metric comparisons into preference pairs. This is the alignment data layer: each row says which trajectory should be preferred, which should be rejected, and which metrics explain the choice.",
        "",
        "## Summary",
        "",
        f"- Preference pairs: {summary.pair_count}",
        f"- Scenario coverage: {summary.scenario_count}",
        f"- Mean score margin: {summary.mean_score_margin:.3f}",
        f"- Pairs rejecting an unsafe trajectory: {summary.unsafe_rejection_count}",
        "",
        "## Pair Counts By Scenario Category",
        "",
        "| Category | Pairs |",
        "| --- | ---: |",
    ]
    for category, count in summary.category_counts.items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        (
            "",
            "## Reason Categories",
            "",
            "| Reason | Count |",
            "| --- | ---: |",
        )
    )
    for category, count in summary.reason_category_counts.items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        (
            "",
            "## Examples",
            "",
            "| Scenario | Preferred | Rejected | Margin | Metric reasons |",
            "| --- | --- | --- | ---: | --- |",
        )
    )
    for pair in pairs:
        lines.append(
            "| "
            f"{pair.scenario_id} | `{pair.preferred_trajectory_id}` | "
            f"`{pair.rejected_trajectory_id}` | {pair.score_margin:.3f} | "
            f"{'; '.join(pair.reasons)} |"
        )

    lines.extend(
        (
        "",
        "## Training Shape",
        "",
        "Each JSON pair includes `prompt`, `chosen`, and `rejected` fields, so it can be adapted into a DPO-style preference-tuning row. The current labels are deterministic because they come from transparent metrics, which makes the dataset reproducible and debuggable before introducing learned rewards or policy updates.",
        "",
        "## Next Experiment",
        "",
            "Train a lightweight preference model that predicts the preferred trajectory from metric-derived pairs, then compare its selections with hard-coded metric reranking.",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _preference_pair(
    scenario: Scenario,
    preferred: PlanningScore,
    rejected: PlanningScore,
    margin: float,
) -> PreferencePair:
    metric_deltas = _metric_deltas(preferred, rejected)
    reason_categories, reasons = _reasons(metric_deltas)
    preferred_trajectory = _trajectory_by_id(scenario, preferred.trajectory_id)
    rejected_trajectory = _trajectory_by_id(scenario, rejected.trajectory_id)
    return PreferencePair(
        preference_id=(
            f"{scenario.scenario_id}:{preferred.trajectory_id}>{rejected.trajectory_id}"
        ),
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        tags=scenario.tags,
        preferred_trajectory_id=preferred.trajectory_id,
        rejected_trajectory_id=rejected.trajectory_id,
        preferred_score=preferred.total,
        rejected_score=rejected.total,
        score_margin=margin,
        metric_deltas=metric_deltas,
        reason_categories=reason_categories,
        reasons=reasons,
        prompt=_training_prompt(scenario),
        chosen=_training_response(preferred_trajectory, preferred),
        rejected=_training_response(rejected_trajectory, rejected),
    )


def _metric_deltas(
    preferred: PlanningScore,
    rejected: PlanningScore,
) -> dict[str, float]:
    preferred_vru = preferred.vru_clearance_m if preferred.vru_clearance_m is not None else 0.0
    rejected_vru = rejected.vru_clearance_m if rejected.vru_clearance_m is not None else 0.0
    return {
        "total_score": round(preferred.total - rejected.total, 3),
        "progress_m": round(preferred.progress_m - rejected.progress_m, 3),
        "collision_clearance_m": round(
            preferred.collision_clearance_m - rejected.collision_clearance_m,
            3,
        ),
        "preferred_collision_clearance_m": preferred.collision_clearance_m,
        "rejected_collision_clearance_m": rejected.collision_clearance_m,
        "vru_clearance_m": round(preferred_vru - rejected_vru, 3),
        "offroad_rate_reduction": round(rejected.offroad_rate - preferred.offroad_rate, 3),
        "comfort_cost_reduction": round(
            rejected.comfort_cost - preferred.comfort_cost,
            3,
        ),
        "route_error_reduction_m": round(
            rejected.route_error_m - preferred.route_error_m,
            3,
        ),
        "imitation_error_reduction_m": round(
            rejected.imitation_error_m - preferred.imitation_error_m,
            3,
        ),
    }


def _reasons(metric_deltas: dict[str, float]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    categories: list[str] = []
    reasons: list[str] = []

    if (
        metric_deltas["rejected_collision_clearance_m"] < 0
        <= metric_deltas["preferred_collision_clearance_m"]
    ):
        categories.append("avoids_collision")
        reasons.append("avoids an overlap or negative-clearance trajectory")
    elif metric_deltas["collision_clearance_m"] > 0.25:
        categories.append("improves_clearance")
        reasons.append(
            f"improves collision clearance by {metric_deltas['collision_clearance_m']:.3f} m"
        )

    if metric_deltas["vru_clearance_m"] > 0.25:
        categories.append("improves_vru_clearance")
        reasons.append(
            f"increases vulnerable-road-user clearance by {metric_deltas['vru_clearance_m']:.3f} m"
        )

    if metric_deltas["progress_m"] > 0.5:
        categories.append("improves_progress")
        reasons.append(f"adds {metric_deltas['progress_m']:.3f} m of route progress")

    if metric_deltas["route_error_reduction_m"] > 0.5:
        categories.append("improves_route_completion")
        reasons.append(
            f"reduces final route error by {metric_deltas['route_error_reduction_m']:.3f} m"
        )

    if metric_deltas["comfort_cost_reduction"] > 0.25:
        categories.append("improves_comfort")
        reasons.append(
            f"reduces comfort cost by {metric_deltas['comfort_cost_reduction']:.3f}"
        )

    if metric_deltas["offroad_rate_reduction"] > 0.0:
        categories.append("reduces_offroad")
        reasons.append(
            f"reduces offroad rate by {metric_deltas['offroad_rate_reduction']:.3f}"
        )

    if not reasons:
        categories.append("higher_weighted_score")
        reasons.append("wins on the weighted planning score across metrics")

    return tuple(dict.fromkeys(categories)), tuple(reasons)


def _trajectory_by_id(scenario: Scenario, trajectory_id: str) -> Trajectory:
    for trajectory in scenario.candidates:
        if trajectory.trajectory_id == trajectory_id:
            return trajectory
    raise ValueError(f"Unknown trajectory id for {scenario.scenario_id}: {trajectory_id}")


def _training_prompt(scenario: Scenario) -> str:
    tags = ", ".join(scenario.tags) if scenario.tags else "none"
    return (
        f"Scenario: {scenario.title}\n"
        f"Driving task: {scenario.prompt}\n"
        f"Category: {scenario.category}\n"
        f"Tags: {tags}\n"
        "Choose a five-point ego trajectory that balances collision avoidance, "
        "route progress, route adherence, comfort, and imitation."
    )


def _training_response(trajectory: Trajectory, score: PlanningScore) -> str:
    points = ", ".join(f"({point.x:.1f}, {point.y:.1f})" for point in trajectory.points)
    vru_clearance = (
        "n/a" if score.vru_clearance_m is None else f"{score.vru_clearance_m:.3f} m"
    )
    return (
        f"Trajectory: {trajectory.label} [{trajectory.trajectory_id}]\n"
        f"Points: {points}\n"
        f"Score: {score.total:.3f}\n"
        "Metrics: "
        f"progress={score.progress_m:.3f} m, "
        f"collision_clearance={score.collision_clearance_m:.3f} m, "
        f"vru_clearance={vru_clearance}, "
        f"route_error={score.route_error_m:.3f} m, "
        f"comfort_cost={score.comfort_cost:.3f}"
    )


def _mean(values) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 3)
