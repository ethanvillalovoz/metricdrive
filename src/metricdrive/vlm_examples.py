from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from metricdrive.hard_negatives import augment_scenarios_with_hard_negatives
from metricdrive.meta_actions import infer_meta_action, meta_action_payload
from metricdrive.metrics import PlanningScore, ranked_scores, score_trajectory
from metricdrive.preferences import PreferencePair, generate_preferences
from metricdrive.scenario import Scenario, Trajectory


@dataclass(frozen=True)
class VlmPlanningExample:
    """DPO-style planning row with a structured driving prompt and response pair."""

    example_id: str
    scenario_id: str
    category: str
    tags: tuple[str, ...]
    preferred_trajectory_id: str
    rejected_trajectory_id: str
    preferred_meta_action: str
    rejected_meta_action: str
    score_margin: float
    reason_categories: tuple[str, ...]
    prompt: str
    chosen: str
    rejected: str


def generate_vlm_examples(
    scenarios: tuple[Scenario, ...],
    include_hard_negatives: bool = True,
    min_score_margin: float = 0.0,
) -> tuple[VlmPlanningExample, ...]:
    """Build public-safe VLM planning preference rows from metric rankings."""

    source_scenarios = (
        augment_scenarios_with_hard_negatives(scenarios)
        if include_hard_negatives
        else scenarios
    )
    scenario_by_id = {scenario.scenario_id: scenario for scenario in source_scenarios}
    pairs = generate_preferences(source_scenarios, min_score_margin=min_score_margin)
    return tuple(
        _vlm_example(
            scenario=scenario_by_id[pair.scenario_id],
            pair=pair,
            index=index,
        )
        for index, pair in enumerate(pairs, start=1)
    )


def vlm_examples_payload(examples: tuple[VlmPlanningExample, ...]) -> dict[str, object]:
    return {
        "format": "metricdrive.vlm_examples.v1",
        "example_count": len(examples),
        "examples": [asdict(example) for example in examples],
    }


def json_vlm_examples(examples: tuple[VlmPlanningExample, ...]) -> str:
    return json.dumps(vlm_examples_payload(examples), indent=2) + "\n"


def jsonl_vlm_examples(examples: tuple[VlmPlanningExample, ...]) -> str:
    return "".join(json.dumps(asdict(example), sort_keys=True) + "\n" for example in examples)


def markdown_vlm_examples(
    examples: tuple[VlmPlanningExample, ...],
    limit: int | None = 12,
) -> str:
    shown = examples if limit is None else examples[:limit]
    lines = [
        "# MetricDrive VLM Planning Examples",
        "",
        f"- Example count: {len(examples)}",
        "- Shape: prompt, chosen trajectory response, rejected trajectory response",
        "- Labels: metric-derived preferences with verifiable meta-actions",
        "",
        "| Scenario | Chosen action | Rejected action | Chosen | Rejected | Margin | Reasons |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for example in shown:
        lines.append(
            "| "
            f"{example.scenario_id} | `{example.preferred_meta_action}` | "
            f"`{example.rejected_meta_action}` | `{example.preferred_trajectory_id}` | "
            f"`{example.rejected_trajectory_id}` | {example.score_margin:.3f} | "
            f"{', '.join(example.reason_categories)} |"
        )
    return "\n".join(lines) + "\n"


def write_vlm_examples(
    path: str | Path,
    examples: tuple[VlmPlanningExample, ...],
    output_format: str = "jsonl",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        target.write_text(json_vlm_examples(examples), encoding="utf-8")
    elif output_format == "markdown":
        target.write_text(markdown_vlm_examples(examples, limit=None), encoding="utf-8")
    else:
        target.write_text(jsonl_vlm_examples(examples), encoding="utf-8")


def _vlm_example(
    scenario: Scenario,
    pair: PreferencePair,
    index: int,
) -> VlmPlanningExample:
    score_by_id = {score.trajectory_id: score for score in ranked_scores(scenario)}
    trajectory_by_id = {
        candidate.trajectory_id: candidate for candidate in scenario.candidates
    }
    preferred = trajectory_by_id[pair.preferred_trajectory_id]
    rejected = trajectory_by_id[pair.rejected_trajectory_id]
    preferred_score = score_by_id[pair.preferred_trajectory_id]
    rejected_score = score_by_id[pair.rejected_trajectory_id]
    preferred_action = infer_meta_action(scenario, preferred)
    rejected_action = infer_meta_action(scenario, rejected)
    return VlmPlanningExample(
        example_id=f"vlm_{index:04d}",
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        tags=scenario.tags,
        preferred_trajectory_id=preferred.trajectory_id,
        rejected_trajectory_id=rejected.trajectory_id,
        preferred_meta_action=preferred_action.action_id,
        rejected_meta_action=rejected_action.action_id,
        score_margin=pair.score_margin,
        reason_categories=pair.reason_categories,
        prompt=_planning_prompt(scenario),
        chosen=_planning_response(scenario, preferred, preferred_score),
        rejected=_planning_response(scenario, rejected, rejected_score),
    )


def _planning_prompt(scenario: Scenario) -> str:
    candidate_lines = []
    scores = ranked_scores(scenario)
    score_by_id = {score.trajectory_id: score for score in scores}
    for candidate in scenario.candidates:
        score = score_by_id[candidate.trajectory_id]
        action = infer_meta_action(scenario, candidate)
        candidate_lines.append(
            "- "
            f"{candidate.trajectory_id}: action={action.action_id}, "
            f"source={candidate.source}, progress={score.progress_m:.3f}m, "
            f"collision_clearance={score.collision_clearance_m:.3f}m, "
            f"route_error={score.route_error_m:.3f}m, total={score.total:.3f}"
        )

    return "\n".join(
        (
            "Driving task:",
            scenario.prompt,
            "",
            "Choose the safest useful ego trajectory. Return JSON with fields "
            "`trajectory_id`, `meta_action`, `rationale`, and `metric_checks`.",
            "",
            "Candidate trajectories:",
            *candidate_lines,
        )
    )


def _planning_response(
    scenario: Scenario,
    trajectory: Trajectory,
    score: PlanningScore | None = None,
) -> str:
    resolved_score = score if score is not None else score_trajectory(scenario, trajectory)
    action_payload = meta_action_payload(scenario, trajectory, resolved_score)
    response = {
        "trajectory_id": trajectory.trajectory_id,
        "meta_action": action_payload["action_id"],
        "rationale": _rationale(resolved_score),
        "metric_checks": action_payload["checks"],
    }
    return json.dumps(response, sort_keys=True)


def _rationale(score: PlanningScore) -> str:
    reasons = []
    if score.collision_clearance_m >= 0:
        reasons.append("keeps positive collision clearance")
    else:
        reasons.append("has negative collision clearance")
    if score.vru_clearance_m is not None:
        reasons.append(f"keeps VRU clearance at {score.vru_clearance_m:.3f} m")
    reasons.append(f"makes {score.progress_m:.3f} m of route progress")
    reasons.append(f"ends {score.route_error_m:.3f} m from the goal")
    return "; ".join(reasons)
