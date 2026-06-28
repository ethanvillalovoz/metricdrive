from __future__ import annotations

from dataclasses import asdict, dataclass

from metricdrive.metrics import PlanningScore, score_trajectory
from metricdrive.scenario import Scenario, Trajectory


@dataclass(frozen=True)
class MetaAction:
    """Verifiable high-level planning action attached to a trajectory."""

    action_id: str
    label: str
    description: str
    verification_metrics: tuple[str, ...]


META_ACTIONS: dict[str, MetaAction] = {
    "YIELD_TO_VRU": MetaAction(
        action_id="YIELD_TO_VRU",
        label="Yield to vulnerable road user",
        description="Slow or pause enough to preserve pedestrian/cyclist clearance.",
        verification_metrics=("vru_clearance_m", "collision_clearance_m", "progress_m"),
    ),
    "MAINTAIN_VRU_CLEARANCE": MetaAction(
        action_id="MAINTAIN_VRU_CLEARANCE",
        label="Maintain vulnerable-user clearance",
        description="Pass or follow while keeping extra lateral clearance.",
        verification_metrics=("vru_clearance_m", "collision_clearance_m", "route_error_m"),
    ),
    "NUDGE_AROUND_OBSTACLE": MetaAction(
        action_id="NUDGE_AROUND_OBSTACLE",
        label="Nudge around obstacle",
        description="Leave the nominal lane centerline while preserving drivable-area and obstacle clearance.",
        verification_metrics=("collision_clearance_m", "offroad_rate", "route_error_m"),
    ),
    "MERGE_INTO_GAP": MetaAction(
        action_id="MERGE_INTO_GAP",
        label="Merge into gap",
        description="Enter the target lane only when the adjacent vehicle clearance remains positive.",
        verification_metrics=("collision_clearance_m", "progress_m", "comfort_cost"),
    ),
    "ACCEPT_INTERSECTION_GAP": MetaAction(
        action_id="ACCEPT_INTERSECTION_GAP",
        label="Accept intersection gap",
        description="Wait for a safe oncoming gap before completing an unprotected turn.",
        verification_metrics=("collision_clearance_m", "progress_m", "route_error_m"),
    ),
    "SLOW_FOR_LEAD": MetaAction(
        action_id="SLOW_FOR_LEAD",
        label="Slow for lead vehicle",
        description="Trade progress for safe following clearance and smooth braking.",
        verification_metrics=("collision_clearance_m", "comfort_cost", "progress_m"),
    ),
    "WAIT_OR_HOLD": MetaAction(
        action_id="WAIT_OR_HOLD",
        label="Wait or hold",
        description="Delay progress when the interaction is not yet safe enough.",
        verification_metrics=("progress_m", "collision_clearance_m", "route_error_m"),
    ),
    "PRESS_FOR_PROGRESS": MetaAction(
        action_id="PRESS_FOR_PROGRESS",
        label="Press for progress",
        description="Prioritize route progress, often exposing safety or comfort failures.",
        verification_metrics=("progress_m", "collision_clearance_m", "vru_clearance_m"),
    ),
    "STABILIZE_LATERAL_MOTION": MetaAction(
        action_id="STABILIZE_LATERAL_MOTION",
        label="Stabilize lateral motion",
        description="Reject unnecessary lateral oscillation that hurts comfort or route adherence.",
        verification_metrics=("comfort_cost", "route_error_m", "collision_clearance_m"),
    ),
    "FOLLOW_ROUTE": MetaAction(
        action_id="FOLLOW_ROUTE",
        label="Follow route",
        description="Continue along the route while satisfying the scenario constraints.",
        verification_metrics=("progress_m", "route_error_m", "collision_clearance_m"),
    ),
}


def infer_meta_action(scenario: Scenario, trajectory: Trajectory) -> MetaAction:
    """Infer a compact public-safe meta-action label from scenario and candidate IDs."""

    candidate_id = trajectory.trajectory_id
    category = scenario.category

    if "lateral_wobble" in candidate_id:
        return META_ACTIONS["STABILIZE_LATERAL_MOTION"]
    if candidate_id.startswith("imitation_") or "progress_pressure" in candidate_id:
        return META_ACTIONS["PRESS_FOR_PROGRESS"]
    if any(token in candidate_id for token in ("cautious", "under_commit", "hold", "wait", "stop")):
        return META_ACTIONS["WAIT_OR_HOLD"]
    if category == "pedestrian_crossing" or "yield" in candidate_id:
        return META_ACTIONS["YIELD_TO_VRU"]
    if category == "cyclist_close_pass" or "wide_pass" in candidate_id:
        return META_ACTIONS["MAINTAIN_VRU_CLEARANCE"]
    if category == "blocked_lane" or "nudge" in candidate_id:
        return META_ACTIONS["NUDGE_AROUND_OBSTACLE"]
    if category == "unprotected_left_turn" or "gap_turn" in candidate_id:
        return META_ACTIONS["ACCEPT_INTERSECTION_GAP"]
    if category == "dense_merge" or "merge" in candidate_id:
        return META_ACTIONS["MERGE_INTO_GAP"]
    if category == "hard_braking_lead_vehicle" or "brake" in candidate_id:
        return META_ACTIONS["SLOW_FOR_LEAD"]
    return META_ACTIONS["FOLLOW_ROUTE"]


def meta_action_payload(
    scenario: Scenario,
    trajectory: Trajectory,
    score: PlanningScore | None = None,
) -> dict[str, object]:
    resolved_score = score if score is not None else score_trajectory(scenario, trajectory)
    action = infer_meta_action(scenario, trajectory)
    payload = asdict(action)
    payload["checks"] = {
        "total_score": resolved_score.total,
        "progress_m": resolved_score.progress_m,
        "collision_clearance_m": resolved_score.collision_clearance_m,
        "vru_clearance_m": resolved_score.vru_clearance_m,
        "offroad_rate": resolved_score.offroad_rate,
        "comfort_cost": resolved_score.comfort_cost,
        "route_error_m": resolved_score.route_error_m,
        "imitation_error_m": resolved_score.imitation_error_m,
    }
    return payload
