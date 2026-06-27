from __future__ import annotations

from metricdrive.scenario import AgentTrack, Point, Rect, Scenario, Trajectory


def synthetic_scenarios() -> tuple[Scenario, ...]:
    """Return controlled long-tail planning scenarios for Milestone 1."""

    return (
        _pedestrian_crossing(),
        _unprotected_left_turn(),
        _cyclist_close_pass(),
        _blocked_lane(),
        _dense_merge(),
        _hard_braking_lead_vehicle(),
    )


def _trajectory(
    trajectory_id: str,
    points: tuple[tuple[float, float], ...],
    label: str,
    source: str = "candidate",
) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        points=tuple(Point(x, y) for x, y in points),
        label=label,
        source=source,
    )


def _agent(
    agent_id: str,
    agent_type: str,
    points: tuple[tuple[float, float], ...],
    radius_m: float,
) -> AgentTrack:
    return AgentTrack(
        agent_id=agent_id,
        agent_type=agent_type,
        states=tuple(Point(x, y) for x, y in points),
        radius_m=radius_m,
    )


def _pedestrian_crossing() -> Scenario:
    reference = _trajectory(
        "reference_logged_yield",
        ((0, 0), (1.8, 0), (3.2, 0), (6.0, -0.8), (10.5, -0.2)),
        "Logged yield trajectory",
        source="reference",
    )
    return Scenario(
        scenario_id="synthetic_pedestrian_crossing",
        category="pedestrian_crossing",
        title="Pedestrian crossing with tempting fast progress",
        prompt="Proceed along the route while yielding to a pedestrian crossing from the right.",
        route_goal=Point(12, 0),
        drivable_area=Rect(-1, 13, -3, 3),
        reference=reference,
        candidates=(
            _trajectory(
                "imitation_fast_log",
                ((0, 0), (2.5, 0), (5.4, 0.1), (8.5, 0), (12, 0)),
                "Fast log-like path",
            ),
            _trajectory(
                "metric_aligned_yield",
                ((0, 0), (1.8, 0), (3.2, 0), (6.0, -0.8), (10.5, -0.2)),
                "Yield, then continue",
            ),
            _trajectory(
                "cautious_stop",
                ((0, 0), (1.0, 0), (1.6, 0), (2.0, 0), (2.2, 0)),
                "Overly cautious stop",
            ),
        ),
        agents=(
            _agent(
                "pedestrian_1",
                "pedestrian",
                ((5.6, -2.2), (5.6, -1.1), (5.6, 0.2), (5.6, 1.2), (5.6, 2.1)),
                radius_m=0.4,
            ),
        ),
        tags=("vulnerable_road_user", "yielding", "long_tail"),
    )


def _unprotected_left_turn() -> Scenario:
    reference = _trajectory(
        "reference_wait_for_gap",
        ((0, -2.0), (2.2, -1.5), (3.5, -1.0), (6.6, 2.4), (9.0, 5.0)),
        "Logged gap-acceptance turn",
        source="reference",
    )
    return Scenario(
        scenario_id="synthetic_unprotected_left_turn",
        category="unprotected_left_turn",
        title="Unprotected left turn with oncoming vehicle",
        prompt="Turn left through the intersection after accepting a safe oncoming gap.",
        route_goal=Point(9, 5),
        drivable_area=Rect(-1, 11, -5, 6),
        reference=reference,
        candidates=(
            _trajectory(
                "imitation_aggressive_turn",
                ((0, -2.0), (2.8, -1.2), (5.3, 0.6), (7.5, 3.2), (9.2, 5.0)),
                "Aggressive early turn",
            ),
            _trajectory(
                "metric_aligned_gap_turn",
                ((0, -2.0), (1.8, -1.7), (3.2, -1.2), (6.3, 2.1), (9.0, 5.0)),
                "Wait for gap, then turn",
            ),
            _trajectory(
                "cautious_hold_position",
                ((0, -2.0), (0.8, -1.9), (1.2, -1.8), (1.2, -1.8), (1.2, -1.8)),
                "Hold position",
            ),
        ),
        agents=(
            _agent(
                "oncoming_vehicle",
                "vehicle",
                ((10.0, 0.7), (8.0, 0.7), (5.8, 0.7), (3.6, 0.7), (1.4, 0.7)),
                radius_m=1.1,
            ),
        ),
        tags=("intersection", "right_of_way", "long_tail"),
    )


def _cyclist_close_pass() -> Scenario:
    reference = _trajectory(
        "reference_wide_pass",
        ((0, 0), (2.0, -0.4), (4.3, -1.1), (7.8, -1.2), (12.0, -0.4)),
        "Logged wide pass",
        source="reference",
    )
    return Scenario(
        scenario_id="synthetic_cyclist_close_pass",
        category="cyclist_close_pass",
        title="Cyclist close pass on urban lane",
        prompt="Pass the cyclist while maintaining clearance and route progress.",
        route_goal=Point(12, 0),
        drivable_area=Rect(-1, 13, -3, 3),
        reference=reference,
        candidates=(
            _trajectory(
                "imitation_close_pass",
                ((0, 0), (2.4, 0), (4.8, 0), (8.2, 0), (12.0, 0)),
                "Close pass",
            ),
            _trajectory(
                "metric_aligned_wide_pass",
                ((0, 0), (2.0, -0.4), (4.3, -1.1), (7.8, -1.2), (12.0, -0.4)),
                "Wide pass",
            ),
            _trajectory(
                "cautious_follow",
                ((0, 0), (1.8, 0), (3.2, 0), (4.0, 0), (4.4, 0)),
                "Follow cyclist",
            ),
        ),
        agents=(
            _agent(
                "cyclist_1",
                "cyclist",
                ((2.2, 1.1), (4.2, 1.1), (6.2, 1.1), (8.2, 1.1), (10.2, 1.1)),
                radius_m=0.6,
            ),
        ),
        tags=("vulnerable_road_user", "cyclist", "clearance"),
    )


def _blocked_lane() -> Scenario:
    reference = _trajectory(
        "reference_nudge_left",
        ((0, 0), (2.4, -0.3), (4.8, -1.7), (8.0, -1.7), (12.0, -0.3)),
        "Logged nudge around blockage",
        source="reference",
    )
    return Scenario(
        scenario_id="synthetic_blocked_lane",
        category="blocked_lane",
        title="Blocked lane with construction object",
        prompt="Continue around a blocked lane without leaving the drivable region.",
        route_goal=Point(12, 0),
        drivable_area=Rect(-1, 13, -3, 3),
        reference=reference,
        candidates=(
            _trajectory(
                "imitation_straight_blocked",
                ((0, 0), (2.5, 0), (5.3, 0), (8.2, 0), (12.0, 0)),
                "Straight through blockage",
            ),
            _trajectory(
                "metric_aligned_nudge",
                ((0, 0), (2.4, -0.3), (4.8, -1.7), (8.0, -1.7), (12.0, -0.3)),
                "Nudge around blockage",
            ),
            _trajectory(
                "cautious_stop_behind",
                ((0, 0), (1.7, 0), (2.8, 0), (3.5, 0), (3.8, 0)),
                "Stop behind blockage",
            ),
        ),
        agents=(
            _agent(
                "construction_block",
                "static_obstacle",
                ((6.0, 0), (6.0, 0), (6.0, 0), (6.0, 0), (6.0, 0)),
                radius_m=0.8,
            ),
        ),
        tags=("blocked_lane", "nudge", "static_obstacle"),
    )


def _dense_merge() -> Scenario:
    reference = _trajectory(
        "reference_gap_merge",
        ((0, -1.4), (2.0, -1.3), (4.2, -1.0), (6.0, -0.7), (10.8, 0.0)),
        "Logged merge into gap",
        source="reference",
    )
    return Scenario(
        scenario_id="synthetic_dense_merge",
        category="dense_merge",
        title="Dense merge with adjacent-lane gap",
        prompt="Merge into the route lane without forcing adjacent traffic to brake hard.",
        route_goal=Point(12, 0),
        drivable_area=Rect(-1, 13, -3, 3),
        reference=reference,
        candidates=(
            _trajectory(
                "imitation_force_merge",
                ((0, -1.4), (2.5, -1.0), (4.8, -0.2), (7.8, 0.0), (12.0, 0.0)),
                "Force merge",
            ),
            _trajectory(
                "metric_aligned_gap_merge",
                ((0, -1.4), (2.0, -1.3), (4.2, -1.0), (6.0, -0.7), (10.8, 0.0)),
                "Merge into gap",
            ),
            _trajectory(
                "cautious_wait_merge",
                ((0, -1.4), (1.5, -1.4), (2.5, -1.4), (3.2, -1.2), (4.0, -1.0)),
                "Wait for larger gap",
            ),
        ),
        agents=(
            _agent(
                "lead_vehicle",
                "vehicle",
                ((5.0, 0.4), (6.5, 0.4), (8.0, 0.4), (10.5, 0.4), (13.2, 0.4)),
                radius_m=0.8,
            ),
            _agent(
                "trailing_vehicle",
                "vehicle",
                ((-2.0, 0.4), (0.0, 0.4), (2.0, 0.4), (4.0, 0.4), (6.0, 0.4)),
                radius_m=0.8,
            ),
        ),
        tags=("merge", "dense_traffic", "interactive"),
    )


def _hard_braking_lead_vehicle() -> Scenario:
    reference = _trajectory(
        "reference_smooth_brake",
        ((0, 0), (2.0, 0), (4.0, 0), (5.2, 0), (5.8, 0)),
        "Logged smooth braking",
        source="reference",
    )
    return Scenario(
        scenario_id="synthetic_hard_braking_lead_vehicle",
        category="hard_braking_lead_vehicle",
        title="Lead vehicle hard braking",
        prompt="Maintain progress while braking smoothly behind a suddenly slowing lead vehicle.",
        route_goal=Point(8, 0),
        drivable_area=Rect(-1, 13, -3, 3),
        reference=reference,
        candidates=(
            _trajectory(
                "imitation_maintain_speed",
                ((0, 0), (3.0, 0), (6.0, 0), (9.0, 0), (12.0, 0)),
                "Maintain speed",
            ),
            _trajectory(
                "metric_aligned_smooth_brake",
                ((0, 0), (2.0, 0), (4.0, 0), (5.2, 0), (5.8, 0)),
                "Smooth brake",
            ),
            _trajectory(
                "cautious_hard_stop",
                ((0, 0), (1.5, 0), (2.0, 0), (2.1, 0), (2.1, 0)),
                "Abrupt stop",
            ),
        ),
        agents=(
            _agent(
                "lead_vehicle",
                "vehicle",
                ((5.5, 0), (7.0, 0), (8.0, 0), (8.5, 0), (8.8, 0)),
                radius_m=1.1,
            ),
        ),
        tags=("following", "hard_braking", "comfort"),
    )
