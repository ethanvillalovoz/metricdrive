from __future__ import annotations

from dataclasses import dataclass

from metricdrive.metrics import (
    min_distance_to_point,
    progress_score,
    smoothness_cost,
)
from metricdrive.scenario import DemoScenario, Point, Trajectory


@dataclass(frozen=True)
class CandidateScore:
    trajectory_id: str
    total: float
    progress: float
    vru_clearance: float
    obstacle_clearance: float
    comfort_penalty: float


def built_in_demo_scenario() -> DemoScenario:
    """Return a tiny pedestrian-yield scenario for CLI smoke tests."""

    return DemoScenario(
        scenario_id="toy_pedestrian_yield",
        route_goal=Point(12.0, 0.0),
        vru_position=Point(5.6, 0.2),
        obstacle_position=Point(7.0, -1.2),
        candidates=(
            Trajectory(
                trajectory_id="imitate_fast_log",
                points=(
                    Point(0.0, 0.0),
                    Point(2.5, 0.0),
                    Point(5.4, 0.1),
                    Point(8.5, 0.0),
                    Point(12.0, 0.0),
                ),
            ),
            Trajectory(
                trajectory_id="yield_then_progress",
                points=(
                    Point(0.0, 0.0),
                    Point(1.8, 0.0),
                    Point(3.2, 0.0),
                    Point(6.0, -0.8),
                    Point(10.5, -0.2),
                ),
            ),
            Trajectory(
                trajectory_id="overly_cautious_stop",
                points=(
                    Point(0.0, 0.0),
                    Point(1.0, 0.0),
                    Point(1.6, 0.0),
                    Point(2.0, 0.0),
                    Point(2.2, 0.0),
                ),
            ),
        ),
    )


def score_candidate(scenario: DemoScenario, trajectory: Trajectory) -> CandidateScore:
    progress = progress_score(trajectory, scenario.route_goal)
    vru_clearance = min_distance_to_point(trajectory, scenario.vru_position)
    obstacle_clearance = min_distance_to_point(trajectory, scenario.obstacle_position)
    comfort_penalty = smoothness_cost(trajectory)
    vru_safety_penalty = max(0.0, 1.5 - vru_clearance) * 6.0

    total = (
        progress
        + min(vru_clearance, 2.0) * 2.0
        + min(obstacle_clearance, 2.0)
        - comfort_penalty * 1.25
        - vru_safety_penalty
    )
    return CandidateScore(
        trajectory_id=trajectory.trajectory_id,
        total=round(total, 3),
        progress=round(progress, 3),
        vru_clearance=round(vru_clearance, 3),
        obstacle_clearance=round(obstacle_clearance, 3),
        comfort_penalty=round(comfort_penalty, 3),
    )


def ranked_demo_scores() -> tuple[CandidateScore, ...]:
    scenario = built_in_demo_scenario()
    return tuple(
        sorted(
            (score_candidate(scenario, candidate) for candidate in scenario.candidates),
            key=lambda score: score.total,
            reverse=True,
        )
    )
