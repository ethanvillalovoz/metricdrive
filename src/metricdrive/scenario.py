from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """2D point in ego-centric meters."""

    x: float
    y: float


@dataclass(frozen=True)
class Trajectory:
    """Candidate ego trajectory."""

    trajectory_id: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class DemoScenario:
    """Small scenario fixture for the initial metric demo."""

    scenario_id: str
    route_goal: Point
    vru_position: Point
    obstacle_position: Point
    candidates: tuple[Trajectory, ...]
