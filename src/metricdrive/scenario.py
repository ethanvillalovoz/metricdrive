from __future__ import annotations

from dataclasses import dataclass, field


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
    label: str = ""
    source: str = "candidate"


@dataclass(frozen=True)
class DemoScenario:
    """Small scenario fixture for the initial metric demo."""

    scenario_id: str
    route_goal: Point
    vru_position: Point
    obstacle_position: Point
    candidates: tuple[Trajectory, ...]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned drivable region in ego-centric meters."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, point: Point) -> bool:
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
        )


@dataclass(frozen=True)
class AgentTrack:
    """A non-ego actor track sampled at the same horizon as candidates."""

    agent_id: str
    agent_type: str
    states: tuple[Point, ...]
    radius_m: float = 0.8


@dataclass(frozen=True)
class Scenario:
    """Long-tail planning scenario used by the Milestone 1 evaluator."""

    scenario_id: str
    category: str
    title: str
    prompt: str
    route_goal: Point
    drivable_area: Rect
    reference: Trajectory
    candidates: tuple[Trajectory, ...]
    agents: tuple[AgentTrack, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
