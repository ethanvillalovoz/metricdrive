from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from metricdrive.metrics import ranked_scores, route_axis_progress
from metricdrive.scenario import Scenario, Trajectory


class Planner(Protocol):
    """Common planner interface for benchmarkable trajectory selection."""

    planner_id: str
    label: str

    def plan(self, scenario: Scenario) -> Trajectory:
        """Return one ego trajectory for the scenario."""


@dataclass(frozen=True)
class ReferenceImitationPlanner:
    """Returns the logged reference trajectory."""

    planner_id: str = "reference_imitation"
    label: str = "Reference imitation"

    def plan(self, scenario: Scenario) -> Trajectory:
        return scenario.reference


@dataclass(frozen=True)
class ProgressOnlyPlanner:
    """Selects the most route-progressive candidate while ignoring safety."""

    planner_id: str = "progress_only"
    label: str = "Progress only"

    def plan(self, scenario: Scenario) -> Trajectory:
        return max(
            scenario.candidates,
            key=lambda candidate: route_axis_progress(candidate, scenario.route_goal),
        )


@dataclass(frozen=True)
class MetricRerankPlanner:
    """Selects the candidate with the best transparent planning score."""

    planner_id: str = "metric_rerank"
    label: str = "Metric rerank"

    def plan(self, scenario: Scenario) -> Trajectory:
        best = ranked_scores(scenario)[0]
        for candidate in scenario.candidates:
            if candidate.trajectory_id == best.trajectory_id:
                return candidate
        raise ValueError(f"Best candidate is missing from scenario: {best.trajectory_id}")


def default_planners() -> tuple[Planner, ...]:
    return (
        ReferenceImitationPlanner(),
        ProgressOnlyPlanner(),
        MetricRerankPlanner(),
    )
