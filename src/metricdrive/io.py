from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metricdrive.scenario import AgentTrack, Point, Rect, Scenario, Trajectory


def save_scenarios(path: str | Path, scenarios: tuple[Scenario, ...]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([scenario_to_dict(scenario) for scenario in scenarios], indent=2)
        + "\n",
        encoding="utf-8",
    )


def load_scenarios(path: str | Path) -> tuple[Scenario, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(scenario_from_dict(item) for item in data)


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "title": scenario.title,
        "prompt": scenario.prompt,
        "route_goal": point_to_dict(scenario.route_goal),
        "drivable_area": {
            "min_x": scenario.drivable_area.min_x,
            "max_x": scenario.drivable_area.max_x,
            "min_y": scenario.drivable_area.min_y,
            "max_y": scenario.drivable_area.max_y,
        },
        "reference": trajectory_to_dict(scenario.reference),
        "candidates": [trajectory_to_dict(candidate) for candidate in scenario.candidates],
        "agents": [agent_to_dict(agent) for agent in scenario.agents],
        "tags": list(scenario.tags),
    }


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    area = data["drivable_area"]
    return Scenario(
        scenario_id=data["scenario_id"],
        category=data["category"],
        title=data["title"],
        prompt=data["prompt"],
        route_goal=point_from_dict(data["route_goal"]),
        drivable_area=Rect(
            min_x=area["min_x"],
            max_x=area["max_x"],
            min_y=area["min_y"],
            max_y=area["max_y"],
        ),
        reference=trajectory_from_dict(data["reference"]),
        candidates=tuple(trajectory_from_dict(item) for item in data["candidates"]),
        agents=tuple(agent_from_dict(item) for item in data["agents"]),
        tags=tuple(data.get("tags", ())),
    )


def point_to_dict(point: Point) -> dict[str, float]:
    return {"x": point.x, "y": point.y}


def point_from_dict(data: dict[str, Any]) -> Point:
    return Point(float(data["x"]), float(data["y"]))


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "label": trajectory.label,
        "source": trajectory.source,
        "points": [point_to_dict(point) for point in trajectory.points],
    }


def trajectory_from_dict(data: dict[str, Any]) -> Trajectory:
    return Trajectory(
        trajectory_id=data["trajectory_id"],
        label=data.get("label", ""),
        source=data.get("source", "candidate"),
        points=tuple(point_from_dict(point) for point in data["points"]),
    )


def agent_to_dict(agent: AgentTrack) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "agent_type": agent.agent_type,
        "radius_m": agent.radius_m,
        "states": [point_to_dict(point) for point in agent.states],
    }


def agent_from_dict(data: dict[str, Any]) -> AgentTrack:
    return AgentTrack(
        agent_id=data["agent_id"],
        agent_type=data["agent_type"],
        radius_m=float(data.get("radius_m", 0.8)),
        states=tuple(point_from_dict(point) for point in data["states"]),
    )
