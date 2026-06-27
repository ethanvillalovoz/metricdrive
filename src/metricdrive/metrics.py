from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from metricdrive.scenario import AgentTrack, Point, Scenario, Trajectory

EGO_RADIUS_M = 1.0


def distance(left: Point, right: Point) -> float:
    return hypot(left.x - right.x, left.y - right.y)


def min_distance_to_point(trajectory: Trajectory, point: Point) -> float:
    return min(distance(candidate_point, point) for candidate_point in trajectory.points)


def path_length(trajectory: Trajectory) -> float:
    if len(trajectory.points) < 2:
        return 0.0
    return sum(
        distance(previous, current)
        for previous, current in zip(trajectory.points, trajectory.points[1:])
    )


def smoothness_cost(trajectory: Trajectory) -> float:
    """Approximate comfort cost from heading changes in a 2D path."""

    if len(trajectory.points) < 3:
        return 0.0

    cost = 0.0
    for first, second, third in zip(
        trajectory.points,
        trajectory.points[1:],
        trajectory.points[2:],
    ):
        first_step = distance(first, second)
        second_step = distance(second, third)
        if first_step == 0 or second_step == 0:
            continue

        vx1 = (second.x - first.x) / first_step
        vy1 = (second.y - first.y) / first_step
        vx2 = (third.x - second.x) / second_step
        vy2 = (third.y - second.y) / second_step
        cost += distance(Point(vx1, vy1), Point(vx2, vy2))
    return cost


def progress_score(trajectory: Trajectory, goal: Point) -> float:
    if not trajectory.points:
        return 0.0
    start_distance = distance(trajectory.points[0], goal)
    end_distance = distance(trajectory.points[-1], goal)
    return max(0.0, start_distance - end_distance)


@dataclass(frozen=True)
class PlanningScore:
    """Interpretable score for one trajectory candidate."""

    scenario_id: str
    category: str
    trajectory_id: str
    total: float
    progress_m: float
    collision_clearance_m: float
    vru_clearance_m: float | None
    offroad_rate: float
    comfort_cost: float
    route_error_m: float
    imitation_error_m: float
    components: dict[str, float]


def offroad_rate(trajectory: Trajectory, scenario: Scenario) -> float:
    if not trajectory.points:
        return 1.0
    offroad_points = sum(
        not scenario.drivable_area.contains(point) for point in trajectory.points
    )
    return offroad_points / len(trajectory.points)


def route_error(trajectory: Trajectory, goal: Point) -> float:
    if not trajectory.points:
        return float("inf")
    return distance(trajectory.points[-1], goal)


def mean_point_error(left: Trajectory, right: Trajectory) -> float:
    paired_points = tuple(zip(left.points, right.points))
    if not paired_points:
        return float("inf")
    return sum(
        distance(left_point, right_point) for left_point, right_point in paired_points
    ) / len(paired_points)


def route_axis_progress(trajectory: Trajectory, goal: Point) -> float:
    """Return signed progress along the route axis without safety awareness."""

    if not trajectory.points:
        return 0.0

    start = trajectory.points[0]
    end = trajectory.points[-1]
    route_x = goal.x - start.x
    route_y = goal.y - start.y
    route_length = hypot(route_x, route_y)
    if route_length == 0:
        return 0.0

    unit_x = route_x / route_length
    unit_y = route_y / route_length
    displacement_x = end.x - start.x
    displacement_y = end.y - start.y
    return (displacement_x * unit_x) + (displacement_y * unit_y)


def min_agent_clearance(
    trajectory: Trajectory,
    agents: tuple[AgentTrack, ...],
    agent_types: set[str] | None = None,
) -> float:
    """Return minimum same-step clearance after subtracting approximate radii."""

    best = float("inf")
    for agent in agents:
        if agent_types is not None and agent.agent_type not in agent_types:
            continue
        if not agent.states:
            continue
        for index, ego_point in enumerate(trajectory.points):
            agent_point = agent.states[min(index, len(agent.states) - 1)]
            clearance = distance(ego_point, agent_point) - EGO_RADIUS_M - agent.radius_m
            best = min(best, clearance)
    return 20.0 if best == float("inf") else best


def min_center_distance_to_agents(
    trajectory: Trajectory,
    agents: tuple[AgentTrack, ...],
    agent_types: set[str],
) -> float | None:
    best = float("inf")
    for agent in agents:
        if agent.agent_type not in agent_types or not agent.states:
            continue
        for index, ego_point in enumerate(trajectory.points):
            agent_point = agent.states[min(index, len(agent.states) - 1)]
            best = min(best, distance(ego_point, agent_point))
    return None if best == float("inf") else best


def score_trajectory(scenario: Scenario, trajectory: Trajectory) -> PlanningScore:
    progress = progress_score(trajectory, scenario.route_goal)
    collision_clearance = min_agent_clearance(trajectory, scenario.agents)
    vru_clearance = min_center_distance_to_agents(
        trajectory,
        scenario.agents,
        {"pedestrian", "cyclist"},
    )
    offroad = offroad_rate(trajectory, scenario)
    comfort = smoothness_cost(trajectory)
    final_route_error = route_error(trajectory, scenario.route_goal)
    imitation = mean_point_error(trajectory, scenario.reference)

    components = {
        "progress": round(progress * 0.8, 3),
        "collision_clearance": round(
            min(max(collision_clearance, 0.0), 2.0) * 2.0
            - max(0.0, -collision_clearance) * 25.0,
            3,
        ),
        "vru_clearance": 0.0,
        "offroad": round(-offroad * 20.0, 3),
        "comfort": round(-comfort * 1.25, 3),
        "route_error": round(-final_route_error * 0.3, 3),
        "imitation": round(-imitation * 0.25, 3),
    }

    if vru_clearance is not None:
        components["vru_clearance"] = round(
            min(max(vru_clearance, 0.0), 5.0) * 1.5
            - max(0.0, 2.0 - vru_clearance) * 6.0,
            3,
        )

    return PlanningScore(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        trajectory_id=trajectory.trajectory_id,
        total=round(sum(components.values()), 3),
        progress_m=round(progress, 3),
        collision_clearance_m=round(collision_clearance, 3),
        vru_clearance_m=None if vru_clearance is None else round(vru_clearance, 3),
        offroad_rate=round(offroad, 3),
        comfort_cost=round(comfort, 3),
        route_error_m=round(final_route_error, 3),
        imitation_error_m=round(imitation, 3),
        components=components,
    )


def ranked_scores(scenario: Scenario) -> tuple[PlanningScore, ...]:
    return tuple(
        sorted(
            (score_trajectory(scenario, candidate) for candidate in scenario.candidates),
            key=lambda score: score.total,
            reverse=True,
        )
    )


def best_score(scenario: Scenario) -> PlanningScore:
    return ranked_scores(scenario)[0]
