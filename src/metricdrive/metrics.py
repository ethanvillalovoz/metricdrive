from __future__ import annotations

from math import hypot

from metricdrive.scenario import Point, Trajectory


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
