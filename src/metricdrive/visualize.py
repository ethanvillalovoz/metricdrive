from __future__ import annotations

from html import escape

from metricdrive.metrics import ranked_scores
from metricdrive.scenario import AgentTrack, Point, Rect, Scenario, Trajectory


def scenario_svg(scenario: Scenario, width: int = 900, height: int = 460) -> str:
    scores = ranked_scores(scenario)
    best_id = scores[0].trajectory_id
    colors = {
        "best": "#0f766e",
        "candidate": "#64748b",
        "reference": "#2563eb",
        "agent": "#dc2626",
        "vru": "#c026d3",
    }
    margin = 42
    bounds = scenario.drivable_area

    def project(point: Point) -> tuple[float, float]:
        x = margin + (point.x - bounds.min_x) / (bounds.max_x - bounds.min_x) * (
            width - margin * 2
        )
        y = height - margin - (point.y - bounds.min_y) / (bounds.max_y - bounds.min_y) * (
            height - margin * 2
        )
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(scenario.title)}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#0f172a}",
        ".small{font-size:13px}.label{font-size:14px;font-weight:700}",
        ".road{fill:#f8fafc;stroke:#94a3b8;stroke-width:2}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _rect_svg(bounds, project, css_class="road"),
        f'<text x="{margin}" y="26" class="label">{escape(scenario.title)}</text>',
        f'<text x="{margin}" y="44" class="small">{escape(scenario.category)}</text>',
    ]

    parts.extend(_grid_svg(bounds, project))
    parts.append(_trajectory_svg(scenario.reference, project, colors["reference"], 3, "6 5"))
    for candidate in scenario.candidates:
        is_best = candidate.trajectory_id == best_id
        parts.append(
            _trajectory_svg(
                candidate,
                project,
                colors["best"] if is_best else colors["candidate"],
                5 if is_best else 2,
                "" if is_best else "4 6",
            )
        )
    for agent in scenario.agents:
        parts.append(_agent_svg(agent, project, colors))

    goal_x, goal_y = project(scenario.route_goal)
    parts.append(
        f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="7" fill="#facc15" stroke="#854d0e" stroke-width="2"/>'
    )
    parts.append(f'<text x="{goal_x + 10:.1f}" y="{goal_y - 10:.1f}" class="small">goal</text>')

    legend_y = height - 18
    parts.extend(
        (
            _legend_item(42, legend_y, colors["best"], "metric top candidate"),
            _legend_item(220, legend_y, colors["reference"], "logged reference"),
            _legend_item(370, legend_y, colors["agent"], "vehicle/obstacle"),
            _legend_item(535, legend_y, colors["vru"], "VRU"),
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def planner_comparison_svg(
    scenario: Scenario,
    selected_trajectories: dict[str, Trajectory],
    width: int = 900,
    height: int = 460,
) -> str:
    """Render selected planner trajectories for one scenario."""

    margin = 42
    bounds = scenario.drivable_area
    colors = {
        "reference_imitation": "#2563eb",
        "progress_only": "#dc2626",
        "metric_rerank": "#0f766e",
    }
    labels = {
        "reference_imitation": "reference imitation",
        "progress_only": "progress only",
        "metric_rerank": "metric rerank",
    }

    def project(point: Point) -> tuple[float, float]:
        x = margin + (point.x - bounds.min_x) / (bounds.max_x - bounds.min_x) * (
            width - margin * 2
        )
        y = height - margin - (point.y - bounds.min_y) / (bounds.max_y - bounds.min_y) * (
            height - margin * 2
        )
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(scenario.title)} planner comparison">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#0f172a}",
        ".small{font-size:13px}.label{font-size:14px;font-weight:700}",
        ".road{fill:#f8fafc;stroke:#94a3b8;stroke-width:2}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _rect_svg(bounds, project, css_class="road"),
        f'<text x="{margin}" y="26" class="label">{escape(scenario.title)}</text>',
        f'<text x="{margin}" y="44" class="small">planner benchmark</text>',
    ]

    parts.extend(_grid_svg(bounds, project))
    for planner_id, trajectory in selected_trajectories.items():
        parts.append(
            _trajectory_svg(
                trajectory,
                project,
                colors.get(planner_id, "#64748b"),
                5 if planner_id == "metric_rerank" else 3,
                "5 5" if planner_id != "metric_rerank" else "",
            )
        )
    for agent in scenario.agents:
        parts.append(
            _agent_svg(
                agent,
                project,
                {"agent": "#334155", "vru": "#c026d3"},
            )
        )

    goal_x, goal_y = project(scenario.route_goal)
    parts.append(
        f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="7" fill="#facc15" stroke="#854d0e" stroke-width="2"/>'
    )
    parts.append(f'<text x="{goal_x + 10:.1f}" y="{goal_y - 10:.1f}" class="small">goal</text>')

    legend_y = height - 18
    legend_x = 42
    for planner_id in selected_trajectories:
        parts.append(
            _legend_item(
                legend_x,
                legend_y,
                colors.get(planner_id, "#64748b"),
                labels.get(planner_id, planner_id),
            )
        )
        legend_x += 170
    parts.append(_legend_item(legend_x, legend_y, "#c026d3", "VRU"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _rect_svg(
    rect: Rect,
    project: callable,
    css_class: str,
) -> str:
    top_left = project(Point(rect.min_x, rect.max_y))
    bottom_right = project(Point(rect.max_x, rect.min_y))
    x = top_left[0]
    y = top_left[1]
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    return (
        f'<rect class="{css_class}" x="{x:.1f}" y="{y:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" rx="4"/>'
    )


def _grid_svg(rect: Rect, project: callable) -> list[str]:
    lines: list[str] = []
    x = int(rect.min_x)
    while x <= int(rect.max_x):
        x1, y1 = project(Point(x, rect.min_y))
        x2, y2 = project(Point(x, rect.max_y))
        lines.append(f'<line class="grid" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        x += 2
    y = int(rect.min_y)
    while y <= int(rect.max_y):
        x1, y1 = project(Point(rect.min_x, y))
        x2, y2 = project(Point(rect.max_x, y))
        lines.append(f'<line class="grid" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        y += 2
    return lines


def _trajectory_svg(
    trajectory: Trajectory,
    project: callable,
    color: str,
    stroke_width: int,
    dash: str,
) -> str:
    points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (project(point) for point in trajectory.points)
    )
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    start_x, start_y = project(trajectory.points[0])
    end_x, end_y = project(trajectory.points[-1])
    label = escape(trajectory.label or trajectory.trajectory_id)
    return "\n".join(
        (
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>',
            f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="4" fill="{color}"/>',
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="5" fill="#ffffff" stroke="{color}" stroke-width="2"/>',
            f'<text x="{end_x + 8:.1f}" y="{end_y + 4:.1f}" class="small">{label}</text>',
        )
    )


def _agent_svg(agent: AgentTrack, project: callable, colors: dict[str, str]) -> str:
    color = colors["vru"] if agent.agent_type in {"pedestrian", "cyclist"} else colors["agent"]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (project(point) for point in agent.states))
    circles = []
    for index, state in enumerate(agent.states):
        x, y = project(state)
        radius = 6 if index < len(agent.states) - 1 else 8
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="0.75"/>'
        )
    label_x, label_y = project(agent.states[-1])
    return "\n".join(
        (
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3 5" opacity="0.7"/>',
            *circles,
            f'<text x="{label_x + 10:.1f}" y="{label_y + 4:.1f}" class="small">{escape(agent.agent_type)}</text>',
        )
    )


def _legend_item(x: int, y: int, color: str, label: str) -> str:
    return (
        f'<circle cx="{x}" cy="{y - 4}" r="5" fill="{color}"/>'
        f'<text x="{x + 10}" y="{y}" class="small">{escape(label)}</text>'
    )
