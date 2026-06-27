from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from metricdrive.metrics import PlanningScore, ranked_scores
from metricdrive.scenario import Scenario
from metricdrive.visualize import scenario_svg


def scores_payload(scenarios: tuple[Scenario, ...]) -> dict[str, object]:
    return {
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "title": scenario.title,
                "prompt": scenario.prompt,
                "tags": list(scenario.tags),
                "scores": [asdict(score) for score in ranked_scores(scenario)],
            }
            for scenario in scenarios
        ],
    }


def json_scores(scenarios: tuple[Scenario, ...]) -> str:
    return json.dumps(scores_payload(scenarios), indent=2) + "\n"


def markdown_scores(scenarios: tuple[Scenario, ...]) -> str:
    lines = [
        "# MetricDrive Synthetic Scenario Scores",
        "",
        "| Scenario | Category | Top candidate | Score | Progress | Collision clearance | VRU clearance | Offroad |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        best = ranked_scores(scenario)[0]
        lines.append(_score_row(scenario.title, scenario.category, best))
    return "\n".join(lines) + "\n"


def generate_milestone_report(
    scenarios: tuple[Scenario, ...],
    output_path: str | Path,
    assets_dir: str | Path,
) -> None:
    output = Path(output_path)
    assets = Path(assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    asset_prefix = Path(assets.name)
    lines = [
        "# Milestone 1: Synthetic Scenario Core",
        "",
        "MetricDrive now has a controlled evaluation world: six long-tail driving scenarios, transparent planning metrics, ranked candidate trajectories, and SVG visualizations.",
        "",
        "The point of this milestone is not model training yet. It is the measurement harness that makes later imitation and preference-alignment experiments meaningful.",
        "",
        "## Summary",
        "",
        "| Scenario | Category | Top candidate | Score | Progress | Collision clearance | VRU clearance | Offroad |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        best = ranked_scores(scenario)[0]
        lines.append(_score_row(scenario.title, scenario.category, best))

    lines.extend(
        (
            "",
            "## Scenario Gallery",
            "",
        )
    )

    for scenario in scenarios:
        asset_name = f"{scenario.scenario_id}.svg"
        (assets / asset_name).write_text(scenario_svg(scenario), encoding="utf-8")
        lines.extend(
            (
                f"### {scenario.title}",
                "",
                f"![{scenario.title}]({(asset_prefix / asset_name).as_posix()})",
                "",
                scenario.prompt,
                "",
                "| Rank | Candidate | Score | Progress | Collision clearance | VRU clearance | Comfort | Imitation error |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for rank, score in enumerate(ranked_scores(scenario), start=1):
            lines.append(
                "| "
                f"{rank} | `{score.trajectory_id}` | {score.total:.3f} | "
                f"{score.progress_m:.3f} | {score.collision_clearance_m:.3f} | "
                f"{_optional(score.vru_clearance_m)} | {score.comfort_cost:.3f} | "
                f"{score.imitation_error_m:.3f} |"
            )
        lines.append("")

    lines.extend(
        (
            "## Next Experiment",
            "",
            "Use this harness to compare three methods: imitation baseline, metric reranking, and metric-derived preference alignment. The expected tradeoff is that metric alignment improves clearance and offroad behavior while sometimes increasing imitation error.",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _score_row(title: str, category: str, score: PlanningScore) -> str:
    return (
        "| "
        f"{title} | {category} | `{score.trajectory_id}` | {score.total:.3f} | "
        f"{score.progress_m:.3f} | {score.collision_clearance_m:.3f} | "
        f"{_optional(score.vru_clearance_m)} | {score.offroad_rate:.3f} |"
    )


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
