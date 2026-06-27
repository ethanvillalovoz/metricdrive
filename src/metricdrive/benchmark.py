from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from metricdrive.metrics import PlanningScore, score_trajectory
from metricdrive.planners import Planner, default_planners
from metricdrive.scenario import Scenario
from metricdrive.visualize import planner_comparison_svg


@dataclass(frozen=True)
class PlannerRun:
    planner_id: str
    planner_label: str
    scenario_id: str
    category: str
    trajectory_id: str
    score: PlanningScore


@dataclass(frozen=True)
class PlannerSummary:
    planner_id: str
    planner_label: str
    scenario_count: int
    mean_total: float
    mean_progress_m: float
    mean_collision_clearance_m: float
    mean_vru_clearance_m: float | None
    mean_offroad_rate: float
    mean_comfort_cost: float
    mean_route_error_m: float
    mean_imitation_error_m: float
    unsafe_collision_count: int


@dataclass(frozen=True)
class BenchmarkResult:
    runs: tuple[PlannerRun, ...]
    summaries: tuple[PlannerSummary, ...]


def run_benchmark(
    scenarios: tuple[Scenario, ...],
    planners: tuple[Planner, ...] | None = None,
) -> BenchmarkResult:
    selected_planners = planners or default_planners()
    runs = tuple(
        _run_one(planner, scenario)
        for planner in selected_planners
        for scenario in scenarios
    )
    summaries = tuple(_summarize(planner, runs) for planner in selected_planners)
    return BenchmarkResult(runs=runs, summaries=summaries)


def benchmark_payload(benchmark: BenchmarkResult) -> dict[str, object]:
    return {
        "planner_count": len(benchmark.summaries),
        "scenario_count": len({run.scenario_id for run in benchmark.runs}),
        "summaries": [asdict(summary) for summary in benchmark.summaries],
        "runs": [asdict(run) for run in benchmark.runs],
    }


def json_benchmark(benchmark: BenchmarkResult) -> str:
    return json.dumps(benchmark_payload(benchmark), indent=2) + "\n"


def markdown_benchmark(benchmark: BenchmarkResult) -> str:
    lines = [
        "# MetricDrive Planner Benchmark",
        "",
        "## Aggregate",
        "",
        "| Planner | Mean score | Progress | Collision clearance | VRU clearance | Offroad | Imitation error | Unsafe cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in benchmark.summaries:
        lines.append(_summary_row(summary))

    lines.extend(
        (
            "",
            "## Per Scenario",
            "",
            "| Scenario | Planner | Selected trajectory | Score | Progress | Collision clearance | VRU clearance | Imitation error |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for run in sorted(benchmark.runs, key=lambda item: (item.scenario_id, item.planner_id)):
        lines.append(_run_row(run))
    return "\n".join(lines) + "\n"


def generate_benchmark_report(
    scenarios: tuple[Scenario, ...],
    output_path: str | Path,
    assets_dir: str | Path,
    planners: tuple[Planner, ...] | None = None,
) -> None:
    selected_planners = planners or default_planners()
    benchmark = run_benchmark(scenarios, selected_planners)
    output = Path(output_path)
    assets = Path(assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    asset_prefix = Path(assets.name)

    lines = [
        "# Milestone 2: Baseline Planner Benchmark",
        "",
        "MetricDrive now compares planner behavior instead of only ranking candidates. This benchmark keeps the methods intentionally simple so the tradeoffs are inspectable.",
        "",
        "## Methods",
        "",
        "- `reference_imitation`: returns the logged reference trajectory.",
        "- `progress_only`: maximizes route-axis progress and ignores safety.",
        "- `metric_rerank`: selects the highest-scoring candidate under MetricDrive's planning metrics.",
        "",
        "## Aggregate Results",
        "",
        "| Planner | Mean score | Progress | Collision clearance | VRU clearance | Offroad | Imitation error | Unsafe cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in benchmark.summaries:
        lines.append(_summary_row(summary))

    progress_summary = _summary_by_id(benchmark, "progress_only")
    metric_summary = _summary_by_id(benchmark, "metric_rerank")
    lines.extend(
        (
            "",
            "## Takeaway",
            "",
            f"The progress-only baseline selects {progress_summary.unsafe_collision_count} unsafe trajectory candidate(s), while metric reranking selects {metric_summary.unsafe_collision_count}. Metric reranking keeps useful progress while explicitly trading against collision, VRU, route, and comfort costs.",
            "",
            "## Per-Scenario Planner Choices",
            "",
            "| Scenario | Planner | Selected trajectory | Score | Progress | Collision clearance | VRU clearance | Imitation error |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for run in sorted(benchmark.runs, key=lambda item: (item.scenario_id, item.planner_id)):
        lines.append(_run_row(run))

    lines.extend(("", "## Planner Comparison Gallery", ""))
    for scenario in scenarios:
        selected = {
            planner.planner_id: planner.plan(scenario) for planner in selected_planners
        }
        asset_name = f"benchmark_{scenario.scenario_id}.svg"
        (assets / asset_name).write_text(
            planner_comparison_svg(scenario, selected),
            encoding="utf-8",
        )
        lines.extend(
            (
                f"### {scenario.title}",
                "",
                f"![{scenario.title}]({(asset_prefix / asset_name).as_posix()})",
                "",
            )
        )

    lines.extend(
        (
            "## Next Experiment",
            "",
            "Use the benchmark to create metric-derived preference pairs: compare progress-only or sampled candidates against metric-reranked candidates, then train a lightweight preference model or policy objective to recover the metric-aware choice without hard-coded reranking.",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _run_one(planner: Planner, scenario: Scenario) -> PlannerRun:
    trajectory = planner.plan(scenario)
    score = score_trajectory(scenario, trajectory)
    return PlannerRun(
        planner_id=planner.planner_id,
        planner_label=planner.label,
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        trajectory_id=trajectory.trajectory_id,
        score=score,
    )


def _summarize(planner: Planner, runs: tuple[PlannerRun, ...]) -> PlannerSummary:
    planner_runs = tuple(run for run in runs if run.planner_id == planner.planner_id)
    scores = tuple(run.score for run in planner_runs)
    vru_values = tuple(
        score.vru_clearance_m for score in scores if score.vru_clearance_m is not None
    )
    return PlannerSummary(
        planner_id=planner.planner_id,
        planner_label=planner.label,
        scenario_count=len(scores),
        mean_total=_mean(score.total for score in scores),
        mean_progress_m=_mean(score.progress_m for score in scores),
        mean_collision_clearance_m=_mean(score.collision_clearance_m for score in scores),
        mean_vru_clearance_m=None if not vru_values else _mean(vru_values),
        mean_offroad_rate=_mean(score.offroad_rate for score in scores),
        mean_comfort_cost=_mean(score.comfort_cost for score in scores),
        mean_route_error_m=_mean(score.route_error_m for score in scores),
        mean_imitation_error_m=_mean(score.imitation_error_m for score in scores),
        unsafe_collision_count=sum(score.collision_clearance_m < 0 for score in scores),
    )


def _summary_by_id(benchmark: BenchmarkResult, planner_id: str) -> PlannerSummary:
    for summary in benchmark.summaries:
        if summary.planner_id == planner_id:
            return summary
    raise ValueError(f"Missing planner summary: {planner_id}")


def _summary_row(summary: PlannerSummary) -> str:
    return (
        "| "
        f"{summary.planner_label} | {summary.mean_total:.3f} | "
        f"{summary.mean_progress_m:.3f} | {summary.mean_collision_clearance_m:.3f} | "
        f"{_optional(summary.mean_vru_clearance_m)} | {summary.mean_offroad_rate:.3f} | "
        f"{summary.mean_imitation_error_m:.3f} | {summary.unsafe_collision_count} |"
    )


def _run_row(run: PlannerRun) -> str:
    score = run.score
    return (
        "| "
        f"{run.scenario_id} | {run.planner_label} | `{run.trajectory_id}` | "
        f"{score.total:.3f} | {score.progress_m:.3f} | "
        f"{score.collision_clearance_m:.3f} | {_optional(score.vru_clearance_m)} | "
        f"{score.imitation_error_m:.3f} |"
    )


def _mean(values) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 3)


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
