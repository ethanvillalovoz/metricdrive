from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from metricdrive import __version__
from metricdrive.benchmark import (
    generate_benchmark_report,
    json_benchmark,
    markdown_benchmark,
    run_benchmark,
)
from metricdrive.demo import built_in_demo_scenario, ranked_demo_scores
from metricdrive.hard_negatives import (
    DEFAULT_HARD_NEGATIVE_EPOCHS,
    generate_hard_negative_report,
    json_hard_negative_experiment,
    markdown_hard_negative_experiment,
    run_hard_negative_experiment,
)
from metricdrive.io import load_scenarios, save_scenarios
from metricdrive.learning import (
    generate_ablation_report,
    generate_learning_report,
    json_ablation_study,
    json_learning,
    markdown_ablation_study,
    markdown_learning,
    run_ablation_study,
    run_learning_experiment,
)
from metricdrive.preferences import (
    generate_preference_report,
    generate_preferences,
    json_preferences,
    markdown_preferences,
    save_preferences,
)
from metricdrive.report import generate_milestone_report, json_scores, markdown_scores
from metricdrive.samples import synthetic_scenarios
from metricdrive.visualize import scenario_svg


DEFAULT_SYNTHETIC_OUTPUT = "data/processed/synthetic_scenarios.json"
DEFAULT_REPORT_OUTPUT = "docs/reports/milestone_1.md"
DEFAULT_REPORT_ASSETS = "docs/reports/assets"
DEFAULT_BENCHMARK_OUTPUT = "docs/reports/milestone_2.md"
DEFAULT_PREFERENCES_OUTPUT = "data/processed/preferences.json"
DEFAULT_PREFERENCES_REPORT = "docs/reports/milestone_3.md"
DEFAULT_LEARNING_REPORT = "docs/reports/milestone_3_learned_model.md"
DEFAULT_ABLATION_REPORT = "docs/reports/milestone_3_ablation_study.md"
DEFAULT_HARD_NEGATIVE_REPORT = "docs/reports/milestone_3_hard_negatives.md"


def demo(output_format: str) -> int:
    scenario = built_in_demo_scenario()
    scores = ranked_demo_scores()

    if output_format == "json":
        payload = {
            "scenario": asdict(scenario),
            "scores": [asdict(score) for score in scores],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"MetricDrive demo: {scenario.scenario_id}")
    print()
    print("| Rank | Candidate | Total | Progress | VRU clearance | Obstacle clearance | Comfort penalty |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, score in enumerate(scores, start=1):
        print(
            "| "
            f"{rank} | `{score.trajectory_id}` | {score.total:.3f} | "
            f"{score.progress:.3f} | {score.vru_clearance:.3f} | "
            f"{score.obstacle_clearance:.3f} | {score.comfort_penalty:.3f} |"
        )
    return 0


def spec() -> int:
    path = Path(__file__).resolve().parents[2] / "docs" / "research_spec.md"
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def generate(output_path: str) -> int:
    scenarios = synthetic_scenarios()
    save_scenarios(output_path, scenarios)
    print(f"Generated {len(scenarios)} synthetic scenario(s) at {output_path}")
    return 0


def _load_or_synthetic(input_path: str | None):
    if input_path:
        return load_scenarios(input_path)
    return synthetic_scenarios()


def score(input_path: str | None, output_format: str) -> int:
    scenarios = _load_or_synthetic(input_path)
    if output_format == "json":
        print(json_scores(scenarios), end="")
    else:
        print(markdown_scores(scenarios), end="")
    return 0


def render(
    scenario_id: str,
    output_path: str | None,
    input_path: str | None,
) -> int:
    scenarios = _load_or_synthetic(input_path)
    scenario = _scenario_by_id(scenarios, scenario_id)
    svg = scenario_svg(scenario)
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(svg, encoding="utf-8")
        print(f"Rendered {scenario_id} to {output_path}")
    else:
        print(svg, end="")
    return 0


def report(output_path: str, assets_dir: str, input_path: str | None) -> int:
    scenarios = _load_or_synthetic(input_path)
    generate_milestone_report(
        scenarios=scenarios,
        output_path=output_path,
        assets_dir=assets_dir,
    )
    print(f"Generated Milestone 1 report at {output_path}")
    return 0


def benchmark(
    input_path: str | None,
    output_format: str,
    report_path: str | None,
    assets_dir: str,
) -> int:
    scenarios = _load_or_synthetic(input_path)
    if report_path:
        generate_benchmark_report(
            scenarios=scenarios,
            output_path=report_path,
            assets_dir=assets_dir,
        )
        print(f"Generated Milestone 2 benchmark report at {report_path}")
        return 0

    result = run_benchmark(scenarios)
    if output_format == "json":
        print(json_benchmark(result), end="")
    else:
        print(markdown_benchmark(result), end="")
    return 0


def preferences(
    input_path: str | None,
    output_format: str,
    output_path: str | None,
    report_path: str | None,
    min_score_margin: float,
) -> int:
    scenarios = _load_or_synthetic(input_path)
    pairs = generate_preferences(scenarios, min_score_margin=min_score_margin)

    if report_path:
        generate_preference_report(pairs, report_path)
        print(f"Generated Milestone 3 preference report at {report_path}")
        return 0

    if output_path:
        save_preferences(output_path, pairs)
        print(f"Exported {len(pairs)} preference pair(s) to {output_path}")
        return 0

    if output_format == "json":
        print(json_preferences(pairs), end="")
    else:
        print(markdown_preferences(pairs), end="")
    return 0


def learned(
    input_path: str | None,
    output_format: str,
    report_path: str | None,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> int:
    scenarios = _load_or_synthetic(input_path)

    if report_path:
        generate_learning_report(
            scenarios=scenarios,
            output_path=report_path,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        print(f"Generated learned preference report at {report_path}")
        return 0

    result = run_learning_experiment(
        scenarios=scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    if output_format == "json":
        print(json_learning(result), end="")
    else:
        print(markdown_learning(result), end="")
    return 0


def ablations(
    input_path: str | None,
    output_format: str,
    report_path: str | None,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> int:
    scenarios = _load_or_synthetic(input_path)

    if report_path:
        generate_ablation_report(
            scenarios=scenarios,
            output_path=report_path,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        print(f"Generated ablation report at {report_path}")
        return 0

    study = run_ablation_study(
        scenarios=scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    if output_format == "json":
        print(json_ablation_study(study), end="")
    else:
        print(markdown_ablation_study(study), end="")
    return 0


def hard_negatives(
    input_path: str | None,
    output_format: str,
    report_path: str | None,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> int:
    scenarios = _load_or_synthetic(input_path)

    if report_path:
        generate_hard_negative_report(
            scenarios=scenarios,
            output_path=report_path,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        print(f"Generated hard negative stress report at {report_path}")
        return 0

    experiment = run_hard_negative_experiment(
        scenarios=scenarios,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    if output_format == "json":
        print(json_hard_negative_experiment(experiment), end="")
    else:
        print(markdown_hard_negative_experiment(experiment), end="")
    return 0


def _scenario_by_id(scenarios, scenario_id: str):
    for scenario in scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    valid_ids = ", ".join(scenario.scenario_id for scenario in scenarios)
    raise SystemExit(f"Unknown scenario id: {scenario_id}. Valid ids: {valid_ids}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="metricdrive",
        description="Metric-derived planning alignment experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the tiny trajectory metric demo.",
    )
    demo_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )

    subparsers.add_parser(
        "spec",
        help="Print the research spec.",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic long-tail scenarios.",
    )
    generate_parser.add_argument(
        "--output",
        default=DEFAULT_SYNTHETIC_OUTPUT,
        help="Scenario JSON path to write.",
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score trajectory candidates for synthetic or saved scenarios.",
    )
    score_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    score_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )

    render_parser = subparsers.add_parser(
        "render",
        help="Render one scenario as SVG.",
    )
    render_parser.add_argument(
        "scenario_id",
        help="Scenario id to render.",
    )
    render_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    render_parser.add_argument(
        "--output",
        help="SVG path to write. Defaults to stdout.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate the Milestone 1 Markdown report and SVG assets.",
    )
    report_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    report_parser.add_argument(
        "--output",
        default=DEFAULT_REPORT_OUTPUT,
        help="Markdown report path to write.",
    )
    report_parser.add_argument(
        "--assets-dir",
        default=DEFAULT_REPORT_ASSETS,
        help="Directory for generated SVG assets.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare baseline planners over the scenario set.",
    )
    benchmark_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    benchmark_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format when not writing a report.",
    )
    benchmark_parser.add_argument(
        "--report",
        help="Optional Markdown report path to write.",
    )
    benchmark_parser.add_argument(
        "--assets-dir",
        default=DEFAULT_REPORT_ASSETS,
        help="Directory for generated SVG report assets.",
    )

    preferences_parser = subparsers.add_parser(
        "preferences",
        help="Generate metric-derived trajectory preference pairs.",
    )
    preferences_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    preferences_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format when not writing a file or report.",
    )
    preferences_parser.add_argument(
        "--output",
        help=f"Optional JSON path to write, such as {DEFAULT_PREFERENCES_OUTPUT}.",
    )
    preferences_parser.add_argument(
        "--report",
        help=f"Optional Markdown report path to write, such as {DEFAULT_PREFERENCES_REPORT}.",
    )
    preferences_parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.0,
        help="Only emit pairs whose preferred score exceeds rejected score by this margin.",
    )

    learned_parser = subparsers.add_parser(
        "learned",
        help="Train and evaluate a learned preference reward model.",
    )
    learned_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    learned_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format when not writing a report.",
    )
    learned_parser.add_argument(
        "--report",
        help=f"Optional Markdown report path to write, such as {DEFAULT_LEARNING_REPORT}.",
    )
    learned_parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Preference-training epochs.",
    )
    learned_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Pairwise logistic learning rate.",
    )
    learned_parser.add_argument(
        "--l2",
        type=float,
        default=0.001,
        help="L2 weight decay.",
    )

    ablations_parser = subparsers.add_parser(
        "ablations",
        help="Run objective ablations for the learned preference model.",
    )
    ablations_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    ablations_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format when not writing a report.",
    )
    ablations_parser.add_argument(
        "--report",
        help=f"Optional Markdown report path to write, such as {DEFAULT_ABLATION_REPORT}.",
    )
    ablations_parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Preference-training epochs per ablation.",
    )
    ablations_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Pairwise logistic learning rate.",
    )
    ablations_parser.add_argument(
        "--l2",
        type=float,
        default=0.001,
        help="L2 weight decay.",
    )

    hard_negatives_parser = subparsers.add_parser(
        "hard-negatives",
        help="Generate and evaluate hard negative trajectory candidates.",
    )
    hard_negatives_parser.add_argument(
        "--input",
        help="Scenario JSON path. Defaults to built-in synthetic scenarios.",
    )
    hard_negatives_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format when not writing a report.",
    )
    hard_negatives_parser.add_argument(
        "--report",
        help=f"Optional Markdown report path to write, such as {DEFAULT_HARD_NEGATIVE_REPORT}.",
    )
    hard_negatives_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_HARD_NEGATIVE_EPOCHS,
        help="Preference-training epochs for stress evaluation.",
    )
    hard_negatives_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Pairwise logistic learning rate.",
    )
    hard_negatives_parser.add_argument(
        "--l2",
        type=float,
        default=0.001,
        help="L2 weight decay.",
    )

    args = parser.parse_args()
    if args.command == "demo":
        return demo(output_format=args.format)
    if args.command == "spec":
        return spec()
    if args.command == "generate":
        return generate(output_path=args.output)
    if args.command == "score":
        return score(input_path=args.input, output_format=args.format)
    if args.command == "render":
        return render(
            scenario_id=args.scenario_id,
            output_path=args.output,
            input_path=args.input,
        )
    if args.command == "report":
        return report(
            output_path=args.output,
            assets_dir=args.assets_dir,
            input_path=args.input,
        )
    if args.command == "benchmark":
        return benchmark(
            input_path=args.input,
            output_format=args.format,
            report_path=args.report,
            assets_dir=args.assets_dir,
        )
    if args.command == "preferences":
        return preferences(
            input_path=args.input,
            output_format=args.format,
            output_path=args.output,
            report_path=args.report,
            min_score_margin=args.min_score_margin,
        )
    if args.command == "learned":
        return learned(
            input_path=args.input,
            output_format=args.format,
            report_path=args.report,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    if args.command == "ablations":
        return ablations(
            input_path=args.input,
            output_format=args.format,
            report_path=args.report,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    if args.command == "hard-negatives":
        return hard_negatives(
            input_path=args.input,
            output_format=args.format,
            report_path=args.report,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
