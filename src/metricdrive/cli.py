from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from metricdrive import __version__
from metricdrive.demo import built_in_demo_scenario, ranked_demo_scores


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

    args = parser.parse_args()
    if args.command == "demo":
        return demo(output_format=args.format)
    if args.command == "spec":
        return spec()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
