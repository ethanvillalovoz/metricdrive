from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metricdrive.benchmark import (
    generate_benchmark_report,
    markdown_benchmark,
    run_benchmark,
)
from metricdrive.samples import synthetic_scenarios


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_runs_all_default_planners(self) -> None:
        benchmark = run_benchmark(synthetic_scenarios())

        self.assertEqual(len(benchmark.summaries), 3)
        self.assertEqual(len(benchmark.runs), 18)

    def test_metric_rerank_has_fewer_unsafe_cases_than_progress_only(self) -> None:
        benchmark = run_benchmark(synthetic_scenarios())
        summaries = {summary.planner_id: summary for summary in benchmark.summaries}

        self.assertLess(
            summaries["metric_rerank"].unsafe_collision_count,
            summaries["progress_only"].unsafe_collision_count,
        )

    def test_markdown_benchmark_mentions_takeaway_methods(self) -> None:
        markdown = markdown_benchmark(run_benchmark(synthetic_scenarios()))

        self.assertIn("MetricDrive Planner Benchmark", markdown)
        self.assertIn("Reference imitation", markdown)
        self.assertIn("Progress only", markdown)
        self.assertIn("Metric rerank", markdown)

    def test_generate_benchmark_report_writes_comparison_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "milestone_2.md"
            assets = Path(tmpdir) / "assets"

            generate_benchmark_report(synthetic_scenarios(), output, assets)

            self.assertTrue(output.exists())
            self.assertTrue((assets / "benchmark_synthetic_pedestrian_crossing.svg").exists())
            self.assertIn("Baseline Planner Benchmark", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
