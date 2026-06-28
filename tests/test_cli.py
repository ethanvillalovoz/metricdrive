from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from metricdrive.cli import main


class CliTests(unittest.TestCase):
    def test_demo_outputs_ranked_candidates(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("demo")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive demo", stdout.getvalue())
        self.assertIn("yield_then_progress", stdout.getvalue())

    def test_spec_prints_research_question(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("spec")

        self.assertEqual(exit_code, 0)
        self.assertIn("Research Question", stdout.getvalue())

    def test_benchmark_outputs_planner_table(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("benchmark")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive Planner Benchmark", stdout.getvalue())
        self.assertIn("Progress only", stdout.getvalue())
        self.assertIn("Metric rerank", stdout.getvalue())

    def test_preferences_outputs_pair_table(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("preferences")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive Preference Pairs", stdout.getvalue())
        self.assertIn("metric_aligned_yield", stdout.getvalue())

    def test_learned_outputs_model_summary(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("learned", "--epochs", "80")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive Learned Preference Model", stdout.getvalue())
        self.assertIn("Leave-one-scenario-out match rate", stdout.getvalue())

    def test_ablations_outputs_objective_table(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("ablations", "--epochs", "80")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive Objective Ablations", stdout.getvalue())
        self.assertIn("Progress only", stdout.getvalue())

    def test_hard_negatives_outputs_stress_summary(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("hard-negatives", "--epochs", "20")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive Hard Negative Stress Test", stdout.getvalue())
        self.assertIn("generated_progress_pressure", stdout.getvalue())

    def test_export_demo_writes_static_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "demo"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main_from_args(
                    "export-demo",
                    "--output",
                    str(output),
                    "--epochs",
                    "5",
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "index.html").exists())
            self.assertIn("Generated MetricDrive static demo", stdout.getvalue())

    def test_vlm_examples_outputs_jsonl(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("vlm-examples", "--limit", "2")

        self.assertEqual(exit_code, 0)
        self.assertIn("vlm_0001", stdout.getvalue())
        self.assertIn("preferred_meta_action", stdout.getvalue())

    def test_rl_align_outputs_policy_summary(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_from_args("rl-align", "--epochs", "10")

        self.assertEqual(exit_code, 0)
        self.assertIn("MetricDrive RL Alignment Analogue", stdout.getvalue())
        self.assertIn("Metric-RL aligned policy", stdout.getvalue())


def main_from_args(*args: str) -> int:
    import sys

    original_argv = sys.argv
    sys.argv = ["metricdrive", *args]
    try:
        return main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
