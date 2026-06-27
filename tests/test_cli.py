from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

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
