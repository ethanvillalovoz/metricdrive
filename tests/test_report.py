from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metricdrive.report import generate_milestone_report, json_scores, markdown_scores
from metricdrive.samples import synthetic_scenarios


class ReportTests(unittest.TestCase):
    def test_markdown_scores_include_top_candidate(self) -> None:
        markdown = markdown_scores(synthetic_scenarios())

        self.assertIn("Synthetic Scenario Scores", markdown)
        self.assertIn("metric_aligned_yield", markdown)

    def test_json_scores_include_scenario_count(self) -> None:
        payload = json_scores(synthetic_scenarios())

        self.assertIn('"scenario_count": 6', payload)

    def test_generate_milestone_report_writes_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "milestone_1.md"
            assets = Path(tmpdir) / "assets"

            generate_milestone_report(synthetic_scenarios(), output, assets)

            self.assertTrue(output.exists())
            self.assertTrue((assets / "synthetic_pedestrian_crossing.svg").exists())
            self.assertIn("Milestone 1", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
