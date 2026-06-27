from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metricdrive.preferences import (
    generate_preference_report,
    generate_preferences,
    json_preferences,
    preference_summary,
    save_preferences,
)
from metricdrive.samples import synthetic_scenarios


class PreferenceTests(unittest.TestCase):
    def test_generate_preferences_creates_pairwise_rankings(self) -> None:
        pairs = generate_preferences(synthetic_scenarios())

        self.assertEqual(len(pairs), 18)
        self.assertTrue(all(pair.score_margin > 0 for pair in pairs))
        self.assertTrue(all(pair.reasons for pair in pairs))

    def test_summary_counts_unsafe_rejections(self) -> None:
        summary = preference_summary(generate_preferences(synthetic_scenarios()))

        self.assertEqual(summary.scenario_count, 6)
        self.assertGreater(summary.unsafe_rejection_count, 0)
        self.assertIn("avoids_collision", summary.reason_category_counts)

    def test_json_preferences_has_stable_format(self) -> None:
        payload = json.loads(json_preferences(generate_preferences(synthetic_scenarios())))

        self.assertEqual(payload["format"], "metricdrive.preferences.v1")
        self.assertEqual(payload["summary"]["pair_count"], 18)
        self.assertIn("Driving task:", payload["pairs"][0]["prompt"])
        self.assertIn("Trajectory:", payload["pairs"][0]["chosen"])
        self.assertIn("Trajectory:", payload["pairs"][0]["rejected"])

    def test_save_preferences_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "preferences.json"

            save_preferences(output, generate_preferences(synthetic_scenarios()))

            self.assertTrue(output.exists())
            self.assertIn("metricdrive.preferences.v1", output.read_text(encoding="utf-8"))

    def test_generate_preference_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "milestone_3.md"

            generate_preference_report(generate_preferences(synthetic_scenarios()), output)

            self.assertTrue(output.exists())
            self.assertIn("Metric-Derived Preferences", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
