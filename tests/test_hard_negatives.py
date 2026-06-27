from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metricdrive.hard_negatives import (
    GENERATED_SOURCE,
    augment_scenarios_with_hard_negatives,
    generate_hard_negative_report,
    generate_hard_negatives,
    json_hard_negative_experiment,
    run_hard_negative_experiment,
)
from metricdrive.samples import synthetic_scenarios


class HardNegativeTests(unittest.TestCase):
    def test_generate_hard_negatives_adds_three_candidates(self) -> None:
        generated = generate_hard_negatives(synthetic_scenarios()[0])

        self.assertEqual(len(generated), 3)
        self.assertTrue(all(candidate.source == GENERATED_SOURCE for candidate in generated))
        self.assertIn(
            "generated_progress_pressure",
            {candidate.trajectory_id for candidate in generated},
        )

    def test_augment_scenarios_expands_candidate_set(self) -> None:
        augmented = augment_scenarios_with_hard_negatives(synthetic_scenarios())
        generated_count = sum(
            candidate.source == GENERATED_SOURCE
            for scenario in augmented
            for candidate in scenario.candidates
        )

        self.assertEqual(sum(len(scenario.candidates) for scenario in augmented), 36)
        self.assertEqual(generated_count, 18)

    def test_hard_negative_experiment_reports_stress_fit(self) -> None:
        experiment = run_hard_negative_experiment(synthetic_scenarios(), epochs=20)
        summary = experiment.summary

        self.assertEqual(summary.generated_candidate_count, 18)
        self.assertEqual(summary.augmented_candidate_count, 36)
        self.assertEqual(summary.preference_pair_count, 90)
        self.assertGreaterEqual(summary.learned_correct_pair_count, 85)
        self.assertEqual(summary.learned_heldout_match_count, 6)
        self.assertEqual(summary.learned_heldout_unsafe_count, 0)

    def test_json_hard_negative_experiment_has_stable_format(self) -> None:
        payload = json.loads(
            json_hard_negative_experiment(
                run_hard_negative_experiment(synthetic_scenarios(), epochs=20)
            )
        )

        self.assertEqual(payload["format"], "metricdrive.hard_negatives.v1")
        self.assertEqual(payload["summary"]["preference_pair_count"], 90)
        self.assertIn(
            "generated_progress_pressure",
            {row["trajectory_id"] for row in payload["generated_candidates"]},
        )

    def test_generate_hard_negative_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "milestone_3_hard_negatives.md"

            generate_hard_negative_report(synthetic_scenarios(), output, epochs=20)

            self.assertTrue(output.exists())
            self.assertIn("Hard Negative Stress Test", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
