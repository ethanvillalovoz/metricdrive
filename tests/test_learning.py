from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metricdrive.learning import (
    LearnedPreferencePlanner,
    generate_learning_report,
    json_learning,
    run_learning_experiment,
    train_preference_model,
)
from metricdrive.metrics import ranked_scores
from metricdrive.samples import synthetic_scenarios


class LearningTests(unittest.TestCase):
    def test_train_preference_model_learns_interpretable_weights(self) -> None:
        model = train_preference_model(synthetic_scenarios(), epochs=80)

        self.assertGreater(model.weights["progress"], 0)
        self.assertGreater(model.weights["collision_clearance"], 0)
        self.assertTrue(all(weight >= 0 for weight in model.weights.values()))

    def test_learned_planner_matches_metric_best_candidates(self) -> None:
        scenarios = synthetic_scenarios()
        planner = LearnedPreferencePlanner(train_preference_model(scenarios, epochs=80))

        for scenario in scenarios:
            with self.subTest(scenario=scenario.scenario_id):
                selected = planner.plan(scenario)
                metric_best = ranked_scores(scenario)[0]
                self.assertEqual(selected.trajectory_id, metric_best.trajectory_id)

    def test_learning_experiment_reports_heldout_recovery(self) -> None:
        result = run_learning_experiment(synthetic_scenarios(), epochs=80)

        self.assertEqual(result.preference_fit.correct_pair_count, 18)
        self.assertEqual(result.training_selection_summary.metric_match_count, 6)
        self.assertEqual(result.heldout_selection_summary.metric_match_count, 6)
        self.assertEqual(result.heldout_selection_summary.unsafe_collision_count, 0)

    def test_json_learning_has_stable_format(self) -> None:
        payload = json.loads(
            json_learning(run_learning_experiment(synthetic_scenarios(), epochs=80))
        )

        self.assertEqual(payload["format"], "metricdrive.learning.v1")
        self.assertEqual(payload["preference_fit"]["pair_count"], 18)
        self.assertIn(
            "learned_preference",
            {row["planner_id"] for row in payload["benchmark"]["runs"]},
        )

    def test_generate_learning_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "milestone_3_learned_model.md"

            generate_learning_report(synthetic_scenarios(), output, epochs=80)

            self.assertTrue(output.exists())
            self.assertIn("Learned Preference Model", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
