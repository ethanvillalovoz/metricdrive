from __future__ import annotations

import unittest

from metricdrive.metrics import best_score
from metricdrive.samples import synthetic_scenarios


class SyntheticScenarioTests(unittest.TestCase):
    def test_synthetic_set_has_six_long_tail_cases(self) -> None:
        scenarios = synthetic_scenarios()

        self.assertEqual(len(scenarios), 6)
        self.assertEqual(len({scenario.scenario_id for scenario in scenarios}), 6)
        self.assertTrue(all(len(scenario.candidates) == 3 for scenario in scenarios))

    def test_metric_aligned_candidate_wins_each_scenario(self) -> None:
        for scenario in synthetic_scenarios():
            with self.subTest(scenario=scenario.scenario_id):
                self.assertTrue(
                    best_score(scenario).trajectory_id.startswith("metric_aligned"),
                )


if __name__ == "__main__":
    unittest.main()
