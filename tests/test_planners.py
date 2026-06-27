from __future__ import annotations

import unittest

from metricdrive.planners import (
    MetricRerankPlanner,
    ProgressOnlyPlanner,
    ReferenceImitationPlanner,
)
from metricdrive.samples import synthetic_scenarios


class PlannerTests(unittest.TestCase):
    def test_reference_planner_returns_reference(self) -> None:
        scenario = synthetic_scenarios()[0]

        trajectory = ReferenceImitationPlanner().plan(scenario)

        self.assertEqual(trajectory.trajectory_id, scenario.reference.trajectory_id)

    def test_progress_only_ignores_safety_on_pedestrian_case(self) -> None:
        scenario = synthetic_scenarios()[0]

        trajectory = ProgressOnlyPlanner().plan(scenario)

        self.assertEqual(trajectory.trajectory_id, "imitation_fast_log")

    def test_metric_rerank_selects_metric_aligned_candidate(self) -> None:
        for scenario in synthetic_scenarios():
            with self.subTest(scenario=scenario.scenario_id):
                trajectory = MetricRerankPlanner().plan(scenario)
                self.assertTrue(trajectory.trajectory_id.startswith("metric_aligned"))


if __name__ == "__main__":
    unittest.main()
