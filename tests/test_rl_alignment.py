from __future__ import annotations

import json
import unittest

from metricdrive.rl_alignment import json_rl_alignment, run_rl_alignment
from metricdrive.samples import synthetic_scenarios


class RlAlignmentTests(unittest.TestCase):
    def test_rl_alignment_recovers_metric_best_under_hard_negatives(self) -> None:
        result = run_rl_alignment(synthetic_scenarios(), epochs=10)
        summaries = {summary.method_id: summary for summary in result.summaries}

        self.assertEqual(summaries["rl_aligned"].metric_match_count, 6)
        self.assertEqual(summaries["rl_aligned"].unsafe_collision_count, 0)
        self.assertEqual(summaries["metric_reward"].metric_match_count, 6)
        self.assertEqual(summaries["token_match"].unsafe_collision_count, 6)
        self.assertEqual(summaries["progress_reward"].unsafe_collision_count, 6)

    def test_json_rl_alignment_has_stable_format(self) -> None:
        payload = json.loads(json_rl_alignment(run_rl_alignment(synthetic_scenarios(), epochs=10)))

        self.assertEqual(payload["format"], "metricdrive.rl_alignment.v1")
        self.assertIn("rl_aligned", {row["method_id"] for row in payload["summaries"]})


if __name__ == "__main__":
    unittest.main()
