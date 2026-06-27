from __future__ import annotations

import unittest

from metricdrive.demo import ranked_demo_scores


class DemoTests(unittest.TestCase):
    def test_metric_ranking_prefers_yielding_candidate(self) -> None:
        scores = ranked_demo_scores()

        self.assertEqual(scores[0].trajectory_id, "yield_then_progress")
        self.assertGreater(scores[0].vru_clearance, scores[-1].vru_clearance)


if __name__ == "__main__":
    unittest.main()
