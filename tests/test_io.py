from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metricdrive.io import load_scenarios, save_scenarios
from metricdrive.samples import synthetic_scenarios


class ScenarioIoTests(unittest.TestCase):
    def test_save_and_load_scenarios_round_trips_ids(self) -> None:
        scenarios = synthetic_scenarios()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            save_scenarios(path, scenarios)

            loaded = load_scenarios(path)

        self.assertEqual(
            [scenario.scenario_id for scenario in loaded],
            [scenario.scenario_id for scenario in scenarios],
        )
        self.assertEqual(loaded[0].candidates[0].trajectory_id, scenarios[0].candidates[0].trajectory_id)


if __name__ == "__main__":
    unittest.main()
