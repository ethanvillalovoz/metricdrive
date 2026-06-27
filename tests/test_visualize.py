from __future__ import annotations

import unittest

from metricdrive.samples import synthetic_scenarios
from metricdrive.visualize import scenario_svg


class VisualizeTests(unittest.TestCase):
    def test_scenario_svg_contains_expected_elements(self) -> None:
        svg = scenario_svg(synthetic_scenarios()[0])

        self.assertIn("<svg", svg)
        self.assertIn("metric top candidate", svg)
        self.assertIn("pedestrian", svg)


if __name__ == "__main__":
    unittest.main()
