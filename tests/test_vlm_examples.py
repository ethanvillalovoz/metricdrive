from __future__ import annotations

import json
import unittest

from metricdrive.samples import synthetic_scenarios
from metricdrive.vlm_examples import (
    generate_vlm_examples,
    json_vlm_examples,
    jsonl_vlm_examples,
)


class VlmExampleTests(unittest.TestCase):
    def test_generate_vlm_examples_uses_augmented_preferences(self) -> None:
        examples = generate_vlm_examples(synthetic_scenarios())

        self.assertEqual(len(examples), 90)
        self.assertIn("Driving task:", examples[0].prompt)
        self.assertIn("trajectory_id", examples[0].chosen)
        self.assertTrue(examples[0].preferred_meta_action)
        self.assertTrue(examples[0].rejected_meta_action)

    def test_jsonl_vlm_examples_has_one_record_per_line(self) -> None:
        examples = generate_vlm_examples(synthetic_scenarios())[:3]
        lines = jsonl_vlm_examples(examples).strip().splitlines()

        self.assertEqual(len(lines), 3)
        payload = json.loads(lines[0])
        self.assertEqual(payload["example_id"], "vlm_0001")
        self.assertIn("metric_checks", json.loads(payload["chosen"]))

    def test_json_vlm_examples_has_stable_format(self) -> None:
        payload = json.loads(json_vlm_examples(generate_vlm_examples(synthetic_scenarios())[:2]))

        self.assertEqual(payload["format"], "metricdrive.vlm_examples.v1")
        self.assertEqual(payload["example_count"], 2)


if __name__ == "__main__":
    unittest.main()
