# Examples

This directory contains small public-safe artifacts generated from the built-in
MetricDrive scenarios.

## VLM Planning Rows

`vlm_planning_examples.jsonl` is generated with:

```bash
metricdrive vlm-examples --limit 12 --output docs/examples/vlm_planning_examples.jsonl
```

Each line contains:

- `prompt`: driving task, candidate trajectory summaries, and expected response
  schema,
- `chosen`: JSON trajectory response preferred by the metric objective,
- `rejected`: JSON trajectory response rejected by the metric objective,
- `preferred_meta_action` and `rejected_meta_action`: verifiable high-level
  action labels,
- `score_margin` and `reason_categories`: compact metric supervision.
