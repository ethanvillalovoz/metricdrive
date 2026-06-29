# MetricDrive Portfolio Report

MetricDrive is a public, laptop-scale research artifact for metric-aligned
autonomous-driving trajectory planning. It demonstrates a complete planning
alignment loop on controlled long-tail scenarios: candidate trajectories,
transparent metrics, preference pairs, learned reward selection, hard-negative
stress tests, VLM-style planning rows, and a tiny reward-optimization analogue.

## What It Shows

| Layer | Evidence |
| --- | --- |
| Scenario core | Six long-tail scenario families with SVG trajectory visualizations |
| Metric scoring | Progress, collision clearance, VRU clearance, offroad, comfort, route error, imitation error |
| Preference data | 90 prompt/chosen/rejected pairs after hard-negative augmentation |
| Learned reward | 89/90 preference fit and 6/6 held-out metric-best recovery |
| Stress testing | No-collision and progress-only ablations select unsafe candidates on the augmented set |
| VLM interface | Public-safe planning examples with structured prompts, chosen/rejected JSON, and meta-actions |
| RL analogue | Metric-reward policy optimization recovers 6/6 metric-best choices with zero unsafe selections |
| Public demo | Static explorer generated from reproducible experiment data and mirrored at `ethanvillalovoz.com/metricdrive` |

## Why It Matters

The central planning question is not whether a model can copy a logged future.
It is whether the model can choose a future that satisfies the behavior metrics
that matter: safety, route intent, progress, vulnerable-road-user clearance, and
comfort.

MetricDrive creates a small, inspectable version of that alignment loop. It is
designed to be credible as a portfolio project because every result is
reproducible from local commands, and every claim stays inside public-safe
implementation boundaries.

## Current Result Snapshot

| Experiment | Result |
| --- | --- |
| Original scenario set | 6 scenarios, 18 hand-authored candidates |
| Hard-negative stress set | 36 total candidates, 18 generated hard negatives |
| Preference pairs | 90 metric-derived pairs |
| Learned reward | 89/90 pairwise fit |
| Held-out recovery | 6/6 metric-best choices, zero unsafe selections |
| No-collision ablation | 2/6 metric-best choices, 3 unsafe selections |
| Progress-only ablation | 1/6 metric-best choices, 5 unsafe selections |
| Token-match proxy | 0/6 metric-best choices, 6 unsafe selections |
| Metric-RL aligned policy | 6/6 metric-best choices, zero unsafe selections |

## Demo Workflow

```bash
metricdrive export-demo --output docs/demo
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/` locally, or use the public portfolio mirror at
`https://ethanvillalovoz.com/metricdrive/`, to inspect scenarios, trajectory
scores, learned reward selections, hard-negative ablations, VLM examples, and
RL policy summaries.

## Public-Safe Positioning

MetricDrive is Waymo-aligned in topic and research vocabulary, but it is
independent. It does not use private data, private metrics, internal systems, or
non-public implementation details. The goal is to show taste and engineering
judgment around a public version of planning-metric alignment.

## Next Work

- Add richer verifiable meta-actions and per-action metric checks.
- Export a small paper-style benchmark table from the VLM and RL commands.
- Add optional public-data or Waymax-style adapters without making the default
  demo heavyweight.
