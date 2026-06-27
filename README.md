# MetricDrive

MetricDrive is a research-oriented autonomous-driving planning project about
training planners toward driving metrics, not only logged-trajectory imitation.

The core question:

> When the logged human trajectory is not the only acceptable future, can
> metric-derived preferences teach a planner to choose safer long-tail
> trajectories?

This repository starts as a laptop-scale, public-data research artifact. It is
inspired by public work on multimodal/VLM driving planners, preference-aligned
planning, and Waymo-style scenario evaluation, but it is independent and is not
affiliated with Waymo.

## Why This Exists

Pure imitation can make a planner good at matching logged trajectories while
still under-optimizing the behavior people actually care about: avoiding
collisions, respecting route intent, yielding to vulnerable road users, staying
comfortable, and making progress.

MetricDrive is designed to test a smaller, reproducible version of the modern
planning-alignment loop:

1. Represent a driving scene as a bird's-eye-view state and structured prompt.
2. Produce future ego trajectory candidates.
3. Score candidates with interpretable planning metrics.
4. Convert metric rankings into trajectory preferences.
5. Compare imitation-only planning against metric-preference alignment.

## Current Status

Milestone 3D hard negative stress test.

The repository currently includes:

- a one-page research spec,
- related-work notes,
- a staged roadmap,
- data and reproducibility policy,
- a tiny standard-library demo of metric-scored trajectory candidates,
- six synthetic long-tail driving scenario families,
- transparent planning metrics and candidate ranking,
- SVG bird's-eye-view scenario rendering,
- a generated Milestone 1 report,
- reference imitation, progress-only, and metric-rerank planner baselines,
- aggregate/per-scenario benchmark reporting,
- a generated Milestone 2 planner comparison report,
- metric-derived preference pairs with score margins, reasons, and DPO-style
  prompt/chosen/rejected records,
- a generated Milestone 3 preference-data report,
- a learned preference reward model trained from pairwise trajectory
  preferences,
- leave-one-scenario-out learned-model evaluation,
- a generated Milestone 3B learned preference model report,
- objective ablations showing which metric terms prevent unsafe or overly
  cautious selections,
- a generated Milestone 3C objective ablation report,
- generated hard-negative trajectory candidates that stress safety/progress
  tradeoffs,
- augmented preference-pair stress evaluation over 36 trajectory candidates,
- a generated Milestone 3D hard negative stress-test report,
- CI for the initial Python package.

The next implementation milestone is verifiable meta-actions connected to
trajectory candidates and metric checks.

See [docs/reports/milestone_1.md](docs/reports/milestone_1.md) for the current
scenario gallery and score tables. See
[docs/reports/milestone_2.md](docs/reports/milestone_2.md) for the baseline
planner benchmark. See [docs/reports/milestone_3.md](docs/reports/milestone_3.md)
for the metric-derived preference dataset. See
[docs/reports/milestone_3_learned_model.md](docs/reports/milestone_3_learned_model.md)
for the learned preference model evaluation. See
[docs/reports/milestone_3_ablation_study.md](docs/reports/milestone_3_ablation_study.md)
for the objective ablation study. See
[docs/reports/milestone_3_hard_negatives.md](docs/reports/milestone_3_hard_negatives.md)
for the hard negative stress test.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
metricdrive demo
python3 -m unittest discover
```

The demo prints a small trajectory-ranking example using collision, progress,
comfort, and vulnerable-road-user clearance terms. It is a smoke test for the
evaluation vocabulary, not the final planner.

## Milestone 1 Commands

Generate the controlled synthetic scenario set:

```bash
metricdrive generate --output data/processed/synthetic_scenarios.json
```

Score built-in or saved scenarios:

```bash
metricdrive score
metricdrive score --input data/processed/synthetic_scenarios.json --format json
```

Render one scenario as SVG:

```bash
metricdrive render synthetic_pedestrian_crossing --output outputs/pedestrian_crossing.svg
```

Generate the first milestone report and SVG gallery:

```bash
metricdrive report --output docs/reports/milestone_1.md --assets-dir docs/reports/assets
```

Compare baseline planners:

```bash
metricdrive benchmark
metricdrive benchmark --format json
metricdrive benchmark --report docs/reports/milestone_2.md --assets-dir docs/reports/assets
```

Generate metric-derived preference pairs:

```bash
metricdrive preferences
metricdrive preferences --format json
metricdrive preferences --output data/processed/preferences.json
metricdrive preferences --report docs/reports/milestone_3.md
```

Train and evaluate the learned preference model:

```bash
metricdrive learned
metricdrive learned --format json
metricdrive learned --report docs/reports/milestone_3_learned_model.md
```

Run objective ablations:

```bash
metricdrive ablations
metricdrive ablations --format json
metricdrive ablations --report docs/reports/milestone_3_ablation_study.md
```

Generate and evaluate hard negative candidates:

```bash
metricdrive hard-negatives
metricdrive hard-negatives --format json
metricdrive hard-negatives --report docs/reports/milestone_3_hard_negatives.md
```

## Research Plan

See [docs/research_spec.md](docs/research_spec.md) for the one-page plan.

The intended experimental ladder:

1. **Synthetic scenario core**: define scenarios, metrics, SVG rendering, and
   report generation.
2. **Imitation baseline**: train a planner to match reference future
   trajectories.
3. **Metric reranking**: sample candidate trajectories and choose the best one
   under planning metrics.
4. **Preference alignment**: create metric-derived preference pairs and train a
   planner or reward model to prefer safer candidates.
5. **Verifiable meta-actions**: add intermediate actions such as
   `YIELD_TO_VRU`, `SLOW_FOR_CUT_IN`, and `NUDGE_AROUND_OBSTACLE`, then verify
   them against trajectory metrics.
6. **Public-data integration**: add optional Waymo Open Motion / Waymax or
   other public benchmark slices without requiring heavy downloads for the
   default demo.

## Planned Metrics

- collision or overlap risk
- off-road / drivable-area violation
- route progress
- route adherence
- vulnerable-road-user clearance
- comfort / jerk
- kinematic feasibility
- imitation distance to logged trajectory
- long-tail scenario category coverage

## Repository Layout

```text
src/metricdrive/       Python package and CLI
tests/                 Unit tests for the initial scaffold
docs/                  Research spec, roadmap, related work, data policy
data/raw/              Local raw data mount point, ignored by git
data/processed/        Local generated data, ignored by git
notebooks/             Exploratory notebooks, kept out of the core path
scripts/               Future experiment and data-prep entrypoints
```

## Related Public Work

MetricDrive is positioned around public research threads including EMMA,
S4-Driver, WOD-E2E, Poutine, DriveMA, Waymax, NAVSIM, PlanT, VAD, and UniAD.
Short notes live in [docs/related_work.md](docs/related_work.md).

## Non-Goals

- This is not a self-driving system.
- This does not claim Waymo-level scale or performance.
- This does not use private company data or non-public project details.
- This does not attempt to reproduce an internal Waymo internship project.

## License

MIT License. See [LICENSE](LICENSE).
