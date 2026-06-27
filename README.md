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

Planning and scaffold phase.

The repository currently includes:

- a one-page research spec,
- related-work notes,
- a staged roadmap,
- data and reproducibility policy,
- a tiny standard-library demo of metric-scored trajectory candidates,
- CI for the initial Python package.

The first implementation milestone is a synthetic long-tail scenario generator,
BEV renderer, imitation baseline, and metric-reranking baseline.

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

## Research Plan

See [docs/research_spec.md](docs/research_spec.md) for the one-page plan.

The intended experimental ladder:

1. **Imitation baseline**: train a planner to match reference future
   trajectories.
2. **Metric reranking**: sample candidate trajectories and choose the best one
   under planning metrics.
3. **Preference alignment**: create metric-derived preference pairs and train a
   planner or reward model to prefer safer candidates.
4. **Verifiable meta-actions**: add intermediate actions such as
   `YIELD_TO_VRU`, `SLOW_FOR_CUT_IN`, and `NUDGE_AROUND_OBSTACLE`, then verify
   them against trajectory metrics.
5. **Public-data integration**: add optional Waymo Open Motion / Waymax or
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
