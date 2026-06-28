# MetricDrive Research Spec

## Title

Metric-derived preference and reward alignment for long-tail autonomous-driving
trajectory planning.

## Research Question

When logged human trajectories are not the only acceptable future, can
metric-derived preferences teach a small planner to choose safer long-tail
trajectories than pure imitation learning?

## Motivation

Autonomous-driving planners are often trained with imitation objectives that
reward matching logged trajectories. That is useful, but imperfect: two safe
trajectories can differ from the log, and a log-like trajectory can still score
poorly under collision, route, comfort, or vulnerable-road-user metrics.

MetricDrive tests a public, laptop-scale version of metric-aligned planning:
convert scenario metrics into trajectory preferences, train a lightweight reward
model, export VLM-style planning examples, and evaluate whether a reward-aligned
policy improves safety-relevant behavior.

## Hypothesis

Metric-derived preference alignment will improve collision, off-road, route, and
VRU-clearance metrics on long-tail scenarios compared with imitation-only
training, while sometimes trading off imitation distance or comfort.

## Data Plan

Start with synthetic scenarios that expose controlled long-tail behaviors:

- unprotected left turns
- pedestrian crossings
- cyclist close passes
- blocked-lane nudges
- dense merges
- hard-braking lead vehicles

Then add optional public-data integrations, prioritizing small reproducible
slices over full-dataset downloads.

## Model Plan

Milestone models:

1. Heuristic candidate generator.
2. Imitation baseline over reference trajectories.
3. Metric-reranked candidate planner.
4. Preference-aligned planner trained from metric-ranked trajectory pairs.
5. VLM-style prompt/chosen/rejected planning examples.
6. Meta-action planner with verifiable intermediate actions.
7. Tiny reward-optimization analogue over candidate policies.

## Metrics

- overlap / collision risk
- off-road or drivable-area violation
- route progress
- route adherence
- vulnerable-road-user clearance
- comfort / jerk
- kinematic feasibility
- imitation displacement
- long-tail category breakdown

## Experiments

Primary table:

| Method | Collision | Off-road | Progress | VRU clearance | Comfort | Imitation error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Imitation baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Metric reranking | TBD | TBD | TBD | TBD | TBD | TBD |
| Preference aligned | TBD | TBD | TBD | TBD | TBD | TBD |

Required ablations:

- remove VRU clearance term
- remove comfort term
- compare metric reranking vs learned preference alignment
- compare token-match/progress proxies against metric-reward optimization
- evaluate by scenario category, not only aggregate score

## Success Criteria

The project is successful if a reader can reproduce a tiny experiment and see:

- why imitation alone fails on selected scenarios,
- how metric-derived preferences are created,
- which metrics improve,
- which metrics regress,
- where the method still fails.

## Limitations

- Synthetic scenarios are not a substitute for real-world validation.
- Metric weights encode assumptions and may overfit specific behaviors.
- Open-loop improvement does not guarantee closed-loop driving quality.
- Public laptop-scale models cannot represent production autonomous-driving
  systems.
