# Milestone 3B: Learned Preference Model

MetricDrive now trains a lightweight reward model from the metric-derived preference pairs. The model learns non-negative weights over interpretable metric components, then acts as a planner by choosing the candidate with the highest learned utility.

## Method

- Generate all pairwise metric preferences within each scenario.
- Represent each candidate with normalized planning-score components.
- Train a Bradley-Terry/logistic preference model on chosen-vs-rejected feature differences.
- Clamp learned weights to be non-negative so the tiny synthetic set cannot learn an inverted safety preference.
- Evaluate both in-sample selection and leave-one-scenario-out generalization.

## Learned Weights

| Feature | Weight |
| --- | ---: |
| progress | 5.380 |
| collision_clearance | 4.811 |
| vru_clearance | 0.464 |
| offroad | 0.000 |
| comfort | 0.001 |
| route_error | 2.017 |
| imitation | 1.036 |

## Results

- Pairwise preference accuracy: 18/18 (1.000)
- Mean preferred probability: 0.968
- Training selection match rate: 6/6 (1.000)
- Leave-one-scenario-out match rate: 6/6 (1.000)
- Leave-one-scenario-out unsafe selections: 0

## Held-Out Scenario Choices

| Scenario | Learned selection | Metric best | Utility | Metric score | Match | Unsafe |
| --- | --- | --- | ---: | ---: | --- | --- |
| synthetic_pedestrian_crossing | `metric_aligned_yield` | `metric_aligned_yield` | 5.186 | 11.416 | yes | no |
| synthetic_unprotected_left_turn | `metric_aligned_gap_turn` | `metric_aligned_gap_turn` | 6.102 | 10.128 | yes | no |
| synthetic_cyclist_close_pass | `metric_aligned_wide_pass` | `metric_aligned_wide_pass` | 5.613 | 13.405 | yes | no |
| synthetic_blocked_lane | `metric_aligned_nudge` | `metric_aligned_nudge` | 5.433 | 8.259 | yes | no |
| synthetic_dense_merge | `metric_aligned_gap_merge` | `metric_aligned_gap_merge` | 5.216 | 9.141 | yes | no |
| synthetic_hard_braking_lead_vehicle | `metric_aligned_smooth_brake` | `metric_aligned_smooth_brake` | 3.054 | 5.780 | yes | no |

## Planner Benchmark With Learned Reward

| Planner | Mean score | Progress | Collision clearance | VRU clearance | Unsafe cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference imitation | 9.647 | 10.312 | 0.633 | 2.188 | 0 |
| Progress only | -21.782 | 10.547 | -1.083 | 0.662 | 6 |
| Metric rerank | 9.688 | 10.312 | 0.663 | 2.188 | 0 |
| Learned preference | 9.688 | 10.312 | 0.663 | 2.188 | 0 |

## Takeaway

The learned preference planner recovers the metric-rerank choices on the controlled scenario set and on leave-one-scenario-out evaluation. This turns the project from hard-coded metric scoring into a small, inspectable alignment loop: metrics create preferences, preferences train a reward model, and the reward model selects trajectories.

## Next Experiment

Add objective ablations and harder generated negatives, then scale the same preference-learning interface to optional public motion-data slices.
