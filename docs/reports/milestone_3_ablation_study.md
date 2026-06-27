# Milestone 3C: Objective Ablation Study

MetricDrive now tests which objective terms matter by retraining the learned preference model with individual metric components removed or isolated. The same leave-one-scenario-out protocol is used for every ablation.

## Method

- Start from the learned Bradley-Terry preference reward model.
- Retrain with selected metric components removed or isolated.
- Evaluate held-out scenario selections against the metric-rerank choice.
- Track unsafe selections and metric-score gaps to expose failure modes.

## Summary

| Objective | Active features | Held-out match | Unsafe | Mean score | Score gap | Pairwise fit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full objective | progress, collision_clearance, vru_clearance, offroad, comfort, route_error, imitation | 6/6 (1.000) | 0 | 9.688 | 0.000 | 18/18 (1.000) |
| No collision term | progress, vru_clearance, offroad, comfort, route_error, imitation | 4/6 (0.667) | 1 | 6.793 | -2.895 | 13/18 (0.722) |
| No VRU clearance | progress, collision_clearance, offroad, comfort, route_error, imitation | 6/6 (1.000) | 0 | 9.688 | 0.000 | 18/18 (1.000) |
| No progress | collision_clearance, vru_clearance, offroad, comfort, route_error, imitation | 5/6 (0.833) | 0 | 9.299 | -0.389 | 18/18 (1.000) |
| No route error | progress, collision_clearance, vru_clearance, offroad, comfort, imitation | 6/6 (1.000) | 0 | 9.688 | 0.000 | 18/18 (1.000) |
| No imitation | progress, collision_clearance, vru_clearance, offroad, comfort, route_error | 6/6 (1.000) | 0 | 9.688 | 0.000 | 18/18 (1.000) |
| No comfort | progress, collision_clearance, vru_clearance, offroad, route_error, imitation | 6/6 (1.000) | 0 | 9.688 | 0.000 | 18/18 (1.000) |
| Progress only | progress | 0/6 (0.000) | 6 | -21.782 | -31.470 | 8/18 (0.444) |
| Safety only | collision_clearance, vru_clearance, offroad | 2/6 (0.333) | 0 | 6.169 | -3.519 | 14/18 (0.778) |

## Held-Out Failure Cases

| Objective | Scenario | Selected | Metric best | Score gap | Unsafe |
| --- | --- | --- | --- | ---: | --- |
| No collision term | synthetic_pedestrian_crossing | `cautious_stop` | `metric_aligned_yield` | -3.643 | no |
| No collision term | synthetic_dense_merge | `imitation_force_merge` | `metric_aligned_gap_merge` | -13.726 | yes |
| No progress | synthetic_hard_braking_lead_vehicle | `cautious_hard_stop` | `metric_aligned_smooth_brake` | -2.335 | no |
| Progress only | synthetic_pedestrian_crossing | `imitation_fast_log` | `metric_aligned_yield` | -42.068 | yes |
| Progress only | synthetic_unprotected_left_turn | `imitation_aggressive_turn` | `metric_aligned_gap_turn` | -42.006 | yes |
| Progress only | synthetic_cyclist_close_pass | `imitation_close_pass` | `metric_aligned_wide_pass` | -20.227 | yes |
| Progress only | synthetic_blocked_lane | `imitation_straight_blocked` | `metric_aligned_nudge` | -26.364 | yes |
| Progress only | synthetic_dense_merge | `imitation_force_merge` | `metric_aligned_gap_merge` | -13.726 | yes |
| Progress only | synthetic_hard_braking_lead_vehicle | `imitation_maintain_speed` | `metric_aligned_smooth_brake` | -44.430 | yes |
| Safety only | synthetic_pedestrian_crossing | `cautious_stop` | `metric_aligned_yield` | -3.643 | no |
| Safety only | synthetic_cyclist_close_pass | `cautious_follow` | `metric_aligned_wide_pass` | -7.436 | no |
| Safety only | synthetic_blocked_lane | `cautious_stop_behind` | `metric_aligned_nudge` | -7.699 | no |
| Safety only | synthetic_hard_braking_lead_vehicle | `cautious_hard_stop` | `metric_aligned_smooth_brake` | -2.335 | no |

## Takeaway

Collision clearance is the most brittle single term: removing it leaves 1 unsafe held-out selection and drops match rate to 0.667. A progress-only objective recreates the dangerous baseline, selecting unsafe trajectories in 6 of 6 held-out scenarios. Safety-only avoids collisions but matches only 2/6 metric-rerank choices, showing why the aligned objective needs both safety and progress terms.

## Next Experiment

Generate harder negatives that force tradeoffs between progress, collision clearance, vulnerable-road-user clearance, comfort, and route adherence.
