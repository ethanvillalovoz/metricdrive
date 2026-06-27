# Milestone 3D: Hard Negative Stress Test

MetricDrive now augments every synthetic scenario with generated hard negatives. These are deterministic trajectory perturbations designed to create tighter safety, progress, route, and comfort tradeoffs than the original hand-authored candidates.

## Generation Strategy

- `generated_progress_pressure`: blends the metric-best trajectory toward the highest-progress candidate.
- `generated_under_commit`: shortens the metric-best trajectory to test low-progress or conflict-zone hesitation.
- `generated_lateral_wobble`: adds alternating lateral motion to test comfort and route robustness.

## Summary

- Original candidates: 18
- Generated hard negatives: 18
- Augmented candidates: 36
- Metric-derived preference pairs: 90
- Unsafe generated candidates: 6
- Near-miss generated candidates within 2 score points: 4
- Learned reward pairwise fit: 89/90 (0.989)
- Learned reward held-out recovery: 6/6
- Learned reward held-out unsafe selections: 0

## Generated Candidate Scores

| Scenario | Candidate | Rank | Score | Gap | Clearance | Progress | Tags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic_pedestrian_crossing | `generated_progress_pressure` | 2 | 8.526 | -2.890 | 0.238 | 11.016 | metric_tradeoff |
| synthetic_pedestrian_crossing | `generated_under_commit` | 3 | 7.903 | -3.513 | 1.003 | 6.509 | low_progress, route_error |
| synthetic_pedestrian_crossing | `generated_lateral_wobble` | 5 | -0.093 | -11.509 | -0.088 | 10.224 | unsafe_collision, comfort_stress |
| synthetic_unprotected_left_turn | `generated_progress_pressure` | 2 | 8.533 | -1.595 | 0.156 | 11.332 | near_miss |
| synthetic_unprotected_left_turn | `generated_under_commit` | 6 | -40.866 | -50.994 | -1.756 | 7.069 | unsafe_collision, low_progress, route_error |
| synthetic_unprotected_left_turn | `generated_lateral_wobble` | 3 | 6.424 | -3.704 | 0.898 | 10.652 | comfort_stress |
| synthetic_cyclist_close_pass | `generated_progress_pressure` | 2 | 11.595 | -1.810 | 0.298 | 11.740 | near_miss |
| synthetic_cyclist_close_pass | `generated_under_commit` | 3 | 8.759 | -4.646 | 0.860 | 7.433 | low_progress, route_error |
| synthetic_cyclist_close_pass | `generated_lateral_wobble` | 5 | 5.038 | -8.367 | 0.001 | 10.850 | comfort_stress |
| synthetic_blocked_lane | `generated_progress_pressure` | 3 | 0.953 | -7.306 | -0.293 | 11.805 | unsafe_collision |
| synthetic_blocked_lane | `generated_under_commit` | 5 | -6.220 | -14.479 | -0.348 | 7.436 | unsafe_collision, low_progress, route_error |
| synthetic_blocked_lane | `generated_lateral_wobble` | 2 | 5.749 | -2.510 | 0.414 | 10.950 | comfort_stress |
| synthetic_dense_merge | `generated_progress_pressure` | 2 | 8.882 | -0.259 | 0.220 | 11.301 | near_miss |
| synthetic_dense_merge | `generated_under_commit` | 6 | -12.727 | -21.868 | -0.637 | 6.751 | unsafe_collision, low_progress, route_error |
| synthetic_dense_merge | `generated_lateral_wobble` | 3 | 4.568 | -4.573 | 0.146 | 10.742 | comfort_stress |
| synthetic_hard_braking_lead_vehicle | `generated_progress_pressure` | 5 | -25.611 | -31.391 | -1.270 | 7.970 | unsafe_collision |
| synthetic_hard_braking_lead_vehicle | `generated_under_commit` | 2 | 5.233 | -0.547 | 3.104 | 3.596 | near_miss, low_progress, route_error |
| synthetic_hard_braking_lead_vehicle | `generated_lateral_wobble` | 4 | 0.578 | -5.202 | 0.992 | 5.676 | comfort_stress |

## Stress Ablations On Augmented Set

| Objective | Held-out match | Unsafe | Score gap | Pairwise fit |
| --- | ---: | ---: | ---: | ---: |
| Full objective | 6/6 (1.000) | 0 | 0.000 | 0.989 |
| No collision term | 2/6 (0.333) | 3 | -12.521 | 0.700 |
| No VRU clearance | 5/6 (0.833) | 0 | -0.481 | 0.978 |
| No progress | 5/6 (0.833) | 0 | -0.091 | 0.967 |
| No route error | 6/6 (1.000) | 0 | 0.000 | 0.989 |
| No imitation | 6/6 (1.000) | 0 | 0.000 | 0.989 |
| No comfort | 5/6 (0.833) | 0 | -0.867 | 0.944 |
| Progress only | 1/6 (0.167) | 5 | -22.296 | 0.544 |
| Safety only | 2/6 (0.333) | 0 | -2.654 | 0.789 |

## Takeaway

The generated negatives expand the preference set from 18 to 90 pairs while preserving a clean learned-reward result: the full learned objective still recovers every held-out metric-best trajectory with zero unsafe selections. The stress ablations become sharper on the augmented set, especially no-collision and progress-only objectives, which confirms the hard negatives are exposing the intended failure modes.

## Next Experiment

Define a verifiable meta-action vocabulary, such as `YIELD_TO_VRU`, `NUDGE_AROUND_OBSTACLE`, and `SLOW_FOR_LEAD`, then connect those actions to generated trajectory candidates and metric checks.
