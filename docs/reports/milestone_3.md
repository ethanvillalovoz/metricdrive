# Milestone 3: Metric-Derived Preferences

MetricDrive now converts trajectory metric comparisons into preference pairs. This is the alignment data layer: each row says which trajectory should be preferred, which should be rejected, and which metrics explain the choice.

## Summary

- Preference pairs: 18
- Scenario coverage: 6
- Mean score margin: 20.980
- Pairs rejecting an unsafe trajectory: 13

## Pair Counts By Scenario Category

| Category | Pairs |
| --- | ---: |
| blocked_lane | 3 |
| cyclist_close_pass | 3 |
| dense_merge | 3 |
| hard_braking_lead_vehicle | 3 |
| pedestrian_crossing | 3 |
| unprotected_left_turn | 3 |

## Reason Categories

| Reason | Count |
| --- | ---: |
| avoids_collision | 12 |
| improves_clearance | 2 |
| improves_comfort | 2 |
| improves_progress | 7 |
| improves_route_completion | 7 |
| improves_vru_clearance | 4 |

## Examples

| Scenario | Preferred | Rejected | Margin | Metric reasons |
| --- | --- | --- | ---: | --- |
| synthetic_pedestrian_crossing | `metric_aligned_yield` | `cautious_stop` | 3.643 | adds 8.287 m of route progress; reduces final route error by 8.287 m |
| synthetic_pedestrian_crossing | `metric_aligned_yield` | `imitation_fast_log` | 42.068 | avoids an overlap or negative-clearance trajectory; increases vulnerable-road-user clearance by 1.816 m |
| synthetic_pedestrian_crossing | `cautious_stop` | `imitation_fast_log` | 38.425 | avoids an overlap or negative-clearance trajectory; increases vulnerable-road-user clearance by 3.571 m |
| synthetic_unprotected_left_turn | `metric_aligned_gap_turn` | `cautious_hold_position` | 12.778 | improves collision clearance by 0.533 m; adds 10.348 m of route progress; reduces final route error by 10.348 m |
| synthetic_unprotected_left_turn | `metric_aligned_gap_turn` | `imitation_aggressive_turn` | 42.006 | avoids an overlap or negative-clearance trajectory |
| synthetic_unprotected_left_turn | `cautious_hold_position` | `imitation_aggressive_turn` | 29.228 | avoids an overlap or negative-clearance trajectory; reduces comfort cost by 0.521 |
| synthetic_cyclist_close_pass | `metric_aligned_wide_pass` | `cautious_follow` | 7.436 | adds 7.200 m of route progress; reduces final route error by 7.200 m |
| synthetic_cyclist_close_pass | `metric_aligned_wide_pass` | `imitation_close_pass` | 20.227 | avoids an overlap or negative-clearance trajectory; increases vulnerable-road-user clearance by 1.235 m |
| synthetic_cyclist_close_pass | `cautious_follow` | `imitation_close_pass` | 12.791 | avoids an overlap or negative-clearance trajectory; increases vulnerable-road-user clearance by 1.360 m |
| synthetic_blocked_lane | `metric_aligned_nudge` | `cautious_stop_behind` | 7.699 | adds 7.900 m of route progress; reduces final route error by 7.900 m |
| synthetic_blocked_lane | `metric_aligned_nudge` | `imitation_straight_blocked` | 26.364 | avoids an overlap or negative-clearance trajectory |
| synthetic_blocked_lane | `cautious_stop_behind` | `imitation_straight_blocked` | 18.665 | avoids an overlap or negative-clearance trajectory |
| synthetic_dense_merge | `metric_aligned_gap_merge` | `cautious_wait_merge` | 9.611 | avoids an overlap or negative-clearance trajectory; adds 6.862 m of route progress; reduces final route error by 6.862 m |
| synthetic_dense_merge | `metric_aligned_gap_merge` | `imitation_force_merge` | 13.726 | avoids an overlap or negative-clearance trajectory; reduces comfort cost by 0.375 |
| synthetic_dense_merge | `cautious_wait_merge` | `imitation_force_merge` | 4.115 | improves collision clearance by 0.524 m |
| synthetic_hard_braking_lead_vehicle | `metric_aligned_smooth_brake` | `cautious_hard_stop` | 2.335 | adds 3.700 m of route progress; reduces final route error by 3.700 m |
| synthetic_hard_braking_lead_vehicle | `metric_aligned_smooth_brake` | `imitation_maintain_speed` | 44.430 | avoids an overlap or negative-clearance trajectory; adds 1.800 m of route progress; reduces final route error by 1.800 m |
| synthetic_hard_braking_lead_vehicle | `cautious_hard_stop` | `imitation_maintain_speed` | 42.095 | avoids an overlap or negative-clearance trajectory |

## Training Shape

Each JSON pair includes `prompt`, `chosen`, and `rejected` fields, so it can be adapted into a DPO-style preference-tuning row. The current labels are deterministic because they come from transparent metrics, which makes the dataset reproducible and debuggable before introducing learned rewards or policy updates.

## Next Experiment

Export these metric-derived pairs as public-safe VLM planning rows, then compare token-match proxies against metric-reward policy optimization.
