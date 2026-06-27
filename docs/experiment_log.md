# Experiment Log

Use this file for short, reproducible experiment notes.

## Template

### YYYY-MM-DD - Experiment Name

- Goal:
- Command:
- Data:
- Method:
- Metrics:
- Result:
- What changed next:

### 2026-06-27 - Learned Preference Reward Model

- Goal: train a small reward model from metric-derived trajectory preference pairs.
- Command: `metricdrive learned --report docs/reports/milestone_3_learned_model.md`
- Data: six built-in synthetic long-tail scenarios with three candidate trajectories each.
- Method: non-negative linear Bradley-Terry preference model over normalized planning-score components.
- Metrics: pairwise preference accuracy, training selection match rate, leave-one-scenario-out match rate, unsafe selection count.
- Result: 18/18 pairwise preferences fit, 6/6 training selections matched metric reranking, 6/6 leave-one-scenario-out selections matched metric reranking, and zero held-out unsafe selections.
- What changed next: add objective ablations and harder generated negatives before scaling to optional public-data slices.
