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

### 2026-06-27 - Objective Ablation Study

- Goal: identify which learned reward terms prevent unsafe or overly cautious selections.
- Command: `metricdrive ablations --report docs/reports/milestone_3_ablation_study.md`
- Data: same six synthetic long-tail scenarios and metric-derived preference pairs.
- Method: retrain the preference model with individual objective terms removed or isolated, then rerun leave-one-scenario-out selection.
- Metrics: held-out metric-rerank match rate, unsafe selection count, mean selected metric score, metric-score gap, pairwise preference fit.
- Result: the full objective matched 6/6 held-out metric-rerank choices with zero unsafe selections; removing collision clearance dropped to 4/6 and produced one unsafe dense-merge selection; progress-only selected unsafe trajectories in 6/6 held-out scenarios; safety-only avoided unsafe choices but matched only 2/6 metric-rerank choices.
- What changed next: generate harder negatives that force richer tradeoffs between collision clearance, VRU clearance, progress, comfort, and route adherence.

### 2026-06-27 - Hard Negative Stress Test

- Goal: generate tighter trajectory negatives and test whether the learned reward still recovers metric-aligned choices.
- Command: `metricdrive hard-negatives --report docs/reports/milestone_3_hard_negatives.md`
- Data: six synthetic scenarios augmented from 18 original candidates to 36 total candidates.
- Method: add generated progress-pressure, under-commit, and lateral-wobble candidates to every scenario, then regenerate metric-derived preference pairs and rerun learned-model plus ablation evaluation.
- Metrics: generated unsafe/near-miss counts, preference-pair count, learned pairwise fit, held-out metric-rerank match rate, held-out unsafe selections, and stress-ablation failures.
- Result: the augmented set produced 90 preference pairs; the learned reward fit 89/90 pairs, matched 6/6 held-out metric-rerank choices, and made zero unsafe held-out selections. On the stressed set, no-collision matched only 2/6 held-out choices with 3 unsafe selections, while progress-only matched 1/6 with 5 unsafe selections.
- What changed next: define verifiable meta-actions and connect them to generated trajectory candidates plus metric checks.
