# MetricDrive Recruiting Packet

## One-Line Pitch

Built MetricDrive, a public autonomous-driving planning lab that converts
long-tail trajectory choices into metric-derived preferences, learned rewards,
hard-negative stress tests, VLM-style planning rows, and a tiny metric-reward
post-training analogue.

## Resume Bullets

- Built a dependency-light autonomous-driving planning evaluation framework in
  Python with synthetic long-tail scenarios, trajectory metrics, preference
  generation, learned reward modeling, hard-negative stress testing, and static
  GitHub Pages demo export.
- Created 90 metric-derived prompt/chosen/rejected planning preference rows from
  36 trajectory candidates, including generated hard negatives that expose
  unsafe progress-only and no-collision objectives.
- Trained and evaluated a lightweight Bradley-Terry reward model that fit 89/90
  preference pairs and recovered 6/6 held-out metric-best trajectory choices
  with zero unsafe selections.
- Added a public-safe VLM/RL alignment surface: structured planning examples
  with verifiable meta-actions and a tiny metric-reward policy optimization
  analogue that outperforms token-match and progress-only baselines.

## Interview Talking Points

- **Problem framing:** imitation can reward log matching even when another
  trajectory is safer under planning metrics.
- **Research bridge:** MetricDrive is a small public analogue for the loop of
  representation design, metric-derived preferences, learned reward selection,
  and reward-aligned post-training.
- **Engineering scope:** no private data, no heavyweight model dependency, and
  every default command runs on a laptop.
- **Failure analysis:** hard negatives make objective failures visible:
  progress-only and token-match proxies choose unsafe candidates on the stressed
  set, while metric reward and the RL-aligned policy recover the safe metric
  choices.
- **Next work:** connect the current meta-actions to richer public-data slices
  or Waymax-style simulation checks.

## Public Links

- Demo: https://ethanvillalovoz.github.io/metricdrive/
- Repository: https://github.com/ethanvillalovoz/metricdrive
- Portfolio report: `docs/reports/portfolio_report.md`
- Hard-negative report: `docs/reports/milestone_3_hard_negatives.md`
