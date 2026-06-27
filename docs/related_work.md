# Related Work Notes

These notes track the public research context MetricDrive is designed to learn
from. They are not a claim of affiliation or reproduction.

## EMMA

**Paper:** [EMMA: End-to-End Multimodal Model for Autonomous Driving](https://arxiv.org/abs/2410.23262)

Why it matters: EMMA frames autonomous driving as a multimodal sequence problem
where perception, prediction, mapping, and planning can share a common model
interface. MetricDrive borrows the representation question: what should a
driving planner see, and how should it express future motion?

## S4-Driver

**Project:** [S4-Driver](https://s4-driver.github.io/)  
**Paper:** [S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation](https://arxiv.org/abs/2505.24139)

Why it matters: S4-Driver is close to the modern Waymo-style direction of
large multimodal driving models with spatio-temporal representations. MetricDrive
uses this as motivation for a small public analogue focused on the planning
alignment loop rather than model scale.

## WOD-E2E

**Paper:** [WOD-E2E: A Large-Scale Real-World Benchmark for End-to-End Autonomous Driving](https://arxiv.org/abs/2510.26125)

Why it matters: WOD-E2E highlights long-tail planning and preference-style
evaluation. MetricDrive's "RFS-lite" idea turns transparent planning metrics
into candidate preferences when human rater labels are unavailable.

## Poutine

**Paper:** [Poutine: Vision-Language-Trajectory Pretraining and Reinforcement Learning Post-Training for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.11234)

Why it matters: Poutine connects multimodal driving models with post-training
against planning preferences. MetricDrive starts with metric-derived preferences
because they are public, reproducible, and inspectable.

## DriveMA

**Paper:** [DriveMA: Driving with Verifiable Meta-Actions](https://arxiv.org/abs/2605.31271)

Why it matters: Meta-actions make planner decisions easier to inspect than raw
waypoints alone. MetricDrive plans to test small verifiable actions such as
`YIELD_TO_VRU` and `SLOW_FOR_CUT_IN`.

## Waymax

**Repository:** [waymo-research/waymax](https://github.com/waymo-research/waymax)

Why it matters: Waymax provides a public reference for Waymo-style scenario
simulation and metrics. MetricDrive should treat it as an optional integration,
not a requirement for the default demo.

## NAVSIM

**Paper:** [NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking](https://arxiv.org/abs/2406.15349)

Why it matters: NAVSIM sits between simple open-loop prediction and full
closed-loop driving. It is useful context for making MetricDrive's evaluation
more realistic over time.

## Classical and Planning-Oriented Baselines

- [PlanT](https://arxiv.org/abs/2210.14222)
- [VAD](https://arxiv.org/abs/2303.12077)
- [UniAD](https://arxiv.org/abs/2212.10156)

Why they matter: These projects give non-VLM context for planning-oriented
autonomous-driving evaluation and help keep MetricDrive from becoming only a
language-model story.
