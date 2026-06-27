# Roadmap

## Milestone 0: Research Scaffold

- Write public research spec.
- Add related-work notes and data policy.
- Add minimal package, CLI, and CI.
- Keep project claims honest and reproducible.

Status: implemented.

## Milestone 1: Synthetic Scenario Core

- Define scenario, agent, map, and trajectory schemas.
- Generate controlled long-tail scenario families.
- Render bird's-eye-view scenes.
- Implement first planning metrics.
- Produce a report with metric-ranked candidate trajectories.

Status: implemented for six synthetic long-tail scenario families, transparent
metric ranking, SVG scene rendering, JSON export/load, and a generated
Milestone 1 report.

## Milestone 2: Baseline Planners

- Add imitation baseline.
- Add candidate trajectory generator.
- Add metric-reranking baseline.
- Compare aggregate and per-category results.

Status: implemented for reference imitation, progress-only, and metric-rerank
planner baselines with aggregate/per-scenario benchmark reporting.

## Milestone 3: Preference Alignment

- Generate metric-derived preference pairs.
- Train a preference model or planner objective.
- Compare against imitation and reranking.
- Add ablations for metric terms.

## Milestone 4: Verifiable Meta-Actions

- Define a small meta-action vocabulary.
- Map meta-actions to trajectory candidates.
- Verify actions against scenario metrics.
- Add visual failure analysis.

## Milestone 5: Public Data Integration

- Add optional small public-data slice ingestion.
- Keep raw data outside git.
- Compare synthetic and public-slice behavior.
- Document setup without making heavy data mandatory.

## Milestone 6: Portfolio Release

- Publish paper-style report.
- Add dashboard or static gallery.
- Add screenshots and failure cases.
- Tag a public baseline release.
