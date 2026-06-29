# Quant ML Results Dashboard

[![Results Dashboard](https://img.shields.io/badge/Results-Dashboard-117C6F)](https://devmuniz02.github.io/btc-experiments/)
[![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-public-2F7D32)](https://devmuniz02.github.io/btc-experiments/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2)](https://www.linkedin.com/in/devmuniz)

Public-facing results for an automated Quant ML experiment and production evaluation system. This repository is a showcase of outcomes: anonymized experiment summaries, delayed production snapshots, and the discipline used to keep public reporting separate from restricted training assets.

## Latest Public Results

The public dashboard is the main view:

- [Overview](https://devmuniz02.github.io/btc-experiments/)
- [Experiments](https://devmuniz02.github.io/btc-experiments/experiments.html)
- [Production](https://devmuniz02.github.io/btc-experiments/prod.html)

After each completed run, public-safe artifacts are summarized under:

- [Experiment artifacts](experiments/)
- [Production artifacts](prod/)

If the dashboard shows no public runs yet, the project has been reset and is waiting for the next completed experiment or production update.

## What This Shows

- Validation-only direction ranking, so production selection is not shaped by future test outcomes.
- Frozen Top K test policy, so only robust final-lock candidates can enter production reporting.
- Delayed production reporting, designed to show outcomes without exposing operational internals.
- Public-safe artifact boundaries, with restricted training assets kept outside the public dashboard.

## Public Reporting Contract

The public pages intentionally show only market ids, numbered phases, anonymized candidates, version windows, and delayed production metrics. Phase focus names, internal configuration details, training assets, credentials, and non-public datasets are excluded from the public view.

<!-- RESULTS_SNAPSHOT_START -->
## Results Snapshot

### Experiments

| Market | Phase progress | Top validation BA |
|---|---:|---:|
| btc_1h | 5/16 | 0.7636 |

### Production

No public production results yet.

<!-- RESULTS_SNAPSHOT_END -->
