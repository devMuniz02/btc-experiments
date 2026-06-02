# Quant-Stream Ensemble Specs

Ensembles are standard `model_type: ensemble` requests that read prediction streams already present in
`data/var_X/global_results.parquet`.

## Hyperparameters

```yaml
model_type: ensemble
variation_id: 1
train_mode: static_baseline
hyperparameters:
  models_pool:
    - model_id_a
    - model_id_b
  voting_mechanism: hard_majority
```

## Voting Modes

- `hard_majority`: modal class wins across `0 = sell` and `1 = buy`; exact ties return `0` (`sell`).
- `soft_average`: probabilities are averaged by the class predicted by each member; the highest averaged class wins.
- `unanimity`: returns the agreed `sell` or `buy` class only when all members agree; disagreements return confidence `0`.
