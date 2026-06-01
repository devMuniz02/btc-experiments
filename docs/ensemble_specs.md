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

- `hard_majority`: modal class wins across `-1`, `0`, and `1`; exact ties return `0`.
- `soft_average`: probabilities are averaged by the class predicted by each member; the highest averaged class wins.
- `unanimity`: returns `1` or `-1` only when all members agree on the same non-zero direction; otherwise returns `0`.
