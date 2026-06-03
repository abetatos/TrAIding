# Advances in Financial Machine Learning — Lopez de Prado (2018)

## Reading progress

- [ ] Ch. 1 — Financial Machine Learning as a Distinct Discipline
- [ ] Ch. 2 — Financial Data Structures
- [ ] Ch. 3 — Labeling
- [ ] Ch. 4 — Sample Weights
- [ ] Ch. 5 — Fractional Differentiation
- [ ] Ch. 6 — Ensemble Methods
- [ ] Ch. 7 — Cross-Validation in Finance
- [ ] Ch. 8 — Feature Importance
- [ ] Ch. 18 — Entropy and Microstructural Features
- [ ] Ch. 20 — Backtesting on Synthetic Data

## Notes by chapter

### Ch. 2 — Financial Data Structures
*(notes here)*

### Ch. 3 — Labeling
*(notes here)*
- Triple-barrier method: set profit-take, stop-loss, and time barriers simultaneously
- Meta-labeling: train a secondary model to size/filter positions from a primary signal
- Implementation: `lib/labeling.py`

### Ch. 5 — Fractional Differentiation
*(notes here)*
- Problem: standard diff(1) destroys memory; raw prices are non-stationary
- Solution: fractional diff with d < 1 preserves memory while achieving stationarity
- Find minimum d via ADF test: `lib/features.find_min_d()`

## Key concepts

| Concept | Chapter | Implementation |
|---------|---------|----------------|
| Triple-barrier labeling | 3 | `lib/labeling.triple_barrier_labels` |
| Meta-labeling | 3 | `lib/labeling.meta_labels` |
| Fractional differentiation | 5 | `lib/features.frac_diff` |
| Kyle's lambda | 18 | `lib/features.kyle_lambda` |
| Combinatorial purged CV | 7 | `lib/backtest.walk_forward_splits` |
| Brier score | — | `lib/backtest.brier_score` |
