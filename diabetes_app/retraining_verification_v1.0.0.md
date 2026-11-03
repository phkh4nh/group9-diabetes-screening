# Verification: pipeline reproduces production v1.0.0 (run 20261103T135155Z_cf1f7a, --force, no new data)

- **Status:** CANDIDATE_READY (not promotable — see notes)
- **Started (UTC):** 2026-11-03T13:51:55.110074Z
- **Production model:** v1.0.0
- **Candidate model:** v1.1.0 → `model_registry/candidates/v1.1.0_20261103T135155Z_cf1f7a`

## Data

- Historical: 253,680 rows (`diabetes_binary_health_indicators_BRFSS2015.csv`)
- New labeled records: 0 valid (0 positive / 0 negative), 0 rejected, 0 still unlabeled; 0 not yet used by production
- Training rows: 202,944 historical + 0 new; locked test 50,736; new holdout 0

## Retraining policy not met

- only 0 labeled records not yet used by production (need ≥ 500)
- only 0 new positive outcomes (need ≥ 50)
- only 0 new negative outcomes (need ≥ 50)

## Notes

- ⚠️ --force used: policy bypassed, this candidate can NOT be promoted.

## Evaluation (same rows, candidate vs production)

| Set | Metric | Production | Candidate | Δ |
|---|---|---|---|---|
| locked_test (n=50,736) | PR-AUC | 0.4258 | 0.4258 | +0.0000 |
| locked_test (n=50,736) | ROC-AUC | 0.8280 | 0.8280 | +0.0000 |
| locked_test (n=50,736) | Brier | 0.0987 | 0.0987 | +0.0000 |
| locked_test | Precision@0.5% | 0.6890 | 0.6890 | +0.0000 |
| locked_test | Precision@1% | 0.6614 | 0.6614 | +0.0000 |
| locked_test | Precision@2.5% | 0.6052 | 0.6052 | +0.0000 |
| locked_test | Precision@5% | 0.5416 | 0.5416 | +0.0000 |
| locked_test | Precision@10% | 0.4773 | 0.4773 | +0.0000 |
| locked_test | Precision@20% | 0.3966 | 0.3966 | +0.0000 |

Paired bootstrap 95% interval for the primary PR-AUC difference: [+0.0000, +0.0000]

Candidate 5-fold OOF PR-AUC 0.4384, ROC-AUC 0.8316

## Promotion criteria

| Check | Value | Result |
|---|---|---|
| Primary PR-AUC (locked_test): candidate − production ≥ 0.0 | +0.0000 | PASS |
| Locked-test PR-AUC drop ≤ 0.005 | +0.0000 | PASS |
| Locked-test Brier increase ≤ 0.002 | +0.0000 | PASS |
| Locked-test Precision@0.5% drop ≤ 0.02 | +0.0000 | PASS |
| Locked-test Precision@1% drop ≤ 0.02 | +0.0000 | PASS |
| Locked-test Precision@2.5% drop ≤ 0.02 | +0.0000 | PASS |
| Locked-test Precision@5% drop ≤ 0.02 | +0.0000 | PASS |
| Locked-test Precision@10% drop ≤ 0.02 | +0.0000 | PASS |
| Locked-test Precision@20% drop ≤ 0.02 | +0.0000 | PASS |
