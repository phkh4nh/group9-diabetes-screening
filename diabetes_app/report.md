# Retraining run 20261111T081944Z_b041e9

- **Status:** CANDIDATE_READY (run again with --promote to deploy)
- **Started (UTC):** 2026-11-11T08:19:44.406084Z
- **Production model:** v1.0.0
- **Candidate model:** v1.1.0 → `model_registry/candidates/v1.1.0_20261111T081944Z_b041e9`

## Data

- Historical: 253,680 rows (`diabetes_binary_health_indicators_BRFSS2015.csv`)
- New labeled records: 201,093 valid (31049 positive / 170044 negative), 0 rejected, 0 still unlabeled; 201,093 not yet used by production
- Training rows: 202,944 historical + 160,934 new; locked test 50,736; new holdout 40,159

## Notes

- ⚠️ Historical CSV checksum differs from the reference copy; the locked-test check below decides whether it is still the same data.

## Evaluation (same rows, candidate vs production)

| Set | Metric | Production | Candidate | Δ |
|---|---|---|---|---|
| locked_test (n=50,736) | PR-AUC | 0.4258 | 0.4247 | -0.0011 |
| locked_test (n=50,736) | ROC-AUC | 0.8280 | 0.8280 | -0.0001 |
| locked_test (n=50,736) | Brier | 0.0987 | 0.0992 | +0.0005 |
| locked_test | Precision@0.5% | 0.6890 | 0.6890 | +0.0000 |
| locked_test | Precision@1% | 0.6614 | 0.6693 | +0.0079 |
| locked_test | Precision@2.5% | 0.6052 | 0.6076 | +0.0024 |
| locked_test | Precision@5% | 0.5416 | 0.5357 | -0.0059 |
| locked_test | Precision@10% | 0.4773 | 0.4752 | -0.0022 |
| locked_test | Precision@20% | 0.3966 | 0.3972 | +0.0006 |
| new_holdout (n=40,159) | PR-AUC | 0.4155 | 0.4197 | +0.0042 |
| new_holdout (n=40,159) | ROC-AUC | 0.8085 | 0.8113 | +0.0028 |
| new_holdout (n=40,159) | Brier | 0.1085 | 0.1080 | -0.0005 |
| new_holdout | Precision@0.5% | 0.6517 | 0.6766 | +0.0249 |
| new_holdout | Precision@1% | 0.6294 | 0.6318 | +0.0025 |
| new_holdout | Precision@2.5% | 0.5827 | 0.5847 | +0.0020 |
| new_holdout | Precision@5% | 0.5378 | 0.5433 | +0.0055 |
| new_holdout | Precision@10% | 0.4771 | 0.4858 | +0.0087 |
| new_holdout | Precision@20% | 0.4011 | 0.4058 | +0.0046 |

Paired bootstrap 95% interval for the primary PR-AUC difference: [+0.0026, +0.0063]

## Promotion criteria

| Check | Value | Result |
|---|---|---|
| Primary PR-AUC (new_holdout): candidate − production ≥ 0.0 | +0.0042 | PASS |
| Locked-test PR-AUC drop ≤ 0.005 | +0.0011 | PASS |
| Locked-test Brier increase ≤ 0.002 | +0.0005 | PASS |
| Locked-test Precision@0.5% drop ≤ 0.02 | +0.0000 | PASS |
| Locked-test Precision@1% drop ≤ 0.02 | -0.0079 | PASS |
| Locked-test Precision@2.5% drop ≤ 0.02 | -0.0024 | PASS |
| Locked-test Precision@5% drop ≤ 0.02 | +0.0059 | PASS |
| Locked-test Precision@10% drop ≤ 0.02 | +0.0022 | PASS |
| Locked-test Precision@20% drop ≤ 0.02 | -0.0006 | PASS |
