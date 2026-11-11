# Retraining experiment: BRFSS 2025 (run 20261111T081944Z_b041e9)

Continuous-learning test of `retrain_pipeline.py` with the BRFSS 2025 cohort.
Produced by `notebooks/Retrain_BRFSS2025_Group9.ipynb` on Google Colab, 11 Nov 2026.

**Deployment status: not deployed.** The app still serves v1.0.0. Candidate v1.1.0 passed every
promotion criterion; switching the app to it is a team decision (`--promote`).

## Data

- Historical: BRFSS 2015 (UCI id 891), 253,680 rows. Locked test (50,736 rows) re-created exactly;
  the pipeline reproduced v1.0.0 on Colab with zero score difference.
- New: BRFSS 2025 (CDC, 356,158 records) recoded to the 21 features + `Diabetes_binary`
  (diabetes = 1; prediabetes, gestational-only and no = 0; don't know/refused dropped).
  201,093 complete records kept (31,049 positive, 15.4%).
- 160,934 new records added to training; 40,159 held out (stable hash split), never trained on.

## Results (same rows for both models)

| Evaluation set | Metric | v1.0.0 (2015) | v1.1.0 (2015 + 2025) | Δ |
|---|---|---|---|---|
| 2025 holdout (n = 40,159) | PR-AUC | 0.4155 | 0.4197 | +0.0042 |
| 2025 holdout | ROC-AUC | 0.8085 | 0.8113 | +0.0028 |
| 2025 holdout | Brier | 0.1085 | 0.1080 | −0.0005 |
| 2025 holdout | Precision@0.5% / 1% / 5% / 10% / 20% | 0.652 / 0.629 / 0.538 / 0.477 / 0.401 | 0.677 / 0.632 / 0.543 / 0.486 / 0.406 | all ↑ |
| 2015 locked test (n = 50,736) | PR-AUC | 0.4258 | 0.4247 | −0.0011 |
| 2015 locked test | Brier | 0.0987 | 0.0992 | +0.0005 |

Paired bootstrap 95% interval for the 2025-holdout PR-AUC gain: [+0.0026, +0.0063].

**Reading:** v1.0.0 loses accuracy on the newer cohort (ROC-AUC 0.828 on 2015 → 0.809 on 2025).
Retraining with 2025 data improves results on 2025 at every capacity level while staying
essentially unchanged on the 2015 locked test. The gain is modest.

Full output: `report.md` / `report.json`. Calibration of v1.1.0 on the locked test: `calibration_curve_v1.1.0.png`.
Promotion thresholds in `retraining_config.json` are still marked PROPOSED.
