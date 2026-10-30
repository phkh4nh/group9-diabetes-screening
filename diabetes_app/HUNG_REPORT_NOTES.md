# Hưng's instructions and evidence — report notes (30 Oct 2026)

These notes preserve Hưng's requests without presenting future/planned work as already completed.

## Confirmed evaluation (from `calibration_report.json` + `calibration_table.csv`)

- Evaluated on the **locked test set** with **50,736** records and **13.93%** survey-positive prevalence.
- **Brier score = 0.0986823761** (~0.0987) vs **naive Brier = 0.1199164846** (~0.1199); **Brier skill score = 0.17707414**.
- Reliability/calibration curve has **10 quantile bins**. In each bin the mean predicted output is higher than the observed positive-class rate; the gap increases in higher-score bins. Highest: predicted **56.32%** vs observed **47.73%** (group averages, NOT patient-level prediction).
- **`recalibrated = false`**: this is a calibration **evaluation**, NOT an actual calibration/recalibration step. Do not claim calibration fixed the probabilities.
- Interpret `predict_proba` as a **model-estimated risk score** used chiefly for ranking and selecting a top-K follow-up group. Do **not** say "patient/customer has X% diabetes" or claim the raw score is an accurate clinical disease probability.
- Include calibration evaluation in **Methodology/Evaluation** (Brier + curve), numbers/figure in **Results**, and implications/limitations in **Discussion and 5.1 Deployment**. The report owners for Methodology/Results should be informed to ensure consistency.

## Capacity details from the team's existing bundle

- There are **six capacity levels only**: **0.5%, 1%, 2.5%, 5%, 10%, 20%** (see `capacity_config.json`), i.e. top-K *shares* of a cohort.
- `risk_reference.json` stores **p50, p75, p90, p95, p99, max** only. It does NOT supply exact patient-score cutoffs at all six capacities (e.g. p99.5, p97.5, p80). The app's per-person percentile and eligibility are approximate interpolations; strict top-K requires scoring/ranking the entire target cohort.
- `Precision@K`, `Recall@K` and `Lift@K` are **locked-test group metrics**, not individual probabilities. Demo priority names (Very high / High / Moderate / Standard) are NOT medical risk thresholds.

## Planned architecture and report sections for Khanh

- **5.1 Deployment Strategy:** Streamlit demo loads the globally optimized XGBoost model; accepts 21 BRFSS predictors; returns raw model score and SHAP explanations. Emphasize calibration limitations, non-diagnostic scope, privacy/identifiable patient data restrictions. Describe Google Sheets logging as *planned/pending until connected and end-to-end tested*. In production, real healthcare systems would require privacy/security controls and independent external validation.
- **5.2 Prioritization Workflow:** rank the *cohort* in descending predicted score, select top-K based on actual healthcare follow-up capacity; show the six supported operating shares and their Precision@K/Recall@K/Lift@K, not any claimed medical cutoff. A single-input Streamlit demo offers only approximate percentile comparison, not strict cohort selection.
- **Continuous learning:** Hưng plans an additional Python script alongside his original model code to periodically check new **labeled outcomes**, retrain a candidate and evaluate against the production model. The team's `retraining_config.json` proposes a **30-day** schedule *conditional on sufficient labeled data*, with PR-AUC primary and capacity metrics. Do NOT claim fully automated learning, retraining, or model replacement is working in Streamlit now.
- **Newer data:** Hưng says there are data in **2021 and 2024** and intends to use **2024 data** to check continuous-learning feasibility. This is a **future test**, not a verified 2024 performance result. Confirm the files, provenance, label coding and feature comparability before making claims.
- **Why use BRFSS 2015?** Hưng explicitly asked the team to add a rationale for choosing 2015 despite newer data. **His message does not establish the actual reason.** Request the real project rationale; do not invent one or imply newer datasets are directly compatible without checking.
- **Google Sheets logging:** Hưng wants the feedback sheet available soon; the actual connection and `logger.py` integration belong to Khanh and Việt Anh in the separate evening discussion, not Part A. Prediction ID, timestamp, outcomes, and conditional retraining cannot be claimed operational until validated.

## What Part A changed vs what remains untouched

- Changed: risk-score wording, calibration metrics + image/data in the **For researchers** expander, six-capacity research table, research limitations, updated smoke test and documentation.
- Unchanged: trained model `.ubj` and `.pkl`, SHAP computation, feature schema, 21-question form, capacity configuration, 5-parameter `log_prediction` call, requirements and deployed/cloud/logging status.
