> **Logging cập nhật 30/10:** dùng Apps Script, KHÔNG dùng Google Cloud Service Account. Xem `LOGGING_SETUP_VI.md`.

# Group 9 — Streamlit Diabetes Risk Screening (Calibration-aware edition)

**Role:** Student Streamlit deployment prototype for Data Mining Group 9.
**Source:** Team-trained global XGBoost model with 21 BRFSS predictors. The model is loaded from portable `diabetes_global_xgboost.ubj` (exported from the group's trusted `.pkl` without retraining) to avoid Windows pickle errors.
**Not a diagnostic device:** The source positive class includes survey diabetes/prediabetes status. Do not use identifiable real patient data in a class demo.

## 1. Run on Windows (VS Code / PowerShell)

1. Unzip the supplied Group 9 app ZIP (or merge the Part A patch into your existing `diabetes_app` folder).
2. Open **the inner `diabetes_app` folder** in VS Code via **File → Open Folder**.
3. Open **Terminal → New Terminal**.
4. Run each command separately:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe smoke_test.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use **Python 3.13** to match the serialized model's original environment. If `python --version` differs, install Python 3.13, then use `py -3.13 -m venv .venv` for the venv creation step. You do **not** need to activate the venv.

After the Streamlit command starts, open the **Local URL** shown in the terminal (usually <http://localhost:8501>).

## 2. Run on macOS / Linux

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python smoke_test.py
.venv/bin/python -m streamlit run app.py
```

## 3. Demonstrate the project

1. **Keep all 21 inputs as their prefilled defaults**; these match `deployment_bundle/sample_input.csv`.
2. Choose a healthcare follow-up capacity, e.g. `Top 10%`.
3. Click **See my screening result**.
4. The **model-estimated risk score** should be **1.03%** (underlying XGBoost `predict_proba` output 0.010257108). **This is not a clinically validated individual diabetes probability.**
5. Check the 3 plain-English sections: result probability, top SHAP reasons, and suggested next steps.
6. Open **For researchers: calibration, SHAP and capacity metrics** for the calibration curve, Brier scores, research metrics, and the six capacity levels.
7. Check the optional logging message (not yet connected until `logger.py` arrives).
8. Change some health inputs and predict again to demonstrate interactive behavior.

The app uses **one global model** for all capacity choices. Choosing a capacity does **not** retrain the model or change its raw predicted score. Precision@K / Recall@K values come from the locked test set and describe groups, **not the patient's probability**.


## Why the results page looks different

Based on teammate feedback, the primary output now answers **three user questions**:

1. **What does the result mean?** A model-estimated *risk score* (e.g., 1.03%) used for ranking; not a verified personal clinical probability.
2. **Why this score?** Three highest-absolute-value SHAP factors, translated from feature codes into plain-English survey answers and an increase/decrease direction. The accompanying chart displays SHAP contributions, **not probability percentages**.
3. **What can I do next?** General, non-diagnostic guidance: discuss appropriate screening with a clinician, do not skip routine testing solely because of the model, and maintain healthy habits.

Research metrics (approximated percentile, app-defined tier, capacity selection, Precision@K / Recall@K, numeric SHAP values) are retained inside an optional expander for the group report. The team-trained model, feature order, SHAP background, `.ubj` file and five-argument `log_prediction` contract **have not been modified**.

## Part A update — Hưng's calibration and deployment limitations

The app now bundles exactly Hưng's `calibration_report.json` and `calibration_table.csv`, plus a plotted PNG **generated directly from the table**. Open the research expander after Predict to see:

- Brier score **0.0987** vs naive baseline **0.1199**; Brier skill score **0.1771**; locked-test **n = 50,736**, **10 quantile bins**.
- Original global XGBoost calibration curve and ideal calibration reference line. Predicted scores were above observed class rates in the 10 groups; the gap grows at the high-score end. **`recalibrated = false`**.
- The **six** supported capacities (top **0.5%, 1%, 2.5%, 5%, 10%, 20%**) and their locked-test **group-level** metrics. They are NOT six exact individual score thresholds. Other missing score percentiles are **interpolated**, not measured.
- The model's output is presented as a **risk score for ranking/priority**, not "you have a X% chance of having diabetes".
- The top-three SHAP explanation and general non-diagnostic next steps remain unchanged.

**Deliberately NOT implemented in Part A:** Google Sheets logging (Việt Anh), Streamlit Cloud deployment, automatic retraining, or evaluation on the team's planned newer 2024 data. See `HUNG_REPORT_NOTES.md` for report reminders and unresolved selection rationale for the 2015 dataset.

Run the updated `smoke_test.py` to confirm the model and calibration files; it should print **four PASS lines**. Python dependencies are unchanged.

## 4. Integrate Việt Anh's `logger.py` (29–30 October)

Copy `logger.py` to the **same folder as `app.py`**. The app already calls:

```python
log_prediction(inputs, probability, risk_percentile, top_shap, model_version)
```

- `inputs`: dict of 21 features using exact keys from `feature_schema.json`.
- `probability`: float 0–1.
- `risk_percentile`: float 0–100, approximate from `risk_reference.json`.
- `top_shap`: list of exactly 3 `(feature_name, shap_value)` tuples.
- `model_version`: string from `model_metadata.json`.
- `logger.py` creates `prediction_id` and timestamp, writes columns ordered as `prediction_logging_schema.csv`, and returns the ID.

If `logger.py` isn't present, **no predictions are logged**. If it is present and fails, app shows a logging warning rather than pretending it saved the record.

**Important:** Ask Việt Anh about any additional Python libraries (`gspread`, `google-auth`, etc.) and add them to `requirements.txt` when the actual logger requires them. Store API keys/service account credentials in Streamlit Secrets or environment variables, **never commit credentials to GitHub**.

## 5. Deploy on Streamlit Community Cloud (once integration is ready)

1. Create a GitHub repository, upload the **contents of `diabetes_app`** (with `app.py` at repository root and the `deployment_bundle/` folder), but do not upload `.venv/` or secrets.
2. Go to <https://share.streamlit.io>, sign in, choose to create a new app.
3. Select the repository and branch, set the **main file path** to `app.py` and the Python version to **3.13** if selectable.
4. Add credentials in the app's **Secrets** settings if needed for `logger.py`, then deploy.
5. Test the live page with the sample data and screenshot the input + output for the report.

## 6. Interpretation caveats to include in the report

- **Estimated percentile:** interpolation from p50, p75, p90, p95, p99 and max. Below p50 it is especially approximate; it is **not** an exact rank.
- **Capacity selection:** approximates whether an individual falls inside top K% based on estimated percentile. Actual operational top-K selection requires ranking **all** patients in a cohort.
- **Priority labels:** created for demonstration (Very high/top 2.5%, High/top 10%, Moderate/top 20%, Standard otherwise), not validated clinical classifications.
- **SHAP:** calculated in log-odds with supplied background data; explanations describe model behavior, **not causality**.
- **Calibration:** Hưng evaluated the model on the 50,736-observation locked test set (Brier 0.0987 vs naive baseline 0.1199, 10 quantile bins). The model tends to overestimate observed rates, especially in high-score bins; **it was evaluated, not recalibrated**. Risk score is not a clinically verified personal probability.

## 7. Retraining pipeline (offline, added 03 Nov)

`retrain_pipeline.py` retrains the global XGBoost model outside the app, following `deployment_bundle/retraining_config.json`. The app itself still never retrains. Step-by-step guide (Vietnamese): **`RETRAINING_VI.md`**.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-retrain.txt
.\.venv\Scripts\python.exe retrain_pipeline.py --new-data data\sheet_export.csv
```

On the original data the pipeline reproduces v1.0.0 exactly (identical scores, same Brier, OOF PR-AUC and capacity metrics). A new version replaces `deployment_bundle/` only when every promotion criterion passes **and** `--promote` is given; the old bundle is archived in `model_registry/archive/`.

## Troubleshooting

- `python was not found`: install Python 3.13 and enable adding it to PATH, or try `py -3.13`.
- `No module named streamlit`: run the pip installation command from section 1 first (with ` .\.venv\Scripts\python.exe`).
- `FileNotFoundError` mentioning `deployment_bundle`: the ZIP was not extracted completely or VS Code opened the wrong folder.
- `Portable model checksum mismatch`: use the unmodified `diabetes_global_xgboost.ubj` and `portable_model_checksum.txt` supplied together. The original `.pkl` and original `model_checksum.txt` are retained for provenance only.
- Missing `logger.py`: expected until logging integration; the app still predicts.
- When the terminal shows `Local URL`, leave it open while using the site; press **Ctrl+C** to stop.
