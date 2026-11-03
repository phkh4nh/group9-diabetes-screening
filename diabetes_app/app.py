"""Group 9 – capacity-aware diabetes screening research prototype.

Run: python -m streamlit run app.py
The app loads the team's existing model. It does not train or retrain it.
"""

from __future__ import annotations

import hashlib
import json
import xgboost as xgb
from pathlib import Path
from io import BytesIO

from batch_processing import (
    MAX_UPLOAD_BYTES, validate_cohort, rank_and_select, create_excel_results,
)

import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "deployment_bundle"

st.set_page_config(
    page_title="Group 9 | Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
)

# Việt Anh supplies logger.py later. No fake "saved" message in demo mode.
try:
    from logger import log_prediction  # type: ignore
    LOGGING_CONNECTED = True
except ModuleNotFoundError as exc:
    if exc.name != "logger":
        raise
    LOGGING_CONNECTED = False

    def log_prediction(inputs, probability, risk_percentile, top_shap, model_version):
        return None


@st.cache_resource(show_spinner="Loading model and SHAP explainer...")
def load_bundle():
    # Portable XGBoost model: avoids cross-platform pickle deserialization errors.
    model_path = BUNDLE / "diabetes_global_xgboost.ubj"
    expected = (BUNDLE / "portable_model_checksum.txt").read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Portable model checksum mismatch: use the unmodified .ubj file.")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    def read_json(filename):
        return json.loads((BUNDLE / filename).read_text(encoding="utf-8"))

    schema = read_json("feature_schema.json")
    metadata = read_json("model_metadata.json")
    risk_ref = read_json("risk_reference.json")
    capacity_info = read_json("capacity_config.json")
    features = schema["feature_order"]
    background = pd.read_csv(BUNDLE / "shap_background.csv")[features]
    sample = pd.read_csv(BUNDLE / "sample_input.csv")[features].iloc[0].to_dict()
    if len(features) != 21 or len(background) == 0:
        raise ValueError("Invalid model feature schema or empty SHAP background data.")
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional"
    )
    return model, schema, metadata, risk_ref, capacity_info, sample, explainer


@st.cache_data(show_spinner=False)
def load_calibration_evaluation():
    """Team-supplied locked-test calibration evaluation; does NOT recalibrate scores."""
    report = json.loads((BUNDLE / "calibration_report.json").read_text(encoding="utf-8"))
    bins = pd.read_csv(BUNDLE / "calibration_table.csv")
    required = ["Mean predicted probability", "Observed diabetes rate", "Absolute gap"]
    if not all(col in bins.columns for col in required):
        raise ValueError("Missing columns in calibration_table.csv")
    if report.get("evaluation_set") != "locked_test_set":
        raise ValueError("Calibration results are not identified as locked-test results.")
    if len(bins) != report["n_calibration_bins"]:
        raise ValueError("Calibration bin count does not match the report.")
    if not (bins[required].ge(0).all().all() and bins[required[:2]].le(1).all().all()):
        raise ValueError("Invalid calibration rates in calibration_table.csv")
    if not np.allclose((bins[required[0]] - bins[required[1]]).abs(), bins[required[2]], atol=1e-8):
        raise ValueError("Calibration gaps do not match the supplied mean and observed rates.")
    if report.get("recalibrated") is not False:
        raise ValueError("Unexpected calibration status: verify evaluation files with the team.")
    if not (BUNDLE / "calibration_curve.png").is_file():
        raise ValueError("Missing calibration_curve.png")
    return report, bins


def estimate_percentile(probability: float, reference: dict) -> float:
    """Approximate score percentile from six quantiles, NOT an empirical CDF.

    Especially below p50, a linear estimate from (0,0) to (p50,50) is rough.
    """
    probs = [0.0] + [float(reference[k]) for k in ("p50", "p75", "p90", "p95", "p99", "max")]
    percentiles = [0.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0]
    if any(b <= a for a, b in zip(probs, probs[1:])):
        raise ValueError("Invalid percentile reference: probabilities must be increasing.")
    return float(np.interp(probability, probs, percentiles))


def demo_priority(estimated_percentile: float) -> tuple[str, float | None]:
    """App-created priority names; these are not medical thresholds."""
    if estimated_percentile >= 97.5:
        return "Very high", 0.025
    if estimated_percentile >= 90.0:
        return "High", 0.10
    if estimated_percentile >= 80.0:
        return "Moderate", 0.20
    return "Standard", None


AGE = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+",
}
EDUCATION = {
    1: "Never attended / kindergarten only", 2: "Elementary school",
    3: "Some high school", 4: "High school graduate / GED",
    5: "Some college / technical school", 6: "College graduate",
}
INCOME = {
    1: "Under $10,000", 2: "$10,000–$14,999", 3: "$15,000–$19,999",
    4: "$20,000–$24,999", 5: "$25,000–$34,999", 6: "$35,000–$49,999",
    7: "$50,000–$74,999", 8: "$75,000 or more",
}
GENERAL_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
BINARY_QUESTIONS = {
    "HighBP": ("High blood pressure", "Ever diagnosed with high blood pressure"),
    "HighChol": ("High cholesterol", "Ever diagnosed with high cholesterol"),
    "CholCheck": ("Cholesterol check", "Cholesterol checked in the past five years"),
    "Smoker": ("Smoking history", "Smoked at least 100 cigarettes in a lifetime"),
    "Stroke": ("Previous stroke", "History of stroke"),
    "HeartDiseaseorAttack": ("Heart disease / heart attack", "History of coronary heart disease or heart attack"),
    "PhysActivity": ("Physical activity", "Physical activity in the past 30 days, outside work"),
    "Fruits": ("Daily fruit", "Fruit at least once a day"),
    "Veggies": ("Daily vegetables", "Vegetables at least once a day"),
    "HvyAlcoholConsump": ("Heavy alcohol use", "Heavy-drinking survey indicator"),
    "AnyHealthcare": ("Healthcare coverage", "Any health insurance or healthcare plan"),
    "NoDocbcCost": ("Doctor visit blocked by cost", "Needed but could not afford a doctor in the past year"),
    "DiffWalk": ("Difficulty walking", "Serious difficulty walking or climbing stairs"),
}


# Plain-English feature descriptions for the patient-facing explanation.
# SHAP directions describe a MODEL score shift, not a medical causal effect.
FEATURE_TITLES = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol",
    "CholCheck": "Recent cholesterol check", "BMI": "Body mass index (BMI)",
    "Smoker": "Smoking history", "Stroke": "Previous stroke",
    "HeartDiseaseorAttack": "History of heart disease",
    "PhysActivity": "Physical activity", "Fruits": "Daily fruit intake",
    "Veggies": "Daily vegetable intake", "HvyAlcoholConsump": "Heavy alcohol use",
    "AnyHealthcare": "Healthcare coverage", "NoDocbcCost": "Unable to see a doctor due to cost",
    "GenHlth": "Overall self-rated health", "MentHlth": "Poor mental-health days",
    "PhysHlth": "Poor physical-health days", "DiffWalk": "Difficulty walking",
    "Sex": "Sex (survey coding)", "Age": "Age group",
    "Education": "Education level", "Income": "Household income",
}


def display_answer(name: str, value: int) -> str:
    """Turn a BRFSS feature code into a reader-friendly survey response."""
    if name in BINARY_QUESTIONS:
        return "Yes" if int(value) == 1 else "No"
    if name == "Sex":
        return "Male" if int(value) == 1 else "Female"
    if name == "Age":
        return AGE[int(value)]
    if name == "Education":
        return EDUCATION[int(value)]
    if name == "Income":
        return INCOME[int(value)]
    if name == "GenHlth":
        return GENERAL_HEALTH[int(value)]
    if name in ("MentHlth", "PhysHlth"):
        return f"{int(value)} of the past 30 days"
    return str(value)


def yes_no_input(name: str, sample: dict, inputs: dict) -> None:
    label, explanation = BINARY_QUESTIONS[name]
    inputs[name] = st.selectbox(
        label, options=[0, 1], index=int(sample[name]),
        format_func=lambda code: "Yes" if code == 1 else "No",
        help=explanation, key=f"feature_{name}",
    )


def category_input(name: str, label: str, categories: dict, sample: dict, inputs: dict) -> None:
    keys = list(categories)
    inputs[name] = st.selectbox(
        label, options=keys, index=keys.index(int(sample[name])),
        format_func=lambda code: f"{code} · {categories[code]}",
        key=f"feature_{name}",
    )


st.title("Diabetes Risk Screening")
st.caption("Data Mining · Group 9 · 2015 BRFSS dataset · Research demonstration")
st.info(
    "This research tool generates a **model-estimated risk score** for the survey's "
    "diabetes/prediabetes category, to help illustrate limited-capacity screening. "
    "The score is **not a clinically validated probability of having diabetes**, "
    "and this app does not diagnose disease or replace clinical assessment."
)

try:
    model, schema, metadata, reference, capacity_info, sample, explainer = load_bundle()
    calibration_report, calibration_bins = load_calibration_evaluation()
except Exception as err:
    st.error(f"Could not load the model bundle: {err}")
    st.stop()

features = schema["feature_order"]

screening_mode = st.radio(
    "Choose a screening workflow",
    ["Single patient", "Batch upload (Excel/CSV)"],
    horizontal=True,
    help="Use individual screening for one questionnaire; upload an Excel/CSV cohort for Top-K selection.",
)

if screening_mode == "Single patient":
    st.subheader("1 · Health questionnaire")
    st.caption(
        "Fields start with the supplied `sample_input.csv` example. "
        "Click Predict immediately to reproduce the team's sample result (1.03%)."
    )
    inputs = {}
    with st.form("patient_form"):
        with st.expander("A. General health & demographics", expanded=True):
            left, right = st.columns(2)
            with left:
                inputs["Sex"] = st.selectbox(
                    "Sex (dataset coding)", [0, 1], index=int(sample["Sex"]),
                    format_func=lambda code: "Female" if code == 0 else "Male",
                    key="feature_Sex",
                )
                category_input("Age", "Age group", AGE, sample, inputs)
                category_input("Education", "Education level", EDUCATION, sample, inputs)
                category_input("Income", "Annual household income (original USD bands)", INCOME, sample, inputs)
            with right:
                category_input("GenHlth", "Self-reported general health", GENERAL_HEALTH, sample, inputs)
                inputs["BMI"] = st.number_input(
                    "Body Mass Index (BMI)", min_value=12, max_value=98,
                    value=int(sample["BMI"]), step=1, key="feature_BMI",
                )
                inputs["MentHlth"] = st.number_input(
                    "Poor mental-health days (last 30)", min_value=0, max_value=30,
                    value=int(sample["MentHlth"]), step=1, key="feature_MentHlth",
                )
                inputs["PhysHlth"] = st.number_input(
                    "Poor physical-health days (last 30)", min_value=0, max_value=30,
                    value=int(sample["PhysHlth"]), step=1, key="feature_PhysHlth",
                )

        with st.expander("B. Health history & healthcare access", expanded=True):
            left, right = st.columns(2)
            with left:
                for f in ("HighBP", "HighChol", "CholCheck", "Stroke"):
                    yes_no_input(f, sample, inputs)
            with right:
                for f in ("HeartDiseaseorAttack", "DiffWalk", "AnyHealthcare", "NoDocbcCost"):
                    yes_no_input(f, sample, inputs)

        with st.expander("C. Lifestyle", expanded=True):
            left, right = st.columns(2)
            with left:
                for f in ("Smoker", "PhysActivity", "Fruits"):
                    yes_no_input(f, sample, inputs)
            with right:
                for f in ("Veggies", "HvyAlcoholConsump"):
                    yes_no_input(f, sample, inputs)

        submitted = st.form_submit_button("See my screening result", type="primary", use_container_width=True)

    if submitted:
        try:
            numeric_inputs = {name: int(inputs[name]) for name in features}
            for name in features:
                spec = schema["features"][name]
                if not spec["min"] <= numeric_inputs[name] <= spec["max"]:
                    raise ValueError(f"{name} is outside the supported range.")
            X = pd.DataFrame([numeric_inputs], columns=features)
            probability = float(model.predict_proba(X)[0, 1])
            percentile = estimate_percentile(probability, reference)
            priority, _ = demo_priority(percentile)

            shap_values = np.asarray(explainer.shap_values(X))[0]
            if len(shap_values) != len(features):
                raise ValueError("Unexpected SHAP output length.")
            top_indices = np.argsort(-np.abs(shap_values))[:3]
            top_shap = [(features[i], float(shap_values[i])) for i in top_indices]

            prediction_id = None
            logging_error = None
            logging_success = False
            if LOGGING_CONNECTED:
                try:
                    logging_success, returned_id, logging_error = log_prediction(
                        numeric_inputs, probability, percentile, top_shap, metadata["model_version"]
                    )
                    if logging_success:
                        prediction_id = returned_id
                    else:
                        logging_error = logging_error or "Unknown Google Sheets error"
                except Exception as err:
                    logging_error = str(err)

            st.session_state["last_prediction"] = {
                "probability": probability,
                "percentile": percentile,
                "priority": priority,
                "top_shap": top_shap,
                "input_values": numeric_inputs,
                "prediction_id": prediction_id,
                "logging_error": logging_error,
                "logging_success": logging_success,
            }
        except Exception as err:
            st.error(f"Prediction failed: {err}")

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        st.divider()
        st.header("Your screening result")
        st.caption("Based on the answers entered when you last clicked the prediction button.")

        # 1. The XGBoost predict_proba output is shown as a model SCORE in percent form.
        # Locked-test calibration is imperfect; this is not a validated personal risk %.
        st.subheader("1 · What does your result mean?")
        st.metric("Model-estimated risk score", f"{result['probability']:.2%}")
        st.write(
            "This is the model's **score for the survey diabetes/prediabetes class**, "
            "shown on a 0–100% scale. It is mainly useful for **comparing/ranking "
            "screening priority across people**. It is **not** a clinically validated "
            "percentage chance that you have diabetes or will develop it."
        )
        st.caption(
            "The research team's locked-test calibration evaluation found imperfect "
            "agreement between predicted scores and observed rates, particularly at "
            "higher scores. See 'For researchers' for the calibration evidence."
        )
        if result["probability"] < float(reference["p50"]):
            st.caption(
                "For context: this model score is below the median score in the supplied "
                "reference distribution. A lower score does not rule out diabetes."
            )
        else:
            st.caption(
                "For context: this model score is at or above the median score in the "
                "supplied reference distribution. This alone cannot confirm diabetes."
            )

        # 2. Translate the exact same SHAP top 3 into readable survey features.
        st.subheader("2 · Why did the model give this result?")
        st.write(
            "These are the **three answers that influenced this prediction most** "
            "according to SHAP. A factor can push the model score up or down "
            "relative to its baseline."
        )
        top = result["top_shap"]
        for feature, shap_value in top:
            name = FEATURE_TITLES.get(feature, feature)
            response = display_answer(feature, result["input_values"][feature])
            direction = "Increased the model score ↑" if shap_value > 0 else (
                "Decreased the model score ↓" if shap_value < 0 else "No estimated change"
            )
            st.markdown(f"**{name}** — {response}  \n{direction}")

        shap_table = pd.DataFrame(
            {"Factor": [FEATURE_TITLES.get(f, f) for f, _ in top],
             "Contribution": [v for _, v in top]}
        ).set_index("Factor")
        st.bar_chart(shap_table, horizontal=True)
        st.caption(
            "SHAP explains the model's calculation, not what caused a person's health condition. "
            "The bar values use a statistical log-odds scale — they are **not percentages**."
        )

        # 3. Non-diagnostic general next steps, without declaring a medical urgency tier.
        st.subheader("3 · What can you do next?")
        st.markdown(
            """- **Use this as a screening discussion starter**, not proof that you do or do not have diabetes.
    - **Ask a healthcare professional whether you need blood sugar testing** (such as HbA1c or fasting plasma glucose), especially if you have symptoms or known risk factors. Even a lower model score does not replace recommended screening.
    - **Focus on sustainable habits** such as regular physical activity, balanced eating, and avoiding smoking. These are general health suggestions, not a treatment plan based on this score."""
        )
        st.warning(
            "Class project only. Do not use this app to make diagnosis or treatment decisions "
            "or upload identifiable patient records."
        )

        # Leave research metrics available for the group report and the assessor.
        with st.expander("For researchers: calibration, SHAP and capacity metrics"):
            st.markdown("**Full research-oriented result**")
            c1, c2 = st.columns(2)
            c1.metric("Estimated model-score percentile", f"≈ {result['percentile']:.1f} / 100")
            c2.metric("Demo priority group", result["priority"])
            st.caption(
                "Percentile is estimated by interpolation from sparse reference quantiles "
                "(especially rough below the median). Priority labels are demo conventions, "
                "not validated medical risk categories."
            )
            st.caption(
                "No K% is needed for an individual prediction. Accurate capacity-based "
                "selection is available in Batch upload, where all records are ranked together."
            )
            st.dataframe(
                pd.DataFrame(
                    [{"Feature": f, "SHAP value (log-odds)": round(v, 4)} for f, v in top]
                ), hide_index=True, use_container_width=True,
            )

            # All six capacity scenarios are provided by Hưng's capacity_config.json.
            # The file gives top-K operating SHARES, not exact patient score cutoffs.
            st.markdown("**Six research capacity scenarios (locked-test group metrics)**")
            capacity_table = pd.DataFrame(capacity_info["locked_test_results"])
            st.dataframe(
                pd.DataFrame({
                    "Top K": capacity_table["capacity_pct"],
                    "Precision@K": capacity_table["global_precision_at_k"].map(lambda x: f"{x:.2%}"),
                    "Recall@K": capacity_table["global_recall_at_k"].map(lambda x: f"{x:.2%}"),
                    "Lift@K": capacity_table["global_lift_at_k"].map(lambda x: f"{x:.2f}×"),
                }),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "These figures come from Hưng's locked test set, NOT the current patient "
                "or an uploaded file. K% is applied to an actual ranked patient list in Batch upload."
            )

            st.divider()
            st.markdown("**Calibration on the locked test set**")
            a, b, c = st.columns(3)
            a.metric("Model Brier score", f"{calibration_report['brier_score']:.4f}")
            b.metric("Naive baseline Brier", f"{calibration_report['naive_brier_score']:.4f}")
            c.metric("Locked-test observations", f"{calibration_report['n_test']:,}")
            st.caption(
                "Lower Brier scores are better. The model has a lower Brier score "
                "than the naive baseline, but this does NOT mean its probabilities "
                "are perfectly calibrated."
            )
            st.image(
                str(BUNDLE / "calibration_curve.png"),
                caption="Team-supplied calibration data: 10 quantile bins, global XGBoost vs ideal calibration (45° line).",
                use_container_width=True,
            )
            st.write(
                "Across these locked-test bins, predicted scores are **higher** than "
                "the observed positive-class rates; the gap is larger in the "
                "highest-score bins. The chart describes **groups**, not an individual's "
                "true disease probability."
            )
            st.caption(
                f"Source: calibration_report.json + calibration_table.csv · "
                f"{calibration_report['n_calibration_bins']} quantile bins · "
                f"Brier skill score {calibration_report['brier_skill_score']:.3f} · "
                "Model was evaluated, NOT recalibrated."
            )

        if result["logging_error"]:
            st.error(f"Prediction succeeded, but the result was NOT saved to Google Sheets: {result['logging_error']}")
        elif result["prediction_id"] is not None:
            st.success(f"Logged successfully · prediction_id: {result['prediction_id']}")
        elif not LOGGING_CONNECTED:
            st.caption("Demo logging not connected yet (Việt Anh's logger.py); no prediction was saved.")

else:
    st.subheader("Batch screening · Excel/CSV cohort upload")
    st.write(
        "Upload an **anonymous** list of patients. The app scores every valid row, "
        "ranks the cohort from highest to lowest model score, and selects as many "
        "patients as the hospital can handle. The app calculates the effective top-K% automatically."
    )
    st.warning(
        "Classroom demonstration only: do not upload real patient names, IDs, contact "
        "details or confidential medical records. Use anonymous/synthetic survey data."
    )
    st.download_button(
        "Download example Excel (20 synthetic records)",
        data=(BUNDLE / "batch_example_20.xlsx").read_bytes(),
        file_name="group9_batch_example_20.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    with st.form("batch_screening_form"):
        upload = st.file_uploader(
            "Upload Excel (.xlsx) or CSV with 21 BRFSS features", type=["xlsx", "csv"],
            help="Use the example CSV header names. Extra columns are ignored. The first Excel sheet is read.",
        )
        hospital_capacity = st.number_input(
            "Hospital follow-up capacity (number of patients)",
            min_value=0, value=2, step=1,
            help="Enter how many patients the hospital can follow up. "
                 "The app calculates K% from the valid uploaded rows and picks exactly "
                 "that many top-ranked records (or everyone if capacity exceeds the cohort).",
        )
        run_batch = st.form_submit_button("Screen and rank patients", type="primary", use_container_width=True)

    if run_batch:
        st.session_state.pop("batch_screening_result", None)
        if upload is None:
            st.error("Please upload an Excel or CSV file first.")
        else:
            try:
                file_bytes = upload.getvalue()
                if len(file_bytes) > MAX_UPLOAD_BYTES:
                    raise ValueError("File exceeds the 10 MB limit for this classroom demo.")
                if upload.name.lower().endswith(".xlsx"):
                    raw = pd.read_excel(BytesIO(file_bytes), engine="openpyxl", sheet_name=0)
                else:
                    raw = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
                ignored = [c for c in raw.columns if c not in features]
                clean = validate_cohort(raw, schema)
                ranked = rank_and_select(clean, model, features, int(hospital_capacity))
                st.session_state["batch_screening_result"] = {
                    "ranked": ranked,
                    "hospital_capacity": int(hospital_capacity),
                    "ignored": ignored,
                }
            except Exception as err:
                st.error(f"Batch screening failed: {err}")

    if "batch_screening_result" in st.session_state:
        batch = st.session_state["batch_screening_result"]
        ranked = batch["ranked"]
        hospital_capacity = batch["hospital_capacity"]
        selected = ranked[ranked["Selected for follow-up"] == "Yes"]
        k = len(selected) / len(ranked)
        st.divider()
        st.subheader("Cohort screening results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records screened", f"{len(ranked):,}")
        m2.metric("Hospital capacity", f"{hospital_capacity:,} patients")
        m3.metric("Calculated Top-K", f"{k:.2%}")
        m4.metric("Selected for follow-up", f"{len(selected):,}")
        st.caption(
            f"Selected the {len(selected)} highest-ranked records from {len(ranked)} "
            "valid patients. For equal model scores, the earlier spreadsheet row "
            "comes first. Model scores are not clinically validated probabilities."
        )
        if hospital_capacity > len(ranked):
            st.info(
                f"Capacity ({hospital_capacity:,}) exceeds the uploaded cohort "
                f"({len(ranked):,}), so all patients are selected (effective Top-100%)."
            )
        elif hospital_capacity == 0:
            st.info("Hospital capacity is 0, so the app ranks every record but selects no one for follow-up.")
        if batch["ignored"]:
            st.info(
                "Extra columns were ignored for privacy and prediction: "
                + ", ".join(map(str, batch["ignored"][:8]))
                + (" …" if len(batch["ignored"]) > 8 else "")
            )
        st.dataframe(
            ranked[["Source row", "Rank", "Model score (%)", "Selected for follow-up"]],
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "Download ranked Excel results (.xlsx)",
            data=create_excel_results(ranked, hospital_capacity, metadata["model_version"]),
            file_name="group9_top_k_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        capacity_row = next(
            (item for item in capacity_info["locked_test_results"]
             if abs(float(item["capacity"]) - k) < 1e-9), None
        )
        if capacity_row is not None:
            st.caption(
                f"For context only: on the separate locked test set at Top {k * 100:g}%, "
                f"Precision@K = {capacity_row['global_precision_at_k']:.2%}, "
                f"Recall@K = {capacity_row['global_recall_at_k']:.2%}. "
                "These are historical group metrics, not measured performance on this upload."
            )
        else:
            st.caption(
                "Historical locked-test metrics are available only for the six evaluated "
                "K% scenarios; none exactly matches this cohort's calculated capacity. "
                "The app still ranks and selects the correct number of patients."
            )
        st.info(
            "Batch results are available for download; this workflow does not automatically "
            "send uploaded cohorts to Google Sheets or retrain the model."
        )

with st.expander("Research notes & model information"):
    st.markdown(
        """
        - **Target:** `Diabetes_binary` in the group's BRFSS 2015 dataset;
          the positive class combines survey diabetes/prediabetes status.
        - **Priority tiers (demo only):** Very high = estimated top 2.5%;
          High = top 10%; Moderate = top 20%; Standard = outside top 20%.
        - **Capacity selection:** the single-person form does not require K%.
          For Batch upload, the hospital enters an absolute follow-up capacity
          (number of patients). The app computes its effective percentage from
          the validated cohort and selects exactly that many top-ranked records,
          capped by the number of uploaded patients. All capacities use the SAME
          global XGBoost model; they are not different trained models.
        - **Privacy:** do not enter real, identifiable patient records in this
          classroom prototype or a publicly hosted demonstration.
        - **Calibration:** the locked-test Brier evaluation and calibration curve
          show imperfect probability calibration; the score is mainly for
          *ranking and top-K screening*, not validated clinical probability.
        - **Study scope:** trained on BRFSS 2015; the team plans to examine
          available newer data (including 2024) in future checks, not a completed
          validation shown here. The reason for choosing 2015 must be explained
          and verified in the written report.
        - **Continuous learning:** collecting outcomes and conditionally
          retraining/evaluating a candidate model are proposed next steps;
          retraining is NOT running automatically inside this app.
        - **Clinical safety:** this app does not replace testing or a clinician.
        """
    )
    st.caption(f"Model: {metadata['model_name']} · Version {metadata['model_version']}")
