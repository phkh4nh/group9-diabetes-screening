"""Google Apps Script logging for the Group 9 classroom demo (no billing/key JSON).

Contract: log_prediction(inputs, probability, risk_percentile, top_shap, model_version)
          -> (success: bool, prediction_id: str, error: str | None)
Do not send real or identifiable patient data through this classroom demonstration.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

BUNDLE = Path(__file__).resolve().parent / "deployment_bundle"


def _schema_columns():
    with (BUNDLE / "prediction_logging_schema.csv").open(newline="", encoding="utf-8-sig") as fh:
        columns = [line["column"].strip() for line in csv.DictReader(fh)]
    if len(columns) != 34 or len(set(columns)) != 34:
        raise ValueError("The logging schema must have exactly 34 unique columns.")
    return columns


def _prepare_row(inputs, probability, risk_percentile, top_shap, model_version, prediction_id):
    columns = _schema_columns()
    features = columns[3:24]
    if not isinstance(inputs, dict) or set(inputs) != set(features):
        raise ValueError("Inputs must contain exactly the 21 features in feature_schema.json.")
    if len(top_shap) != 3 or len({str(f) for f, _ in top_shap}) != 3:
        raise ValueError("Expected exactly 3 different SHAP features.")
    prob = float(probability)
    pct = float(risk_percentile)
    if not (math.isfinite(prob) and 0 <= prob <= 1 and math.isfinite(pct) and 0 <= pct <= 100):
        raise ValueError("Probability or percentile is out of range.")
    schema = __import__('json').loads((BUNDLE / "feature_schema.json").read_text(encoding="utf-8"))
    clean_inputs = {}
    for name in features:
        val = float(inputs[name])
        rule = schema['features'][name]
        if not math.isfinite(val) or not (rule['min'] <= val <= rule['max']):
            raise ValueError(f"Invalid value for {name}")
        if rule['type'] != 'numeric' and not val.is_integer():
            raise ValueError(f"{name} must be an integer value")
        clean_inputs[name] = int(val) if val.is_integer() else val
    ordered = sorted(top_shap, key=lambda item: abs(float(item[1])), reverse=True)
    data = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": str(model_version),
        **clean_inputs,
        "predicted_probability": prob,
        "risk_percentile": pct,
        "actual_diabetes_outcome": "",
        "outcome_recorded_at": "",
    }
    for index, (feature, value) in enumerate(ordered, 1):
        if feature not in clean_inputs or not math.isfinite(float(value)):
            raise ValueError("Invalid SHAP contribution")
        data[f"top_feature_{index}"] = str(feature)
        data[f"top_feature_{index}_shap"] = float(value)
    if set(data) != set(columns):
        raise ValueError("The row does not match the 34-column logging schema")
    return [data[col] for col in columns]


def log_prediction(inputs, probability, risk_percentile, top_shap, model_version):
    """Send one new record. If saving fails, keep the prediction visible in Streamlit."""
    prediction_id = str(uuid4())
    try:
        row = _prepare_row(inputs, probability, risk_percentile, top_shap,
                           model_version, prediction_id)
        url = str(st.secrets["apps_script_url"]).strip()
        token = str(st.secrets["logging_token"]).strip()
        if not (url.startswith("https://script.google.com/macros/s/") and url.endswith("/exec")):
            raise ValueError("apps_script_url must be the deployed Apps Script /exec URL")
        if len(token) < 24:
            raise ValueError("logging_token must be a long random secret (24+ characters)")
        response = requests.post(url, json={"token": token, "row": row},
                                 timeout=30, allow_redirects=True)
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("Apps Script returned non-JSON. Check deployment Access and /exec URL.") from exc
        if not isinstance(result, dict) or result.get("ok") is not True:
            error = result.get("error", "Unknown Apps Script error") if isinstance(result, dict) else "Invalid response"
            raise RuntimeError(f"Apps Script: {error}")
        if result.get("prediction_id") != prediction_id:
            raise RuntimeError("Apps Script returned an unexpected prediction_id")
        return True, prediction_id, None
    except Exception as exc:
        # Do not report that an uncertain/failed remote write was saved.
        return False, prediction_id, f"{type(exc).__name__}: {exc}"
