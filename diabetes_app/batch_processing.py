"""Cohort-level (top-K) screening simulation for Group 9's Streamlit app.

No clinical decisions, automatic retraining, or Google Sheets batch logging here.
Only anonymous numeric BRFSS features are accepted.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

MAX_BATCH_ROWS = 5000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def validate_cohort(raw: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Check feature columns, ranges and integer encodings, retaining source row numbers."""
    features = schema["feature_order"]
    if raw is None or raw.empty:
        raise ValueError("The uploaded file is empty. Add at least one row with 21 feature values.")
    missing = [f for f in features if f not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Blank spreadsheet rows are harmless; don't silently drop partially blank records.
    subset = raw[features].dropna(how="all").copy()
    if subset.empty:
        raise ValueError("There are no populated patient rows in the file.")
    if len(subset) > MAX_BATCH_ROWS:
        raise ValueError(f"This demo accepts at most {MAX_BATCH_ROWS:,} records per file.")

    issues = []
    clean = pd.DataFrame(index=subset.index)
    for name in features:
        spec = schema["features"][name]
        numeric = pd.to_numeric(subset[name], errors="coerce")
        numeric_arr = numeric.to_numpy(dtype=float)
        valid = (
            np.isfinite(numeric_arr)
            & (numeric_arr >= float(spec["min"]))
            & (numeric_arr <= float(spec["max"]))
        )
        if name != "BMI":
            valid &= numeric_arr == np.floor(numeric_arr)
        bad_rows = subset.index[~valid].tolist()
        if bad_rows:
            display_rows = ", ".join(str(i + 2) for i in bad_rows[:5])
            issues.append(f"{name}: invalid or missing value at spreadsheet row(s) {display_rows}")
        clean[name] = numeric

    if issues:
        summary = "; ".join(issues[:8])
        if len(issues) > 8:
            summary += f"; and {len(issues) - 8} more affected columns"
        raise ValueError("Please correct the data before screening. " + summary)

    clean = clean[features]
    clean.insert(0, "Source row", clean.index.to_numpy() + 2)
    return clean.reset_index(drop=True)


def rank_and_select(clean: pd.DataFrame, model, features: list[str], capacity_count: int) -> pd.DataFrame:
    """Rank the cohort and select exactly min(hospital capacity, cohort size) rows.

    Hospital capacity is an absolute number of patients, not a percentage.
    Zero capacity is supported and selects no patients while keeping all scores.
    """
    if isinstance(capacity_count, (bool, np.bool_)) or not isinstance(capacity_count, (int, np.integer)):
        raise ValueError("Hospital capacity must be a whole number of patients.")
    if capacity_count < 0:
        raise ValueError("Hospital capacity cannot be negative.")
    if clean.empty:
        raise ValueError("The cohort must contain at least one valid patient.")
    feature_frame = clean[features]
    probs = np.asarray(model.predict_proba(feature_frame), dtype=float)
    if probs.shape != (len(clean), 2):
        raise ValueError("Unexpected model prediction shape.")
    scores = probs[:, 1]
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("The model returned invalid scores.")
    n_selected = min(len(clean), int(capacity_count))

    result = clean.copy()
    result["_raw_model_score"] = scores
    result = result.sort_values(
        ["_raw_model_score", "Source row"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    result.insert(1, "Model score (%)", np.round(result["_raw_model_score"].to_numpy() * 100, 4))
    result.drop(columns=["_raw_model_score"], inplace=True)
    result.insert(1, "Rank", np.arange(1, len(result) + 1))
    result.insert(3, "Selected for follow-up", np.where(result["Rank"] <= n_selected, "Yes", "No"))
    return result


def create_excel_results(result: pd.DataFrame, capacity_count: int, model_version: str) -> bytes:
    """Export scores, selections and the absolute hospital capacity to Excel."""
    selected = result[result["Selected for follow-up"] == "Yes"]
    effective_count = len(selected)
    effective_share = effective_count / len(result) if len(result) else 0.0
    notes = pd.DataFrame([
        ("Purpose", "Classroom cohort-level capacity simulation; not clinical triage"),
        ("Selection rule", "Highest-ranked min(requested hospital capacity, uploaded patients) records"),
        ("Requested hospital capacity (patients)", int(capacity_count)),
        ("Uploaded records", len(result)),
        ("Effective follow-up capacity (patients)", effective_count),
        ("Calculated top-K share", f"{effective_share:.4%}"),
        ("Selected records", len(selected)),
        ("Model version", str(model_version)),
        ("Score meaning", "Uncalibrated model-estimated class score; NOT a validated diabetes probability"),
        ("Tie rule", "Same score: earlier source row first"),
        ("Actual outcomes", "Not collected by this batch screening demonstration"),
        ("Model selection", "Same global XGBoost model for all hospital capacity values"),
    ], columns=["Item", "Details"])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Ranked cohort", index=False)
        selected.to_excel(writer, sheet_name="Selected top K", index=False)
        notes.to_excel(writer, sheet_name="Methodology", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions if sheet.title != "Methodology" else None
            sheet.column_dimensions["A"].width = 22
            if sheet.title == "Methodology":
                sheet.column_dimensions["B"].width = 88
    return buffer.getvalue()
