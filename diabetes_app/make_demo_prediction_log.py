"""Build a SIMULATED 'Predictions' Sheet export to exercise retrain_pipeline.py.

The rows are resampled from the historical TRAIN split (never the locked test), so a
candidate trained on them shows how the pipeline behaves, NOT a real model improvement.
Never use this file for an actual model promotion or as a reported result.

Usage: python make_demo_prediction_log.py [--rows 3000] [--out data/demo_prediction_log.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "deployment_bundle"


def build(historical: Path, rows: int, pending: int, seed: int) -> pd.DataFrame:
    schema = json.loads((BUNDLE / "feature_schema.json").read_text(encoding="utf-8"))
    split = json.loads((BUNDLE / "training_params.json").read_text(encoding="utf-8"))["historical_split"]
    features = schema["feature_order"]
    with (BUNDLE / "prediction_logging_schema.csv").open(encoding="utf-8-sig", newline="") as fh:
        columns = [r["column"] for r in csv.DictReader(fh)]

    df = pd.read_csv(historical)
    X, y = df[features].astype(int), df["Diabetes_binary"].astype(int)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=split["test_size"], stratify=y,
                                        random_state=split["random_state"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=rows + pending, replace=False)
    sample = X_tr.iloc[idx].reset_index(drop=True)
    labels = y_tr.iloc[idx].reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(str(BUNDLE / "diabetes_global_xgboost.ubj"))
    prob = model.predict_proba(sample)[:, 1]
    start = datetime(2026, 10, 30, tzinfo=timezone.utc)
    out = pd.DataFrame({c: "" for c in columns}, index=range(len(sample)))
    out["prediction_id"] = [str(UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63)), version=4))
                            for _ in range(len(sample))]
    stamps = [start + timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 30, len(sample))]
    out["timestamp"] = [t.isoformat(timespec="seconds") for t in stamps]
    out["model_version"] = "1.0.0"
    for f in features:
        out[f] = sample[f].astype(str)
    out["predicted_probability"] = prob
    out["risk_percentile"] = 50.0
    for i, f in enumerate(("Age", "HighBP", "GenHlth"), 1):
        out[f"top_feature_{i}"], out[f"top_feature_{i}_shap"] = f, 0.1
    out["actual_diabetes_outcome"] = labels.astype(str)
    out["outcome_recorded_at"] = [(t + timedelta(days=int(d))).isoformat(timespec="seconds")
                                  for t, d in zip(stamps, rng.integers(7, 60, len(sample)))]
    out.loc[rows:, ["actual_diabetes_outcome", "outcome_recorded_at"]] = ""  # still waiting for outcomes

    # A few deliberately broken rows: the pipeline must reject each with a reason.
    bad = out.iloc[:6].copy()
    bad.iloc[0, bad.columns.get_loc("BMI")] = "150"                      # out of range
    bad.iloc[1, bad.columns.get_loc("actual_diabetes_outcome")] = "yes"  # not 0/1
    bad.iloc[2, bad.columns.get_loc("prediction_id")] = "not-a-uuid"
    flipped = "1" if out.loc[3, "actual_diabetes_outcome"] == "0" else "0"
    bad.iloc[3, bad.columns.get_loc("actual_diabetes_outcome")] = flipped  # conflicts with row 3
    bad.iloc[4, bad.columns.get_loc("outcome_recorded_at")] = "2026-01-01T00:00:00+00:00"  # before prediction
    # bad.iloc[5] is an exact repeat of row 5 (same id, same outcome) -> dropped as duplicate
    return pd.concat([out, bad], ignore_index=True)[columns]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--historical", type=Path, default=ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv")
    ap.add_argument("--rows", type=int, default=3000, help="labeled rows")
    ap.add_argument("--pending", type=int, default=200, help="rows still waiting for an outcome")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo_prediction_log_SIMULATED.csv")
    args = ap.parse_args()
    frame = build(args.historical, args.rows, args.pending, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} SIMULATED rows to {args.out}")


if __name__ == "__main__":
    main()
