"""Fast offline tests for retrain_pipeline.py logic (no dataset, no training, < 5 s)."""
import csv
import json
import tempfile
from pathlib import Path

import pandas as pd

import retrain_pipeline as rp

BUNDLE = Path(__file__).resolve().parent / "deployment_bundle"
schema = json.loads((BUNDLE / "feature_schema.json").read_text(encoding="utf-8"))
cfg = json.loads((BUNDLE / "retraining_config.json").read_text(encoding="utf-8"))
with (BUNDLE / "prediction_logging_schema.csv").open(encoding="utf-8-sig", newline="") as fh:
    COLUMNS = [r["column"] for r in csv.DictReader(fh)]
SAMPLE = pd.read_csv(BUNDLE / "sample_input.csv").iloc[0].to_dict()


def log_row(i, outcome="0", **changes):
    row = {c: "" for c in COLUMNS}
    row.update({k: str(int(v)) for k, v in SAMPLE.items()})
    row.update({"prediction_id": f"00000000-0000-4000-8000-{i:012d}", "timestamp": "2026-11-01T10:00:00+00:00",
                "model_version": "1.0.0", "predicted_probability": "0,0103",  # vi-VN decimal comma
                "actual_diabetes_outcome": outcome, "outcome_recorded_at": "2026-11-20T10:00:00+00:00"})
    row.update(changes)
    return row


rows = [log_row(i, outcome=str(i % 2)) for i in range(1, 11)]
rows += [
    log_row(11, outcome=""),                          # pending: not rejected, not used
    log_row(12, BMI="150"),                           # out of range
    log_row(13, outcome="yes"),                       # bad label
    log_row(14, prediction_id="abc"),                 # bad id
    log_row(15, outcome_recorded_at="2026-01-01T00:00:00+00:00"),  # outcome before prediction
    log_row(16, GenHlth="2.5"),                       # not an integer code
    log_row(1, outcome="0"),                          # id 1 again, other outcome -> both copies rejected
    log_row(2, outcome="0"),                          # id 2 again, same outcome -> copy dropped
]

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "export.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    clean, rejected, summary = rp.load_new_labeled([path], schema)

reasons = " | ".join(rejected["reason"])
assert summary["pending_unlabeled"] == 1
for text in ("BMI", "must be 0 or 1", "invalid prediction_id", "before prediction", "GenHlth",
             "conflicting outcomes", "duplicate prediction_id"):
    assert text in reasons, f"missing rejection reason: {text}"
assert len(clean) == 9, len(clean)                    # ids 2–10 kept; id 1 rejected (conflict)
print("PASS: validation rejects bad rows with reasons, keeps pending rows out")

assert "pid:00000000-0000-4000-8000-000000000002" in set(clean["row_key"])
assert not clean["row_key"].duplicated().any()
print("PASS: duplicate prediction_id kept once, conflicting ids rejected entirely")

train_a, hold_a, _ = rp.split_new(clean, cfg)
train_b, hold_b, _ = rp.split_new(clean.sample(frac=1, random_state=1), cfg)
assert set(hold_a["row_key"]) == set(hold_b["row_key"])
print("PASS: holdout assignment is stable regardless of row order")

assert rp.top_k(50736, 0.005) == 254 and rp.top_k(1000, 0.1) == 100
assert rp.bump_minor("1.0.0") == "1.1.0" and rp.bump_minor("1.9.0") == "1.10.0"
print("PASS: top-K rule matches capacity_config.json; version bump")

prod = {"locked_test": {"pr_auc": 0.43, "brier": 0.098,
                        "capacity": [{"capacity_pct": "1%", "precision_at_k": 0.66}]}}
cand = {"locked_test": {"pr_auc": 0.42, "brier": 0.098,
                        "capacity": [{"capacity_pct": "1%", "precision_at_k": 0.66}]}}
checks = rp.promotion_checks(prod, cand, cfg["promotion_criteria"])
assert not all(c["passed"] for c in checks)          # worse PR-AUC must block promotion
print("PASS: a candidate with lower PR-AUC is not promotable")
