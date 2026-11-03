#!/usr/bin/env python3
"""Group 9 — offline retraining pipeline for the global XGBoost screening model.

Implements deployment_bundle/retraining_config.json step by step:

  1. Load the historical BRFSS 2015 data + newly labeled records
     (Google Sheet "Predictions" CSV export and/or a recoded dataset CSV).
  2. Validate schema, ranges, integer codes and outcome labels. Rejected rows are
     written to rejected_rows.csv with a reason; unlabeled log rows are skipped.
  3. Check the retraining policy: 30-day schedule + minimum labeled outcomes.
  4. Re-create the locked test set exactly (80/20 stratified, random_state=42) and
     split the new data into new-train / new-holdout (stratified).
  5. Check that the pipeline reproduces the production model on the original data.
  6. Retrain a candidate with the production hyperparameters (training_params.json).
  7. Evaluate candidate vs production: PR-AUC (primary), ROC-AUC, Brier and
     Precision@K / Recall@K / Lift@K at the six supported capacities.
  8. Apply the promotion criteria. A candidate bundle is always written to
     model_registry/candidates/. deployment_bundle/ is replaced ONLY when all
     criteria pass AND --promote is given; the old bundle is archived first.

The Streamlit app never retrains. Run this script manually or from a scheduler.

Exit codes: 0 = candidate ready / promoted, 2 = skipped by policy,
            3 = promotion criteria failed (production kept), 1 = error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "deployment_bundle"
REGISTRY = ROOT / "model_registry"
RUNS = ROOT / "retraining_runs"
DEFAULT_HISTORICAL = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"
LOG_OUTCOME = "actual_diabetes_outcome"
MODEL_FILE = "diabetes_global_xgboost.ubj"
UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
REPRO_TOLERANCE = 1e-6

# Rebuilt for every candidate. Every other file is copied unchanged from production.
REGENERATED = {
    MODEL_FILE, "portable_model_checksum.txt", "model_metadata.json", "risk_reference.json",
    "capacity_config.json", "calibration_report.json", "calibration_table.csv",
    "calibration_curve.png", "sample_output.json", "training_params.json", "new_data_manifest.json",
}
# Describe v1.0.0 only and are not loaded by the app; kept in the archive on promotion.
PROVENANCE_ONLY = {"diabetes_global_xgboost.pkl", "model_checksum.txt"}


class PipelineError(Exception):
    """A problem that must stop the run before anything is written to production."""


# --------------------------------------------------------------------------- helpers
def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def bump_minor(version: str) -> str:
    parts = str(version).split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise PipelineError(f"model_version must look like MAJOR.MINOR.PATCH, got {version!r}")
    major, minor, _ = (int(p) for p in parts)
    return f"{major}.{minor + 1}.0"


def load_production_model(bundle: Path) -> xgb.XGBClassifier:
    model_path = bundle / MODEL_FILE
    expected = (bundle / "portable_model_checksum.txt").read_text(encoding="utf-8").strip()
    if sha256_file(model_path) != expected:
        raise PipelineError("Production model checksum mismatch: deployment_bundle was modified.")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def scores(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


# ------------------------------------------------------------------ step 1–2: data
def _feature_problems(raw: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric feature frame + per-row reason text ('' = valid).

    Every feature must be an integer code inside feature_schema.json's range: the
    app collects integers and the production model was trained on integer BRFSS codes.
    """
    reasons = pd.Series("", index=raw.index, dtype=object)
    values = pd.DataFrame(index=raw.index)
    for name in schema["feature_order"]:
        rule = schema["features"][name]
        col = raw[name].astype(str).str.strip().str.replace(",", ".", regex=False)
        num = pd.to_numeric(col, errors="coerce")
        arr = num.to_numpy(dtype=float)
        ok = np.isfinite(arr)
        ok[ok] = (arr[ok] >= rule["min"]) & (arr[ok] <= rule["max"]) & (arr[ok] == np.floor(arr[ok]))
        reasons[~ok] = reasons[~ok] + f"{name} missing/out of range [{rule['min']}, {rule['max']}]/not integer; "
        values[name] = num
    return values, reasons


def load_historical(path: Path, schema: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """The trusted training file must pass validation completely (no silent row drops)."""
    if not path.is_file():
        raise PipelineError(
            f"Historical dataset not found: {path}\n"
            "  Put the team's diabetes_binary_health_indicators_BRFSS2015.csv there, "
            "or pass --historical PATH."
        )
    raw = pd.read_csv(path)
    features = schema["feature_order"]
    missing = [c for c in [TARGET, *features] if c not in raw.columns]
    if missing:
        raise PipelineError(f"Historical dataset is missing columns: {', '.join(missing)}")
    values, reasons = _feature_problems(raw, schema)
    labels = pd.to_numeric(raw[TARGET], errors="coerce")
    bad = (reasons != "") | ~labels.isin([0, 1])
    if bad.any():
        raise PipelineError(
            f"Historical dataset failed validation in {int(bad.sum()):,} rows. "
            "Use the unmodified team file; do not edit it to add new records (use --new-data)."
        )
    info = {"file": str(path), "rows": int(len(raw)), "sha256": sha256_file(path)}
    return values.astype("int64"), labels.astype("int64"), info


def load_new_labeled(paths: list[Path], schema: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (valid labeled rows, rejected rows with reasons, summary).

    Two accepted formats:
      * prediction_log: CSV export of the Google Sheet 'Predictions' tab (34 columns).
        Rows whose actual_diabetes_outcome is still blank are pending, not rejected.
      * dataset: any CSV with the 21 features + Diabetes_binary (e.g. recoded newer BRFSS).
    """
    features = schema["feature_order"]
    kept = []
    summary = {"files": [], "pending_unlabeled": 0}
    for path in paths:
        if not path.is_file():
            raise PipelineError(f"New data file not found: {path}")
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        raw.columns = [str(c).strip() for c in raw.columns]
        if LOG_OUTCOME in raw.columns:
            fmt, label_col = "prediction_log", LOG_OUTCOME
        elif TARGET in raw.columns:
            fmt, label_col = "dataset", TARGET
        else:
            raise PipelineError(f"{path.name}: needs '{LOG_OUTCOME}' (Sheet export) or '{TARGET}' column.")
        missing = [c for c in features if c not in raw.columns]
        if fmt == "prediction_log":
            missing += [c for c in ("prediction_id", "timestamp", "outcome_recorded_at") if c not in raw.columns]
        if missing:
            raise PipelineError(f"{path.name}: missing columns {', '.join(missing)}")

        total_rows = int(len(raw))
        raw = raw.copy()
        raw["source_file"] = path.name
        raw["source_row"] = np.arange(len(raw)) + 2  # spreadsheet row number (row 1 = header)
        label_text = raw[label_col].astype(str).str.strip()
        pending = label_text == ""
        if fmt == "prediction_log":
            summary["pending_unlabeled"] += int(pending.sum())
            raw, label_text = raw[~pending], label_text[~pending]
        values, reasons = _feature_problems(raw, schema)
        label = pd.to_numeric(label_text.str.replace(",", ".", regex=False), errors="coerce")
        reasons[~label.isin([0, 1])] += f"{label_col} must be 0 or 1; "

        frame = values.copy()
        frame[TARGET] = label
        frame["source_file"] = raw["source_file"]
        frame["source_row"] = raw["source_row"]
        frame["source_format"] = fmt
        if fmt == "prediction_log":
            pid = raw["prediction_id"].astype(str).str.strip()
            reasons[~pid.str.match(UUID_PATTERN)] += "invalid prediction_id; "
            predicted_at = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True, format="ISO8601")
            outcome_at = pd.to_datetime(raw["outcome_recorded_at"], errors="coerce", utc=True, format="ISO8601")
            reasons[(outcome_at < predicted_at).fillna(False)] += "outcome recorded before prediction; "
            frame["prediction_id"] = pid.str.lower()
            frame["logged_model_version"] = raw.get("model_version", pd.Series("", index=raw.index))
        else:
            frame["prediction_id"] = ""
            frame["logged_model_version"] = ""
        frame["reason"] = reasons
        summary["files"].append({"file": str(path), "format": fmt, "rows": total_rows,
                                 "sha256": sha256_file(path)})
        kept.append(frame)

    if not kept:
        empty = pd.DataFrame(columns=[*features, TARGET])
        return empty, pd.DataFrame(), summary
    data = pd.concat(kept, ignore_index=True)

    # Same prediction_id twice: identical outcome -> keep first; conflicting -> reject all.
    logged = (data["source_format"] == "prediction_log") & (data["reason"] == "")
    if logged.any():
        groups = data[logged].groupby("prediction_id")[TARGET]
        conflicting = groups.nunique()
        conflicting = set(conflicting[conflicting > 1].index)
        conflict_mask = logged & data["prediction_id"].isin(conflicting)
        data.loc[conflict_mask, "reason"] += "same prediction_id with conflicting outcomes; "
        dup_mask = logged & ~conflict_mask & data.duplicated("prediction_id", keep="first")
        data.loc[dup_mask, "reason"] += "duplicate prediction_id; "

    valid = data["reason"] == ""
    clean = data[valid].copy()
    clean[features] = clean[features].astype("int64")
    clean[TARGET] = clean[TARGET].astype("int64")
    clean = add_row_keys(clean, features)
    bad = data[~valid].copy()
    summary.update({"valid_labeled": int(len(clean)), "rejected": int(len(bad)),
                    "positives": int(clean[TARGET].sum()),
                    "negatives": int(len(clean) - clean[TARGET].sum())})
    return clean.reset_index(drop=True), bad.reset_index(drop=True), summary


# ------------------------------------------------------------------- step 3: policy
def policy_problems(summary: dict, metadata: dict, cfg: dict, now: datetime,
                    ignore_schedule: bool) -> tuple[list[str], dict]:
    req = cfg["data_requirements"]
    problems = []
    n_new = summary.get("fresh_labeled", 0)
    pos, neg = summary.get("fresh_positives", 0), summary.get("fresh_negatives", 0)
    if n_new < req["min_new_labeled_rows"]:
        problems.append(f"only {n_new} labeled records not yet used by production (need ≥ {req['min_new_labeled_rows']})")
    if pos < req["min_new_positive_outcomes"]:
        problems.append(f"only {pos} new positive outcomes (need ≥ {req['min_new_positive_outcomes']})")
    if neg < req["min_new_negative_outcomes"]:
        problems.append(f"only {neg} new negative outcomes (need ≥ {req['min_new_negative_outcomes']})")
    last = datetime.fromisoformat(str(metadata["created_at"]).replace("Z", "+00:00"))
    days = (now - last).total_seconds() / 86400
    if days < cfg["schedule_days"] and not ignore_schedule:
        problems.append(f"{days:.1f} days since the production model was trained (schedule: {cfg['schedule_days']} days)")
    return problems, {"days_since_last_training": round(days, 1)}


# ------------------------------------------------------------- step 4–6: training
def locked_split(X: pd.DataFrame, y: pd.Series, spec: dict):
    return train_test_split(X, y, test_size=spec["test_size"], stratify=y,
                            random_state=spec["random_state"])


def _unit_hash(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def add_row_keys(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Stable identity per new record, used for the holdout split and the manifest.

    Sheet rows: prediction_id. Dataset rows: content (21 features + label) + occurrence
    number, split by content only, so identical answers always land on the same side.
    """
    frame = frame.copy()
    if frame.empty:
        frame["row_key"], frame["split_key"] = [], []
        return frame
    content = frame[features + [TARGET]].astype(str).agg(",".join, axis=1)
    occurrence = content.groupby(content).cumcount().astype(str)
    is_log = frame["source_format"] == "prediction_log"
    frame["split_key"] = np.where(is_log, "pid:" + frame["prediction_id"], "row:" + content)
    frame["row_key"] = np.where(is_log, frame["split_key"], frame["split_key"] + "#" + occurrence)
    return frame


def split_new(new: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Hash-based split that never changes between runs.

    A record assigned to the holdout stays there in every future run, so no model version
    is ever evaluated on rows it was trained on, even though the Sheet export keeps growing.
    Returns (train, holdout, holdout_usable); holdout rows are never trained on.
    """
    spec = cfg["split"]
    if new.empty:
        return new, new, False
    in_hold = new["split_key"].map(_unit_hash) < spec["new_data_holdout_fraction"]
    train, hold = new[~in_hold], new[in_hold]
    pos = int(hold[TARGET].sum())
    usable = pos >= spec["min_holdout_positives"] and len(hold) - pos >= spec["min_holdout_positives"]
    return train, hold, usable


def train_model(params: dict, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def oof_metrics(params: dict, X: pd.DataFrame, y: pd.Series, spec: dict) -> dict:
    oof = np.zeros(len(y))
    folds = StratifiedKFold(spec["n_splits"], shuffle=spec["shuffle"], random_state=spec["random_state"])
    for train_idx, valid_idx in folds.split(X, y):
        fold_model = train_model(params, X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = scores(fold_model, X.iloc[valid_idx])
    return {"oof_pr_auc": float(average_precision_score(y, oof)),
            "oof_roc_auc": float(roc_auc_score(y, oof))}


# ---------------------------------------------------------------- step 7: evaluate
def top_k(n: int, capacity: float) -> int:
    return int(math.ceil(round(n * capacity, 9)))  # matches the team's capacity_config.json


def evaluate(y, s, capacities) -> dict:
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, s))
    naive = float(brier_score_loss(y, np.full_like(s, prevalence)))
    order = np.argsort(-s, kind="mergesort")
    cap_rows = []
    for c in capacities:
        k = top_k(len(y), c)
        hits = y[order[:k]]
        precision = float(hits.mean())
        cap_rows.append({"capacity": c, "capacity_pct": f"{c * 100:g}%", "k": k,
                         "precision_at_k": precision,
                         "recall_at_k": float(hits.sum() / y.sum()),
                         "lift_at_k": precision / prevalence})
    return {"n": int(len(y)), "positives": int(y.sum()), "prevalence": prevalence,
            "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "brier": brier, "naive_brier": naive, "brier_skill_score": 1 - brier / naive,
            "capacity": cap_rows}


def paired_bootstrap_pr_auc(y, s_prod, s_cand, n_boot: int, seed: int) -> list[float]:
    """95% interval of (candidate − production) PR-AUC on the same resampled rows."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            diffs.append(average_precision_score(y[idx], s_cand[idx])
                         - average_precision_score(y[idx], s_prod[idx]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def promotion_checks(prod: dict, cand: dict, crit: dict) -> list[dict]:
    primary_set = "new_holdout" if "new_holdout" in cand else "locked_test"
    lt_p, lt_c = prod["locked_test"], cand["locked_test"]
    checks = [{
        "check": f"Primary PR-AUC ({primary_set}): candidate − production ≥ {crit['min_primary_improvement']}",
        "value": cand[primary_set]["pr_auc"] - prod[primary_set]["pr_auc"],
        "passed": cand[primary_set]["pr_auc"] - prod[primary_set]["pr_auc"] >= crit["min_primary_improvement"],
    }, {
        "check": f"Locked-test PR-AUC drop ≤ {crit['max_locked_test_pr_auc_drop']}",
        "value": lt_p["pr_auc"] - lt_c["pr_auc"],
        "passed": lt_p["pr_auc"] - lt_c["pr_auc"] <= crit["max_locked_test_pr_auc_drop"],
    }, {
        "check": f"Locked-test Brier increase ≤ {crit['max_brier_increase']}",
        "value": lt_c["brier"] - lt_p["brier"],
        "passed": lt_c["brier"] - lt_p["brier"] <= crit["max_brier_increase"],
    }]
    for row_p, row_c in zip(lt_p["capacity"], lt_c["capacity"]):
        drop = row_p["precision_at_k"] - row_c["precision_at_k"]
        checks.append({"check": f"Locked-test Precision@{row_p['capacity_pct']} drop ≤ {crit['max_precision_at_k_drop']}",
                       "value": drop, "passed": drop <= crit["max_precision_at_k_drop"]})
    return checks


# ------------------------------------------------------------- step 8: bundle I/O
def plot_calibration(table: pd.DataFrame, version: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.1, 5.7), dpi=165)
    ax.plot([0, 1], [0, 1], linestyle=(0, (5, 2, 1, 2)), linewidth=2.2, color="#1f77b4",
            label="Perfect calibration")
    ax.plot(table["Mean predicted probability"], table["Observed diabetes rate"], marker="o",
            linewidth=2.6, color="#e8711c", label=f"Global XGBoost v{version} ({len(table)} quantile bins)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed survey positive-class rate")
    ax.set_title(f"Calibration Curve — Global XGBoost v{version}", fontsize=15, fontweight="bold", pad=14)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_candidate_bundle(dest: Path, model, version: str, ctx: dict) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    for item in BUNDLE.iterdir():
        if item.is_file() and item.name not in REGENERATED | PROVENANCE_ONLY:
            shutil.copy2(item, dest / item.name)

    model.save_model(str(dest / MODEL_FILE))
    (dest / "portable_model_checksum.txt").write_text(sha256_file(dest / MODEL_FILE) + "\n", encoding="utf-8")

    features = ctx["schema"]["feature_order"]
    sample = pd.read_csv(BUNDLE / "sample_input.csv")[features]
    p_sample = float(scores(model, sample)[0])
    write_json(dest / "sample_output.json", {"predicted_probability": p_sample,
                                             "predicted_probability_pct": round(p_sample * 100, 2)})

    train_s = ctx["candidate_train_scores"]
    risk = {f"p{q}": float(np.quantile(train_s, q / 100)) for q in (50, 75, 90, 95, 99)}
    risk["max"] = float(train_s.max())
    write_json(dest / "risk_reference.json", risk)

    cap = dict(ctx["capacity_cfg"])
    cap["locked_test_results"] = [{
        "capacity": r["capacity"], "capacity_pct": r["capacity_pct"],
        "global_precision_at_k": round(r["precision_at_k"], 4),
        "global_recall_at_k": round(r["recall_at_k"], 4),
        "global_lift_at_k": round(r["lift_at_k"], 2),
    } for r in ctx["cand_eval"]["locked_test"]["capacity"]]
    write_json(dest / "capacity_config.json", cap)

    lt = ctx["cand_eval"]["locked_test"]
    calib_spec = ctx["cfg"]["calibration"]
    observed, predicted = calibration_curve(ctx["y_test"], ctx["candidate_test_scores"],
                                            n_bins=calib_spec["n_bins"], strategy=calib_spec["strategy"])
    table = pd.DataFrame({"Mean predicted probability": predicted, "Observed diabetes rate": observed,
                          "Absolute gap": np.abs(predicted - observed)})
    table.to_csv(dest / "calibration_table.csv", index=False)
    write_json(dest / "calibration_report.json", {
        "evaluation_set": "locked_test_set", "n_test": lt["n"], "diabetes_prevalence": lt["prevalence"],
        "brier_score": lt["brier"], "naive_brier_score": lt["naive_brier"],
        "brier_skill_score": lt["brier_skill_score"], "n_calibration_bins": int(len(table)),
        "calibration_strategy": calib_spec["strategy"], "recalibrated": False, "model_version": version,
    })
    plot_calibration(table, version, dest / "calibration_curve.png")

    write_json(dest / "new_data_manifest.json", ctx["manifest"])

    params_doc = dict(ctx["training_params"])
    params_doc["trained_model_version"] = version
    write_json(dest / "training_params.json", params_doc)

    prod_meta = ctx["prod_meta"]
    meta = dict(prod_meta)
    meta.update({
        "model_version": version,
        "created_at": utc_now_iso(ctx["now"]),
        "status": "candidate",
        "parent_model_version": prod_meta["model_version"],
        "retraining_run_id": ctx["run_id"],
        "hyperparameter_optimizer": "Optuna (search done for v1.0.0; hyperparameters reused, not re-tuned)",
        "train_test_split": "80/20 stratified locked test (historical BRFSS 2015, random_state=42) "
                            "+ new labeled records split train/holdout",
        "oof_pr_auc": ctx["oof"].get("oof_pr_auc"),
        "oof_roc_auc": ctx["oof"].get("oof_roc_auc"),
        "locked_test_pr_auc": lt["pr_auc"],
        "locked_test_roc_auc": lt["roc_auc"],
        "locked_test_brier": lt["brier"],
        "training_data": ctx["training_data"],
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
    })
    if ctx["oof"].get("oof_pr_auc") is None:
        meta["oof_note"] = "Cross-validation skipped (--skip-cv); OOF metrics not computed for this version."
    write_json(dest / "model_metadata.json", meta)


def verify_bundle(bundle: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import smoke_test  # same checks the team runs before a demo
    smoke_test.main(bundle)


def promote(candidate: Path, prod_version: str, run_id: str, now: datetime) -> Path:
    """Archive the current production bundle, then swap the candidate in."""
    archive = REGISTRY / "archive" / f"v{prod_version}_{run_id}"
    staging = ROOT / "deployment_bundle.__staging__"
    old = ROOT / "deployment_bundle.__old__"
    for leftover in (staging, old):
        if leftover.exists():
            raise PipelineError(f"Leftover folder from an interrupted run: {leftover} — inspect and remove it.")
    shutil.copytree(BUNDLE, archive)              # nothing is touched until the archive exists
    shutil.copytree(candidate, staging)
    meta = read_json(staging / "model_metadata.json")
    meta.update({"status": "production", "promoted_at": utc_now_iso(now)})
    write_json(staging / "model_metadata.json", meta)
    BUNDLE.rename(old)
    staging.rename(BUNDLE)
    try:
        verify_bundle(BUNDLE)
    except Exception:
        BUNDLE.rename(staging)
        old.rename(BUNDLE)                         # automatic rollback
        raise
    shutil.rmtree(old)
    shutil.rmtree(staging, ignore_errors=True)
    return archive


# ------------------------------------------------------------------------ report
def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def write_report(run_dir: Path, report: dict) -> None:
    write_json(run_dir / "report.json", report)
    lines = [f"# Retraining run {report['run_id']}", "",
             f"- **Status:** {report['status']}",
             f"- **Started (UTC):** {report['started_at']}",
             f"- **Production model:** v{report['production_version']}"]
    if report.get("candidate_version"):
        lines.append(f"- **Candidate model:** v{report['candidate_version']} → `{report.get('candidate_bundle')}`")
    if report.get("archived_bundle"):
        lines.append(f"- **Previous production archived to:** `{report['archived_bundle']}`")
    lines += ["", "## Data", ""]
    d = report["data"]
    lines.append(f"- Historical: {d['historical']['rows']:,} rows (`{Path(d['historical']['file']).name}`)")
    nd = d.get("new", {})
    lines.append(f"- New labeled records: {nd.get('valid_labeled', 0):,} valid "
                 f"({nd.get('positives', 0)} positive / {nd.get('negatives', 0)} negative), "
                 f"{nd.get('rejected', 0)} rejected, {nd.get('pending_unlabeled', 0)} still unlabeled; "
                 f"{nd.get('fresh_labeled', 0):,} not yet used by production")
    if "split" in d:
        s = d["split"]
        lines.append(f"- Training rows: {s['historical_train']:,} historical + {s['new_train']:,} new; "
                     f"locked test {s['locked_test']:,}; new holdout {s['new_holdout']:,}")
    if report.get("policy_problems"):
        lines += ["", "## Retraining policy not met", ""] + [f"- {p}" for p in report["policy_problems"]]
    if report.get("notes"):
        lines += ["", "## Notes", ""] + [f"- ⚠️ {note}" for note in report["notes"]]
    if "evaluation" in report:
        ev = report["evaluation"]
        lines += ["", "## Evaluation (same rows, candidate vs production)", "",
                  "| Set | Metric | Production | Candidate | Δ |", "|---|---|---|---|---|"]
        for set_name in ("locked_test", "new_holdout"):
            if set_name not in ev["candidate"]:
                continue
            p, c = ev["production"][set_name], ev["candidate"][set_name]
            for key, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("brier", "Brier")):
                lines.append(f"| {set_name} (n={c['n']:,}) | {label} | {fmt(p[key])} | {fmt(c[key])} | {c[key] - p[key]:+.4f} |")
            for rp, rc in zip(p["capacity"], c["capacity"]):
                lines.append(f"| {set_name} | Precision@{rp['capacity_pct']} | {fmt(rp['precision_at_k'])} | "
                             f"{fmt(rc['precision_at_k'])} | {rc['precision_at_k'] - rp['precision_at_k']:+.4f} |")
        if ev.get("primary_pr_auc_diff_ci95"):
            lo, hi = ev["primary_pr_auc_diff_ci95"]
            lines.append(f"\nPaired bootstrap 95% interval for the primary PR-AUC difference: [{lo:+.4f}, {hi:+.4f}]")
        if ev.get("candidate_oof"):
            lines.append(f"\nCandidate 5-fold OOF PR-AUC {fmt(ev['candidate_oof']['oof_pr_auc'])}, "
                         f"ROC-AUC {fmt(ev['candidate_oof']['oof_roc_auc'])}")
    if report.get("checks"):
        lines += ["", "## Promotion criteria", "", "| Check | Value | Result |", "|---|---|---|"]
        for ch in report["checks"]:
            lines.append(f"| {ch['check']} | {ch['value']:+.4f} | {'PASS' if ch['passed'] else 'FAIL'} |")
    if report.get("after_promotion"):
        lines += ["", "## Manual follow-ups after promotion", ""] + [f"- {t}" for t in report["after_promotion"]]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Group 9 retraining pipeline (offline, manual promotion).")
    ap.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL,
                    help="Team BRFSS 2015 CSV (unmodified). Default: data/diabetes_binary_health_indicators_BRFSS2015.csv")
    ap.add_argument("--new-data", type=Path, nargs="*", default=[],
                    help="Google Sheet 'Predictions' CSV export(s) and/or recoded dataset CSV(s) with outcomes")
    ap.add_argument("--promote", action="store_true",
                    help="Replace deployment_bundle/ if every promotion criterion passes")
    ap.add_argument("--ignore-schedule", action="store_true", help="Run before the 30-day schedule is due")
    ap.add_argument("--force", action="store_true",
                    help="Bypass data-sufficiency rules (testing only; a forced candidate is never promoted)")
    ap.add_argument("--skip-cv", action="store_true", help="Skip 5-fold OOF metrics (faster)")
    ap.add_argument("--skip-repro-check", action="store_true", help="Skip the production reproduction check")
    ap.add_argument("--now", help="Override current time (ISO 8601) for schedule testing")
    return ap.parse_args(argv)


def run(args) -> int:
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    schema = read_json(BUNDLE / "feature_schema.json")
    cfg = read_json(BUNDLE / "retraining_config.json")
    prod_meta = read_json(BUNDLE / "model_metadata.json")
    training_params = read_json(BUNDLE / "training_params.json")
    capacity_cfg = read_json(BUNDLE / "capacity_config.json")
    prod_calib = read_json(BUNDLE / "calibration_report.json")
    params = training_params["xgb_params"]
    capacities = capacity_cfg["supported_capacities"]
    features = schema["feature_order"]

    report = {"run_id": run_id, "started_at": utc_now_iso(now), "status": "RUNNING",
              "production_version": prod_meta["model_version"],
              "arguments": {k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v)
                            for k, v in vars(args).items()},
              "data": {}, "notes": []}
    print(f"[run {run_id}] production model v{prod_meta['model_version']}")

    # 1–2 · load + validate
    prod_model = load_production_model(BUNDLE)
    X_hist, y_hist, hist_info = load_historical(args.historical, schema)
    report["data"]["historical"] = hist_info
    ref_sha = training_params.get("historical_dataset", {}).get("sha256")
    if ref_sha and ref_sha != hist_info["sha256"]:
        report["notes"].append("Historical CSV checksum differs from the reference copy; "
                               "the locked-test check below decides whether it is still the same data.")
    new, rejected, new_summary = load_new_labeled(args.new_data, schema)
    report["data"]["new"] = new_summary
    if not rejected.empty:
        rejected.to_csv(run_dir / "rejected_rows.csv", index=False)
    print(f"  historical rows: {len(X_hist):,} · new valid labeled: {new_summary.get('valid_labeled', 0)} "
          f"· rejected: {new_summary.get('rejected', 0)} · pending: {new_summary.get('pending_unlabeled', 0)}")

    # 3 · policy — only records the production model has never used count as "new"
    manifest_path = BUNDLE / "new_data_manifest.json"
    seen = set()
    if manifest_path.is_file():
        prev = read_json(manifest_path)
        seen = set(prev.get("train_keys", [])) | set(prev.get("holdout_keys", []))
    fresh = new[~new["row_key"].isin(seen)] if len(new) else new
    new_summary.update({"already_used_by_production": int(len(new) - len(fresh)),
                        "fresh_labeled": int(len(fresh)), "fresh_positives": int(fresh[TARGET].sum()) if len(fresh) else 0})
    new_summary["fresh_negatives"] = new_summary["fresh_labeled"] - new_summary["fresh_positives"]
    problems, policy_info = policy_problems(new_summary, prod_meta, cfg, now, args.ignore_schedule)
    report["policy"] = policy_info
    report["policy_problems"] = problems
    if problems and not args.force:
        report["status"] = "SKIPPED_POLICY_NOT_MET"
        write_report(run_dir, report)
        print("  SKIPPED — retraining policy not met:\n   - " + "\n   - ".join(problems))
        print(f"  report: {run_dir / 'report.md'}")
        return 2
    if problems:
        report["notes"].append("--force used: policy bypassed, this candidate can NOT be promoted.")
    simulated = any("simulated" in Path(f["file"]).name.lower() for f in new_summary.get("files", []))
    if simulated:
        report["notes"].append("New data comes from a SIMULATED file: pipeline test only, this candidate can NOT "
                               "be promoted and its metrics must not be reported as results.")
    not_promotable = bool(problems) or simulated

    # 4 · splits (locked test must be exactly the team's)
    X_tr, X_te, y_tr, y_te = locked_split(X_hist, y_hist, training_params["historical_split"])
    prod_test = scores(prod_model, X_te)
    integrity = {"n_test": len(y_te) == prod_calib["n_test"],
                 "prevalence": np.isclose(y_te.mean(), prod_calib["diabetes_prevalence"], atol=1e-12),
                 "production_brier": np.isclose(brier_score_loss(y_te, prod_test), prod_calib["brier_score"], atol=1e-10)}
    if not all(integrity.values()):
        raise PipelineError(f"Locked test set could not be reproduced ({integrity}). "
                            "Comparisons would be invalid — check the historical CSV (rows/order).")
    print("  PASS locked test reproduced (n, prevalence and production Brier match calibration_report.json)")
    new_train, new_hold, holdout_usable = split_new(new, cfg)
    if not new.empty and not holdout_usable:
        report["notes"].append(f"New-data holdout has too few outcomes ({len(new_hold)} rows) to evaluate; "
                               "decision relies on the locked test. Holdout rows are still kept out of training.")
    report["data"]["split"] = {"historical_train": int(len(X_tr)), "locked_test": int(len(X_te)),
                               "new_train": int(len(new_train)), "new_holdout": int(len(new_hold))}

    # 5 · reproduction check
    prod_new_rows = prod_meta.get("training_data", {}).get("new_labeled_rows_used", 0)
    if not args.skip_repro_check and prod_new_rows:
        report["notes"].append("Reproduction check skipped: production was itself trained with new records.")
    elif not args.skip_repro_check:
        repro = train_model(params, X_tr, y_tr)
        diff = float(np.abs(scores(repro, X_te) - prod_test).max())
        report["reproduction_max_abs_diff"] = diff
        if diff <= REPRO_TOLERANCE:
            print(f"  PASS pipeline reproduces production v{prod_meta['model_version']} (max |Δscore| = {diff:.2e})")
        else:
            report["notes"].append(f"Pipeline does not exactly reproduce production (max |Δscore| = {diff:.2e}); "
                                   "likely different library versions/hardware. Use requirements-retrain.txt.")
            print(f"  WARN reproduction differs (max |Δscore| = {diff:.2e})")

    # 6 · candidate
    if new_train.empty:
        X_cand, y_cand = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    else:
        X_cand = pd.concat([X_tr, new_train[features].astype("int64")], ignore_index=True)
        y_cand = pd.concat([y_tr, new_train[TARGET].astype("int64")], ignore_index=True)
    print(f"  training candidate on {len(X_cand):,} rows ...")
    cand_model = train_model(params, X_cand, y_cand)
    oof = {} if args.skip_cv else oof_metrics(params, X_cand, y_cand, training_params["cross_validation"])

    # 7 · evaluation
    cand_test = scores(cand_model, X_te)
    ev_prod = {"locked_test": evaluate(y_te, prod_test, capacities)}
    ev_cand = {"locked_test": evaluate(y_te, cand_test, capacities)}
    primary = ("locked_test", y_te.to_numpy(), prod_test, cand_test)
    if holdout_usable:
        y_nh = new_hold[TARGET].to_numpy()
        prod_nh, cand_nh = scores(prod_model, new_hold[features]), scores(cand_model, new_hold[features])
        ev_prod["new_holdout"] = evaluate(y_nh, prod_nh, capacities)
        ev_cand["new_holdout"] = evaluate(y_nh, cand_nh, capacities)
        primary = ("new_holdout", y_nh, prod_nh, cand_nh)
    crit = cfg["promotion_criteria"]
    ci = paired_bootstrap_pr_auc(primary[1], primary[2], primary[3], crit["bootstrap_resamples"], 42)
    report["evaluation"] = {"production": ev_prod, "candidate": ev_cand, "primary_set": primary[0],
                            "primary_pr_auc_diff_ci95": ci, "candidate_oof": oof or None}
    checks = promotion_checks(ev_prod, ev_cand, crit)
    report["checks"] = [{**c, "passed": bool(c["passed"])} for c in checks]
    gate_ok = all(c["passed"] for c in checks)

    # 8 · candidate bundle + optional promotion
    version = bump_minor(prod_meta["model_version"])
    cand_dir = REGISTRY / "candidates" / f"v{version}_{run_id}"
    ctx = {"schema": schema, "cfg": cfg, "prod_meta": prod_meta, "training_params": training_params,
           "capacity_cfg": capacity_cfg, "cand_eval": ev_cand, "y_test": y_te.to_numpy(),
           "candidate_test_scores": cand_test, "candidate_train_scores": scores(cand_model, X_cand),
           "oof": oof, "now": now, "run_id": run_id,
           "manifest": {"model_version": version,
                        "train_keys": sorted(new_train["row_key"]) if len(new_train) else [],
                        "holdout_keys": sorted(new_hold["row_key"]) if len(new_hold) else []},
           "training_data": {"historical_rows_used": int(len(X_tr)), "new_labeled_rows_used": int(len(new_train)),
                             "new_holdout_rows": int(len(new_hold)), "historical": hist_info,
                             "new_sources": new_summary.get("files", [])}}
    write_candidate_bundle(cand_dir, cand_model, version, ctx)
    verify_bundle(cand_dir)
    report.update({"candidate_version": version, "candidate_bundle": str(cand_dir.relative_to(ROOT))})
    print(f"  candidate bundle verified: {cand_dir.relative_to(ROOT)}")

    if not gate_ok:
        report["status"] = "REJECTED_CRITERIA_FAILED"
        code = 3
    elif args.promote and not not_promotable:
        archive = promote(cand_dir, prod_meta["model_version"], run_id, now)
        new_sample = read_json(BUNDLE / "sample_output.json")["predicted_probability_pct"]
        report.update({"status": "PROMOTED", "archived_bundle": str(archive.relative_to(ROOT)),
                       "after_promotion": [
                           f"README.md §3 and app.py caption still say 1.03% — new sample score is {new_sample}%.",
                           "Re-read the calibration narrative in app.py (over-estimation wording) against the new curve.",
                           "Tell report owners (Methodology/Results/5.1) the new version and metrics.",
                           f"Rollback: copy {archive.relative_to(ROOT)} back to deployment_bundle/.",
                       ]})
        code = 0
    else:
        report["status"] = "CANDIDATE_READY" + (" (not promotable — see notes)" if not_promotable else
                                                " (run again with --promote to deploy)")
        code = 0
    write_report(run_dir, report)
    print(f"  STATUS: {report['status']}")
    for c in report["checks"]:
        print(f"   {'PASS' if c['passed'] else 'FAIL'}  {c['check']}  ({c['value']:+.4f})")
    print(f"  report: {(run_dir / 'report.md').relative_to(ROOT)}")
    return code


def main(argv=None) -> int:
    try:
        return run(parse_args(argv))
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
