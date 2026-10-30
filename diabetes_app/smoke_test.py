"""Fast integrity check of the exact deployment model against sample_output.json."""

import hashlib
import json
import xgboost as xgb
from pathlib import Path

import numpy as np
import pandas as pd
import shap

BUNDLE = Path(__file__).resolve().parent / "deployment_bundle"


def main():
    model_file = BUNDLE / "diabetes_global_xgboost.ubj"
    expected_checksum = (BUNDLE / "portable_model_checksum.txt").read_text().strip()
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == expected_checksum, "Portable model checksum mismatch"

    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    feature_order = json.loads((BUNDLE / "feature_schema.json").read_text())["feature_order"]
    X = pd.read_csv(BUNDLE / "sample_input.csv")[feature_order]
    expected = json.loads((BUNDLE / "sample_output.json").read_text())["predicted_probability"]
    actual = float(model.predict_proba(X)[0, 1])
    assert np.isclose(actual, expected, atol=1e-7), f"Prediction mismatch: {actual} != {expected}"

    background = pd.read_csv(BUNDLE / "shap_background.csv")[feature_order]
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = np.asarray(explainer.shap_values(X))
    assert shap_values.shape == (1, 21), f"SHAP shape unexpected: {shap_values.shape}"
    # Hưng's locked-test calibration summary; no model recalibration or retraining.
    report = json.loads((BUNDLE / "calibration_report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(BUNDLE / "calibration_table.csv")
    assert report["evaluation_set"] == "locked_test_set"
    assert report["n_test"] == 50736
    assert report["recalibrated"] is False
    assert len(table) == report["n_calibration_bins"] == 10
    assert np.isclose(report["brier_score"], 0.09868237609635581)
    assert np.isclose(report["naive_brier_score"], 0.11991648458189615)
    assert np.allclose((table["Mean predicted probability"] - table["Observed diabetes rate"]).abs(), table["Absolute gap"], atol=1e-8)
    assert (BUNDLE / "calibration_curve.png").is_file()

    print("PASS · portable UBJ checksum verified")
    print(f"PASS · predicted probability {actual:.10f} ({actual:.2%}) matches sample_output.json")
    print("PASS · SHAP returned 21 contributions")
    print("PASS · calibration data (50,736 locked-test rows, 10 bins, Brier 0.0987) validated")
    print("Note: this script tests the model, not Streamlit's browser UI or Google Sheets logging.")


if __name__ == "__main__":
    main()
