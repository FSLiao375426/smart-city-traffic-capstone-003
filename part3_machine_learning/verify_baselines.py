"""Export portable evidence from actual local MLflow runs and persisted models."""
import hashlib
import json
import logging
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging

logger = logging.getLogger(__name__)


def verify():
    folder = PROJECT_ROOT / "part3_machine_learning/results"
    manifest = json.loads((folder / "training_manifest.json").read_text())
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = mlflow.MlflowClient()
    metrics = pd.read_csv(folder / "validation_metrics.csv", keep_default_na=False, na_values=["NA"])
    predictions = pd.read_csv(folder / "validation_predictions.csv", keep_default_na=False, na_values=["NA"])
    if len(metrics) != 6 or len(predictions) != 6 * 8713:
        raise ValueError("Incomplete baseline evidence")
    if not pd.to_datetime(predictions.date_time).dt.year.eq(2017).all():
        raise ValueError("Prediction evidence contains observations outside validation")
    exported = []
    for model in manifest["runs"]:
        path = PROJECT_ROOT / model["model_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != model["model_sha256"]:
            raise ValueError("Model hash differs from the recorded run")
        run = client.get_run(model["run_id"])
        if run.info.status != "FINISHED":
            raise ValueError("An experiment did not finish")
        row = metrics.loc[metrics.model.eq(model["model"])].iloc[0]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "mae", "r2"):
            if pd.notna(row[key]):
                np.testing.assert_allclose(run.data.metrics["validation_" + key], row[key], atol=5.1e-11, rtol=0)
        artifacts = client.list_artifacts(model["run_id"], "model")
        if not any(item.path.endswith("model.joblib") for item in artifacts):
            raise ValueError("MLflow model artifact missing")
        exported.append({"model": model["model"], "run_id": model["run_id"], "status": run.info.status,
                         "start_time_ms": run.info.start_time, "end_time_ms": run.info.end_time,
                         "params": run.data.params, "tags": {key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.")},
                         "metrics": run.data.metrics, "model_sha256": digest,
                         "model_artifacts": [item.path for item in artifacts]})
    result = {"dataset_sha256": manifest["dataset_sha256"], "validation_rows_per_model": 8713,
              "test_scored": False, "completed_runs": exported,
              "checks": ["Six real FINISHED MLflow runs", "Persisted model hashes match", "Saved metrics match MLflow", "Predictions contain only 2017 validation hours"]}
    (folder / "mlflow_evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Verified and exported six real MLflow runs; final test remains unscored")


if __name__ == "__main__":
    configure_logging(PROJECT_ROOT / "part3_machine_learning/verify_baselines.log")
    try:
        verify()
    except Exception:
        logger.error("Baseline evidence verification failed", exc_info=True)
        sys.exit(1)
