"""Fit train-only baselines, score 2017 validation and track actual runs in MLflow."""
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from part2_python.pipeline import PROJECT_ROOT, configure_logging
from part3_machine_learning.dataset import FEATURES, FORBIDDEN_FEATURES

logger = logging.getLogger(__name__)
BASE = PROJECT_ROOT / "part3_machine_learning"
SEED = 42


def classification_metrics(truth, prediction, score):
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, score))}


def run_training():
    source = PROJECT_ROOT / "data/processed/traffic_supervised.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata_path = BASE / "results/dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if source_hash != metadata["output_sha256"] or list(FEATURES) != metadata["features"]:
        raise ValueError("Dataset does not match the verified feature metadata")
    if set(FEATURES) & FORBIDDEN_FEATURES:
        raise ValueError("Target leakage in the input schema")
    frame = pd.read_csv(source, keep_default_na=False)
    train = frame.loc[frame.split.eq("train")].copy()
    validation = frame.loc[frame.split.eq("validation")].copy()
    if len(train) != 25329 or len(validation) != 8713:
        raise ValueError("Unexpected chronological split sizes")
    if train.date_time.max() >= validation.date_time.min():
        raise ValueError("Training and validation periods overlap")
    # Test observations are never passed to fit, predict, scoring or model selection.
    del frame
    x_train, x_validation = train.loc[:, FEATURES], validation.loc[:, FEATURES]
    models = [
        ("classification", "dummy_prior", DummyClassifier(strategy="prior")),
        ("classification", "logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))),
        ("classification", "random_forest_classifier", RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, class_weight="balanced", n_jobs=2, random_state=SEED)),
        ("regression", "dummy_mean", DummyRegressor(strategy="mean")),
        ("regression", "linear_regression", make_pipeline(StandardScaler(), LinearRegression())),
        ("regression", "random_forest_regressor", RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
    ]
    track = BASE / "mlflow"
    track.mkdir(exist_ok=True)
    uri = "sqlite:///" + (track / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    experiment_name = "capstone003_same_hour_baselines"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        experiment_name, artifact_location=(track / "artifacts").as_uri())
    logger.info("Train %d hours; validation %d hours; %d common inputs; test untouched", len(train), len(validation), len(FEATURES))
    metrics_rows, predictions, manifests = [], [], []
    for task, name, model in models:
        target = "high_risk" if task == "classification" else "traffic_volume"
        target_train, target_validation = train[target], validation[target]
        model_folder = BASE / "models" / name
        model_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Fitting %s (%s)", name, task)
        with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
            mlflow.set_tags({"task": task, "evaluation_split": "validation_2017",
                             "prediction_mode": "conditional_same_hour", "label_status": "constructed_proxy_not_accidents",
                             "model_version": "v1", "dataset_sha256": source_hash})
            mlflow.log_params({"random_seed": SEED, "train_hours": len(train), "validation_hours": len(validation),
                               "fit_end": metadata["fit_end"], "feature_count": len(FEATURES),
                               "target": target, "classification_threshold": 0.5 if task == "classification" else "not_applicable"})
            params = {key: str(value) for key, value in model.get_params(deep=True).items() if "__" in key or not hasattr(value, "fit")}
            mlflow.log_dict(params, "model_parameters.json")
            started = time.perf_counter()
            model.fit(x_train, target_train)
            fit_seconds = time.perf_counter() - started
            estimate = model.predict(x_validation)
            prediction_frame = pd.DataFrame({"date_time": validation.date_time, "task": task,
                                             "model": name, "actual": target_validation, "prediction": estimate})
            if task == "classification":
                probability = model.predict_proba(x_validation)[:, 1]
                metrics = classification_metrics(target_validation, estimate, probability)
                prediction_frame["proxy_score"] = probability
                matrix = confusion_matrix(target_validation, estimate, labels=[0, 1]).tolist()
                mlflow.log_dict({"labels": [0, 1], "rows": "actual", "columns": "predicted", "matrix": matrix}, "confusion_matrix.json")
            else:
                metrics = {"mae": float(mean_absolute_error(target_validation, estimate)), "r2": float(r2_score(target_validation, estimate))}
            if not np.isfinite(list(metrics.values())).all():
                raise ValueError("Non-finite model metric")
            mlflow.log_metrics({"validation_" + key: value for key, value in metrics.items()})
            mlflow.log_metric("fit_seconds", fit_seconds)
            schema = {"features": list(FEATURES), "dtypes": x_train.dtypes.astype(str).to_dict(),
                      "prediction_mode": metadata["prediction_mode"], "target": target,
                      "input_contract": "Prepared features from dataset.py with frozen Part 2 preprocessing",
                      "dataset_sha256": source_hash, "metrics": metrics, "evaluation_split": "2017_validation"}
            model_path = model_folder / "model.joblib"
            joblib.dump(model, model_path)
            (model_folder / "metadata.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
            # Persisted-model predictions must match the live model on validation.
            reloaded = joblib.load(model_path)
            np.testing.assert_allclose(reloaded.predict(x_validation), estimate, rtol=0, atol=1e-10)
            mlflow.log_artifact(str(model_path), "model")
            mlflow.log_artifact(str(model_folder / "metadata.json"), "model")
            mlflow.log_artifact(str(metadata_path), "data")
            metrics_rows.append({"task": task, "model": name, "split": "validation_2017", **metrics,
                                 "fit_seconds": fit_seconds, "mlflow_run_id": run.info.run_id})
            manifests.append({"model": name, "run_id": run.info.run_id, "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
            predictions.append(prediction_frame)
            logger.info("Validated %s: %s", name, metrics)
    results = pd.DataFrame(metrics_rows)
    results.to_csv(BASE / "results/validation_metrics.csv", index=False, float_format="%.10f", na_rep="NA")
    pd.concat(predictions, ignore_index=True).to_csv(BASE / "results/validation_predictions.csv", index=False, float_format="%.10f", na_rep="NA")
    learned = results.loc[~results.model.str.startswith("dummy_")]
    best_classifier = learned.loc[learned.task.eq("classification")].sort_values("f1", ascending=False).iloc[0].model
    best_regressor = learned.loc[learned.task.eq("regression")].sort_values("mae").iloc[0].model
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "version": "v1",
               "selection_split": "2017_validation", "test_scored": False,
               "selection_rules": {"classification": "highest validation F1 at default 0.5 threshold", "regression": "lowest validation MAE"},
               "validation_candidates": {"classification": best_classifier, "regression": best_regressor},
               "dataset_sha256": source_hash, "runs": manifests,
               "tracking_uri": uri, "experiment_name": experiment_name,
               "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "mlflow": mlflow.__version__, "pandas": pd.__version__, "numpy": np.__version__},
               "limitations": metadata["limitations"]}
    (BASE / "results/training_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Supervised source changed during training")
    logger.info("Saved six validation runs and candidate selection; final test remains untouched")
    return summary


if __name__ == "__main__":
    configure_logging(BASE / "train_baselines.log")
    try:
        run_training()
    except Exception:
        logger.error("Baseline training failed", exc_info=True)
        sys.exit(1)
