"""Score the frozen models on 2018 once; reuse saved predictions for reporting."""
import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, r2_score, roc_auc_score

from part2_python.pipeline import PROJECT_ROOT, configure_logging
from part3_machine_learning.dataset import FEATURES

BASE = PROJECT_ROOT / "part3_machine_learning"
RESULTS = BASE / "results"
DEST = RESULTS / "final_2018"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    truth = rows.high_risk
    predicted = rows.proxy_prediction
    tn, fp, fn, tp = map(int, confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    return {
        "hours": len(rows), "actual_positive_hours": tp + fn,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "mae": float((rows.volume_prediction - rows.traffic_volume).abs().mean()),
    }


def main():
    plan_path = RESULTS / "final_evaluation_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["status"] != "frozen_before_final_scoring" or plan["features"] != list(FEATURES):
        raise ValueError("Review the frozen evaluation plan and feature order")
    if DEST.exists():
        logging.info("Saved evaluation already exists at %s; skipping inference", DEST)
        return
    source = PROJECT_ROOT / "data/processed/traffic_supervised.csv"
    if digest(source) != plan["dataset_sha256"]:
        raise ValueError("Supervised dataset hash mismatch")
    frame = pd.read_csv(source)
    test = frame.loc[frame.split.eq("test")]
    if (frame.split.value_counts().to_dict() != plan["expected_splits"]
            or not test.date_time.str.startswith("2018-").all()
            or not test.date_time.is_unique or not test.date_time.is_monotonic_increasing):
        raise ValueError("Unexpected split or timestamps")
    x = test[plan["features"]]
    models = {}
    for task, record in plan["candidates"].items():
        path = PROJECT_ROOT / Path(record["model_path"].replace("\\", "/"))
        if digest(path) != record["model_sha256"]:
            raise ValueError(f"{task} model hash mismatch")
        models[task] = joblib.load(path)
        if list(models[task].feature_names_in_) != list(x.columns):
            raise ValueError(f"{task} feature names differ from plan")
    classifier = models["classification"]
    rows = test[["date_time", "traffic_volume", "high_risk", *plan["proposed_slices"]]].copy()
    rows["volume_prediction"] = models["regression"].predict(x)
    rows["proxy_prediction"] = classifier.predict(x)
    positive = list(classifier.classes_).index(1)
    rows["proxy_score"] = classifier.predict_proba(x)[:, positive]
    overall = metrics(rows)
    overall.update(
        accuracy=float((rows.high_risk == rows.proxy_prediction).mean()),
        f1=float(f1_score(rows.high_risk, rows.proxy_prediction)),
        roc_auc=float(roc_auc_score(rows.high_risk, rows.proxy_score)),
        r2=float(r2_score(rows.traffic_volume, rows.volume_prediction)),
    )
    slices = {group: {str(v): metrics(rows.loc[rows[group].eq(v)]) for v in values}
              for group, values in plan["proposed_slices"].items()}
    with TemporaryDirectory(dir=RESULTS) as temp:
        output = Path(temp) / DEST.name
        output.mkdir()
        predictions = output / "predictions.csv"
        rows.to_csv(predictions, index=False)
        report = dict(plan_sha256=digest(plan_path), predictions_sha256=digest(predictions),
                      first_hour=rows.date_time.iloc[0], last_hour=rows.date_time.iloc[-1],
                      overall=overall, slices=slices)
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                             encoding="utf-8")
        output.rename(DEST)
    logging.info("Saved frozen-model evaluation for %d observed hours", len(rows))


if __name__ == "__main__":
    configure_logging(BASE / "final_evaluation.log")
    main()
