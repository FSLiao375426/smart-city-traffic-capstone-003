"""Mine train-only context-to-congestion rules; validate frozen rules in 2017."""
import hashlib
import json
import logging
import math
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
import mlflow
import mlxtend
from mlxtend.frequent_patterns import fpgrowth
import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, WEATHER_TYPES, configure_logging

logger = logging.getLogger(__name__)
BASE = PROJECT_ROOT / "part3_machine_learning"
LEVELS = {1: "Low", 2: "Medium", 3: "High", 4: "Severe"}
TIME_BINS = ("night_00_05", "morning_06_09", "midday_10_12", "afternoon_13_16", "evening_17_20", "late_evening_21_23")
MIN_SUPPORT = .01
MIN_ANTECEDENT_HOURS = 500
MIN_CONFIDENCE = .30
MIN_LIFT = 1.20


def transactions(frame):
    """One observed hour is one Boolean basket, preserving multiple weather items."""
    times = pd.to_datetime(frame.date_time, errors="raise")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Expected unique observed hours")
    hours = times.dt.hour
    slot = np.select([hours.lt(6), hours.lt(10), hours.lt(13), hours.lt(17), hours.lt(21)], TIME_BINS[:-1], default=TIME_BINS[-1])
    result = pd.DataFrame(index=frame.index)
    for name in TIME_BINS:
        result["time=" + name] = slot == name
    weekend = times.dt.day_of_week.ge(5)
    result["day=weekday"] = ~weekend
    result["day=weekend"] = weekend
    for name in WEATHER_TYPES:
        values = frame["weather_" + name]
        if not values.isin([0, 1]).all():
            raise ValueError("Weather items must be binary")
        result["weather=" + name] = values.eq(1)
    if not frame.traffic_volume_quartile.isin(LEVELS).all():
        raise ValueError("Invalid congestion level")
    for value, name in LEVELS.items():
        result["congestion=" + name] = frame.traffic_volume_quartile.eq(value)
    return result.astype(bool)


def evaluate_rule(baskets, antecedents, consequent):
    """Compute counts and the three descriptive metrics from exact intersections."""
    a = baskets.loc[:, list(antecedents)].all(axis=1)
    b = baskets[consequent]
    n, n_a, n_b, joint = len(baskets), int(a.sum()), int(b.sum()), int((a & b).sum())
    confidence = joint / n_a if n_a else None
    base = n_b / n if n else None
    return {"total_hours": n, "antecedent_hours": n_a, "consequent_hours": n_b, "joint_hours": joint,
            "support": joint / n if n else None, "confidence": confidence, "base_rate": base,
            "lift": confidence / base if confidence is not None and base else None}


def mine_rules(baskets):
    frequent = fpgrowth(baskets, min_support=MIN_SUPPORT, use_colnames=True, max_len=4, verbose=0)
    rules = []
    for itemset in frequent.itemsets:
        outcomes = [item for item in itemset if item.startswith("congestion=")]
        if len(outcomes) != 1:
            continue
        antecedents = sorted(itemset - set(outcomes))
        if not 1 <= len(antecedents) <= 3:
            continue
        if any(not item.startswith(("time=", "day=", "weather=")) for item in antecedents):
            raise ValueError("Outcome-derived item leaked into an antecedent")
        counts = evaluate_rule(baskets, antecedents, outcomes[0])
        if (counts["joint_hours"] < math.ceil(MIN_SUPPORT * len(baskets))
                or counts["antecedent_hours"] < MIN_ANTECEDENT_HOURS
                or counts["confidence"] < MIN_CONFIDENCE or counts["lift"] < MIN_LIFT):
            continue
        rules.append({"antecedents": antecedents, "consequent": outcomes[0], **counts})
    rules.sort(key=lambda row: (-row["lift"], -row["support"], len(row["antecedents"]), " & ".join(row["antecedents"]), row["consequent"]))
    for number, rule in enumerate(rules, start=1):
        rule["rule_id"] = f"R{number:03d}"
    return frequent, rules


def run_mining():
    source = PROJECT_ROOT / "data/processed/traffic_supervised.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = json.loads((BASE / "results/dataset_metadata.json").read_text())
    if digest != metadata["output_sha256"]:
        raise ValueError("Prepared source differs from verified metadata")
    frame = pd.read_csv(source, keep_default_na=False)
    train = frame.loc[frame.split.eq("train")].copy()
    validation = frame.loc[frame.split.eq("validation")].copy()
    del frame
    training_baskets, validation_baskets = transactions(train), transactions(validation)
    folder = BASE / "results/association_rules"
    folder.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (BASE / "mlflow/mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    exp_name = "capstone003_association_rules"
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else client.create_experiment(exp_name, artifact_location=(BASE / "mlflow/artifacts").as_uri())
    with mlflow.start_run(experiment_id=exp_id, run_name="context_to_congestion_v1") as run:
        mlflow.set_tags({"task": "association_rules", "selection_split": "train_through_2016", "dataset_sha256": digest})
        mlflow.log_params({"min_support": MIN_SUPPORT, "min_antecedent_hours": MIN_ANTECEDENT_HOURS,
                           "min_confidence": MIN_CONFIDENCE, "min_lift": MIN_LIFT,
                           "max_antecedent_items": 3, "train_hours": len(train), "validation_hours": len(validation),
                           "algorithm": "FP-growth", "mlxtend_version": mlxtend.__version__})
        logger.info("Transactions: %d train, %d validation, %d Boolean items; no 2018 transactions", len(train), len(validation), len(training_baskets.columns))
        frequent, rules = mine_rules(training_baskets)
        if not rules:
            raise ValueError("No rules pass the predefined criteria; report without relaxing on validation")
        rows = []
        for rule in rules:
            # IDs and ordering are frozen from TRAINING before looking at validation.
            validated = evaluate_rule(validation_baskets, rule["antecedents"], rule["consequent"])
            rows.append({"rule_id": rule["rule_id"], "antecedent": " AND ".join(rule["antecedents"]),
                         "consequent": rule["consequent"],
                         **{"train_" + key: value for key, value in rule.items() if key not in ("antecedents", "consequent", "rule_id")},
                         **{"validation_" + key: value for key, value in validated.items()}})
        table = pd.DataFrame(rows)
        table.to_csv(folder / "rules_train_validation.csv", index=False, float_format="%.10f", na_rep="NA")
        # At most three per outcome for review, still ordered by training lift only.
        table.groupby("consequent", sort=False).head(3).to_csv(folder / "top_rules_per_level.csv", index=False, float_format="%.10f", na_rep="NA")
        frequent.assign(itemsets=frequent.itemsets.map(lambda items: " AND ".join(sorted(items)))).sort_values("support",ascending=False).to_csv(folder / "frequent_itemsets.csv", index=False, float_format="%.10f")
        examples = []
        for row_number in range(10):
            row = train.iloc[row_number]
            basket = training_baskets.iloc[row_number]
            examples.append({"date_time": row.date_time, "traffic_volume": int(row.traffic_volume),
                             "items": basket.index[basket].tolist()})
        (folder / "transaction_examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
        training_baskets.head(10).astype(int).assign(date_time=train.date_time.head(10).to_numpy()).to_csv(folder / "transaction_encoding_examples.csv", index=False)
        (folder / "frozen_rules.json").write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")
        manifest = {"source_sha256": digest, "train_hours": len(train), "validation_hours": len(validation),
                    "test_transactions_created": False, "algorithm": "FP-growth", "mlxtend_version": mlxtend.__version__,
                    "item_catalog": training_baskets.columns.tolist(), "time_bins": list(TIME_BINS),
                    "congestion_labels": LEVELS, "thresholds": metadata["quartile_thresholds"],
                    "criteria": {"min_support": MIN_SUPPORT, "min_joint_hours": math.ceil(MIN_SUPPORT*len(train)),
                                 "min_antecedent_hours": MIN_ANTECEDENT_HOURS, "min_confidence": MIN_CONFIDENCE,
                                 "min_lift": MIN_LIFT, "max_antecedent_items": 3},
                    "frequent_itemsets": len(frequent), "selected_rules": len(rules), "mlflow_run_id": run.info.run_id,
                    "ranking": "Training lift descending, then training support descending; never validation selection",
                    "limitations": ["Descriptive co-occurrence, not causation or a forecast", "Weather items overlap; one observed hour is one transaction",
                                    "Repeated nested antecedents are not independent discoveries", "Minimum support suppresses rare weather; absent rules do not imply safety",
                                    "Fixed time bins and thresholds affect discoveries", "No multiplicity-adjusted significance claim"]}
        (folder / "mining_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({"frequent_itemsets": len(frequent), "selected_rules": len(rules),
                            "highest_train_lift": float(table.train_lift.max())})
        for path in folder.iterdir():
            if path.suffix in (".csv", ".json"):
                mlflow.log_artifact(str(path), "rules")
        logger.info("Saved %d train-selected rules from %d frequent itemsets; fixed-rule validation only",len(rules),len(frequent))
    completed = client.get_run(manifest["mlflow_run_id"])
    if completed.info.status != "FINISHED":
        raise ValueError("MLflow rule experiment did not finish")
    (folder / "mlflow_evidence.json").write_text(json.dumps({"run_id": completed.info.run_id, "status": completed.info.status,
                                                            "params": completed.data.params, "metrics": completed.data.metrics},indent=2)+"\n",encoding="utf-8")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source changed during mining")
    return table


if __name__ == "__main__":
    configure_logging(BASE / "association_rules.log")
    try:
        run_mining()
    except Exception:
        logger.error("Association rule mining failed",exc_info=True)
        sys.exit(1)
