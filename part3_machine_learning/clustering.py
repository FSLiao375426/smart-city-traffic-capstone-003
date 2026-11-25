"""Compare train-only K-means models and describe observed traffic regimes."""
import hashlib
import json
import logging
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from part2_python.pipeline import PROJECT_ROOT, configure_logging
from part3_machine_learning.dataset import SEVERE_WEATHER, LOW_VISIBILITY

logger = logging.getLogger(__name__)
BASE = PROJECT_ROOT / "part3_machine_learning"
CLUSTER_FEATURES = ("traffic_volume", "hour_sin", "hour_cos", "adverse_weather")
SEED = 42


def cluster_inputs(frame):
    """Use a transparent binary weather context, not a measured severity score."""
    values = frame.loc[:, ["traffic_volume", "hour_sin", "hour_cos"]].copy()
    values["adverse_weather"] = frame[["weather_" + name for name in SEVERE_WEATHER + LOW_VISIBILITY]].eq(1).any(axis=1).astype(int)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("K-means inputs must be finite")
    return values


def summarize(frame, assignments):
    data = frame[["date_time", "traffic_volume", "temp", "is_weekend", "split"]].copy()
    data["cluster"] = assignments
    data["hour"] = pd.to_datetime(data.date_time).dt.hour
    data["adverse_weather"] = cluster_inputs(frame).adverse_weather.to_numpy()
    summary = data.groupby(["split", "cluster"]).agg(
        observed_hours=("traffic_volume", "size"), mean_traffic=("traffic_volume", "mean"),
        median_traffic=("traffic_volume", "median"), mean_temp_k=("temp", "mean"),
        adverse_weather_fraction=("adverse_weather", "mean"), weekend_fraction=("is_weekend", "mean")).reset_index()
    summary["split_fraction"] = summary.observed_hours / summary.groupby("split").observed_hours.transform("sum")
    summary["mean_temp_c"] = summary.pop("mean_temp_k") - 273.15
    profile = data.groupby(["split", "cluster", "hour"]).size().rename("observed_hours").reset_index()
    return summary, profile


def run_clustering():
    source = PROJECT_ROOT / "data/processed/traffic_supervised.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = json.loads((BASE / "results/dataset_metadata.json").read_text())
    if source_hash != metadata["output_sha256"]:
        raise ValueError("Prepared dataset does not match metadata")
    all_data = pd.read_csv(source, keep_default_na=False)
    # Keep final-test rows out of all K-means fits, predictions and plots.
    data = all_data.loc[all_data.split.isin(["train", "validation"])].copy()
    del all_data
    train_mask = data.split.eq("train")
    train = data.loc[train_mask]
    x_train = cluster_inputs(train)
    sample_indices = np.random.default_rng(SEED).choice(len(train), size=min(2000, len(train)), replace=False)
    folder = BASE / "results/clustering"
    folder.mkdir(parents=True, exist_ok=True)
    tracking = BASE / "mlflow"
    tracking.mkdir(exist_ok=True)
    uri = "sqlite:///" + (tracking / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    name = "capstone003_traffic_kmeans"
    experiment = client.get_experiment_by_name(name)
    exp_id = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(tracking / "artifacts").as_uri())
    candidates, fitted = [], {}
    logger.info("K-means: %d training hours; k=2..6; same %d training samples for silhouette", len(train), len(sample_indices))
    for k in range(2, 7):
        model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=SEED))
        with mlflow.start_run(experiment_id=exp_id, run_name=f"kmeans_k{k}") as run:
            mlflow.set_tags({"task": "unsupervised_traffic_clustering", "selection_split": "train_through_2016", "dataset_sha256": source_hash})
            mlflow.log_params({"k": k, "n_init": 10, "random_seed": SEED, "train_hours": len(train),
                               "silhouette_sample_hours": len(sample_indices), "feature_count": 4,
                               "fit_end": metadata["fit_end"]})
            with threadpool_limits(limits=2):
                model.fit(x_train)
                scaled = model.named_steps["standardscaler"].transform(x_train)
                labels = model.named_steps["kmeans"].labels_
                score = silhouette_score(scaled[sample_indices], labels[sample_indices])
            sizes = np.bincount(labels, minlength=k)
            metrics = {"train_sample_silhouette": float(score), "train_inertia": float(model.named_steps["kmeans"].inertia_),
                       "minimum_cluster_fraction": float(sizes.min() / len(train))}
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"features": list(CLUSTER_FEATURES), "weather_definition": list(SEVERE_WEATHER + LOW_VISIBILITY),
                             "weather_type": "binary category assumption, not measured severity", "cluster_counts": sizes.tolist()}, "cluster_definition.json")
            candidates.append({"k": k, **metrics, "mlflow_run_id": run.info.run_id})
            fitted[k] = model
            logger.info("k=%d silhouette=%.5f inertia=%.2f minimum share=%.3f", k, score, metrics["train_inertia"], metrics["minimum_cluster_fraction"])
    comparison = pd.DataFrame(candidates)
    best_row = comparison.sort_values(["train_sample_silhouette", "k"], ascending=[False, True]).iloc[0]
    best_k = int(best_row.k)
    model = fitted[best_k]
    labels = model.predict(cluster_inputs(data))
    summary, profile = summarize(data, labels)
    assignments = data[["date_time", "split"]].copy()
    assignments["cluster"] = labels
    comparison.to_csv(folder / "k_comparison.csv", index=False, float_format="%.10f")
    summary.to_csv(folder / "cluster_summary.csv", index=False, float_format="%.10f")
    profile.to_csv(folder / "cluster_hour_profile.csv", index=False)
    assignments.to_csv(folder / "cluster_assignments.csv", index=False)
    model_folder = BASE / "models/kmeans"
    model_folder.mkdir(exist_ok=True)
    model_path = model_folder / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    np.testing.assert_array_equal(joblib.load(model_path).predict(cluster_inputs(data)), labels)
    centers = model.named_steps["standardscaler"].inverse_transform(model.named_steps["kmeans"].cluster_centers_)
    pd.DataFrame(centers, columns=CLUSTER_FEATURES).rename_axis("cluster").reset_index().to_csv(folder / "cluster_centers.csv", index=False, float_format="%.10f")
    evidence = []
    for candidate in candidates:
        run = client.get_run(candidate["mlflow_run_id"])
        if run.info.status != "FINISHED":
            raise ValueError("Unfinished clustering run")
        evidence.append({"k": candidate["k"], "run_id": run.info.run_id, "status": run.info.status,
                         "params": run.data.params, "metrics": run.data.metrics})
    with mlflow.start_run(run_id=str(best_row.mlflow_run_id)):
        mlflow.set_tag("selected_by_train_silhouette", "true")
        mlflow.log_artifact(str(model_path), "selected_model")
        for path in (folder / "k_comparison.csv", folder / "cluster_summary.csv", folder / "cluster_centers.csv"):
            mlflow.log_artifact(str(path), "selected_results")
    manifest = {"selected_k": best_k, "selection_rule": "Highest silhouette on the same 2000 seeded training samples; ties prefer smaller k",
                "features": list(CLUSTER_FEATURES), "source_sha256": source_hash, "fit_end": metadata["fit_end"],
                "sample_indices_sha256": hashlib.sha256(sample_indices.astype("int64").tobytes()).hexdigest(),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "test_assigned": False,
                "tracking_uri": uri, "runs": evidence,
                "limitations": ["Traffic descriptions, not future forecasts or accident classes", "Arbitrary cluster IDs; name only after inspecting profiles",
                                "K-means assumes Euclidean geometry and depends on scaling/feature choice", "Binary adverse weather is a category assumption", "Silhouette is an internal training-sample measure, not external validity"]}
    (folder / "clustering_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (model_folder / "metadata.json").write_text(json.dumps({key: manifest[key] for key in ("selected_k", "features", "fit_end", "limitations")}, indent=2) + "\n", encoding="utf-8")
    save_plot(comparison, profile, best_k)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Source changed during clustering")
    logger.info("Selected k=%d by training silhouette; saved %d train/validation assignments; no 2018 assignment",best_k,len(assignments))
    return manifest


def save_plot(comparison, profile, best_k):
    """Render only the saved training evidence, without another model fit."""
    plot_folder = BASE / "figures"
    plot_folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(comparison.k, comparison.train_sample_silhouette, "o-", color="#176B9A")
    axes[0].set(title="Training-sample silhouette", xlabel="Number of clusters k", ylabel="Silhouette (higher is better)", xticks=range(2, 7))
    train_profile = profile.loc[profile.split.eq("train")]
    for cluster in range(best_k):
        part = train_profile.loc[train_profile.cluster.eq(cluster)].set_index("hour").observed_hours.reindex(range(24), fill_value=0)
        axes[1].plot(part.index, part / part.sum(), label=f"Cluster {cluster}")
    axes[1].set(title=f"Selected k={best_k}: time-of-day profiles", xlabel="Source-clock hour", ylabel="Fraction within each cluster", xticks=range(0,24,4))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01,1))
    fig.suptitle("K-means traffic regimes | fitted through 2016", fontsize=13)
    fig.text(.04,.02,"Four standardized inputs: traffic, hour sin/cos, binary adverse weather. No proxy labels or 2018 test rows used.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    figure = plot_folder / "kmeans_selection_profiles.png"
    fig.savefig(figure,dpi=160)
    plt.close(fig)
    logger.info("Saved figure %s",figure)


if __name__ == "__main__":
    configure_logging(BASE / "clustering.log")
    try:
        run_clustering()
    except Exception:
        logger.error("K-means workflow failed",exc_info=True)
        sys.exit(1)
