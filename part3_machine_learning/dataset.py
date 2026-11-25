"""Build conditional same-hour supervised data without changing Part 2 outputs."""
import hashlib
import json
import logging
import sys

import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, WEATHER_TYPES, configure_logging

logger = logging.getLogger(__name__)
SEVERE_WEATHER = ("Snow", "Thunderstorm", "Squall")
LOW_VISIBILITY = ("Fog", "Mist", "Haze", "Smoke")
FEATURES = (
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_holiday", "temp_z", "clouds_all_z", "rain_1h", "snow_1h",
    *(f"weather_{name}" for name in WEATHER_TYPES),
)
FORBIDDEN_FEATURES = {"traffic_volume", "traffic_volume_quartile", "high_risk",
                      "is_severe_weather", "is_low_visibility", "risky_weather", "split"}


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Map already fitted quartile levels and nonexclusive weather flags to a proxy."""
    times = pd.to_datetime(frame.date_time, errors="raise")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Expected unique chronologically ordered observed hours")
    if not frame.traffic_volume_quartile.isin([1, 2, 3, 4]).all():
        raise ValueError("Quartile levels must be 1 to 4")
    weather_columns = [f"weather_{name}" for name in WEATHER_TYPES]
    if not frame[weather_columns].isin([0, 1]).all().all():
        raise ValueError("Weather flags must be binary")
    result = frame.copy()
    weekday = times.dt.day_of_week
    result["day_of_week_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["is_severe_weather"] = result[[f"weather_{name}" for name in SEVERE_WEATHER]].eq(1).any(axis=1).astype("int8")
    result["is_low_visibility"] = result[[f"weather_{name}" for name in LOW_VISIBILITY]].eq(1).any(axis=1).astype("int8")
    result["risky_weather"] = (result.is_severe_weather.eq(1) | result.is_low_visibility.eq(1)).astype("int8")
    result["high_risk"] = (result.traffic_volume_quartile.ge(3) & result.risky_weather.eq(1)).astype("int8")
    result["split"] = np.select([times.lt("2017-01-01"), times.lt("2018-01-01")], ["train", "validation"], default="test")
    if times.ge("2019-01-01").any():
        raise ValueError("Unexpected data after the specified 2018 test period")
    if not np.isfinite(result.loc[:, FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite")
    if FORBIDDEN_FEATURES.intersection(FEATURES):
        raise RuntimeError("Target-derived input detected")
    return result


def build_dataset():
    folder = PROJECT_ROOT / "data/processed"
    source = folder / "traffic_model_features.csv"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source, keep_default_na=False)
    result = add_targets(frame)
    train = result.split.eq("train")
    scaling = json.loads((folder / "scaling_values.json").read_text())
    quartiles = json.loads((folder / "quartile_values.json").read_text())
    imputation = json.loads((folder / "imputation_values.json").read_text())
    fit_end = "2016-12-31 23:59:59"
    if any(item["fit_end"] != fit_end for item in (scaling, quartiles, imputation)):
        raise ValueError("All preprocessing reference cutoffs must match the training period")
    actual_thresholds = result.loc[train, "traffic_volume"].quantile([.25, .5, .75]).to_numpy()
    np.testing.assert_allclose(actual_thresholds, list(quartiles["thresholds"].values()), rtol=0, atol=1e-10)
    expected_levels = np.searchsorted(actual_thresholds, result.traffic_volume.to_numpy(), side="left") + 1
    np.testing.assert_array_equal(result.traffic_volume_quartile, expected_levels)
    for column in ("temp", "clouds_all"):
        reference = result.loc[train, column]
        stats = scaling["columns"][column]
        np.testing.assert_allclose([stats["mean"], stats["std"]], [reference.mean(), reference.std(ddof=0)], rtol=0, atol=1e-10)
        np.testing.assert_allclose(result[column + "_z"], (result[column] - stats["mean"]) / stats["std"], rtol=0, atol=5.1e-9)
    pd.testing.assert_frame_equal(result.loc[:, frame.columns], frame)
    output = folder / "traffic_supervised.csv"
    result.to_csv(output, index=False, float_format="%.8f")
    metadata = {
        "prediction_mode": "Conditional same-hour estimate with observed weather; not a forward forecast",
        "source": str(source.relative_to(PROJECT_ROOT)), "source_sha256": before,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "fit_end": fit_end, "features": list(FEATURES),
        "classification_target": "high_risk = (quartile >= 3) AND risky_weather",
        "regression_target": "traffic_volume", "severe_weather": list(SEVERE_WEATHER),
        "low_visibility_weather": list(LOW_VISIBILITY), "quartile_thresholds": quartiles["thresholds"],
        "shape": list(result.shape), "split_counts": result.split.value_counts().to_dict(),
        "pre_test_target_summary": {},
        "limitations": ["Constructed proxy, not real accident outcomes or probabilities",
                        "Weather used in the proxy also appears as inputs: performance partly reproduces the label rule",
                        "Low visibility is a category-based assumption, not a measured visibility value",
                        "Missing hours are absent; no lag features or row-based next-hour targets"],
    }
    for split in ("train", "validation"):
        part = result.loc[result.split.eq(split)]
        if part.high_risk.nunique() != 2:
            raise ValueError(f"Both proxy classes required in {split}")
        metadata["pre_test_target_summary"][split] = {
            "positive_hours": int(part.high_risk.sum()), "positive_fraction": float(part.high_risk.mean())}
        logger.info("%s: %d hours, %d proxy positives", split, len(part), part.high_risk.sum())
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("Source features changed")
    (PROJECT_ROOT / "part3_machine_learning/results").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "part3_machine_learning/results/dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s: %d rows, %d columns; %d common inputs; no test metrics computed", output.name, *result.shape, len(FEATURES))
    return result, metadata


if __name__ == "__main__":
    configure_logging(PROJECT_ROOT / "part3_machine_learning/dataset.log")
    try:
        build_dataset()
    except Exception:
        logger.error("Supervised data preparation failed", exc_info=True)
        sys.exit(1)
