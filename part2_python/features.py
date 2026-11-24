"""Add calendar and cyclic-hour features without altering cleaned observations.

Run after cleaning with: python -m part2_python.features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys

import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging

logger = logging.getLogger(__name__)
TIME_FEATURE_COLUMNS = ("hour", "day_of_week", "is_weekend")
CYCLIC_HOUR_COLUMNS = ("hour_sin", "hour_cos")
SCALE_COLUMNS = ("temp", "clouds_all")


def add_scaled_features(hourly: pd.DataFrame, fit_end: str) -> tuple[pd.DataFrame, dict]:
    """Fit population statistics on the reference period, then transform all rows."""
    times = pd.to_datetime(hourly["date_time"], errors="raise")
    cutoff = pd.Timestamp(fit_end)
    if times.isna().any() or pd.isna(cutoff):
        raise ValueError("Missing timestamp or fit_end")
    reference = times.le(cutoff)
    if not reference.any():
        raise ValueError("Reference period is empty")
    result = hourly.copy()
    metadata = {"fit_end": fit_end, "reference_hours": int(reference.sum()),
                "ddof": 0, "columns": {}}
    for column in SCALE_COLUMNS:
        output = f"{column}_z"
        if output in result.columns:
            raise ValueError(f"Scaled column already exists: {output}")
        values = pd.to_numeric(hourly[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
        mean = float(values.loc[reference].mean())
        std = float(values.loc[reference].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"{column} has no positive reference standard deviation")
        result[output] = (values - mean) / std
        metadata["columns"][column] = {"mean": mean, "std": std}
        logger.info("Scaled %s: reference=%d, fit_end=%s, mean=%.8f, population_std=%.8f",
                    column, reference.sum(), fit_end, mean, std)
    return result, metadata


def add_quartile_levels(hourly: pd.DataFrame, fit_end: str) -> tuple[pd.DataFrame, dict]:
    """Fit traffic quartiles on reference hours; equal thresholds stay in the lower bin.

    The resulting label is a traffic-volume target, never a model input.
    """
    if "traffic_volume_quartile" in hourly.columns:
        raise ValueError("traffic_volume_quartile already exists")
    times = pd.to_datetime(hourly["date_time"], errors="raise")
    cutoff = pd.Timestamp(fit_end)
    if times.isna().any() or pd.isna(cutoff):
        raise ValueError("Missing timestamp or fit_end")
    reference = times.le(cutoff)
    if not reference.any():
        raise ValueError("Reference period is empty")
    volume = pd.to_numeric(hourly["traffic_volume"], errors="raise")
    if not np.isfinite(volume.to_numpy(dtype=float)).all() or volume.lt(0).any():
        raise ValueError("traffic_volume must be finite and nonnegative")
    thresholds = volume.loc[reference].quantile([0.25, 0.5, 0.75], interpolation="linear")
    if not np.all(np.diff(thresholds.to_numpy()) > 0):
        raise ValueError("Distinct quartile thresholds are required for four levels")
    result = hourly.copy()
    result["traffic_volume_quartile"] = (
        np.searchsorted(thresholds.to_numpy(), volume.to_numpy(), side="left") + 1
    ).astype("int8")
    metadata = {"fit_end": fit_end, "reference_hours": int(reference.sum()),
                "source_column": "traffic_volume", "label_column": "traffic_volume_quartile",
                "quantile_interpolation": "linear", "thresholds":
                dict(zip(("q1", "q2", "q3"), map(float, thresholds))),
                "intervals": ["x <= q1", "q1 < x <= q2", "q2 < x <= q3", "x > q3"],
                "meaning": "Traffic-volume proxy levels, not observed accident risk"}
    logger.debug("Traffic quartiles: reference=%d, fit_end=%s, thresholds=%s",
                reference.sum(), fit_end, metadata["thresholds"])
    for name, mask in (("reference", reference), ("holdout", ~reference)):
        logger.info("Quartile counts (%s): %s", name,
                    result.loc[mask, "traffic_volume_quartile"].value_counts().sort_index().to_dict())
    return result, metadata


def add_time_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Keep every input column and row, adding only three calendar fields."""
    if "date_time" not in hourly.columns:
        raise ValueError("Missing required column: date_time")
    present = sorted(set(TIME_FEATURE_COLUMNS) & set(hourly.columns))
    if present:
        raise ValueError(f"Time feature columns already exist: {', '.join(present)}")

    times = pd.to_datetime(hourly["date_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if times.isna().any():
        raise ValueError("date_time contains missing timestamps")
    result = hourly.copy()

    # 保留原始時間欄位；星期一為 0，星期日為 6。
    result["hour"] = times.dt.hour.astype("int8")
    result["day_of_week"] = times.dt.day_of_week.astype("int8")
    result["is_weekend"] = result["day_of_week"].ge(5).astype("int8")

    logger.info("Added %d calendar features to %d rows: %s",
                len(TIME_FEATURE_COLUMNS), len(result), ", ".join(TIME_FEATURE_COLUMNS))
    return result


def add_cyclic_hour_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Represent each integer hour as a point on a fixed 24-hour unit circle."""
    if "hour" not in hourly.columns:
        raise ValueError("Missing required column: hour")
    present = sorted(set(CYCLIC_HOUR_COLUMNS) & set(hourly.columns))
    if present:
        raise ValueError(f"Cyclic hour columns already exist: {', '.join(present)}")
    hours = pd.to_numeric(hourly["hour"], errors="raise")
    if not hours.between(0, 23).all() or hours.mod(1).ne(0).any():
        raise ValueError("hour must contain integers from 0 to 23")

    result = hourly.copy()
    # 一整圈是 24 小時；保留兩個座標，以免只用 sin 時不同時段重合。
    angle = 2 * np.pi * hours / 24
    result["hour_sin"] = np.sin(angle)
    result["hour_cos"] = np.cos(angle)
    logger.info("Added %d cyclic hour features using the fixed 24-hour period",
                len(CYCLIC_HOUR_COLUMNS))
    return result


def main(argv: list[str] | None = None, *, configure: bool = True) -> int:
    """Read the cleaned table and save a separate table plus an execution log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", action="store_true",
                        help="Add temp_z and clouds_all_z using the cleaning reference period")
    parser.add_argument("--quartiles", action="store_true",
                        help="Include scaling and reference-fitted traffic quartile labels")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args(argv)
    args.scale = args.scale or args.quartiles
    if configure:
        configure_logging(PROJECT_ROOT / "part2_python" / "features.log", args.log_level)
    input_path = PROJECT_ROOT / "data" / "processed" / "traffic_hourly.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "traffic_time_features.csv"
    try:
        source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        hourly = pd.read_csv(input_path, keep_default_na=False)
        logger.info("Loaded cleaned hourly table: %d rows, %d columns", *hourly.shape)
        result = add_time_features(hourly)
        result = add_cyclic_hour_features(result)
        if args.scale:
            metadata_path = input_path.parent / "imputation_values.json"
            fit_end = json.loads(metadata_path.read_text(encoding="utf-8"))["fit_end"]
            result, scaling = add_scaled_features(result, fit_end)
            output_path = input_path.parent / "traffic_scaled_features.csv"
        if args.quartiles:
            result, quartiles = add_quartile_levels(result, fit_end)
            output_path = input_path.parent / "traffic_model_features.csv"
        if not result.loc[:, hourly.columns].equals(hourly):
            raise RuntimeError("An original observation changed during feature creation")
        result.to_csv(output_path, index=False, float_format="%.8f")
        if args.scale:
            (input_path.parent / "scaling_values.json").write_text(
                json.dumps(scaling, indent=2) + "\n", encoding="utf-8")
        if args.quartiles:
            (input_path.parent / "quartile_values.json").write_text(
                json.dumps(quartiles, indent=2) + "\n", encoding="utf-8")
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("The cleaned input file changed during feature creation")
        logger.info("Saved %s: %d rows, %d columns; input observations and file unchanged",
                    output_path.name, *result.shape)
    except Exception:
        logger.error("Time feature step failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
