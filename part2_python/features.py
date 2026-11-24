"""Add calendar and cyclic-hour features without altering cleaned observations.

Run after cleaning with: python -m part2_python.features
"""

from __future__ import annotations

import hashlib
import logging
import sys

import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging

logger = logging.getLogger(__name__)
TIME_FEATURE_COLUMNS = ("hour", "day_of_week", "is_weekend")
CYCLIC_HOUR_COLUMNS = ("hour_sin", "hour_cos")


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


def main() -> int:
    """Read the cleaned table and save a separate table plus an execution log."""
    configure_logging(PROJECT_ROOT / "part2_python" / "features.log")
    input_path = PROJECT_ROOT / "data" / "processed" / "traffic_hourly.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "traffic_time_features.csv"
    try:
        source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        hourly = pd.read_csv(input_path, keep_default_na=False)
        logger.info("Loaded cleaned hourly table: %d rows, %d columns", *hourly.shape)
        result = add_time_features(hourly)
        result = add_cyclic_hour_features(result)
        if not result.loc[:, hourly.columns].equals(hourly):
            raise RuntimeError("An original observation changed during feature creation")
        result.to_csv(output_path, index=False, float_format="%.8f")
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
