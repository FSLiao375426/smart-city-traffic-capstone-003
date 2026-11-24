"""Validate observations and build one traffic row per observed hour.

Run from the project root with: python -m part2_python.pipeline
The input CSV is never overwritten. No missing hours are fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = (
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
)
WEATHER_NUMERIC = ("temp", "rain_1h", "snow_1h", "clouds_all")
WEATHER_TYPES = (
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain",
    "Smoke", "Snow", "Squall", "Thunderstorm",
)
DEFAULT_FIT_END = "2016-12-31 23:59:59"
OUTPUT_FILENAMES = (
    "traffic_hourly.csv", "hourly_weather.csv", "coverage_by_year.csv",
    "time_gaps.csv", "quality_report.json", "imputation_values.json",
)


def configure_logging(path: Path, level: str = "INFO") -> None:
    """Configure console and file handlers only from the CLI entry point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime
    handlers = [logging.StreamHandler(), logging.FileHandler(path, mode="w", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level), handlers=handlers, force=True)
    # Keep application DEBUG values useful without third-party font diagnostics.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def validate_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    logger.info("Schema validated: all %d required columns are present", len(EXPECTED_COLUMNS))


def load_observations(path: Path) -> pd.DataFrame:
    """Preserve the literal holiday label 'None', rather than treating it as NA."""
    frame = pd.read_csv(path, keep_default_na=False)
    logger.info("Loaded raw CSV: %d rows, %d columns", *frame.shape)
    validate_schema(frame)  # 檢查欄位後，才進行其他清理。
    return frame


def unique_labels(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values if str(value)}))


def prepare_observations(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardise, deduplicate and mark invalid readings as missing."""
    validate_schema(frame)
    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    report: dict = {"raw_rows": len(frame), "raw_columns": len(frame.columns)}

    categorical_changes = 0
    for column in ("holiday", "weather_main", "weather_description"):
        before = frame[column].astype(str)
        after = before.str.strip().str.replace(r"\s+", " ", regex=True)
        if column == "holiday":
            after = after.mask(after.str.casefold().isin(["none", ""]), "None")
        elif column == "weather_main":
            mapping = {value.casefold(): value for value in WEATHER_TYPES}
            mapped = after.str.casefold().map(mapping)
            if mapped.isna().any():
                raise ValueError(f"Unknown weather_main labels: {sorted(after[mapped.isna()].unique())}")
            after = mapped
        else:
            after = after.str.casefold()
        changed = int(before.ne(after).sum())
        frame[column] = after
        categorical_changes += changed
        if changed:
            logger.warning("Standardised %d rows in %s: inconsistent categorical text", changed, column)
        else:
            logger.info("Categorical text checked: %s; no changes", column)
    report["categorical_cells_standardised"] = categorical_changes

    parsed_dates = pd.to_datetime(frame["date_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"Invalid date_time values: {int(parsed_dates.isna().sum())}")
    if parsed_dates.ne(parsed_dates.dt.floor("h")).any():
        raise ValueError("date_time contains non-hourly timestamps; no rounding was applied")
    frame["date_time"] = parsed_dates
    logger.info("Parsed and validated %d hourly timestamps", len(frame))

    for column in (*WEATHER_NUMERIC, "traffic_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    target = frame["traffic_volume"]
    invalid_target = ~np.isfinite(target) | target.lt(0) | target.mod(1).ne(0)
    if invalid_target.any():
        raise ValueError(f"Invalid traffic_volume values: {int(invalid_target.sum())}; target was not imputed")
    frame["traffic_volume"] = target.astype("int64")

    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    report["exact_duplicates_removed"] = duplicates
    if duplicates:
        logger.warning("Dropped %d duplicate rows: all nine values identical after standardisation", duplicates)
    else:
        logger.info("Exact duplicate check completed: no rows dropped")

    groups = frame.groupby("date_time", sort=True)
    conflict_counts = {column: int(groups[column].nunique(dropna=False).gt(1).sum())
                       for column in (*WEATHER_NUMERIC, "traffic_volume")}
    report["numeric_conflicting_hours"] = conflict_counts
    if conflict_counts["traffic_volume"]:
        raise ValueError("Conflicting traffic_volume values within an hour; manual review is required")
    logger.info("Traffic consistency checked: same-hour traffic volumes agree")

    invalid_counts = {}
    for column in WEATHER_NUMERIC:
        values = frame[column]
        invalid = ~np.isfinite(values)
        if column == "temp":
            invalid |= values.le(0)
        elif column == "rain_1h":
            invalid |= values.lt(0) | values.gt(9000)
        elif column == "snow_1h":
            invalid |= values.lt(0)
        else:
            invalid |= values.lt(0) | values.gt(100)
        affected = int(invalid.sum())
        invalid_counts[column] = affected
        frame.loc[invalid, column] = np.nan
        if affected:
            logger.warning("Marked %d rows invalid in %s: missing/non-finite or rule violation", affected, column)
        else:
            logger.info("Validity checks completed: %s; no readings changed", column)
    report["invalid_observation_counts"] = invalid_counts
    return frame.sort_values("date_time", kind="stable"), report


def build_hourly_tables(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Separate traffic at hourly grain from possibly multiple weather labels."""
    groups = observations.groupby("date_time", sort=True)
    if groups["traffic_volume"].nunique().gt(1).any():
        raise ValueError("Cannot collapse conflicting same-hour traffic values")
    hourly = groups.agg(traffic_volume=("traffic_volume", "first"), source_row_count=("traffic_volume", "size"))

    # 先去除同小時重複的數值組合，避免天氣描述的筆數為數值加權。
    readings = observations.loc[:, ["date_time", *WEATHER_NUMERIC]].drop_duplicates()
    hourly = hourly.join(readings.groupby("date_time")[list(WEATHER_NUMERIC)].median())
    hourly["numeric_readings_differ"] = readings.groupby("date_time").size().gt(1).astype("int8")
    for column in WEATHER_NUMERIC:
        hourly[f"{column}_imputed"] = hourly[column].isna().astype("int8")

    weather = (observations.groupby(["date_time", "weather_main"], sort=True)["weather_description"]
               .agg(unique_labels).rename("weather_descriptions").reset_index())
    indicators = pd.crosstab(weather["date_time"], weather["weather_main"]).clip(upper=1)
    for label in WEATHER_TYPES:
        hourly[f"weather_{label}"] = indicators.reindex(hourly.index)[label].fillna(0).astype("int8") if label in indicators else 0
    hourly["weather_main_labels"] = groups["weather_main"].agg(unique_labels)
    hourly["weather_description_labels"] = groups["weather_description"].agg(unique_labels)
    hourly["n_weather_conditions"] = weather.groupby("date_time").size().reindex(hourly.index)

    original_holiday = groups["holiday"].agg(lambda values: unique_labels(values[values.ne("None")]))
    original_holiday = original_holiday.mask(original_holiday.eq(""), "None")
    dated_holidays = observations.loc[observations["holiday"].ne("None"), ["date_time", "holiday"]].copy()
    dated_holidays["date"] = dated_holidays["date_time"].dt.date
    holiday_by_date = dated_holidays.groupby("date")["holiday"].agg(unique_labels)
    mapped_holidays = pd.Series(hourly.index.date, index=hourly.index).map(holiday_by_date)
    hourly["holiday"] = mapped_holidays.astype("string").fillna("None")
    hourly["is_holiday"] = hourly["holiday"].ne("None").astype("int8")
    changed_holidays = int(original_holiday.ne(hourly["holiday"]).sum())
    if changed_holidays:
        logger.warning("Updated %d hourly holiday labels: propagated known labels within the same calendar date", changed_holidays)

    extra_rows = len(observations) - len(hourly)
    if extra_rows:
        logger.warning("Collapsed %d additional weather-observation rows to %d unique traffic hours; weather labels retained separately", extra_rows, len(hourly))
    logger.info("Hourly tables built: %d traffic hours and %d hour/weather pairs", len(hourly), len(weather))
    if not hourly.index.is_unique:
        raise ValueError("Hourly date_time index is not unique")
    report = {
        "rows_after_exact_deduplication": len(observations),
        "unique_observed_hours": len(hourly),
        "hour_weather_pairs": len(weather),
        "holiday_labels_extended_hours": changed_holidays,
        "holiday_hours": int(hourly["is_holiday"].sum()),
        "zero_traffic_hours_preserved": int(hourly["traffic_volume"].eq(0).sum()),
        "clear_clouds_overlap_hours": int((hourly["weather_Clear"].eq(1) & hourly["weather_Clouds"].eq(1)).sum()),
    }
    return hourly, weather, report


def impute_monthly(hourly: pd.DataFrame, fit_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Fit monthly medians only on the reference period, then apply them."""
    hourly = hourly.copy()
    reference = hourly.loc[hourly.index <= fit_end]
    if reference.empty:
        raise ValueError("No observations are available on or before fit_end")
    parameters: dict = {"fit_end": str(fit_end), "reference_hours": len(reference), "columns": {}}
    # 明確依月份迴圈；補值的統計量不使用保留的 2017/2018 資料。
    for column in WEATHER_NUMERIC:
        monthly = reference[column].groupby(reference.index.month).median()
        fallback = reference[column].median()
        if pd.isna(fallback):
            raise ValueError(f"No valid reference readings for {column}")
        parameters["columns"][column] = {
            "monthly_medians": {str(int(month)): float(value) for month, value in monthly.dropna().items()},
            "fallback_median": float(fallback),
        }
        logger.debug("Monthly imputation parameters for %s: %s", column, parameters["columns"][column])
        for month in range(1, 13):
            mask = hourly.index.month == month
            missing = mask & hourly[column].isna()
            count = int(missing.sum())
            if not count:
                continue
            value = monthly.get(month, np.nan)
            if pd.isna(value):
                value = fallback
                logger.warning("Month %d has no valid %s reference readings; using reference-period global median", month, column)
            hourly.loc[missing, column] = float(value)
            logger.warning("Imputed %d hourly %s readings in month %d using reference-period median %.4f", count, column, month, value)
    if hourly[list(WEATHER_NUMERIC)].isna().any().any():
        raise ValueError("Weather fields remain missing after imputation")
    return hourly, parameters


def coverage_tables(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    coverage_rows = []
    for year, data in hourly.groupby(hourly.index.year):
        start = pd.Timestamp(year=int(year), month=1, day=1)
        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
        calendar_hours = int((end - start).total_seconds() / 3600)
        coverage_rows.append({
            "year": int(year), "observed_hours": len(data),
            "full_calendar_year_hours": calendar_hours,
            "calendar_coverage_pct": 100 * len(data) / calendar_hours,
            "first_observed_hour": str(data.index.min()), "last_observed_hour": str(data.index.max()),
            "traffic_sum_observed": int(data["traffic_volume"].sum()),
            "traffic_mean_per_observed_hour": float(data["traffic_volume"].mean()),
        })
    gaps = []
    for previous, current in zip(hourly.index[:-1], hourly.index[1:]):
        hours = int((current - previous).total_seconds() / 3600)
        if hours > 1:
            gaps.append({"missing_start": str(previous + pd.Timedelta(hours=1)),
                         "missing_end": str(current - pd.Timedelta(hours=1)), "missing_hours": hours - 1})
    periods = pd.period_range(hourly.index.min().to_period("M"), hourly.index.max().to_period("M"), freq="M")
    observed = set(hourly.index.to_period("M"))
    missing_months = [str(period) for period in periods if period not in observed]
    gap_frame = pd.DataFrame(gaps, columns=["missing_start", "missing_end", "missing_hours"])
    return pd.DataFrame(coverage_rows), gap_frame, missing_months


def run_pipeline(input_path: Path, output_dir: Path, fit_end: pd.Timestamp) -> dict:
    if input_path.resolve() in {(output_dir / name).resolve() for name in OUTPUT_FILENAMES}:
        raise ValueError("Refusing to overwrite the raw input with a generated output")
    raw_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    observations = load_observations(input_path)
    observations, report = prepare_observations(observations)
    hourly, weather, hourly_report = build_hourly_tables(observations)
    report.update(hourly_report)
    hourly, parameters = impute_monthly(hourly, fit_end)
    coverage, gaps, missing_months = coverage_tables(hourly)
    report.update({
        "input_file": input_path.name, "input_sha256": raw_hash,
        "fit_end": str(fit_end), "date_min": str(hourly.index.min()), "date_max": str(hourly.index.max()),
        "missing_full_months": missing_months,
        "missing_hours_inside_observed_range": int(gaps["missing_hours"].sum()),
        "longest_missing_interval_hours": int(gaps["missing_hours"].max()) if not gaps.empty else 0,
        "hourly_imputation_counts": {column: int(hourly[f"{column}_imputed"].sum()) for column in WEATHER_NUMERIC},
    })
    logger.warning("Coverage audit: %d full months absent and %d unobserved hours inside the time range; no hours fabricated", len(missing_months), report["missing_hours_inside_observed_range"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"traffic_hourly.csv": hourly.reset_index(), "hourly_weather.csv": weather,
              "coverage_by_year.csv": coverage, "time_gaps.csv": gaps}
    for filename, table in tables.items():
        path = output_dir / filename
        if path.resolve() == input_path.resolve():
            raise ValueError("Refusing to overwrite the raw input")
        table.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.8f")
        logger.info("Saved %s: %d rows, %d columns", path.relative_to(output_dir.parent), *table.shape)
    for filename, payload in {"quality_report.json": report, "imputation_values.json": parameters}.items():
        (output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved processing metadata: %s", filename)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != raw_hash:
        raise RuntimeError("Raw input changed during processing")
    logger.info("Pipeline completed: %d observed hours; raw input SHA-256 unchanged", len(hourly))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "Metro_Interstate_Traffic_Volume.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--log-file", type=Path, default=PROJECT_ROOT / "part2_python" / "pipeline.log")
    parser.add_argument("--fit-end", default=DEFAULT_FIT_END)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--all", action="store_true", help="Run cleaning, features and all three figures with one event log")
    args = parser.parse_args(argv)
    if args.all and args.output_dir.resolve() != (PROJECT_ROOT / "data/processed").resolve():
        parser.error("--all requires the default output directory used by downstream modules")
    output_paths = {(args.output_dir / name).resolve() for name in OUTPUT_FILENAMES}
    if args.all:
        output_paths.update((PROJECT_ROOT / "data/processed" / name).resolve() for name in (
            "traffic_time_features.csv", "traffic_scaled_features.csv", "traffic_model_features.csv",
            "scaling_values.json", "quartile_values.json"))
        output_paths.update(path.resolve() for path in (PROJECT_ROOT / "part2_python/figures").glob("*"))
    if args.input.resolve() in output_paths | {args.log_file.resolve()}:
        parser.error("The raw input must differ from all generated output and log paths")
    if args.log_file.resolve() in output_paths:
        parser.error("The log path must differ from the generated output paths")
    configure_logging(args.log_file, args.log_level)
    try:
        run_pipeline(args.input, args.output_dir, pd.Timestamp(args.fit_end))
        if args.all:
            from part2_python import features, visualize
            for flags in ([], ["--scale"], ["--quartiles"]):
                logger.info("Starting feature stage: %s", flags or "time only")
                if features.main(flags, configure=False) != 0:
                    raise RuntimeError("Feature stage failed")
            logger.info("Starting visualization stage")
            if visualize.main(configure=False) != 0:
                raise RuntimeError("Visualization stage failed")
            logger.info("End-to-end pipeline completed: raw CSV, cleaning, features and three figures")
    except Exception:
        # CLI 邊界攔截並記錄失敗；不是未指定型別的 bare except。
        logger.error("Pipeline failed; no successful completion was recorded", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
