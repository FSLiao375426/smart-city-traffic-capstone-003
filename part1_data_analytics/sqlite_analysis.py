"""Load raw/hourly data and run yearly, change and holiday SQLite queries.

Run after cleaning with: python -m part1_data_analytics.sqlite_analysis
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import logging
import sqlite3
import sys

import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging, load_observations

logger = logging.getLogger(__name__)
PART_ROOT = PROJECT_ROOT / "part1_data_analytics"
QUERY_OUTPUTS = (
    ("01_yearly_traffic.sql", "yearly_traffic_2012_2017.csv"),
    ("02_yearly_changes.sql", "yearly_changes_2012_2017.csv"),
    ("03_holiday_temperature.sql", "holiday_temperature_2015_2017.csv"),
)


def run_analysis() -> dict:
    raw_path = PROJECT_ROOT / "Metro_Interstate_Traffic_Volume.csv"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "traffic_hourly.csv"
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (raw_path, hourly_path)}
    raw = load_observations(raw_path)
    hourly = pd.read_csv(hourly_path, keep_default_na=False)
    required = {"date_time", "traffic_volume", "temp", "temp_imputed", "holiday"}
    if missing := sorted(required - set(hourly.columns)):
        raise ValueError(f"Missing hourly columns: {', '.join(missing)}")
    if hourly["date_time"].isna().any() or not hourly["date_time"].is_unique:
        raise ValueError("The hourly table must contain unique, non-missing timestamps")
    traffic = pd.to_numeric(hourly["traffic_volume"], errors="raise")
    if not traffic.ge(0).all() or not traffic.mod(1).eq(0).all():
        raise ValueError("Hourly traffic must contain non-negative integer values")
    logger.info("Validated hourly input: %d rows and %d columns", *hourly.shape)

    database_path = PART_ROOT / "traffic.sqlite"
    results_dir = PART_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    query_metadata = []
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            # 只重建本流程產生的兩張表；原始 CSV 與清理 CSV 保持不變。
            raw.to_sql("traffic_raw", connection, if_exists="replace", index=False,
                       dtype={"date_time": "TEXT", "traffic_volume": "INTEGER"})
            hourly.to_sql("traffic_hourly", connection, if_exists="replace", index=False,
                          dtype={"date_time": "TEXT", "traffic_volume": "INTEGER"})
            connection.execute("CREATE INDEX idx_traffic_raw_time ON traffic_raw(date_time)")
            connection.execute("CREATE UNIQUE INDEX idx_traffic_hourly_time ON traffic_hourly(date_time)")
            raw_rows, raw_hours = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT date_time) FROM traffic_raw").fetchone()
            hourly_rows, hourly_hours = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT date_time) FROM traffic_hourly").fetchone()
            if (raw_rows, hourly_rows, hourly_hours) != (len(raw), len(hourly), len(hourly)):
                raise RuntimeError("SQLite load counts do not match the source tables")
            if raw_hours != hourly_hours:
                raise RuntimeError("Raw and hourly timestamp coverage differs")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
            logger.info("SQLite load verified: %d raw rows; %d distinct hourly rows", raw_rows, hourly_rows)
            for query_name, output_name in QUERY_OUTPUTS:
                query_path = PART_ROOT / "sql" / query_name
                query = query_path.read_text(encoding="utf-8")
                table = pd.read_sql_query(query, connection)
                if query_name == "02_yearly_changes.sql":
                    table[["previous_year", "traffic_change"]] = table[
                        ["previous_year", "traffic_change"]].astype("Int64")
                results[output_name] = table
                query_metadata.append({
                    "file": query_name, "sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
                    "output": output_name, "rows": len(table), "columns": len(table.columns),
                })
                logger.info("Executed %s: %d rows, %d columns", query_name, *table.shape)
            annual = results["yearly_traffic_2012_2017.csv"]
            if annual["year"].tolist() != list(range(2012, 2018)):
                raise ValueError("The annual query did not return all six required years")
            holiday = results["holiday_temperature_2015_2017.csv"]
            if int(holiday["date_basis"].eq("calendar_date").sum()) != 6:
                raise ValueError("The holiday query did not retain all six requested calendar dates")

    for output_name, table in results.items():
        table.to_csv(results_dir / output_name, index=False, float_format="%.2f", na_rep="NA")
        logger.info("Saved %s", output_name)
    for path, digest in source_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input changed during SQLite processing: {path.name}")
    report = {
        "sqlite_version": sqlite3.sqlite_version,
        "sources": [{"file": str(path.relative_to(PROJECT_ROOT)), "sha256": digest}
                    for path, digest in source_hashes.items()],
        "raw_rows": raw_rows, "raw_distinct_hours": raw_hours,
        "hourly_rows": hourly_rows, "hourly_distinct_hours": hourly_hours,
        "sqlite_integrity": integrity, "query_file": "01_yearly_traffic.sql", "queries": query_metadata,
        "output_years": annual["year"].tolist(), "input_files_unchanged": True,
    }
    (results_dir / "load_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("SQLite queries completed: %d annual years and %d holiday/date-basis rows",
                len(annual), len(holiday))
    logger.info("SQLite step completed; input hashes unchanged")
    return report


def main() -> int:
    configure_logging(PART_ROOT / "sqlite_analysis.log")
    try:
        run_analysis()
    except Exception:
        logger.error("SQLite analysis failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
