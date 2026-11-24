"""Query processed traffic: python -m part2_python.cli --help."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import re
import sys

import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging

logger = logging.getLogger(__name__)


class QueryParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parse_time(value: str, *, day: bool = False) -> pd.Timestamp:
    pattern = r"\d{4}-\d{2}-\d{2}" if day else r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    if not re.fullmatch(pattern, value):
        raise ValueError("Use YYYY-MM-DD" if day else "Use YYYY-MM-DD HH:MM:SS")
    stamp = pd.Timestamp(datetime.strptime(value, "%Y-%m-%d" if day else "%Y-%m-%d %H:%M:%S"))
    if not day and (stamp.minute or stamp.second):
        raise ValueError("Queries must specify an exact hour (minutes and seconds = 00)")
    return stamp


def query(frame: pd.DataFrame, args: argparse.Namespace) -> dict:
    times = pd.to_datetime(frame.date_time, errors="raise")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Processed data must contain unique observed timestamps")
    if args.command == "at":
        part = frame.loc[times.eq(parse_time(args.time))]
    else:
        start, end = parse_time(args.start, day=True), parse_time(args.end, day=True)
        if start > end:
            raise ValueError("start must be on or before end")
        part = frame.loc[times.ge(start) & times.lt(end + pd.Timedelta(days=1))]
    if part.empty:
        logger.warning("No observed hours match this query; missing data is not zero traffic")
        return {"status": "no_data", "observed_hours": 0, "mean_traffic": None,
                "message": "No observed data; this does not imply zero traffic."}
    if args.command == "at":
        row = part.iloc[0]
        return {"status": "ok", "date_time": str(row.date_time),
                "traffic_volume": int(row.traffic_volume), "temp_c": float(row.temp - 273.15),
                "clouds_all": float(row.clouds_all),
                "traffic_volume_quartile": int(row.traffic_volume_quartile)}
    if args.command == "summary":
        return {"status": "ok", "observed_hours": len(part),
                "mean_traffic": float(part.traffic_volume.mean()),
                "min_traffic": int(part.traffic_volume.min()), "max_traffic": int(part.traffic_volume.max()),
                "quartile_counts": {str(i): int(part.traffic_volume_quartile.eq(i).sum()) for i in range(1, 5)}}
    weekend = pd.to_datetime(part.date_time).dt.day_of_week.ge(5)
    groups = {}
    for label, mask in (("weekday", ~weekend), ("weekend", weekend)):
        volume = part.loc[mask, "traffic_volume"]
        groups[label] = {"observed_hours": len(volume),
                         "mean_traffic": float(volume.mean()) if len(volume) else None}
    return {"status": "ok", "observed_hours": len(part), "groups": groups,
            "note": "Means per observed hour; year/season composition is not adjusted."}


def main(argv: list[str] | None = None) -> int:
    configure_logging(PROJECT_ROOT / "part2_python" / "cli.log")
    actual = list(sys.argv[1:] if argv is None else argv)
    logger.info("CLI invoked with arguments: %s", actual)
    parser = QueryParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    at = commands.add_parser("at", help="Traffic and weather at an exact observed hour")
    at.add_argument("--time", required=True, help="YYYY-MM-DD HH:MM:SS")
    for name, help_text in (("summary", "Traffic summary and fixed quartile counts"),
                            ("compare", "Weekday/weekend means per observed hour")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD")
        command.add_argument("--end", required=True, help="Inclusive YYYY-MM-DD")
    try:
        args = parser.parse_args(actual)
        frame = pd.read_csv(PROJECT_ROOT / "data/processed/traffic_model_features.csv", keep_default_na=False)
        result = query(frame, args)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        logger.info("CLI command %s completed: %s", args.command, result["status"])
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        logger.error("Cannot complete query: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
