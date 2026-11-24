"""Query boundaries, absence and user error behaviour."""
import argparse
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import pandas as pd

from part2_python.cli import main, query, parse_time


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "date_time": ["2017-01-06 23:00:00", "2017-01-07 00:00:00", "2017-01-07 23:00:00", "2017-01-08 00:00:00"],
            "traffic_volume": [0, 100, 300, 400], "temp": [273.15] * 4,
            "clouds_all": [50] * 4, "traffic_volume_quartile": [1, 1, 2, 3]})

    def test_inclusive_date_range_and_counts(self):
        result = query(self.frame, argparse.Namespace(command="summary", start="2017-01-07", end="2017-01-07"))
        self.assertEqual(result["observed_hours"], 2)
        self.assertEqual(result["mean_traffic"], 200)
        self.assertEqual(result["quartile_counts"], {"1": 1, "2": 1, "3": 0, "4": 0})

    def test_zero_is_observed_and_absence_is_null(self):
        result = query(self.frame, argparse.Namespace(command="at", time="2017-01-06 23:00:00"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["traffic_volume"], 0)
        missing = query(self.frame, argparse.Namespace(command="at", time="2015-01-01 00:00:00"))
        self.assertEqual(missing["status"], "no_data")
        self.assertIsNone(missing["mean_traffic"])

    def test_single_group_comparison(self):
        result = query(self.frame, argparse.Namespace(command="compare", start="2017-01-07", end="2017-01-07"))
        self.assertEqual(result["groups"]["weekday"], {"observed_hours": 0, "mean_traffic": None})
        self.assertEqual(result["groups"]["weekend"]["mean_traffic"], 200)

    def test_invalid_input_has_error_without_traceback(self):
        for flags in (["at", "--time", "bad"], ["unknown"], ["summary", "--start", "2017-02-01", "--end", "2017-01-01"]):
            with patch("part2_python.cli.configure_logging"), patch("part2_python.cli.pd.read_csv", return_value=self.frame):
                with self.assertLogs("part2_python.cli", level="INFO") as events:
                    self.assertEqual(main(flags), 2)
                self.assertTrue(any("ERROR" in line for line in events.output))
                self.assertFalse(any("Traceback" in line for line in events.output))
        for value in ("2017-02-30", "2017-1-1", "2017-01-01junk"):
            with self.assertRaises(ValueError):
                parse_time(value, day=True)

    def test_cli_stdout_is_only_json_answer(self):
        output = io.StringIO()
        with patch("part2_python.cli.configure_logging"), patch("part2_python.cli.pd.read_csv", return_value=self.frame), redirect_stdout(output):
            self.assertEqual(main(["at", "--time", "2017-01-06 23:00:00"]), 0)
        self.assertEqual(json.loads(output.getvalue())["traffic_volume"], 0)


if __name__ == "__main__":
    unittest.main()
