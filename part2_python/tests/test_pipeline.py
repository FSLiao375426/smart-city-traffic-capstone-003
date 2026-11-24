"""Checks for data-loss risks, conflicting observations and temporal leakage."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from part2_python.pipeline import (
    WEATHER_NUMERIC, build_hourly_tables, impute_monthly,
    main, prepare_observations, validate_schema,
)

logger = logging.getLogger(__name__)


def observation(time, weather="Clear", volume=0, temperature=280.0):
    return {"holiday": "None", "temp": temperature, "rain_1h": 0.0,
            "snow_1h": 0.0, "clouds_all": 0, "weather_main": weather,
            "weather_description": weather.lower(), "date_time": time,
            "traffic_volume": volume}


class PipelineTests(unittest.TestCase):
    def test_schema_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_schema(pd.DataFrame({"traffic_volume": [10]}))

    def test_one_hour_keeps_zero_traffic_and_all_weather_labels(self):
        row = observation("2016-01-01 00:00:00")
        cloudy = observation("2016-01-01 00:00:00", "Clouds")
        prepared, report = prepare_observations(pd.DataFrame([row, row, cloudy]))
        hourly, weather, _ = build_hourly_tables(prepared)
        self.assertEqual(report["exact_duplicates_removed"], 1)
        self.assertEqual(len(hourly), 1)
        self.assertEqual(int(hourly["traffic_volume"].iloc[0]), 0)
        self.assertEqual(set(weather["weather_main"]), {"Clear", "Clouds"})
        self.assertEqual(int(hourly["weather_Clear"].iloc[0]), 1)
        self.assertEqual(int(hourly["weather_Clouds"].iloc[0]), 1)

    def test_conflicting_traffic_is_not_silently_averaged(self):
        rows = [observation("2016-01-01 00:00:00", volume=100),
                observation("2016-01-01 00:00:00", "Rain", volume=200)]
        with self.assertRaisesRegex(ValueError, "Conflicting traffic_volume"):
            prepare_observations(pd.DataFrame(rows))

    def test_repeated_weather_descriptions_do_not_reweight_numeric_median(self):
        rows = [observation("2016-01-01 00:00:00", temperature=270),
                observation("2016-01-01 00:00:00", "Fog", temperature=270),
                observation("2016-01-01 00:00:00", "Clouds", temperature=290)]
        prepared, _ = prepare_observations(pd.DataFrame(rows))
        hourly, weather, _ = build_hourly_tables(prepared)
        self.assertEqual(float(hourly["temp"].iloc[0]), 280.0)
        self.assertEqual(int(hourly["numeric_readings_differ"].iloc[0]), 1)
        self.assertEqual(len(weather), 3)

    def test_monthly_imputation_excludes_held_out_future(self):
        times = pd.to_datetime(["2016-01-01", "2016-01-02", "2016-02-01", "2018-01-01", "2018-01-02"])
        hourly = pd.DataFrame(index=times)
        hourly["temp"] = [280.0, 284.0, 250.0, 330.0, float("nan")]
        for column in WEATHER_NUMERIC[1:]:
            hourly[column] = 0.0
        result, parameters = impute_monthly(hourly, pd.Timestamp("2016-12-31 23:59:59"))
        self.assertEqual(result.loc[times[-1], "temp"], 282.0)
        self.assertEqual(parameters["columns"]["temp"]["monthly_medians"]["1"], 282.0)
        self.assertEqual(parameters["columns"]["temp"]["monthly_medians"]["2"], 250.0)

    def test_known_holiday_extends_only_within_its_date(self):
        rows = [observation("2016-01-01 00:00:00"), observation("2016-01-01 01:00:00"),
                observation("2016-01-02 00:00:00")]
        rows[0]["holiday"] = "New Years Day"
        prepared, _ = prepare_observations(pd.DataFrame(rows))
        hourly, _, _ = build_hourly_tables(prepared)
        self.assertEqual(hourly["is_holiday"].tolist(), [1, 1, 0])

    def test_log_path_cannot_overwrite_raw_input(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.csv"
            path.write_text("original input\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as result:
                main(["--input", str(path), "--log-file", str(path)])
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "original input\n")


if __name__ == "__main__":
    unittest.main()
