"""Independent standard-library and SQL checks of the saved real-data analysis."""
import csv
from fractions import Fraction
import json
import sqlite3
import statistics
import unittest

import pandas as pd

from part2_python.pipeline import PROJECT_ROOT


class SavedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_ROOT / "data/processed/traffic_hourly.csv").open(newline="") as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.folder = PROJECT_ROOT / "part1_data_analytics/results"

    def test_population_statistics_and_observed_correlation(self):
        volumes = [int(r["traffic_volume"]) for r in self.rows]
        result = pd.read_csv(self.folder / "descriptive_statistics.csv").set_index("metric").value
        for key, expected in (("mean", statistics.mean(volumes)), ("median", statistics.median(volumes)),
                              ("population_std_ddof0", statistics.pstdev(volumes)),
                              ("population_variance_ddof0", statistics.pvariance(volumes))):
            self.assertAlmostEqual(result[key], expected, delta=1e-8)
        observed = [r for r in self.rows if r["temp_imputed"] == "0"]
        correlation = statistics.correlation([float(r["temp"]) for r in observed], [int(r["traffic_volume"]) for r in observed])
        saved = pd.read_csv(self.folder / "temperature_correlation.csv").iloc[0]
        self.assertEqual(int(saved.paired_hours), 40565)
        self.assertAlmostEqual(saved.pearson_r, correlation, delta=1e-11)

    def test_sql_probabilities_and_exact_odds(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE hours (v INTEGER, c INTEGER, k INTEGER, t REAL)")
            connection.executemany("INSERT INTO hours VALUES (?,?,?,?)", [
                (int(r["traffic_volume"]), int(r["weather_Clear"]), int(r["weather_Clouds"]), float(r["temp"])) for r in self.rows])
            counts = connection.execute("SELECT count(*),sum(v>5500),sum(c=1),sum(v>5500 AND c=1),sum(v>5500 AND t>292) FROM hours").fetchone()
            self.assertEqual(counts, (40575, 6107, 13371, 1763, 1699))
            groups = connection.execute("SELECT c,k,sum(v>5500),sum(v<=5500) FROM hours WHERE c<>k GROUP BY c,k ORDER BY c DESC").fetchall()
        self.assertEqual(groups, [(1, 0, 1762, 11602), (0, 1, 2586, 12534)])
        summary = json.loads((self.folder / "statistical_validation.json").read_text())
        expected = Fraction(1762 * 12534, 11602 * 2586)
        self.assertAlmostEqual(summary["clear_vs_clouds_crude_odds_ratio"], float(expected), delta=1e-12)
        saved = pd.read_csv(self.folder / "probabilities.csv")
        expected_rates = [Fraction(6107,40575), Fraction(13371,40575), Fraction(1763,40575),
                          Fraction(1763,6107), Fraction(1763,13371), Fraction(1699,6107)]
        for value, fraction in zip(saved.probability, expected_rates):
            self.assertAlmostEqual(value, float(fraction), delta=1e-11)

    def test_standardization_weights_and_saved_regression_values(self):
        strata = pd.read_csv(self.folder / "weather_strata.csv")
        rates = pd.read_csv(self.folder / "weather_standardized_rates.csv")
        expected = {("hour", "clear_without_clouds"): .152798557348,
                    ("hour", "clouds_without_clear"): .152303632743,
                    ("hour_weekend", "clear_without_clouds"): .156450559875,
                    ("hour_weekend", "clouds_without_clear"): .149887561203}
        for row in rates.itertuples():
            part = strata.loc[strata.method.eq(row.method) & strata.group.eq(row.group)]
            self.assertAlmostEqual(part.common_weight.sum(), 1, delta=1e-10)
            computed = ((part.congested_hours / part.observed_hours) * part.common_weight).sum()
            self.assertAlmostEqual(computed, row.standardized_congestion_probability, delta=1e-10)
            self.assertAlmostEqual(row.standardized_congestion_probability, expected[(row.method,row.group)], delta=1e-11)


if __name__ == "__main__":
    unittest.main()
