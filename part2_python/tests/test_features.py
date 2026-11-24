"""Reference-period isolation and saved feature acceptance checks."""
import json
import unittest

import numpy as np
import pandas as pd

from part2_python.features import add_scaled_features, add_quartile_levels
from part2_python.pipeline import PROJECT_ROOT


class QuartileTests(unittest.TestCase):
    def test_boundaries_and_holdout_isolation(self):
        frame = pd.DataFrame({"date_time": ["2016-01-01"] * 5 + ["2017-01-01"] * 6,
                              "traffic_volume": [0, 10, 20, 30, 40, 10, 11, 20, 21, 30, 31]})
        result, metadata = add_quartile_levels(frame, "2016-12-31")
        self.assertEqual(metadata["thresholds"], {"q1": 10., "q2": 20., "q3": 30.})
        self.assertEqual(result.traffic_volume_quartile.iloc[5:].tolist(), [1, 2, 2, 3, 3, 4])
        frame.loc[5:, "traffic_volume"] = 99999
        changed, stats = add_quartile_levels(frame, "2016-12-31")
        self.assertEqual(metadata, stats)
        pd.testing.assert_frame_equal(result.iloc[:5], changed.iloc[:5])
        for bad, cutoff in [(frame, "2015-01-01"), (frame.assign(traffic_volume=1), "2016-12-31"),
                            (frame.assign(traffic_volume=np.nan), "2016-12-31"),
                            (frame.assign(traffic_volume=-1), "2016-12-31")]:
            with self.assertRaises(ValueError):
                add_quartile_levels(bad, cutoff)

    def test_saved_quartiles(self):
        folder = PROJECT_ROOT / "data" / "processed"
        original = pd.read_csv(folder / "traffic_scaled_features.csv", keep_default_na=False)
        result = pd.read_csv(folder / "traffic_model_features.csv", keep_default_na=False)
        metadata = json.loads((folder / "quartile_values.json").read_text())
        cleaning = json.loads((folder / "imputation_values.json").read_text())
        self.assertEqual(result.shape, (40575, 36))
        pd.testing.assert_frame_equal(result.loc[:, original.columns], original)
        self.assertEqual(metadata["fit_end"], cleaning["fit_end"])
        ref = result.date_time.le(metadata["fit_end"])
        fitted = np.quantile(result.loc[ref, "traffic_volume"], [.25, .5, .75], method="linear")
        np.testing.assert_array_equal(fitted, list(metadata["thresholds"].values()))
        volume = result.traffic_volume
        q1, q2, q3 = fitted
        expected = np.select([volume.le(q1), volume.le(q2), volume.le(q3)], [1, 2, 3], default=4)
        np.testing.assert_array_equal(result.traffic_volume_quartile, expected)
        self.assertEqual(int(ref.sum()), 25329)
        self.assertEqual(result.traffic_volume_quartile.isna().sum(), 0)


class ScalingTests(unittest.TestCase):
    def test_saved_artifact(self):
        folder = PROJECT_ROOT / "data" / "processed"
        original = pd.read_csv(folder / "traffic_time_features.csv", keep_default_na=False)
        scaled = pd.read_csv(folder / "traffic_scaled_features.csv", keep_default_na=False)
        metadata = json.loads((folder / "scaling_values.json").read_text())
        cleaning = json.loads((folder / "imputation_values.json").read_text())
        self.assertEqual(scaled.shape, (40575, 35))
        pd.testing.assert_frame_equal(scaled.loc[:, original.columns], original)
        self.assertEqual(metadata["fit_end"], cleaning["fit_end"])
        reference = pd.to_datetime(scaled.date_time).le(pd.Timestamp(metadata["fit_end"]))
        self.assertEqual(int(reference.sum()), 25329)
        self.assertEqual(int((~reference).sum()), 15246)
        for column in ("temp", "clouds_all"):
            values = scaled.loc[reference, column + "_z"]
            self.assertAlmostEqual(values.mean(), 0, delta=1e-8)
            self.assertAlmostEqual(values.std(ddof=0), 1, delta=1e-8)
            expected = (scaled[column] - metadata["columns"][column]["mean"]) / metadata["columns"][column]["std"]
            np.testing.assert_allclose(scaled[column + "_z"], expected, atol=5.1e-9, rtol=0)

    def test_holdout_cannot_change_fit(self):
        frame = pd.DataFrame({"date_time": ["2016-01-01", "2016-01-02", "2017-01-01"],
                              "temp": [270., 290., 280.], "clouds_all": [0., 100., 50.]})
        first, stats = add_scaled_features(frame, "2016-12-31 23:59:59")
        frame.loc[2, ["temp", "clouds_all"]] = [9999., -9999.]
        second, changed_stats = add_scaled_features(frame, "2016-12-31 23:59:59")
        self.assertEqual(stats, changed_stats)
        pd.testing.assert_frame_equal(first.iloc[:2], second.iloc[:2])
        for cutoff, bad in [("2015-01-01", frame),
                            ("2016-12-31", frame.assign(temp=270.)),
                            ("2016-12-31", frame.assign(temp=np.nan)),
                            ("2016-12-31", frame.assign(temp=np.inf))]:
            with self.assertRaises(ValueError):
                add_scaled_features(bad, cutoff)


if __name__ == "__main__":
    unittest.main()
