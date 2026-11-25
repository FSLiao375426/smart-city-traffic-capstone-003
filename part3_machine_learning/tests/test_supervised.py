"""Proxy-rule boundaries, temporal isolation and persisted baseline evidence."""
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from part2_python.pipeline import PROJECT_ROOT, WEATHER_TYPES
from part3_machine_learning.dataset import add_targets, FEATURES, FORBIDDEN_FEATURES


class DatasetTests(unittest.TestCase):
    def test_proxy_and_multilabel_weather(self):
        frame = pd.DataFrame({"date_time": pd.date_range("2016-01-03", periods=5, freq="D").astype(str),
                              "traffic_volume_quartile": [3, 3, 2, 4, 4]})
        for weather in WEATHER_TYPES:
            frame["weather_" + weather] = 0
        for column in FEATURES:
            if column not in frame.columns and column not in ("day_of_week_sin", "day_of_week_cos"):
                frame[column] = 0.
        frame.loc[[0, 4], "weather_Clear"] = 1
        frame.loc[1, "weather_Snow"] = 1
        frame.loc[[2, 4], "weather_Fog"] = 1
        frame.loc[3, "weather_Thunderstorm"] = 1
        result = add_targets(frame)
        self.assertEqual(result.high_risk.tolist(), [0, 1, 0, 1, 1])
        self.assertEqual(result.is_low_visibility.tolist(), [0, 0, 1, 0, 1])
        pd.testing.assert_frame_equal(result.loc[:, frame.columns], frame)
        np.testing.assert_allclose(result.day_of_week_sin**2 + result.day_of_week_cos**2, 1, atol=1e-12)
        distance = np.hypot(np.diff(result.day_of_week_sin), np.diff(result.day_of_week_cos))
        np.testing.assert_allclose(distance, distance[0], atol=1e-12)
        with self.assertRaises(ValueError):
            add_targets(frame.assign(traffic_volume_quartile=5))
        with self.assertRaises(ValueError):
            add_targets(frame.assign(weather_Fog=2))
        with self.assertRaises(ValueError):
            add_targets(frame.assign(date_time="2016-01-01"))

    def test_real_dataset_fit_and_feature_isolation(self):
        d = pd.read_csv(PROJECT_ROOT / "data/processed/traffic_supervised.csv", keep_default_na=False)
        original = pd.read_csv(PROJECT_ROOT / "data/processed/traffic_model_features.csv", keep_default_na=False)
        self.assertEqual(d.shape, (40575, 43))
        pd.testing.assert_frame_equal(d.loc[:, original.columns], original)
        self.assertEqual(d.split.value_counts().to_dict(), {"train": 25329, "validation": 8713, "test": 6533})
        self.assertFalse(set(FEATURES) & FORBIDDEN_FEATURES)
        self.assertEqual(len(FEATURES), 21)
        self.assertTrue(np.isfinite(d.loc[:, FEATURES].to_numpy()).all())
        times = pd.to_datetime(d.date_time)
        self.assertTrue(times.loc[d.split.eq("train")].lt("2017-01-01").all())
        self.assertTrue(times.loc[d.split.eq("validation")].dt.year.eq(2017).all())
        self.assertTrue(times.loc[d.split.eq("test")].dt.year.eq(2018).all())
        train = d.loc[d.split.eq("train")]
        thresholds = train.traffic_volume.quantile([.25,.5,.75]).to_numpy()
        np.testing.assert_array_equal(thresholds,[1204,3339,4932])
        for column in ("temp_z", "clouds_all_z"):
            self.assertAlmostEqual(train[column].mean(),0,delta=1e-8)
            self.assertAlmostEqual(train[column].std(ddof=0),1,delta=1e-8)
        self.assertEqual(int(train.high_risk.sum()),2257)
        self.assertEqual(int(d.loc[d.split.eq("validation"),"high_risk"].sum()),1003)
        for partition in ("train","validation"):
            part = d.loc[d.split.eq(partition)]
            expected = (part.traffic_volume.gt(3339) & part.risky_weather.eq(1)).astype(int)
            np.testing.assert_array_equal(part.high_risk,expected)


class TrainingEvidenceTests(unittest.TestCase):
    def test_validation_predictions_and_metrics(self):
        folder = PROJECT_ROOT / "part3_machine_learning/results"
        predictions = pd.read_csv(folder / "validation_predictions.csv")
        metrics = pd.read_csv(folder / "validation_metrics.csv")
        self.assertEqual(len(metrics),6)
        self.assertEqual(len(predictions),6*8713)
        self.assertTrue(pd.to_datetime(predictions.date_time).dt.year.eq(2017).all())
        for row in metrics.itertuples():
            p = predictions.loc[predictions.model.eq(row.model)]
            self.assertEqual(len(p),8713)
            self.assertEqual(p.date_time.nunique(),8713)
            if row.task=="classification":
                self.assertAlmostEqual(row.f1,f1_score(p.actual,p.prediction,zero_division=0),delta=1e-9)
                self.assertAlmostEqual(row.roc_auc,roc_auc_score(p.actual,p.proxy_score),delta=1e-8)
            else:
                self.assertAlmostEqual(row.mae,mean_absolute_error(p.actual,p.prediction),delta=1e-8)
        manifest=json.loads((folder / "training_manifest.json").read_text())
        self.assertEqual(len(manifest["runs"]),6)
        self.assertNotIn("test_metrics",manifest)


if __name__ == "__main__":
    unittest.main()
