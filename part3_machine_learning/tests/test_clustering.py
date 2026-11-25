"""K-means input isolation, train-only preprocessing and saved assignment checks."""
import hashlib
import json
import unittest

import joblib
import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT
from part3_machine_learning.clustering import cluster_inputs, CLUSTER_FEATURES


class ClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(PROJECT_ROOT / "data/processed/traffic_supervised.csv")
        cls.folder = PROJECT_ROOT / "part3_machine_learning/results/clustering"

    def test_no_target_labels_used(self):
        data = self.data.loc[self.data.split.eq("train")].iloc[:100].copy()
        original = cluster_inputs(data)
        data["high_risk"] = 999
        data["traffic_volume_quartile"] = 999
        data["risky_weather"] = 999
        pd.testing.assert_frame_equal(original,cluster_inputs(data))
        self.assertEqual(list(original.columns),list(CLUSTER_FEATURES))
        self.assertIn("traffic_volume",original.columns)
        self.assertTrue(original.adverse_weather.isin([0,1]).all())

    def test_fit_and_model_reload(self):
        path = PROJECT_ROOT / "part3_machine_learning/models/kmeans/model.joblib"
        model = joblib.load(path)
        train = self.data.loc[self.data.split.eq("train")]
        inputs = cluster_inputs(train)
        scaler = model.named_steps["standardscaler"]
        self.assertEqual(int(scaler.n_samples_seen_),25329)
        np.testing.assert_allclose(scaler.mean_,inputs.mean().to_numpy(),atol=1e-8,rtol=0)
        np.testing.assert_allclose(scaler.var_,inputs.var(ddof=0).to_numpy(),atol=1e-7,rtol=0)
        assignments=pd.read_csv(self.folder / "cluster_assignments.csv")
        self.assertEqual(len(assignments),34042)
        self.assertEqual(assignments.date_time.nunique(),34042)
        self.assertFalse(assignments.split.eq("test").any())
        self.assertTrue(pd.to_datetime(assignments.date_time).lt("2018-01-01").all())
        permitted=self.data.loc[self.data.split.isin(["train","validation"])]
        np.testing.assert_array_equal(model.predict(cluster_inputs(permitted)),assignments.cluster)
        manifest=json.loads((self.folder / "clustering_manifest.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest["model_sha256"])

    def test_selection_and_summary_counts(self):
        comparison=pd.read_csv(self.folder / "k_comparison.csv")
        manifest=json.loads((self.folder / "clustering_manifest.json").read_text())
        self.assertEqual(comparison.k.tolist(),[2,3,4,5,6])
        selected=int(comparison.sort_values(["train_sample_silhouette","k"],ascending=[False,True]).iloc[0].k)
        self.assertEqual(manifest["selected_k"],selected)
        self.assertEqual(len(manifest["runs"]),5)
        self.assertTrue(all(run["status"]=="FINISHED" for run in manifest["runs"]))
        assignments=pd.read_csv(self.folder / "cluster_assignments.csv")
        summary=pd.read_csv(self.folder / "cluster_summary.csv")
        for row in summary.itertuples():
            mask=assignments.split.eq(row.split)&assignments.cluster.eq(row.cluster)
            self.assertEqual(int(mask.sum()),row.observed_hours)
            dates=assignments.loc[mask,"date_time"]
            original=self.data.loc[self.data.date_time.isin(dates)]
            self.assertAlmostEqual(row.mean_traffic,original.traffic_volume.mean(),delta=1e-7)
        self.assertEqual(int(summary.loc[summary.split.eq("train"),"observed_hours"].sum()),25329)
        self.assertEqual(int(summary.loc[summary.split.eq("validation"),"observed_hours"].sum()),8713)


if __name__=="__main__":
    unittest.main()
