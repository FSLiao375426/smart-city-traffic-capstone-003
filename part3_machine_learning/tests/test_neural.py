"""Neural validation, preprocessing and SHAP decomposition evidence."""
import hashlib
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score

from part2_python.pipeline import PROJECT_ROOT
from part3_machine_learning.dataset import FEATURES


class NeuralEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=PROJECT_ROOT / "part3_machine_learning"
        cls.folder=cls.base / "results/neural"
        cls.manifest=json.loads((cls.folder / "neural_manifest.json").read_text())

    def test_validation_metrics_and_frozen_test(self):
        p=pd.read_csv(self.folder / "validation_predictions.csv")
        self.assertEqual(len(p),8713)
        self.assertEqual(p.date_time.nunique(),8713)
        self.assertTrue(pd.to_datetime(p.date_time).dt.year.eq(2017).all())
        self.assertFalse(self.manifest["test_scored"])
        self.assertAlmostEqual(mean_absolute_error(p.actual,p.prediction),self.manifest["metrics"]["validation_mae"],delta=1e-8)
        self.assertAlmostEqual(r2_score(p.actual,p.prediction),self.manifest["metrics"]["validation_r2"],delta=1e-10)
        history=pd.read_csv(self.folder / "learning_history.csv")
        self.assertEqual(len(history),self.manifest["completed_epochs"])
        self.assertLessEqual(len(history),80)
        self.assertGreaterEqual(self.manifest["restored_best_epoch"],1)
        self.assertLessEqual(self.manifest["restored_best_epoch"],len(history))
        evidence=json.loads((self.folder / "mlflow_evidence.json").read_text())
        self.assertEqual(evidence["status"],"FINISHED")
        self.assertAlmostEqual(evidence["metrics"]["validation_mae"],self.manifest["metrics"]["validation_mae"],delta=1e-10)

    def test_scalers_fit_training_only_and_saved_model_hash(self):
        scalers=joblib.load(self.base / "models/neural_regressor/preprocessing.joblib")
        data=pd.read_csv(PROJECT_ROOT / "data/processed/traffic_supervised.csv")
        train=data.loc[data.split.eq("train")]
        self.assertEqual(int(scalers["x_scaler"].n_samples_seen_),25329)
        self.assertEqual(int(scalers["y_scaler"].n_samples_seen_),25329)
        self.assertEqual(scalers["features"],list(FEATURES))
        np.testing.assert_allclose(scalers["x_scaler"].mean_,train.loc[:,FEATURES].mean().to_numpy(),atol=1e-8,rtol=0)
        self.assertAlmostEqual(scalers["y_scaler"].mean_[0],train.traffic_volume.mean(),delta=1e-8)
        path=self.base / "models/neural_regressor/model.keras"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.manifest["model_sha256"])

    def test_shap_cases_and_additivity(self):
        p=pd.read_csv(self.folder / "shap_reconstruction.csv")
        v=pd.read_csv(self.folder / "shap_contributions.csv")
        cases=pd.read_csv(self.folder / "shap_input_cases.csv")
        self.assertEqual(len(p),24)
        self.assertEqual(v.shape,(24,22))
        self.assertEqual(v.date_time.tolist(),p.date_time.tolist())
        self.assertEqual(cases.date_time.tolist(),p.date_time.tolist())
        self.assertTrue(pd.to_datetime(p.date_time).dt.year.eq(2017).all())
        self.assertEqual(len(self.manifest["background_dates"]),64)
        self.assertTrue((pd.to_datetime(self.manifest["background_dates"]).year <= 2016).all())
        np.testing.assert_allclose(p.base_value+v.loc[:,FEATURES].sum(axis=1),p.prediction,atol=.01,rtol=0)
        np.testing.assert_allclose(p.reconstructed_prediction,p.prediction,atol=.01,rtol=0)
        self.assertLess(self.manifest["shap_max_additivity_error"],.01)


if __name__=="__main__":
    unittest.main()
