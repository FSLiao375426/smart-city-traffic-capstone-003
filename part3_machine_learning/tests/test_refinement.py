"""Fold-fitted imputation isolation, bounded selection and fresh-process evidence."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from part2_python.pipeline import PROJECT_ROOT,WEATHER_TYPES
from part3_machine_learning.refinement import fit_state,feature_frame,GROUPS,ELIGIBLE_GROUPS


class RefinementTests(unittest.TestCase):
    def test_fold_imputation_restores_flags_and_ignores_later_values(self):
        frame=pd.DataFrame({"date_time":["2013-01-01 00:00:00","2013-01-02 00:00:00","2013-02-01 00:00:00"],
                            "temp":[270.,280.,999.],"temp_imputed":[0,0,1],
                            "clouds_all":[0.,100.,50.],"rain_1h":[0.,0.,0.],"snow_1h":[0.,0.,0.],"is_holiday":[0,0,0]})
        for weather in WEATHER_TYPES:
            frame["weather_"+weather]=0
        state=fit_state(frame)
        self.assertEqual(state["imputation"]["temp"]["fallback_median"],275.)
        self.assertEqual(state["scaling"]["temp"]["mean"],275.)
        self.assertNotIn("2",state["imputation"]["temp"]["monthly_medians"])
        saved=json.dumps(state,sort_keys=True)
        later=frame.iloc[[2]].copy()
        later["date_time"]="2015-02-01 00:00:00"
        later["temp"]=9999.
        values=feature_frame(later,state)
        self.assertEqual(float(values.temp_z.iloc[0]),0.)
        self.assertEqual(json.dumps(state,sort_keys=True),saved)
        changed=frame.copy()
        changed.loc[2,"temp"]=-9999.
        self.assertEqual(fit_state(changed),state)

    def test_selection_is_within_pre2017_data(self):
        folder=PROJECT_ROOT / "part3_machine_learning/results/refinement"
        manifest=json.loads((folder / "refinement_manifest.json").read_text())
        trials=pd.read_csv(folder / "inner_fold_results.csv")
        self.assertEqual(len(trials),14)
        self.assertTrue(trials.fit_max.lt(trials.assessment_min).all())
        self.assertTrue(trials.assessment_max.lt("2017-01-01").all())
        self.assertEqual(set(trials.fold),{"through2014_to2015","through2015_to2016"})
        ablation=pd.read_csv(folder / "feature_ablation.csv")
        self.assertEqual(set(ablation.feature_group),set(GROUPS))
        selected=ablation.loc[ablation.eligible_final_candidate].sort_values(["mean_fold_mae","feature_count","feature_group"]).iloc[0].feature_group
        self.assertEqual(manifest["feature_group"],selected)
        self.assertIn(selected,ELIGIBLE_GROUPS)
        architecture=pd.read_csv(folder / "architecture_comparison.csv")
        self.assertEqual(len(architecture),3)
        self.assertEqual(manifest["features"],list(GROUPS[selected]))
        self.assertFalse(manifest["test_scored"])
        evidence=json.loads((folder / "mlflow_evidence.json").read_text())
        self.assertEqual(len(evidence),15)
        self.assertTrue(all(item["status"]=="FINISHED" for item in evidence))

    def test_refined_validation_and_training_fit_counts(self):
        folder=PROJECT_ROOT / "part3_machine_learning/results/refinement"
        manifest=json.loads((folder / "refinement_manifest.json").read_text())
        p=pd.read_csv(folder / "validation_predictions.csv")
        self.assertEqual(len(p),8713)
        self.assertTrue(pd.to_datetime(p.date_time).dt.year.eq(2017).all())
        self.assertAlmostEqual(mean_absolute_error(p.actual,p.prediction),manifest["metrics"]["validation_mae"],delta=1e-8)
        bundle=joblib.load(PROJECT_ROOT / "part3_machine_learning/models/refined_neural/preprocessing.joblib")
        self.assertEqual(bundle["state"]["train_hours"],25329)
        self.assertLess(bundle["state"]["fit_max"],"2017-01-01")
        self.assertEqual(int(bundle["x_scaler"].n_samples_seen_),25329)
        self.assertEqual(int(bundle["y_scaler"].n_samples_seen_),25329)

    def test_fresh_process_bundle_reloads(self):
        for result in ("neural","refinement"):
            p=PROJECT_ROOT / "part3_machine_learning/results" / result / "reload_evidence.json"
            evidence=json.loads(p.read_text())
            self.assertEqual(evidence["validation_hours"],8713)
            self.assertTrue(evidence["preprocessing_loaded_from_disk"])
            self.assertTrue(evidence["model_loaded_from_disk"])
            self.assertLessEqual(evidence["max_absolute_prediction_difference"],1e-6)
            self.assertFalse(evidence["test_scored"])


if __name__=="__main__":
    unittest.main()
