"""Constrained departure windows, input validation and support-cohort checks."""
import json
import unittest

import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT,WEATHER_TYPES
from part3_machine_learning.dataset import FEATURES
from part3_machine_learning.recommend import validate_request,scenario_features,recommend,load_bundle


class HourModel:
    def predict(self,x):
        hours=np.rint(np.mod(np.arctan2(x.hour_sin,x.hour_cos)*24/(2*np.pi),24))
        return hours*100


def example_bundle():
    history=pd.DataFrame({"hour":[8,9,10,11],"is_weekend":[0]*4,"is_holiday":[0]*4,"traffic_volume":[900]*4})
    for name in WEATHER_TYPES:
        history["weather_"+name]=int(name=="Clear")
    return {"model":HourModel(),"features":list(FEATURES),"support_history":history,"model_name":"test_model","model_run_id":"test",
            "imputation":{"columns":{column:{"monthly_medians":{"6":value},"fallback_median":value}
                                       for column,value in (("temp",293.15),("clouds_all",1.),("rain_1h",0.),("snow_1h",0.))}},
            "scaling":{"columns":{"temp":{"mean":280.,"std":10.},"clouds_all":{"mean":50.,"std":25.}}},
            "weather_ranges":{"temp":[260.,310.],"clouds_all":[0.,100.],"rain_1h":[0.,10.],"snow_1h":[0.,1.]}}


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.request={"date":"2017-06-05","start_hour":8,"end_hour":12,"window_hours":2,
                      "weather":["Clear"],"is_holiday":0,"temp_c":20.,"clouds_all":1.,"rain_1h":0.,"snow_1h":0.,"min_support_hours":1}

    def test_allowed_window_and_minimum_mean(self):
        result=recommend(self.request,example_bundle())
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["best_window"]["start_hour"],8)
        self.assertEqual(result["best_window"]["end_hour"],10)
        self.assertAlmostEqual(result["best_window"]["mean_estimated_traffic"],850.)
        self.assertEqual(len(result["ranked_supported_windows"]),3)
        for window in result["ranked_supported_windows"]:
            self.assertGreaterEqual(window["start_hour"],8)
            self.assertLessEqual(window["end_hour"],12)
            self.assertEqual(window["end_hour"]-window["start_hour"],2)
        gap=example_bundle()
        gap["support_history"]=gap["support_history"].loc[gap["support_history"].hour.ne(9)]
        after_gap=recommend(self.request,gap)
        self.assertEqual(after_gap["best_window"]["start_hour"],10)
        self.assertEqual(len(after_gap["ranked_supported_windows"]),1)

    def test_unsupported_weather_and_unknown_numeric_defaults(self):
        bundle=example_bundle()
        result=recommend({**self.request,"min_support_hours":2},bundle)
        self.assertEqual(result["status"],"insufficient_support")
        self.assertIsNone(result["best_window"])
        self.assertEqual(recommend({**self.request,"weather":["Clear","Clouds"]},bundle)["status"],"insufficient_support")
        missing=self.request.copy()
        del missing["temp_c"]
        x,_,_,assumptions,outside=scenario_features(missing,bundle)
        self.assertAlmostEqual(float(x.temp_z.iloc[0]),1.315)
        self.assertEqual(len(assumptions),1)
        self.assertFalse(outside)
        self.assertEqual(list(x.columns),list(FEATURES))
        self.assertNotIn("traffic_volume",x.columns)
        self.assertEqual(recommend({**self.request,"temp_c":50.},bundle)["status"],"outside_training_weather_range")

    def test_invalid_inputs(self):
        for changes in ({"date":"2017-02-30"},{"date":"2017-6-5"},{"start_hour":12},{"window_hours":5},
                        {"weather":["Unknown"]},{"weather":["Clear","Clear"]},{"clouds_all":101.},
                        {"temp_c":float("nan")},{"rain_1h":-1.},{"is_holiday":True},{"min_support_hours":0}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_request({**self.request,**changes})

    def test_weather_disclosure_before_estimate(self):
        for source, label in (("forecast","使用者提供的氣象預報"), ("assumption","使用者設定的假設"),
                              ("unspecified","來源未註明")):
            request = {**self.request, "weather_source":source}
            del request["temp_c"]
            conditions, estimate = recommend(request,example_bundle())["recommendation"].split("\n",1)
            self.assertIn(label, conditions)
            self.assertIn("20°C（訓練資料中位數補值）", conditions)
            self.assertIn("Clear", conditions)
            self.assertIn("850", estimate)
        with self.assertRaises(ValueError):
            validate_request({**self.request,"weather_source":"unknown"})

    def test_real_saved_examples_and_train_only_bundle(self):
        bundle=load_bundle()
        history=bundle["support_history"]
        self.assertEqual(len(history),25329)
        self.assertTrue(history.date_time.lt("2017-01-01").all())
        folder=PROJECT_ROOT / "part3_machine_learning/results/recommendation"
        for name in ("weekday_clear","weekend_clear","weekday_rain","rare_squall"):
            request=json.loads((folder / "requests" / (name+".json")).read_text())
            saved=json.loads((folder / (name+".json")).read_text(encoding="utf-8"))
            actual=recommend(request,bundle)
            self.assertEqual(actual["status"],saved["status"])
            if saved["best_window"] is None:
                self.assertIsNone(actual["best_window"])
            else:
                for key in ("start_hour","end_hour","minimum_support_hours"):
                    self.assertEqual(actual["best_window"][key],saved["best_window"][key])
                self.assertAlmostEqual(actual["best_window"]["mean_estimated_traffic"],saved["best_window"]["mean_estimated_traffic"],delta=1e-8)
            if actual["status"]=="ok":
                windows=actual["ranked_supported_windows"]
                self.assertEqual(windows[0],min(windows,key=lambda w:(w["mean_estimated_traffic"],w["start_hour"])))
                self.assertTrue(all(w["minimum_support_hours"]>=30 for w in windows))
        rare=json.loads((folder / "rare_squall.json").read_text(encoding="utf-8"))
        self.assertEqual(rare["status"],"insufficient_support")
        self.assertIsNone(rare["best_window"])


if __name__=="__main__":
    unittest.main()
