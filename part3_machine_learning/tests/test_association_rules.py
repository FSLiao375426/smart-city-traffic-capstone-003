"""Transaction boundaries, exact rule metrics and source-isolated rule evidence."""
import json
import math
import unittest

import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, WEATHER_TYPES
from part3_machine_learning.association_rules import transactions, evaluate_rule


class AssociationRuleTests(unittest.TestCase):
    def test_hour_bins_and_nonexclusive_weather(self):
        hours = [0,5,6,9,10,12,13,16,17,20,21,23]
        frame = pd.DataFrame({"date_time": [f"2016-01-04 {h:02}:00:00" for h in hours],
                              "traffic_volume_quartile": [1,2,3,4]*3})
        for weather in WEATHER_TYPES:
            frame["weather_"+weather] = 0
        frame.loc[0,["weather_Rain","weather_Snow"]]=1
        basket = transactions(frame)
        self.assertEqual(basket.shape,(12,23))
        time_cols = [c for c in basket if c.startswith("time=")]
        day_cols = [c for c in basket if c.startswith("day=")]
        outcome_cols = [c for c in basket if c.startswith("congestion=")]
        self.assertTrue(basket[time_cols].sum(axis=1).eq(1).all())
        self.assertTrue(basket[day_cols].sum(axis=1).eq(1).all())
        self.assertTrue(basket[outcome_cols].sum(axis=1).eq(1).all())
        self.assertTrue(basket["day=weekday"].all())
        self.assertTrue(basket.loc[0,"weather=Rain"] and basket.loc[0,"weather=Snow"])
        expected=["night_00_05","night_00_05","morning_06_09","morning_06_09",
                  "midday_10_12","midday_10_12","afternoon_13_16","afternoon_13_16",
                  "evening_17_20","evening_17_20","late_evening_21_23","late_evening_21_23"]
        for index,name in enumerate(expected):
            self.assertTrue(basket.loc[index,"time="+name])

    def test_exact_metrics_and_absent_antecedent(self):
        basket=pd.DataFrame({"a":[1,1,1,1,0,0,0,0,0,0],
                             "b":[1,1,1,0,1,1,0,0,0,0],"never":[0]*10}).astype(bool)
        scores=evaluate_rule(basket,["a"],"b")
        self.assertEqual(scores["joint_hours"],3)
        self.assertEqual(scores["support"],.3)
        self.assertEqual(scores["confidence"],.75)
        self.assertEqual(scores["lift"],1.5)
        missing=evaluate_rule(basket,["never"],"b")
        self.assertIsNone(missing["confidence"])
        self.assertIsNone(missing["lift"])

    def test_saved_rules_match_direct_source_counts(self):
        source=pd.read_csv(PROJECT_ROOT / "data/processed/traffic_supervised.csv")
        train=transactions(source.loc[source.split.eq("train")])
        validation=transactions(source.loc[source.split.eq("validation")])
        folder=PROJECT_ROOT / "part3_machine_learning/results/association_rules"
        frozen=json.loads((folder / "frozen_rules.json").read_text())
        saved=pd.read_csv(folder / "rules_train_validation.csv")
        manifest=json.loads((folder / "mining_manifest.json").read_text())
        self.assertEqual(len(frozen),len(saved))
        self.assertEqual(len(train),25329)
        self.assertEqual(len(validation),8713)
        self.assertFalse(manifest["test_transactions_created"])
        self.assertEqual(set(saved.consequent),{"congestion=Low","congestion=Medium","congestion=High","congestion=Severe"})
        self.assertTrue(saved.train_lift.is_monotonic_decreasing)
        for rule,row in zip(frozen,saved.itertuples()):
            self.assertEqual(rule["rule_id"],row.rule_id)
            self.assertTrue(all(item.startswith(("time=","day=","weather=")) for item in rule["antecedents"]))
            self.assertLessEqual(len(rule["antecedents"]),3)
            self.assertGreaterEqual(row.train_joint_hours,math.ceil(.01*len(train)))
            self.assertGreaterEqual(row.train_antecedent_hours,500)
            self.assertGreaterEqual(row.train_confidence,.3-1e-10)
            self.assertGreaterEqual(row.train_lift,1.2-1e-10)
            for split,basket in (("train",train),("validation",validation)):
                actual=evaluate_rule(basket,rule["antecedents"],rule["consequent"])
                for key in ("joint_hours","antecedent_hours","consequent_hours","total_hours"):
                    self.assertEqual(getattr(row,split+"_"+key),actual[key])
                for key in ("support","confidence","base_rate","lift"):
                    self.assertAlmostEqual(getattr(row,split+"_"+key),actual[key],delta=5.1e-10)
        evidence=json.loads((folder / "mlflow_evidence.json").read_text())
        self.assertEqual(evidence["status"],"FINISHED")
        self.assertEqual(int(evidence["metrics"]["selected_rules"]),len(saved))


if __name__=="__main__":
    unittest.main()
