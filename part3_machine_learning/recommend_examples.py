"""Save scenario recommendation examples and an actual MLflow application run."""
import json
import logging
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT","1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY","false")
import mlflow

from part2_python.pipeline import PROJECT_ROOT,configure_logging
from part3_machine_learning.recommend import build_bundle,recommend

logger=logging.getLogger(__name__)
BASE=PROJECT_ROOT / "part3_machine_learning"


def run_examples():
    bundle=build_bundle()
    folder=BASE / "results/recommendation"
    requests=folder / "requests"
    requests.mkdir(parents=True,exist_ok=True)
    common={"date":"2017-06-05","start_hour":8,"end_hour":19,"window_hours":2,
            "weather":["Clear"],"is_holiday":0,"temp_c":20.,"clouds_all":1.,"rain_1h":0.,"snow_1h":0.}
    scenarios={"weekday_clear":common,
               "weekend_clear":{**common,"date":"2017-06-10"},
               "weekday_rain":{**common,"weather":["Rain"],"temp_c":15.,"clouds_all":90.,"rain_1h":1.},
               "rare_squall":{**common,"weather":["Squall"],"clouds_all":75.}}
    uri="sqlite:///"+(BASE / "mlflow/mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client=mlflow.MlflowClient()
    experiment_name="capstone003_recommendation_demo"
    exp=client.get_experiment_by_name(experiment_name)
    exp_id=exp.experiment_id if exp else client.create_experiment(experiment_name,artifact_location=(BASE / "mlflow/artifacts").as_uri())
    results={}
    with mlflow.start_run(experiment_id=exp_id,run_name="constrained_departure_windows_v1") as run:
        mlflow.log_params({"model_name":bundle["model_name"],"model_run_id":bundle["model_run_id"],
                           "prediction_mode":"conditional_same_hour","allowed_hours":"08:00-19:00 end exclusive",
                           "window_hours":2,"min_support_hours":30,"scenario_count":len(scenarios)})
        mlflow.set_tags({"task":"departure_timing_recommendation","uses_live_weather":"false","evaluates_test_labels":"false"})
        for name,request in scenarios.items():
            result=recommend(request,bundle)
            results[name]=result
            (requests / (name+".json")).write_text(json.dumps(request,indent=2)+"\n",encoding="utf-8")
            (folder / (name+".json")).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            logger.info("Example %s: %s; selected %s",name,result["status"],result["best_window"])
        mlflow.log_metrics({"example_count":len(results),"examples_with_recommendation":sum(r["status"]=="ok" for r in results.values())})
        mlflow.log_dict({"examples":{name:{"status":r["status"],"best_window":r["best_window"]} for name,r in results.items()},
                         "interpretation":"Illustrative scenario outputs, not prediction performance or realized travel savings"},"example_summary.json")
        for path in [*folder.glob("*.json"),*requests.glob("*.json"),BASE / "models/recommendation/metadata.json"]:
            mlflow.log_artifact(str(path),"examples")
        manifest={"mlflow_run_id":run.info.run_id,"model_name":bundle["model_name"],"model_run_id":bundle["model_run_id"],
                  "support_training_end":bundle["fit_end"],"example_names":list(scenarios),
                  "date_note":"Historical calendar dates chosen for controlled examples; no observed 2017 outcomes are read",
                  "status_summary":{name:result["status"] for name,result in results.items()},
                  "test_scored":False,"stage":"recommendation prototype, no deployment or travel-time performance claim"}
        (folder / "recommendation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        mlflow.log_artifact(str(folder / "recommendation_manifest.json"),"examples")
    actual=client.get_run(manifest["mlflow_run_id"])
    if actual.info.status!="FINISHED":
        raise ValueError("Recommendation example run did not finish")
    (folder / "mlflow_evidence.json").write_text(json.dumps({"run_id":actual.info.run_id,"status":actual.info.status,
                                                            "params":actual.data.params,"metrics":actual.data.metrics},indent=2)+"\n",encoding="utf-8")
    return results


if __name__=="__main__":
    configure_logging(BASE / "recommend_examples.log")
    try:
        run_examples()
    except Exception:
        logger.error("Recommendation examples failed",exc_info=True)
        sys.exit(1)
