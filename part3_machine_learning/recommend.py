"""Recommend lower-volume departure windows for a supplied same-hour scenario."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT,WEATHER_TYPES
from part3_machine_learning.dataset import FEATURES

logger=logging.getLogger(__name__)
BASE=PROJECT_ROOT / "part3_machine_learning"
BUNDLE_PATH=BASE / "models/recommendation/bundle.joblib"
DEFAULT_MIN_SUPPORT=30
SOURCE_LABELS = {"forecast": "使用者提供的氣象預報", "assumption": "使用者設定的假設",
                 "unspecified": "使用者提供，來源未註明"}


def validate_request(request):
    required={"date","start_hour","end_hour","window_hours","weather","is_holiday"}
    allowed=required|{"temp_c","clouds_all","rain_1h","snow_1h","min_support_hours","weather_source"}
    if not isinstance(request,dict) or required-set(request) or set(request)-allowed:
        raise ValueError("Provide date, start_hour, end_hour, window_hours, weather and is_holiday; no unknown fields")
    if not isinstance(request["date"],str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}",request["date"]):
        raise ValueError("date must use YYYY-MM-DD in the source calendar")
    date=datetime.strptime(request["date"],"%Y-%m-%d")
    if request.get("weather_source", "unspecified") not in tuple(SOURCE_LABELS):
        raise ValueError("weather_source must be forecast, assumption or unspecified")
    for key in ("start_hour","end_hour","window_hours"):
        if type(request[key]) is not int:
            raise ValueError(key+" must be an integer")
    if not 0<=request["start_hour"]<request["end_hour"]<=24:
        raise ValueError("Require 0 <= start_hour < end_hour <= 24, with end exclusive")
    if not 1<=request["window_hours"]<=request["end_hour"]-request["start_hour"]:
        raise ValueError("window_hours must fit inside the permitted interval")
    if type(request["is_holiday"]) is not int or request["is_holiday"] not in (0,1):
        raise ValueError("is_holiday must be explicitly supplied as integer 0 or 1")
    weather=request["weather"]
    if (not isinstance(weather,list) or not weather or any(not isinstance(w,str) or w not in WEATHER_TYPES for w in weather)
            or len(set(weather))!=len(weather)):
        raise ValueError("weather must be a nonempty list of unique known weather categories")
    support=request.get("min_support_hours",DEFAULT_MIN_SUPPORT)
    if type(support) is not int or support<1:
        raise ValueError("min_support_hours must be a positive integer")
    for column in ("temp_c","clouds_all","rain_1h","snow_1h"):
        if column in request:
            value=request[column]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                raise ValueError(column+" must be a finite number")
    if "temp_c" in request and request["temp_c"]<=-273.15:
        raise ValueError("Temperature must be above absolute zero")
    if "clouds_all" in request and not 0<=request["clouds_all"]<=100:
        raise ValueError("clouds_all must be between 0 and 100")
    if any(request.get(column,0)<0 for column in ("rain_1h","snow_1h")):
        raise ValueError("Precipitation cannot be negative")
    return date


def build_bundle():
    manifest=json.loads((BASE / "results/training_manifest.json").read_text())
    selected=manifest["validation_candidates"]["regression"]
    record=next(item for item in manifest["runs"] if item["model"]==selected)
    source=PROJECT_ROOT / record["model_path"]
    if hashlib.sha256(source.read_bytes()).hexdigest()!=record["model_sha256"]:
        raise ValueError("Regression model differs from the recorded candidate")
    if selected!="random_forest_regressor":
        raise ValueError("Review the feature/serving contract before changing model family")
    imputation=json.loads((PROJECT_ROOT / "data/processed/imputation_values.json").read_text())
    scaling=json.loads((PROJECT_ROOT / "data/processed/scaling_values.json").read_text())
    fit_end="2016-12-31 23:59:59"
    if imputation["fit_end"]!=fit_end or scaling["fit_end"]!=fit_end:
        raise ValueError("Serving transformations must match the model training period")
    data_path=PROJECT_ROOT / "data/processed/traffic_supervised.csv"
    if hashlib.sha256(data_path.read_bytes()).hexdigest()!=manifest["dataset_sha256"]:
        raise ValueError("Support data differs from the model training dataset")
    data=pd.read_csv(data_path,keep_default_na=False)
    train=data.loc[data.split.eq("train")].copy()
    keep=["date_time","hour","is_weekend","is_holiday","traffic_volume",*("weather_"+name for name in WEATHER_TYPES)]
    bundle={"model":joblib.load(source),"features":list(FEATURES),"imputation":imputation,"scaling":scaling,
            "support_history":train[keep],"model_name":selected,"model_run_id":record["run_id"],
            "model_sha256":record["model_sha256"],"dataset_sha256":manifest["dataset_sha256"],"fit_end":fit_end,
            "weather_ranges":{column:[float(train[column].min()),float(train[column].max())] for column in ("temp","clouds_all","rain_1h","snow_1h")}}
    if len(train)!=25329 or train.date_time.ge("2017-01-01").any():
        raise ValueError("Only the training period may provide support cohorts")
    BUNDLE_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle,BUNDLE_PATH,compress=3)
    metadata={key:bundle[key] for key in ("features","model_name","model_run_id","model_sha256","dataset_sha256","fit_end","weather_ranges")}
    metadata.update({"support_hours":len(train),"bundle_sha256":hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
                     "mode":"Fixed supplied weather across candidate hours; conditional comparison, not a weather or traffic forecast",
                     "default_min_support_hours":DEFAULT_MIN_SUPPORT,
                     "support_definition":"Same source hour, weekday/weekend, holiday flag and exact multi-hot weather label set; numerical weather values are not matched"})
    (BUNDLE_PATH.parent / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    logger.info("Saved self-contained recommendation bundle from %s; %d training support hours",selected,len(train))
    return bundle


def load_bundle():
    metadata=json.loads((BUNDLE_PATH.parent / "metadata.json").read_text())
    if hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()!=metadata["bundle_sha256"]:
        raise ValueError("Recommendation bundle hash mismatch")
    return joblib.load(BUNDLE_PATH)


def resolve_weather(request, bundle, date):
    supplied={"temp":request.get("temp_c",None),"clouds_all":request.get("clouds_all",None),
              "rain_1h":request.get("rain_1h",None),"snow_1h":request.get("snow_1h",None)}
    if supplied["temp"] is not None:
        supplied["temp"]+=273.15
    assumptions=[]
    for column,value in list(supplied.items()):
        if value is None:
            settings=bundle["imputation"]["columns"][column]
            supplied[column]=float(settings["monthly_medians"].get(str(date.month),settings["fallback_median"]))
            assumptions.append(column+" uses the frozen training-month median/fallback, not a measured or forecast value")
    return supplied, assumptions


def scenario_features(request,bundle):
    date=validate_request(request)
    hours=np.arange(request["start_hour"],request["end_hour"])
    supplied, assumptions=resolve_weather(request,bundle,date)
    features=pd.DataFrame(index=np.arange(len(hours)))
    features["hour_sin"]=np.sin(2*np.pi*hours/24)
    features["hour_cos"]=np.cos(2*np.pi*hours/24)
    features["day_of_week_sin"]=np.sin(2*np.pi*date.weekday()/7)
    features["day_of_week_cos"]=np.cos(2*np.pi*date.weekday()/7)
    features["is_weekend"]=int(date.weekday()>=5)
    features["is_holiday"]=request["is_holiday"]
    for column in ("temp","clouds_all"):
        stats=bundle["scaling"]["columns"][column]
        features[column+"_z"]=(supplied[column]-stats["mean"])/stats["std"]
    for column in ("rain_1h","snow_1h"):
        features[column]=supplied[column]
    for weather in WEATHER_TYPES:
        features["weather_"+weather]=int(weather in request["weather"])
    # Part 2 CSVs and the original forest used eight-decimal prepared features.
    features=features.loc[:,bundle["features"]].round(8)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite prepared scenario features")
    outside=[column for column,value in supplied.items() if not bundle["weather_ranges"][column][0]<=value<=bundle["weather_ranges"][column][1]]
    return features,hours,date,assumptions,outside


def recommend(request,bundle=None):
    if bundle is None:
        bundle=load_bundle()
    features,hours,date,assumptions,outside=scenario_features(request,bundle)
    prediction=np.asarray(bundle["model"].predict(features),dtype=float)
    if prediction.shape!=(len(hours),) or not np.isfinite(prediction).all() or (prediction<0).any():
        raise ValueError("Model returned invalid traffic estimates")
    history=bundle["support_history"]
    common=history.is_weekend.eq(int(date.weekday()>=5))&history.is_holiday.eq(request["is_holiday"])
    for weather in WEATHER_TYPES:
        common &= history["weather_"+weather].eq(int(weather in request["weather"]))
    minimum=request.get("min_support_hours",DEFAULT_MIN_SUPPORT)
    hourly=[]
    for hour,value in zip(hours,prediction):
        cohort=history.loc[common&history.hour.eq(hour)]
        hourly.append({"hour":int(hour),"estimated_traffic":float(value),"support_hours":len(cohort),
                       "historical_mean_traffic":float(cohort.traffic_volume.mean()) if len(cohort) else None,
                       "support_sufficient":bool(len(cohort)>=minimum)})
    windows=[]
    duration=request["window_hours"]
    for offset in range(len(hourly)-duration+1):
        part=hourly[offset:offset+duration]
        if not all(item["support_sufficient"] for item in part):
            continue
        windows.append({"start_hour":part[0]["hour"],"end_hour":part[-1]["hour"]+1,
                        "mean_estimated_traffic":float(np.mean([item["estimated_traffic"] for item in part])),
                        "minimum_support_hours":min(item["support_hours"] for item in part)})
    windows.sort(key=lambda row:(row["mean_estimated_traffic"],row["start_hour"]))
    # Out-of-range numerics may produce estimates but are not promoted to a recommendation.
    if outside:
        status="outside_training_weather_range"
        best=None
        message="指定數值天氣超出訓練參考範圍；保留估計供檢查，但不產生推薦窗口。"
    elif not windows:
        status="insufficient_support"
        best=None
        message="可出發區間內沒有足夠訓練情境支持的完整窗口；不把缺資料解讀為零車流。"
    else:
        status="ok"
        best=windows[0]
        message=(f"在你指定的 {request['start_hour']:02}:00-{request['end_hour']:02}:00 區間與固定天氣情境下，"
                 f"可考慮 {best['start_hour']:02}:00-{best['end_hour']:02}:00 出發；"
                 f"該窗口平均估計車流為 {best['mean_estimated_traffic']:.0f} 輛／小時。這是情境比較，不是即時或提前路況預報。")
    values, _ = resolve_weather(request, bundle, date)
    def display_weather(key, label, value, unit):
        origin = "" if key in request else "（訓練資料中位數補值）"
        return f"{label}{value:g}{unit}{origin}"
    conditions = "、".join([
        display_weather("temp_c", "氣溫 ", values["temp"]-273.15, "°C"),
        display_weather("clouds_all", "雲量 ", values["clouds_all"], "%"),
        display_weather("rain_1h", "每小時雨量 ", values["rain_1h"], " mm"),
        display_weather("snow_1h", "每小時雪量 ", values["snow_1h"], " mm"),
    ])
    source = SOURCE_LABELS[request.get("weather_source", "unspecified")]
    context = (f"{request['date']} {request['start_hour']:02}:00–{request['end_hour']:02}:00 "
               f"各候選小時採用相同天氣條件（{source}）：{'、'.join(request['weather'])}，{conditions}。")
    message = context + "\n" + message
    return {"status":status,"recommendation":message,"request":request,"model":bundle["model_name"],"model_run_id":bundle["model_run_id"],
            "prediction_mode":"conditional_same_hour_scenario","source_calendar_day_type":"weekend" if date.weekday()>=5 else "weekday",
            "weather_assumed_constant_across_hours":True,"numeric_imputation_assumptions":assumptions,
            "outside_training_weather_range":outside,"minimum_support_hours":minimum,
            "support_note":"Training-only cohort: same hour, day type, holiday and exact weather label set; numerical weather is not matched; not a calibrated confidence interval",
            "best_window":best,"ranked_supported_windows":windows,"hourly_estimates":hourly,
            "limitations":["Departure window, not trip duration or route optimization","No live traffic, future weather service or accident probability",
                            "Lower estimated traffic does not guarantee shorter travel time or greater safety","Default support floor 30 is an analyst setting, not a statistical guarantee"]}


class RequestParser(argparse.ArgumentParser):
    def error(self,message):
        raise ValueError(message)


def main(argv=None):
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # A local entry point owns handlers; imported functions do not configure them.
    from part2_python.pipeline import configure_logging
    configure_logging(BASE / "recommend.log")
    parser=RequestParser(description=__doc__)
    parser.add_argument("--build-bundle",action="store_true")
    parser.add_argument("--request",type=Path,help="JSON scenario file")
    try:
        args=parser.parse_args(argv)
        if args.build_bundle:
            build_bundle()
        if args.request:
            request=json.loads(args.request.read_text(encoding="utf-8-sig"))
            logger.info("Recommendation request: %s",request)
            result=recommend(request)
            sys.stdout.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
            logger.info("Recommendation completed: %s",result["status"])
        elif not args.build_bundle:
            raise ValueError("Provide --build-bundle or --request FILE")
    except (ValueError,OSError,KeyError,TypeError) as exc:
        logger.error("Cannot complete recommendation: %s",exc)
        return 2
    return 0


if __name__=="__main__":
    sys.exit(main())
