"""Train-period feature ablations and bounded MLP architecture search."""
import hashlib
import json
import logging
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","0")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT","1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY","false")

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score

from part2_python.pipeline import PROJECT_ROOT,WEATHER_NUMERIC,WEATHER_TYPES,configure_logging
from part3_machine_learning.dataset import FEATURES

logger=logging.getLogger(__name__)
BASE=PROJECT_ROOT / "part3_machine_learning"
TIME_FEATURES=tuple(FEATURES[:6])
NUMERIC_FEATURES=("temp_z","clouds_all_z","rain_1h","snow_1h")
WEATHER_FEATURES=tuple("weather_"+name for name in WEATHER_TYPES)
GROUPS={"full":tuple(FEATURES),"without_weekend":tuple(c for c in FEATURES if c!="is_weekend"),
        "time_only":TIME_FEATURES,"time_numeric":TIME_FEATURES+NUMERIC_FEATURES,
        "time_weather_flags":TIME_FEATURES+WEATHER_FEATURES}
ELIGIBLE_GROUPS=("full","without_weekend","time_weather_flags")
ARCHITECTURES={"64_32":(64,32),"32":(32,),"128_64":(128,64)}
FOLDS=(("through2014_to2015","2015-01-01","2016-01-01"),
       ("through2015_to2016","2016-01-01","2017-01-01"))
SEED=42


def restore_missing(frame):
    data=frame.copy()
    for column in WEATHER_NUMERIC:
        if column+"_imputed" in data:
            data.loc[data[column+"_imputed"].eq(1),column]=np.nan
    return data


def fill_weather(frame,state):
    data=restore_missing(frame)
    months=pd.to_datetime(data.date_time).dt.month
    for column in WEATHER_NUMERIC:
        settings=state["imputation"][column]
        by_month={int(month):value for month,value in settings["monthly_medians"].items()}
        values=months.map(by_month).fillna(settings["fallback_median"])
        data[column]=data[column].fillna(values)
    if not np.isfinite(data[list(WEATHER_NUMERIC)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite weather after fold-fitted imputation")
    return data


def fit_state(training):
    data=restore_missing(training)
    months=pd.to_datetime(data.date_time).dt.month
    state={"fit_min":str(data.date_time.min()),"fit_max":str(data.date_time.max()),
           "train_hours":len(data),"imputation":{},"scaling":{}}
    for column in WEATHER_NUMERIC:
        monthly=data[column].groupby(months).median()
        fallback=float(data[column].median())
        if not np.isfinite(fallback):
            raise ValueError("No training observations for "+column)
        state["imputation"][column]={"monthly_medians":{str(int(month)):float(value) for month,value in monthly.dropna().items()},
                                       "fallback_median":fallback}
    filled=fill_weather(training,state)
    for column in ("temp","clouds_all"):
        mean,std=float(filled[column].mean()),float(filled[column].std(ddof=0))
        if not np.isfinite(std) or std<=0:
            raise ValueError("Invalid training scale for "+column)
        state["scaling"][column]={"mean":mean,"std":std}
    return state


def feature_frame(frame,state):
    data=fill_weather(frame,state)
    times=pd.to_datetime(data.date_time)
    result=pd.DataFrame(index=data.index)
    result["hour_sin"]=np.sin(2*np.pi*times.dt.hour/24)
    result["hour_cos"]=np.cos(2*np.pi*times.dt.hour/24)
    result["day_of_week_sin"]=np.sin(2*np.pi*times.dt.day_of_week/7)
    result["day_of_week_cos"]=np.cos(2*np.pi*times.dt.day_of_week/7)
    result["is_weekend"]=times.dt.day_of_week.ge(5).astype(int)
    result["is_holiday"]=data.is_holiday
    for column in ("temp","clouds_all"):
        settings=state["scaling"][column]
        result[column+"_z"]=(data[column]-settings["mean"])/settings["std"]
    for column in ("rain_1h","snow_1h",*WEATHER_FEATURES):
        result[column]=data[column]
    return result.loc[:,FEATURES]


def make_model(widths,input_width):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=(input_width,)),
                               *[tf.keras.layers.Dense(width,activation="relu") for width in widths],
                               tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss="mse")
    return model


def data_batches(x,y,training=False):
    options=tf.data.Options()
    options.threading.private_threadpool_size=2
    ds=tf.data.Dataset.from_tensor_slices((x.astype("float32"),y.astype("float32")))
    if training:
        ds=ds.shuffle(len(x),seed=SEED)
    return ds.batch(256 if training else 512).with_options(options).prefetch(1)


def predict_bundle(model,x_scaler,y_scaler,features):
    x=x_scaler.transform(features).astype("float32")
    prediction=model(x,training=False).numpy().reshape(-1)
    return prediction.astype(float)*float(y_scaler.scale_[0])+float(y_scaler.mean_[0])


def score_fold(training,validation,group,architecture):
    state=fit_state(training)
    columns=list(GROUPS[group])
    train_x=feature_frame(training,state)[columns]
    val_x=feature_frame(validation,state)[columns]
    x_scaler,y_scaler=StandardScaler(),StandardScaler()
    tx=x_scaler.fit_transform(train_x)
    vx=x_scaler.transform(val_x)
    ty=y_scaler.fit_transform(training[["traffic_volume"]])
    vy=y_scaler.transform(validation[["traffic_volume"]])
    model=make_model(ARCHITECTURES[architecture],len(columns))
    early=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,min_delta=1e-4,restore_best_weights=True)
    history=model.fit(data_batches(tx,ty,True),validation_data=data_batches(vx,vy),epochs=40,
                      callbacks=[early],verbose=0,shuffle=False)
    prediction=predict_bundle(model,x_scaler,y_scaler,val_x)
    return {"mae":float(mean_absolute_error(validation.traffic_volume,prediction)),
            "r2":float(r2_score(validation.traffic_volume,prediction)),"epochs":len(history.history["loss"]),
            "best_epoch":int(early.best_epoch+1),"preprocessing":state}


def run_refinement():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    source=PROJECT_ROOT / "data/processed/traffic_hourly.csv"
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    frame=pd.read_csv(source,keep_default_na=False)
    if frame.date_time.duplicated().any():
        raise ValueError("Duplicate observed timestamps")
    search_data=frame.loc[frame.date_time.lt("2017-01-01")].copy()
    folder=BASE / "results/refinement"
    folder.mkdir(parents=True,exist_ok=True)
    uri="sqlite:///"+(BASE / "mlflow/mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client=mlflow.MlflowClient()
    name="capstone003_train_period_refinement"
    exp=client.get_experiment_by_name(name)
    exp_id=exp.experiment_id if exp else client.create_experiment(name,artifact_location=(BASE / "mlflow/artifacts").as_uri())
    records=[]
    fold_states={}
    fold_states={}
    def compare(group,architecture,phase):
        for fold,split,end in FOLDS:
            train=search_data.loc[search_data.date_time.lt(split)]
            valid=search_data.loc[search_data.date_time.ge(split)&search_data.date_time.lt(end)]
            logger.info("%s: features=%s (%d), architecture=%s, fold=%s, %d/%d hours",phase,group,len(GROUPS[group]),architecture,fold,len(train),len(valid))
            with mlflow.start_run(experiment_id=exp_id,run_name=f"{group}_{architecture}_{fold}") as run:
                mlflow.log_params({"phase":phase,"feature_group":group,"feature_count":len(GROUPS[group]),
                                   "architecture":architecture,"fold":fold,"train_hours":len(train),"validation_hours":len(valid),
                                   "train_before":split,"assessment_before":end,"max_epochs":40,"patience":5,"seed":SEED})
                mlflow.set_tags({"selection_scope":"pre2017_inner_time_splits","source_sha256":digest})
                started=time.perf_counter()
                result=score_fold(train,valid,group,architecture)
                fold_states[fold]=result["preprocessing"]
                fold_states[fold]=result["preprocessing"]
                mlflow.log_metrics({"fold_mae":result["mae"],"fold_r2":result["r2"],"fit_seconds":time.perf_counter()-started,
                                    "epochs":result["epochs"],"best_epoch":result["best_epoch"]})
                mlflow.log_dict(result["preprocessing"],"fold_fitted_preprocessing.json")
                records.append({"phase":phase,"feature_group":group,"feature_count":len(GROUPS[group]),
                                "architecture":architecture,"fold":fold,"mae":result["mae"],"r2":result["r2"],
                                "best_epoch":result["best_epoch"],"epochs":result["epochs"],"train_hours":len(train),
                                "assessment_hours":len(valid),"fit_max":result["preprocessing"]["fit_max"],
                                "assessment_min":str(valid.date_time.min()),"assessment_max":str(valid.date_time.max()),
                                "mlflow_run_id":run.info.run_id})
                logger.info("Fold MAE %.2f; best epoch %d",result["mae"],result["best_epoch"])
    for group in GROUPS:
        compare(group,"64_32","feature_ablation")
    ablation=pd.DataFrame(records).groupby(["feature_group","feature_count","architecture"]).agg(
        mean_fold_mae=("mae","mean"),folds=("fold","size"),mae_std=("mae","std")).reset_index()
    ablation["eligible_final_candidate"]=ablation.feature_group.isin(ELIGIBLE_GROUPS)
    ablation=ablation.sort_values(["mean_fold_mae","feature_count","feature_group"])
    eligible=ablation.loc[ablation.eligible_final_candidate]
    group=str(eligible.iloc[0].feature_group)
    for architecture in ("32","128_64"):
        compare(group,architecture,"architecture_search")
    trials=pd.DataFrame(records)
    architectures=trials.loc[trials.feature_group.eq(group)].groupby("architecture").agg(
        mean_fold_mae=("mae","mean"),folds=("fold","size"),mean_best_epoch=("best_epoch","mean")).reset_index()
    selected=architectures.sort_values(["mean_fold_mae","architecture"]).iloc[0]
    architecture=str(selected.architecture)
    matching=trials.loc[trials.feature_group.eq(group)&trials.architecture.eq(architecture)]
    fixed_epochs=max(1,int(np.floor(matching.best_epoch.median()+.5)))
    logger.info("Frozen by inner comparisons: group=%s architecture=%s epochs=%d; now fitting full pre2017 training",group,architecture,fixed_epochs)
    ablation.to_csv(folder / "feature_ablation.csv",index=False,float_format="%.10f")
    architectures.sort_values("mean_fold_mae").to_csv(folder / "architecture_comparison.csv",index=False,float_format="%.10f")
    trials.to_csv(folder / "inner_fold_results.csv",index=False,float_format="%.10f")
    (folder / "fold_preprocessing.json").write_text(json.dumps(fold_states,indent=2)+"\n",encoding="utf-8")
    (folder / "fold_preprocessing.json").write_text(json.dumps(fold_states,indent=2)+"\n",encoding="utf-8")
    validation=frame.loc[frame.date_time.ge("2017-01-01")&frame.date_time.lt("2018-01-01")].copy()
    del frame
    state=fit_state(search_data)
    columns=list(GROUPS[group])
    x_train=feature_frame(search_data,state).loc[:,columns]
    x_validation=feature_frame(validation,state).loc[:,columns]
    x_scaler,y_scaler=StandardScaler(),StandardScaler()
    tx=x_scaler.fit_transform(x_train)
    ty=y_scaler.fit_transform(search_data[["traffic_volume"]])
    model=make_model(ARCHITECTURES[architecture],len(columns))
    model_folder=BASE / "models/refined_neural"
    model_folder.mkdir(exist_ok=True)
    with mlflow.start_run(experiment_id=exp_id,run_name="frozen_configuration_2017_check") as run:
        mlflow.log_params({"feature_group":group,"features":len(columns),"architecture":architecture,"fixed_epochs":fixed_epochs,
                           "selection":"inner_2015_2016_mean_mae","no_2017_early_stopping":True,"seed":SEED})
        model.fit(data_batches(tx,ty,True),epochs=fixed_epochs,verbose=0,shuffle=False)
        prediction=predict_bundle(model,x_scaler,y_scaler,x_validation)
        metrics={"validation_mae":float(mean_absolute_error(validation.traffic_volume,prediction)),
                 "validation_r2":float(r2_score(validation.traffic_volume,prediction))}
        mlflow.log_metrics(metrics)
        model.save(model_folder / "model.keras")
        joblib.dump({"state":state,"features":columns,"x_scaler":x_scaler,"y_scaler":y_scaler},model_folder / "preprocessing.joblib")
        pd.DataFrame({"date_time":validation.date_time,"actual":validation.traffic_volume,"prediction":prediction}).to_csv(folder / "validation_predictions.csv",index=False,float_format="%.10f")
        config={"version":"v2_inner_time_search","source_sha256":digest,"feature_group":group,"features":columns,"architecture":list(ARCHITECTURES[architecture]),
                "model_sha256":hashlib.sha256((model_folder / "model.keras").read_bytes()).hexdigest(),
                "preprocessing_sha256":hashlib.sha256((model_folder / "preprocessing.joblib").read_bytes()).hexdigest(),
                "fixed_epochs":fixed_epochs,"inner_mean_mae":float(selected.mean_fold_mae),"selection_data_before":"2017-01-01",
                "eligible_feature_groups":list(ELIGIBLE_GROUPS),"folds":FOLDS,"metrics":metrics,"test_scored":False,
                "fit_max":state["fit_max"],"fit_hours":len(search_data),"outer_validation_hours":len(validation),
                "final_run_id":run.info.run_id,"trial_count":len(records),
                "selection_method":"Two-stage bounded search: fixed-architecture feature ablation then selected-group architecture comparison; equal mean of two fold MAEs",
                "limitations":["Not exhaustive AutoML or global feature/architecture optimum","One seed, two time folds; 2015 has missing months",
                                "Inner assessment periods also select early-stopping epochs; scores are selection evidence",
                                "2017 was already used for earlier models and is not a pristine outer test","No further selection based on this 2017 check; 2018 remains reserved",
                                "Ablations and architecture choices interact; not all joint combinations searched"]}
        (folder / "refinement_manifest.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
        (model_folder / "metadata.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
        for path in [*folder.glob("*.csv"),folder / "refinement_manifest.json",*model_folder.glob("*")]:
            mlflow.log_artifact(str(path),"evidence")
        logger.info("Frozen refined MLP validation: MAE %.2f, R2 %.4f",metrics["validation_mae"],metrics["validation_r2"])
    evidence=[]
    for run_id in [record["mlflow_run_id"] for record in records]+[config["final_run_id"]]:
        actual=client.get_run(run_id)
        if actual.info.status!="FINISHED":
            raise ValueError("Incomplete refinement run")
        evidence.append({"run_id":run_id,"status":actual.info.status,"params":actual.data.params,"metrics":actual.data.metrics})
    (folder / "mlflow_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise RuntimeError("Hourly source changed during refinement")
    logger.info("Saved %d inner runs plus one frozen-config check; no 2018 scoring",len(records))


if __name__=="__main__":
    configure_logging(BASE / "refinement.log")
    try:
        run_refinement()
    except Exception:
        logger.error("Feature/architecture refinement failed",exc_info=True)
        sys.exit(1)
