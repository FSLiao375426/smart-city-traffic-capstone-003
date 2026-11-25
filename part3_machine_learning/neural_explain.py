"""Train a same-hour MLP regressor and explain a validation sample with Kernel SHAP."""
import hashlib
import json
import logging
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from part2_python.pipeline import PROJECT_ROOT, configure_logging
from part3_machine_learning.dataset import FEATURES

logger = logging.getLogger(__name__)
BASE = PROJECT_ROOT / "part3_machine_learning"
SEED = 42


def run_neural():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    source = PROJECT_ROOT / "data/processed/traffic_supervised.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = json.loads((BASE / "results/dataset_metadata.json").read_text())
    if digest != metadata["output_sha256"]:
        raise ValueError("Prepared dataset differs from the source metadata")
    data = pd.read_csv(source, keep_default_na=False)
    train = data.loc[data.split.eq("train")].copy()
    validation = data.loc[data.split.eq("validation")].copy()
    del data
    x_train, x_validation = train.loc[:, FEATURES], validation.loc[:, FEATURES]
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    tx = x_scaler.fit_transform(x_train).astype("float32")
    vx = x_scaler.transform(x_validation).astype("float32")
    ty = y_scaler.fit_transform(train[["traffic_volume"]]).astype("float32")
    vy = y_scaler.transform(validation[["traffic_volume"]]).astype("float32")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train_ds = tf.data.Dataset.from_tensor_slices((tx,ty)).shuffle(len(tx),seed=SEED).batch(256).with_options(options).prefetch(1)
    val_ds = tf.data.Dataset.from_tensor_slices((vx,vy)).batch(512).with_options(options)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(len(FEATURES),)),
                                 tf.keras.layers.Dense(64,activation="relu"),
                                 tf.keras.layers.Dense(32,activation="relu"),
                                 tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss="mse",metrics=["mae"])
    class EpochLog(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            if epoch==0 or (epoch+1)%10==0:
                logger.info("Epoch %d: train MAE %.2f; validation MAE %.2f vehicles/hour",epoch+1,
                            logs["mae"]*y_scaler.scale_[0],logs["val_mae"]*y_scaler.scale_[0])
    early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,min_delta=1e-4,restore_best_weights=True)
    result_folder=BASE / "results/neural"
    model_folder=BASE / "models/neural_regressor"
    figures=BASE / "figures"
    for path in (result_folder,model_folder,figures):
        path.mkdir(parents=True,exist_ok=True)
    uri="sqlite:///"+(BASE / "mlflow/mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client=mlflow.MlflowClient()
    experiment_name="capstone003_neural_regression_shap"
    exp=client.get_experiment_by_name(experiment_name)
    exp_id=exp.experiment_id if exp else client.create_experiment(experiment_name,artifact_location=(BASE / "mlflow/artifacts").as_uri())
    with mlflow.start_run(experiment_id=exp_id,run_name="mlp_64_32_kernel_shap_v1") as run:
        mlflow.set_tags({"task":"neural_regression_with_explainability","prediction_mode":"conditional_same_hour",
                         "dataset_sha256":digest,"selection_split":"2017_validation"})
        mlflow.log_params({"hidden_layers":"64,32","activation":"relu","optimizer":"Adam","learning_rate":.001,
                           "batch_size":256,"max_epochs":80,"early_stopping_patience":8,"early_stopping_monitor":"val_loss",
                           "random_seed":SEED,"train_hours":len(train),"validation_hours":len(validation),
                           "target":"traffic_volume","feature_count":len(FEATURES),"parameter_count":model.count_params(),
                           "shap_background_hours":64,"shap_validation_hours":24,"shap_nsamples":256})
        logger.info("Training MLP (%d parameters), %d training hours; test period excluded",model.count_params(),len(train))
        started=time.perf_counter()
        history=model.fit(train_ds,validation_data=val_ds,epochs=80,callbacks=[early_stop,EpochLog()],verbose=0,shuffle=False)
        fit_seconds=time.perf_counter()-started
        def raw_predict(array,network=model):
            inputs=pd.DataFrame(np.asarray(array),columns=list(FEATURES))
            scaled=x_scaler.transform(inputs).astype("float32")
            standardized=network(scaled,training=False).numpy().reshape(-1)
            return standardized.astype(float)*float(y_scaler.scale_[0])+float(y_scaler.mean_[0])
        train_prediction=raw_predict(x_train.to_numpy())
        prediction=raw_predict(x_validation.to_numpy())
        metrics={"train_mae":float(mean_absolute_error(train.traffic_volume,train_prediction)),
                 "train_r2":float(r2_score(train.traffic_volume,train_prediction)),
                 "validation_mae":float(mean_absolute_error(validation.traffic_volume,prediction)),
                 "validation_r2":float(r2_score(validation.traffic_volume,prediction)),"fit_seconds":fit_seconds}
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError("Non-finite neural network metric")
        mlflow.log_metrics(metrics)
        hist=pd.DataFrame(history.history)
        hist.insert(0,"epoch",np.arange(1,len(hist)+1))
        hist["train_mae_vehicles_per_hour"]=hist.mae*y_scaler.scale_[0]
        hist["validation_mae_vehicles_per_hour"]=hist.val_mae*y_scaler.scale_[0]
        hist.to_csv(result_folder / "learning_history.csv",index=False,float_format="%.10f")
        pd.DataFrame({"date_time":validation.date_time,"actual":validation.traffic_volume,"prediction":prediction,
                      "absolute_error":np.abs(validation.traffic_volume-prediction)}).to_csv(result_folder / "validation_predictions.csv",index=False,float_format="%.10f")
        model_path=model_folder / "model.keras"
        model.save(model_path)
        joblib.dump({"x_scaler":x_scaler,"y_scaler":y_scaler,"features":list(FEATURES)},model_folder / "preprocessing.joblib")
        reloaded=tf.keras.models.load_model(model_path,compile=False)
        np.testing.assert_allclose(raw_predict(x_validation.to_numpy(),network=reloaded),prediction,atol=1e-6,rtol=0)
        logger.info("Neural metrics: %s; computing Kernel SHAP on 24 fixed validation cases",metrics)
        rng=np.random.default_rng(SEED)
        background_index=np.sort(rng.choice(len(train),64,replace=False))
        explained_index=np.sort(rng.choice(len(validation),24,replace=False))
        background=x_train.iloc[background_index]
        explained=x_validation.iloc[explained_index]
        # Kernel SHAP explains this actual neural model; no surrogate is substituted.
        explainer=shap.KernelExplainer(raw_predict,background.to_numpy(),feature_names=list(FEATURES))
        np.random.seed(SEED)
        values=np.asarray(explainer.shap_values(explained.to_numpy(),nsamples=256,l1_reg="num_features(21)",silent=True))
        if values.ndim==3 and values.shape[-1]==1:
            values=values[...,0]
        if values.shape!=(24,len(FEATURES)) or not np.isfinite(values).all():
            raise ValueError("Unexpected SHAP values")
        base=float(np.asarray(explainer.expected_value).reshape(-1)[0])
        explained_predictions=raw_predict(explained.to_numpy())
        reconstructed=base+values.sum(axis=1)
        np.testing.assert_allclose(reconstructed,explained_predictions,atol=.01,rtol=0)
        contributions=pd.DataFrame(values,columns=list(FEATURES))
        contributions.insert(0,"date_time",validation.iloc[explained_index].date_time.to_numpy())
        contributions.to_csv(result_folder / "shap_contributions.csv",index=False,float_format="%.10f")
        cases=explained.reset_index(drop=True)
        cases.insert(0,"date_time",validation.iloc[explained_index].date_time.to_numpy())
        cases.to_csv(result_folder / "shap_input_cases.csv",index=False,float_format="%.10f")
        pd.DataFrame({"date_time":cases.date_time,"base_value":base,"prediction":explained_predictions,
                      "reconstructed_prediction":reconstructed,"actual":validation.iloc[explained_index].traffic_volume.to_numpy()}).to_csv(result_folder / "shap_reconstruction.csv",index=False,float_format="%.10f")
        importance=pd.DataFrame({"feature":list(FEATURES),"mean_abs_shap":np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
        importance.to_csv(result_folder / "shap_sample_importance.csv",index=False,float_format="%.10f")
        baseline=pd.read_csv(BASE / "results/validation_metrics.csv",na_values=["NA"])
        comparison=baseline.loc[baseline.task.eq("regression"),["model","mae","r2"]].copy()
        comparison=pd.concat([comparison,pd.DataFrame([{"model":"neural_mlp_64_32","mae":metrics["validation_mae"],"r2":metrics["validation_r2"]}])],ignore_index=True)
        comparison.to_csv(result_folder / "regression_comparison.csv",index=False,float_format="%.10f")
        manifest={"dataset_sha256":digest,"features":list(FEATURES),"architecture":[21,64,32,1],
                  "metrics":metrics,"completed_epochs":len(hist),"minimum_observed_val_loss_epoch":int(hist.val_loss.idxmin()+1),
                  "restored_best_epoch":int(early_stop.best_epoch+1),"early_stopping_min_delta":1e-4,
                  "test_scored":False,"mlflow_run_id":run.info.run_id,"versions":{"tensorflow":tf.__version__,"shap":shap.__version__},
                  "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  "background_dates":train.iloc[background_index].date_time.tolist(),"explained_dates":cases.date_time.tolist(),
                  "shap_expected_value":base,"shap_max_additivity_error":float(np.max(np.abs(reconstructed-explained_predictions))),
                  "limitations":["Same-hour conditional estimate, not a forward forecast","Validation used for early stopping; not an unbiased final test",
                                  "Kernel SHAP estimates explain this neural model, not causal effects","24 cases and 64 background rows are illustrative, not a full-population importance audit",
                                  "Correlated cyclic/weather features and masked synthetic combinations affect SHAP allocations",
                                  "Fixed 256 masks per explained case; no robustness or calibration guarantee"]}
        (result_folder / "neural_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        (model_folder / "metadata.json").write_text(json.dumps({"features":list(FEATURES),"target":"traffic_volume","prediction_mode":metadata["prediction_mode"],"metrics":metrics},indent=2)+"\n",encoding="utf-8")
        save_figures(hist,importance,values[0],cases.iloc[0].date_time,base,explained_predictions[0],figures)
        for path in [*result_folder.glob("*.csv"),*result_folder.glob("*.json"),*model_folder.glob("*")]:
            mlflow.log_artifact(str(path),"evidence" if path.parent==result_folder else "model")
        for name in ("neural_learning_curve.png","neural_shap_sample_importance.png","neural_shap_local.png"):
            mlflow.log_artifact(str(figures / name),"figures")
    finished=client.get_run(manifest["mlflow_run_id"])
    if finished.info.status!="FINISHED":
        raise ValueError("MLflow neural experiment unfinished")
    (result_folder / "mlflow_evidence.json").write_text(json.dumps({"run_id":finished.info.run_id,"status":finished.info.status,
                                                                 "params":finished.data.params,"metrics":finished.data.metrics},indent=2)+"\n",encoding="utf-8")
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise RuntimeError("Source changed during neural experiment")
    logger.info("Saved neural model, three figures and SHAP checks; no final-test predictions")


def save_figures(history,importance,local_values,date,base,prediction,folder):
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(history.epoch,history.train_mae_vehicles_per_hour,label="Training (within-epoch)")
    ax.plot(history.epoch,history.validation_mae_vehicles_per_hour,label="2017 validation")
    ax.set(title="MLP learning curves",xlabel="Epoch",ylabel="MAE (vehicles/hour)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "neural_learning_curve.png",dpi=160)
    plt.close(fig)
    part=importance.head(10).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(part.feature,part.mean_abs_shap,color="#176B9A")
    ax.set(title="Neural model: SHAP importance on 24 validation cases",xlabel="Mean absolute SHAP (vehicles/hour)")
    fig.text(.03,.015,"64 training background rows; sampled explanations, not causal or full-population importance.",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(folder / "neural_shap_sample_importance.png",dpi=160)
    plt.close(fig)
    indices=np.argsort(np.abs(local_values))[-10:]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(np.arange(len(indices)),local_values[indices],color=["#C05A30" if value>0 else "#176B9A" for value in local_values[indices]])
    ax.set_yticks(np.arange(len(indices)),[str(FEATURES[index]) for index in indices])
    ax.axvline(0,color="#555555",linewidth=.7)
    ax.set(title=f"Neural SHAP example | {date}",xlabel="Contribution to model output (vehicles/hour)")
    fig.text(.03,.015,f"Baseline {base:.1f} + all 21 contributions = prediction {prediction:.1f}; top 10 shown. Not causal effects.",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(folder / "neural_shap_local.png",dpi=160)
    plt.close(fig)
    logger.info("Saved three neural/SHAP figures in %s",folder)


if __name__=="__main__":
    configure_logging(BASE / "neural_explain.log")
    logging.getLogger("shap").setLevel(logging.WARNING)
    try:
        run_neural()
    except Exception:
        logger.error("Neural regression or SHAP explanation failed",exc_info=True)
        sys.exit(1)
