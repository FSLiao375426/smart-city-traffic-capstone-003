"""Fresh-process model + disk preprocessing inference checks for neural bundles."""
import hashlib
import json
import logging
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","0")
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from part2_python.pipeline import PROJECT_ROOT,configure_logging
from part3_machine_learning.refinement import feature_frame

logger=logging.getLogger(__name__)
BASE=PROJECT_ROOT / "part3_machine_learning"


def verify_bundle(name):
    refined=name=="refined_neural"
    model_folder=BASE / "models" / name
    result_folder=BASE / "results" / ("refinement" if refined else "neural")
    model_path=model_folder / "model.keras"
    state_path=model_folder / "preprocessing.joblib"
    # Neither this model object nor these scalers existed before this process.
    model=tf.keras.models.load_model(model_path,compile=False)
    bundle=joblib.load(state_path)
    source=PROJECT_ROOT / "data/processed" / ("traffic_hourly.csv" if refined else "traffic_supervised.csv")
    data=pd.read_csv(source,keep_default_na=False)
    if refined:
        data=data.loc[data.date_time.ge("2017-01-01")&data.date_time.lt("2018-01-01")]
        features=feature_frame(data,bundle["state"]).loc[:,bundle["features"]]
    else:
        data=data.loc[data.split.eq("validation")]
        features=data.loc[:,bundle["features"]]
    x=bundle["x_scaler"].transform(features).astype("float32")
    estimate=model(x,training=False).numpy().reshape(-1).astype(float)
    prediction=estimate*float(bundle["y_scaler"].scale_[0])+float(bundle["y_scaler"].mean_[0])
    saved=pd.read_csv(result_folder / "validation_predictions.csv")
    if len(saved)!=8713 or len(data)!=8713 or saved.date_time.tolist()!=data.date_time.tolist():
        raise ValueError("Validation ordering or row counts differ")
    np.testing.assert_allclose(prediction,saved.prediction.to_numpy(),rtol=0,atol=1e-6)
    result={"model":name,"fresh_process_id":os.getpid(),"validation_hours":len(saved),
            "max_absolute_prediction_difference":float(np.max(np.abs(prediction-saved.prediction.to_numpy()))),
            "absolute_tolerance":1e-6,"relative_tolerance":0,"test_scored":False,
            "preprocessing_loaded_from_disk":True,"model_loaded_from_disk":True,
            "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "preprocessing_sha256":hashlib.sha256(state_path.read_bytes()).hexdigest(),
            "scope":"Reloaded model/scalers and prepared validation features; refined model also reloads fitted imputation/scaling state"}
    (result_folder / "reload_evidence.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    logger.info("Fresh-process bundle %s: %d predictions match; max absolute difference %.10g",name,len(saved),result["max_absolute_prediction_difference"])
    return result


if __name__=="__main__":
    configure_logging(BASE / "verify_neural_reload.log")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.experimental.enable_op_determinism()
        for name in ("neural_regressor","refined_neural"):
            verify_bundle(name)
    except Exception:
        logger.error("Fresh-process neural bundle verification failed",exc_info=True)
        sys.exit(1)
