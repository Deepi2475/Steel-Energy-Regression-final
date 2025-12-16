import os, yaml, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.h2o
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

EXP_NAME = "steel_energy_models"
DATA_DIR = "data/processed"

def load_data():
    with open(os.path.join(DATA_DIR, "feature_schema.yaml"), "r") as f:
        schema = yaml.safe_load(f)
    pre = joblib.load(os.path.join(DATA_DIR, "preprocess_pipeline.pkl"))
    feature_cols = schema["feature_cols"]; target = schema["target"]

    def prep(split):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
        X = pre.transform(df[feature_cols])
        y = df[target].values
        return X, y
    return schema, pre, prep

def main():
    mlflow.set_experiment(EXP_NAME)
    h2o.init()

    schema, pre, prep = load_data()
    X_train, y_train = prep("train")
    X_val, y_val = prep("validate")
    X_test, y_test = prep("test")

    # Convert to H2OFrame
    x_cols = [f"f_{i}" for i in range(X_train.shape[1])]
    train_h2o = h2o.H2OFrame(pd.DataFrame(X_train, columns=x_cols))
    train_h2o[schema["target"]] = h2o.H2OFrame(y_train.reshape(-1,1))
    val_h2o = h2o.H2OFrame(pd.DataFrame(X_val, columns=x_cols))
    val_h2o[schema["target"]] = h2o.H2OFrame(y_val.reshape(-1,1))
    test_h2o = h2o.H2OFrame(pd.DataFrame(X_test, columns=x_cols))
    test_h2o[schema["target"]] = h2o.H2OFrame(y_test.reshape(-1,1))

    params = {
        "ntrees": 300,
        "max_depth": 6,
        "learn_rate": 0.05,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "seed": 42,
    }

    with mlflow.start_run(run_name="h2o_gbm"):
        mlflow.log_params(params)
        model = H2OGradientBoostingEstimator(**params)
        model.train(x=x_cols, y=schema["target"], training_frame=train_h2o, validation_frame=val_h2o)

        # Evaluate
        val_pred = model.predict(val_h2o).as_data_frame().values.flatten()
        test_pred = model.predict(test_h2o).as_data_frame().values.flatten()

        for split, y_true, y_pred in [("val", y_val, val_pred), ("test", y_test, test_pred)]:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mlflow.log_metric(f"{split}_rmse", rmse)
            mlflow.log_metric(f"{split}_mae", mae)
            mlflow.log_metric(f"{split}_r2", r2)

        mlflow.h2o.log_model(model, "model", registered_model_name="steel_energy_gbm")

    print("H2O GBM training completed.")

if __name__ == "__main__":
    main()
