import os, yaml, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

EXP_NAME = "steel_energy_models"
DATA_DIR = "data/processed"
ART_DIR = "mlflow_artifacts"

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

def eval_and_log(y_true, y_pred, split_label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mlflow.log_metric(f"{split_label}_rmse", rmse)
    mlflow.log_metric(f"{split_label}_mae", mae)
    mlflow.log_metric(f"{split_label}_r2", r2)
    return rmse, mae, r2

def plot_residuals(y_true, y_pred, name):
    plt.figure(figsize=(6,4))
    resid = y_true - y_pred
    sns.histplot(resid, bins=50, kde=True)
    plt.title(f"Residuals - {name}")
    os.makedirs(ART_DIR, exist_ok=True)
    pth = os.path.join(ART_DIR, f"residuals_{name}.png")
    plt.savefig(pth); plt.close()
    mlflow.log_artifact(pth)

def main():
    mlflow.set_experiment(EXP_NAME)
    schema, pre, prep = load_data()
    X_train, y_train = prep("train")
    X_val, y_val = prep("validate")
    X_test, y_test = prep("test")

    params = {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="random_forest_regressor"):
        mlflow.log_params(params)
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        eval_and_log(y_val, val_pred, "val")
        eval_and_log(y_test, test_pred, "test")

        plot_residuals(y_val, val_pred, "rf_val")
        plot_residuals(y_test, test_pred, "rf_test")

        mlflow.sklearn.log_model(model, "model", registered_model_name="steel_energy_randomforest")
        mlflow.log_artifact(os.path.join(DATA_DIR, "preprocess_pipeline.pkl"))

    print("RandomForest training completed.")

if __name__ == "__main__":
    main()
