import os
import yaml
import joblib
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow.pyfunc import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = "data/processed"

def main():
    # 1) Load schema and preprocessing pipeline
    with open(os.path.join(DATA_DIR, "feature_schema.yaml"), "r") as f:
        schema = yaml.safe_load(f)
    pre = joblib.load(os.path.join(DATA_DIR, "preprocess_pipeline.pkl"))

    feature_cols, target = schema["feature_cols"], schema["target"]

    # 2) Load reference and new data
    df_ref = pd.read_csv(os.path.join(DATA_DIR, "drift_reference.csv"))
    df_new = pd.read_csv(os.path.join(DATA_DIR, "drift_target.csv"))

    # 3) Data drift report
    data_drift_report = Report([DataDriftPreset()])
    eval_drift = data_drift_report.run(df_new[feature_cols], df_ref[feature_cols])
    eval_drift.save_html(os.path.join(DATA_DIR, "evidently_data_drift.html"))

    # 4) Performance drift (manual metrics)
    model = load_model("models:/steel_energy_gbm@champion")

    # Preprocess features and predict
    X_t = pre.transform(df_new[feature_cols])
    X_df = pd.DataFrame(X_t, columns=[f"f_{i}" for i in range(X_t.shape[1])])
    preds = model.predict(X_df)

    df_new_perf = df_new.copy()
    df_new_perf["prediction"] = preds.iloc[:, 0].to_numpy()

    # Compute metrics manually (RMSE via np.sqrt)
    mse = mean_squared_error(df_new_perf[target], df_new_perf["prediction"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df_new_perf[target], df_new_perf["prediction"])

    print("✅ Drift reports saved successfully")
    print(f"Performance metrics on new data -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
