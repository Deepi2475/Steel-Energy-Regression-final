# scripts/09_eval_new_window.py
import os, yaml, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.pyfunc import load_model

DATA_DIR = "data/processed"
MODELS = [
    ("steel_energy_gbm", "champion"),
    ("steel_energy_dnn", "staging"),
    ("steel_energy_randomforest", "staging"),
]

def load_schema_pre():
    with open(os.path.join(DATA_DIR, "feature_schema.yaml"), "r") as f:
        schema = yaml.safe_load(f)
    pre = joblib.load(os.path.join(DATA_DIR, "preprocess_pipeline.pkl"))
    return schema, pre

def predict_h2o_pyfunc(model, X_t):
    # rename to f_i (as in training)
    x_cols = [f"f_{i}" for i in range(X_t.shape[1])]
    X_df = pd.DataFrame(X_t, columns=x_cols)
    pred_df = model.predict(X_df)
    return pred_df.iloc[:, 0].to_numpy()

def main():
    schema, pre = load_schema_pre()
    feature_cols, target = schema["feature_cols"], schema["target"]

    df_new = pd.read_csv(os.path.join(DATA_DIR, "drift_target.csv"))
    X_raw = df_new[feature_cols]
    y_true = df_new[target].values

    rows = []
    for name, alias in MODELS:
        model = load_model(f"models:/{name}@{alias}")

        # Preprocess consistently
        X_t = pre.transform(X_raw)

        if ("gbm" in name) or ("dnn" in name):
            y_pred = predict_h2o_pyfunc(model, X_t)
        else:
            y_pred = model.predict(X_t)
            if hasattr(y_pred, "ravel"):
                y_pred = y_pred.ravel()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        rows.append({"model": name, "alias": alias, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"{name}@{alias} -> RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "new_window_metrics.csv"), index=False)
    print("Saved new_window_metrics.csv")

if __name__ == "__main__":
    main()
