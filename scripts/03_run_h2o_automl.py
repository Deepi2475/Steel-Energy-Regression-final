import os
import yaml
import pandas as pd
import joblib
import h2o
from h2o.automl import H2OAutoML

DATA_DIR = "data/processed"
RESULTS_DIR = "reports"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load schema
    with open(os.path.join(DATA_DIR, "feature_schema.yaml"), "r") as f:
        schema = yaml.safe_load(f)

    target = schema["target"]
    feature_cols = schema["feature_cols"]

    pre = joblib.load(os.path.join(DATA_DIR, "preprocess_pipeline.pkl"))

    # Load data
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "validate.csv"))

    X_train = train[feature_cols]
    y_train = train[target]
    X_val = val[feature_cols]
    y_val = val[target]

    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)

    # Start H2O
    h2o.init(
        max_mem_size="4G",
        nthreads=-1,
        port=54321
    )

    # Build H2O frames
    x_cols = [f"f_{i}" for i in range(X_train_t.shape[1])]

    train_h2o = h2o.H2OFrame(pd.DataFrame(X_train_t, columns=x_cols))
    train_h2o[target] = h2o.H2OFrame(y_train.to_numpy().reshape(-1, 1))

    val_h2o = h2o.H2OFrame(pd.DataFrame(X_val_t, columns=x_cols))
    val_h2o[target] = h2o.H2OFrame(y_val.to_numpy().reshape(-1, 1))

    # Force numeric target
    train_h2o[target] = train_h2o[target].asnumeric()
    val_h2o[target] = val_h2o[target].asnumeric()

    # Run AutoML
    aml = H2OAutoML(
        max_runtime_secs=1800,
        seed=42,
        sort_metric="RMSE"
    )

    aml.train(
        x=x_cols,
        y=target,
        training_frame=train_h2o,
        leaderboard_frame=val_h2o
    )

    # Save leaderboard
    lb = aml.leaderboard.as_data_frame()
    lb.to_csv(os.path.join(RESULTS_DIR, "h2o_leaderboard.csv"), index=False)

    # Select top 3 distinct algorithm types
    def model_type(mid: str) -> str:
        if mid.startswith("XGBoost"):
            return "xgboost"
        if mid.startswith("GBM"):
            return "h2o_gbm"
        if mid.startswith("DRF"):
            return "random_forest"
        if mid.startswith("GLM"):
            return "glm"
        if mid.startswith("DeepLearning"):
            return "dnn"
        if mid.startswith("StackedEnsemble"):
            return "stacked_ensemble"
        return "other"

    selected = []
    seen = set()

    for _, row in lb.iterrows():
        mtype = model_type(row["model_id"])
        if mtype not in seen and mtype not in {"stacked_ensemble", "other"}:
            selected.append({
                "model_id": row["model_id"],
                "type": mtype,
                "rmse": row["rmse"],
                "mae": row.get("mae")
            })
            seen.add(mtype)
        if len(selected) == 3:
            break

    pd.DataFrame(selected).to_csv(
        os.path.join(RESULTS_DIR, "selected_models.csv"),
        index=False
    )

    print("Selected top 3 algorithm types:", [s["type"] for s in selected])

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
