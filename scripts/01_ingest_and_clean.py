import os
import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
import joblib

DATA_RAW = "data/raw/steel_energy.csv"
DATA_DIR = "data/processed"
PIPELINE_PATH = os.path.join(DATA_DIR, "preprocess_pipeline.pkl")
SCHEMA_PATH = os.path.join(DATA_DIR, "feature_schema.yaml")

def parse_datetime(df, col):
    # The dataset has a combined date/time string like "1/1/2018 0:15"
    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    return df

def normalize_text(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def cap_outliers_iqr(df, numeric_cols, factor=3.0):
    for c in numeric_cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[c] = df[c].clip(lower, upper)
    return df

def load_config():
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = load_config()

    df = pd.read_csv(DATA_RAW)
    # Rename columns if needed to match config
    # Ensure target and datetime exist
    assert cfg["target"] in df.columns, f"Missing target {cfg['target']}"
    assert cfg["datetime_col"] in df.columns, f"Missing datetime {cfg['datetime_col']}"

    # Parse datetime
    df = parse_datetime(df, cfg["datetime_col"])
    df = df.sort_values(cfg["datetime_col"]).reset_index(drop=True)

    # Normalize text fields
    df = normalize_text(df, cfg["categorical_cols"])

    # Basic cleaning: remove impossible negatives for certain features (domain-aware)
    non_negative = [cfg["target"], "CO2(tCO2)", "NSM"]
    for c in non_negative:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # Outlier capping
    if cfg.get("outlier_cap", {}).get("method") == "iqr":
        df = cap_outliers_iqr(df, cfg["numeric_cols"], factor=cfg["outlier_cap"].get("factor", 3.0))

    # Build preprocessing pipeline for features (exclude target, datetime)
    feature_cols = cfg["numeric_cols"] + cfg["categorical_cols"]
    X = df[feature_cols]
    y = df[cfg["target"]]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_imputer, cfg["numeric_cols"]),
            ("cat", Pipeline(steps=[("impute", cat_imputer), ("encode", encoder)]), cfg["categorical_cols"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Fit the preprocessor on full dataset (fits encoders/imputers, but models will be trained on train split later)
    preprocessor.fit(X)

    # Persist pipeline and schema
    joblib.dump(preprocessor, PIPELINE_PATH)

    schema = {
        "datetime_col": cfg["datetime_col"],
        "target": cfg["target"],
        "numeric_cols": cfg["numeric_cols"],
        "categorical_cols": cfg["categorical_cols"],
        "feature_cols": feature_cols,
    }
    with open(SCHEMA_PATH, "w") as f:
        yaml.safe_dump(schema, f)

    # Save a cleaned full CSV for splitting
    df.to_csv(os.path.join(DATA_DIR, "cleaned_full.csv"), index=False)
    print("Cleaned dataset and preprocessing pipeline saved.")

if __name__ == "__main__":
    main()

