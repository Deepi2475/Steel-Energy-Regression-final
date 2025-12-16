# scripts/08_prepare_new_window.py
import os, yaml
import pandas as pd

DATA_DIR = "data/processed"

def main():
    with open(os.path.join(DATA_DIR, "feature_schema.yaml"), "r") as f:
        schema = yaml.safe_load(f)

    # Load all splits you already have
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "validate.csv"))
    df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Concatenate into one full dataset
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    dt_col = schema.get("datetime_col", "date")
    df_all[dt_col] = pd.to_datetime(df_all[dt_col])
    df_all = df_all.sort_values(dt_col)

    # Split into reference vs. newer window
    cutoff_idx = int(len(df_all) * 0.8)
    df_ref = df_all.iloc[:cutoff_idx].copy()
    df_new = df_all.iloc[cutoff_idx:].copy()

    df_ref.to_csv(os.path.join(DATA_DIR, "drift_reference.csv"), index=False)
    df_new.to_csv(os.path.join(DATA_DIR, "drift_target.csv"), index=False)

    print("Saved drift_reference.csv and drift_target.csv")

if __name__ == "__main__":
    main()
