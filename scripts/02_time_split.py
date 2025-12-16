import os
import pandas as pd
import yaml

DATA_DIR = "data/processed"

def main():
    with open(os.path.join(DATA_DIR, "feature_schema.yaml"), "r") as f:
        schema = yaml.safe_load(f)
    dt_col = schema["datetime_col"]

    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_full.csv"))
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col)

    t_min, t_max = df[dt_col].min(), df[dt_col].max()
    total_range = (t_max - t_min).total_seconds()
    t_train_end = t_min + pd.to_timedelta(total_range * 0.35, unit="s")
    t_val_end = t_min + pd.to_timedelta(total_range * 0.70, unit="s")  # 35% train + 35% val

    train_df = df[df[dt_col] <= t_train_end]
    val_df = df[(df[dt_col] > t_train_end) & (df[dt_col] <= t_val_end)]
    test_df = df[df[dt_col] > t_val_end]

    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "validate.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"Saved splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == "__main__":
    main()

