import os
import joblib
import yaml
import mlflow
from mlflow import pyfunc

DATA_DIR = "data/processed"

def load_preprocessor():
    return joblib.load(os.path.join(DATA_DIR, "preprocess_pipeline.pkl"))

def load_model_by_name(name: str, alias: str | None = None):
    if alias:
        uri = f"models:/{name}@{alias}"
    else:
        uri = f"models:/{name}@champion"  # default to champion
    print(f"Loading model from URI: {uri}")
    return pyfunc.load_model(uri)



