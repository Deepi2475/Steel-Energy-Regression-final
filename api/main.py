from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
import os
import yaml
import pandas as pd
from .schemas import PredictRequest, PredictResponse
from .loader import load_preprocessor, load_model_by_name

import mlflow
from dotenv import load_dotenv


load_dotenv("config/mlflow.env")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

app = FastAPI(title="Steel Energy Prediction API")

preprocessor = load_preprocessor()

# Helper: convert incoming records to DataFrame with expected columns
def records_to_df(records):
    # Ensure aliases are mapped properly
    rows = []
    for r in records:
        item = r.model_dump(by_alias=True)
        rows.append(item)
    df = pd.DataFrame(rows)
    return df

import yaml
import pandas as pd
from datetime import datetime, timezone
from fastapi import HTTPException

def predict_common(model_name, alias, req: PredictRequest):
    try:
        df = records_to_df(req.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    model = load_model_by_name(model_name, alias)

    with open("data/processed/feature_schema.yaml", "r") as f:
        schema = yaml.safe_load(f)

    if "gbm" in model_name or "dnn" in model_name:
        X = df[schema["feature_cols"]]
        X_t = preprocessor.transform(X)
        x_cols = [f"f_{i}" for i in range(X_t.shape[1])]
        X_df = pd.DataFrame(X_t, columns=x_cols)

        pred_df = model.predict(X_df)
        preds = pred_df.iloc[:, 0].to_numpy().tolist()
    else:
        X = df[schema["feature_cols"]]
        X_t = preprocessor.transform(X)
        preds = model.predict(X_t).ravel().tolist()

    return PredictResponse(
        model_name=model_name,
        model_version=alias,
        timestamp=datetime.now(timezone.utc).isoformat(),
        predictions=preds,
    )



@app.post("/predict_model1", response_model=PredictResponse)
def predict_model1(req: PredictRequest):
    return predict_common("steel_energy_gbm", "champion", req)

@app.post("/predict_model2", response_model=PredictResponse)
def predict_model2(req: PredictRequest):
    return predict_common("steel_energy_dnn", "staging", req)

@app.post("/predict_model3", response_model=PredictResponse)
def predict_model3(req: PredictRequest):
    return predict_common("steel_energy_randomforest", "staging", req)

@app.get("/")
def root():
    return {"message": "Steel Energy Prediction API", "endpoints": ["/predict_model1", "/predict_model2", "/predict_model3"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

