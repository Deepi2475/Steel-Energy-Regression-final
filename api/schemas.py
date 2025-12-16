from pydantic import BaseModel, Field
from typing import List, Optional

class FeatureRecord(BaseModel):
    Lagging_Current_Reactive_Power_kVarh: float = Field(..., alias="Lagging_Current_Reactive.Power_kVarh")
    Leading_Current_Reactive_Power_kVarh: float = Field(..., alias="Leading_Current_Reactive_Power_kVarh")
    CO2_tCO2: float = Field(..., alias="CO2(tCO2)")
    Lagging_Current_Power_Factor: float = Field(..., alias="Lagging_Current_Power_Factor")
    Leading_Current_Power_Factor: float = Field(..., alias="Leading_Current_Power_Factor")
    NSM: float = Field(..., alias="NSM")
    WeekStatus: str = Field(..., alias="WeekStatus")
    Day_of_week: str = Field(..., alias="Day_of_week")
    Load_Type: str = Field(..., alias="Load_Type")

    class Config:
        populate_by_name = True


class PredictRequest(BaseModel):
    records: List[FeatureRecord]

class PredictResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    timestamp: str
    predictions: List[float]

