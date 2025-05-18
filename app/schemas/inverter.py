from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# Inverter şemaları
class InverterBase(BaseModel):
    name: str
    location: Optional[str] = None
    description: Optional[str] = None

class InverterCreate(InverterBase):
    pass

class InverterUpdate(InverterBase):
    name: Optional[str] = None

class InverterInDB(InverterBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

# InverterData şemaları
class InverterDataBase(BaseModel):
    inverter_id: int
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    power_output: float
    temperature: Optional[float] = None
    irradiance: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None

class InverterDataCreate(InverterDataBase):
    pass

class InverterDataUpdate(BaseModel):
    power_output: Optional[float] = None
    temperature: Optional[float] = None
    irradiance: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None

class InverterDataInDB(InverterDataBase):
    id: int

    model_config = ConfigDict(from_attributes=True)

# InverterPrediction şemaları
class InverterPredictionBase(BaseModel):
    inverter_id: int
    prediction_timestamp: datetime
    predicted_power_output: float
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    features: Optional[Dict[str, Any]] = None

class InverterPredictionCreate(InverterPredictionBase):
    pass

class InverterPredictionInDB(InverterPredictionBase):
    id: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

# Model şemaları
class ModelBase(BaseModel):
    inverter_id: int
    version: str
    file_path: str
    metrics: Optional[Dict[str, Any]] = None

class ModelCreate(ModelBase):
    pass

class ModelInDB(ModelBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Pagination şeması
class PaginationInfo(BaseModel):
    total_records: int
    skip: int
    limit: int
    has_more: bool

# Response şemaları
class Inverter(InverterInDB):
    pass

class InverterData(InverterDataInDB):
    pass

class InverterPrediction(InverterPredictionInDB):
    pass

class Model(ModelInDB):
    pass

# Listeleme şemaları
class InverterWithData(Inverter):
    data: List[InverterData] = []
    pagination: Optional[PaginationInfo] = None

class InverterWithPredictions(Inverter):
    predictions: List[InverterPrediction] = []

class InverterComplete(Inverter):
    data: List[InverterData] = []
    predictions: List[InverterPrediction] = [] 