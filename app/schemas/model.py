from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ModelBase(BaseModel):
    """Model temel şeması"""
    inverter_id: int
    version: str
    model_type: str = "RandomForestRegressor"
    is_active: bool = True

class ModelCreate(ModelBase):
    """Model oluşturma şeması"""
    pass

class ModelInfo(ModelBase):
    """Model bilgi şeması"""
    id: int
    model_path: str
    metrics: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ModelMetrics(BaseModel):
    """Model metrikleri şeması"""
    inverter_id: int
    exists: bool
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    created_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    message: Optional[str] = None

class ModelTrainingResponse(BaseModel):
    """Model eğitim yanıt şeması"""
    success: bool
    message: str
    inverter_id: int
    status: str
    model_info: Optional[Dict[str, Any]] = None 