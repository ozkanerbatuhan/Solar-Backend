from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class DateRange(BaseModel):
    """Tarih aralığı şeması"""
    model_config = ConfigDict(from_attributes=True)
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    days: Optional[int] = None

class DataStatistics(BaseModel):
    """Veri istatistikleri şeması"""
    model_config = ConfigDict(from_attributes=True)
    total_records: int
    date_range: Optional[DateRange] = None
    inverter_count: int
    inverter_ids: List[int]

class DataUploadResponse(BaseModel):
    """Veri yükleme yanıt şeması"""
    model_config = ConfigDict(from_attributes=True)
    success: bool
    message: str
    processed_rows: int
    statistics: Optional[Dict[str, Any]] = None

class TxtDataUploadResponse(DataUploadResponse):
    """TXT veri yükleme yanıt şeması"""
    conflict_count: Optional[int] = 0
    updated_count: Optional[int] = 0
    total_inverters: Optional[int] = 0
    job_id: Optional[str] = None

class ModelTrainingStatus(BaseModel):
    """Model eğitim durumu şeması"""
    model_config = ConfigDict(from_attributes=True)
    job_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    progress: float  # 0-100 arası
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    inverter_id: Optional[int] = None
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class ModelTrainingResponse(BaseModel):
    """Model eğitim yanıt şeması"""
    model_config = ConfigDict(from_attributes=True)
    success: bool
    message: str
    job_id: str
    status: str 