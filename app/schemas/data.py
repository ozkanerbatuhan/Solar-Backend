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

class ModelLogEntry(BaseModel):
    """Model eğitim günlüğü kaydı şeması"""
    model_config = ConfigDict(from_attributes=True)
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'ERROR', vs.
    message: str

class ModelLog(BaseModel):
    """Model eğitim günlüğü şeması"""
    model_config = ConfigDict(from_attributes=True)
    model_version: str
    inverter_id: int
    created_at: datetime
    data_summary: Dict[str, Any]  # Veri özeti - kullanılan/eksik satır sayısı vs.
    feature_importance: Dict[str, float]  # Özellik önemi
    metrics: Dict[str, Any]  # Model metrikleri
    logs: List[ModelLogEntry]  # Eğitim sırasında oluşturulan günlük kayıtları
    
class ModelLogResponse(BaseModel):
    """Model eğitim günlüğü yanıt şeması"""
    model_config = ConfigDict(from_attributes=True)
    success: bool
    message: str
    log: Optional[ModelLog] = None

class TxtDataUploadJob(BaseModel):
    """TXT veri yükleme job şeması"""
    model_config = ConfigDict(from_attributes=True)
    job_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    progress: float  # 0-100 arası
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    message: Optional[str] = None
    processed_rows: int = 0
    conflict_count: int = 0
    updated_count: int = 0
    total_inverters: int = 0
    file_name: Optional[str] = None 