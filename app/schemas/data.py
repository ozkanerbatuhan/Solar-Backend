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