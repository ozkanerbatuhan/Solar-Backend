from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class DateRange(BaseModel):
    """Tarih aralığı şeması"""
    min_date: datetime
    max_date: datetime
    days: int

class DataStatistics(BaseModel):
    """Veri istatistikleri şeması"""
    total_records: int
    date_range: Optional[DateRange] = None
    inverter_count: int
    inverter_ids: List[int]

class DataUploadResponse(BaseModel):
    """Veri yükleme yanıt şeması"""
    success: bool
    message: str
    processed_rows: int
    statistics: Optional[Dict[str, Any]] = None 