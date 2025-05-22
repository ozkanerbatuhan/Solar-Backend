from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class LogEntry(BaseModel):
    """Log kaydı şeması"""
    model_config = ConfigDict(from_attributes=True)
    timestamp: datetime
    level: str
    message: str

class JobStatus(BaseModel):
    """Job durum şeması"""
    model_config = ConfigDict(from_attributes=True)
    id: str
    status: str
    message: Optional[str] = None

class JobSummary(BaseModel):
    """Job özet şeması"""
    model_config = ConfigDict(from_attributes=True)
    id: str
    type: str
    status: str
    progress: int
    description: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime
    last_message: Optional[str] = None
    error: Optional[str] = None

class JobDetail(BaseModel):
    """Job detay şeması"""
    model_config = ConfigDict(from_attributes=True)
    id: str
    type: str
    status: str
    progress: int
    params: Dict[str, Any] = {}
    description: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime
    logs: List[LogEntry] = []
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_trace: Optional[str] = None

class JobList(BaseModel):
    """Job listesi şeması"""
    model_config = ConfigDict(from_attributes=True)
    total: int
    jobs: List[JobSummary] 