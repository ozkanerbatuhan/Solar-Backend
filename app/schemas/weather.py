from typing import Dict, List, Any, Optional
from datetime import datetime, date
from pydantic import BaseModel, Field

# WeatherData şemaları
class WeatherDataBase(BaseModel):
    """Hava durumu verisi temel şeması"""
    temperature: Optional[float] = None
    shortwave_radiation: Optional[float] = None
    direct_radiation: Optional[float] = None
    diffuse_radiation: Optional[float] = None
    direct_normal_irradiance: Optional[float] = None
    global_tilted_irradiance: Optional[float] = None
    terrestrial_radiation: Optional[float] = None
    relative_humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    visibility: Optional[float] = None

class WeatherData(WeatherDataBase):
    """Hava durumu verisi şeması"""
    id: int
    timestamp: datetime
    is_forecast: bool
    additional_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class WeatherDataCreate(WeatherDataBase):
    pass

class WeatherDataInDB(WeatherDataBase):
    id: int

    class Config:
        from_attributes = True

# WeatherForecast şemaları
class WeatherForecast(WeatherDataBase):
    """Hava durumu tahmini şeması"""
    id: int
    forecast_timestamp: datetime
    created_at: datetime
    forecast_source: str
    additional_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class WeatherForecastCreate(WeatherForecast):
    pass

class WeatherForecastInDB(WeatherForecast):
    pass

# Response şemaları
class WeatherForecast(WeatherForecastInDB):
    pass

# Ek şemalar
class WeatherDataFilter(BaseModel):
    """Hava durumu verisi filtreleme şeması"""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_forecast: Optional[bool] = None

class ForecastRequest(BaseModel):
    """Hava durumu tahmini istek şeması"""
    latitude: float = Field(37.8713, description="Enlem değeri")
    longitude: float = Field(32.4846, description="Boylam değeri")
    forecast_days: int = Field(7, description="Kaç gün sonrası için tahmin", ge=1, le=14)

class OpenMeteoResponse(BaseModel):
    """Open-meteo API yanıt şeması"""
    data: Dict[str, Any]

class CSVUploadResponse(BaseModel):
    success: bool
    message: str
    processed_rows: int
    weather_data_fetched: bool 