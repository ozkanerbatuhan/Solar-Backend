from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base

class WeatherData(Base):
    """Hava durumu ölçüm verilerini saklayan model."""
    __tablename__ = "weather_data"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    temperature = Column(Float, nullable=True)
    shortwave_radiation = Column(Float, nullable=True)
    direct_radiation = Column(Float, nullable=True)
    diffuse_radiation = Column(Float, nullable=True)
    direct_normal_irradiance = Column(Float, nullable=True)
    global_tilted_irradiance = Column(Float, nullable=True)
    terrestrial_radiation = Column(Float, nullable=True)
    relative_humidity = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    visibility = Column(Float, nullable=True)
    additional_data = Column(JSON, nullable=True)  # Ekstra veriler için
    is_forecast = Column(Integer, default=0)  # 0: gerçek ölçüm, 1: tahmin
    
    def __repr__(self):
        return f"<WeatherData at {self.timestamp}>"

class WeatherForecast(Base):
    """Gelecek hava durumu tahminlerini saklayan model."""
    __tablename__ = "weather_forecasts"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_timestamp = Column(DateTime, index=True)  # Tahmin edilen zaman
    created_at = Column(DateTime, default=datetime.utcnow)  # Tahmin yapılan zaman
    temperature = Column(Float, nullable=True)
    shortwave_radiation = Column(Float, nullable=True)
    direct_radiation = Column(Float, nullable=True)
    diffuse_radiation = Column(Float, nullable=True)
    direct_normal_irradiance = Column(Float, nullable=True)
    global_tilted_irradiance = Column(Float, nullable=True)
    terrestrial_radiation = Column(Float, nullable=True)
    relative_humidity = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    visibility = Column(Float, nullable=True)
    additional_data = Column(JSON, nullable=True)  # Ekstra veriler için
    forecast_source = Column(String(100), nullable=True)  # API kaynağı
    
    def __repr__(self):
        return f"<WeatherForecast for {self.forecast_timestamp}>" 