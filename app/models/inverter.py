from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base

class Inverter(Base):
    """İnverter verilerini saklayan model."""
    __tablename__ = "inverters"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    location = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    capacity = Column(Float, nullable=True)  # kW cinsinden kapasite
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    data = relationship("InverterData", back_populates="inverter", cascade="all, delete-orphan")
    predictions = relationship("InverterPrediction", back_populates="inverter", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="inverter", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Inverter {self.name}>"

class InverterData(Base):
    """İnverter ölçüm verilerini saklayan model."""
    __tablename__ = "inverter_data"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    power_output = Column(Float, nullable=False)
    temperature = Column(Float, nullable=True)
    irradiance = Column(Float, nullable=True)
    additional_data = Column(JSON, nullable=True)  # Esnek veri yapısı için JSON alanı
    
    # İlişkiler
    inverter = relationship("Inverter", back_populates="data")
    
    def __repr__(self):
        return f"<InverterData {self.inverter_id} at {self.timestamp}>"

class InverterPrediction(Base):
    """İnverter güç çıktı tahminlerini saklayan model."""
    __tablename__ = "inverter_predictions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    prediction_timestamp = Column(DateTime, nullable=False)  # Tahmin edilen zaman
    predicted_power_output = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)  # Tahmin güven değeri
    features = Column(JSON, nullable=True)  # Tahmin için kullanılan özellikler
    
    # İlişkiler
    inverter = relationship("Inverter", back_populates="predictions")
    
    def __repr__(self):
        return f"<InverterPrediction {self.inverter_id} for {self.prediction_timestamp}>"

# Model sınıfını kaldırıyoruz, çünkü app/models/model.py içinde zaten tanımlandı
# Burada tanımlanmış olması çakışmaya neden oluyor
# Bu sınıfı kaldırmak yerine yorum satırı yapıyoruz, gerekirse referans için

"""
class Model(Base):
    # Makine öğrenimi modellerinin meta verilerini saklayan model.
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=False)
    version = Column(String(50), nullable=False)
    file_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON, nullable=True)  # Model performans metrikleri
    
    # İlişkiler
    inverter = relationship("Inverter") 
""" 