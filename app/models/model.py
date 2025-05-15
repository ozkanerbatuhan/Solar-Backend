from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base

class Model(Base):
    """
    Eğitilmiş makine öğrenimi modellerinin veritabanı tablosu.
    """
    __tablename__ = "models"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=False)
    version = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    model_type = Column(String, nullable=False, default="RandomForestRegressor")
    metrics = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # İlişkiler
    inverter = relationship("Inverter", back_populates="models")

    def __repr__(self):
        return f"<Model(id={self.id}, inverter_id={self.inverter_id}, version={self.version})>" 