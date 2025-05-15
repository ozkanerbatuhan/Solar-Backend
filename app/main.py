from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.routes import inverter_routes, model_routes, data_routes, weather_routes
from app.db.database import engine, Base, get_db
from app.core.config import settings

# Veritabanı tablolarını oluştur
Base.metadata.create_all(bind=engine)

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Solar Inverter Prediction API",
    description="Güneş enerjisi inverterları için güç çıktısı tahmin API'si",
    version="0.1.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (üretimde değiştirilmeli)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API rotalarını ekle
app.include_router(inverter_routes.router, prefix="/api/inverters", tags=["Inverters"])
app.include_router(model_routes.router, prefix="/api/models", tags=["Models"])
app.include_router(data_routes.router, prefix="/api/data", tags=["Data"])
app.include_router(weather_routes.router, prefix="/api/weather", tags=["Weather"])

# Model klasörünü oluştur
os.makedirs(os.path.join(os.path.dirname(__file__), "ml", "models"), exist_ok=True)

@app.get("/")
async def root():
    """API kök endpoint'i"""
    return {
        "message": "Solar Inverter Prediction API",
        "version": "0.1.0",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i"""
    return {
        "status": "healthy",
        "api_version": "0.1.0",
        "database": "connected"
    } 