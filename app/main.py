from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from sqlalchemy import text

# Veritabanı ve modeller
from app.db.database import engine, Base, get_db
from app.models import model, inverter, weather
from app.core.config import settings

# API rotaları
from app.api.routes import inverter_routes, model_routes, data_routes, weather_routes

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Veritabanı tablolarını oluştur
try:
    logger.info("Veritabanı tablolarını oluşturma başlıyor...")
    Base.metadata.create_all(bind=engine)
    logger.info("Veritabanı tabloları başarıyla oluşturuldu.")
except Exception as e:
    logger.error(f"Veritabanı tablolarını oluştururken hata: {e}")
    # Hataya rağmen uygulamayı çalıştırmaya devam et

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
    try:
        # Bağlantı durumunu kontrol etme
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception as e:
        db_status = f"disconnected (error: {str(e)})"
    
    return {
        "status": "healthy",
        "api_version": "0.1.0",
        "database": db_status
    } 