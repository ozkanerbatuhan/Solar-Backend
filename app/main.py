from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from sqlalchemy import text
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Veritabanı ve modeller
from app.db.database import engine, Base, get_db
from app.models import model, inverter, weather

# API rotaları
from app.core.config import settings
from app.api.routes import inverter_routes, model_routes, data_routes, weather_routes

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

def ensure_database_exists():
    """Veritabanının var olduğundan emin ol, yoksa oluştur"""
    try:
        # PostgreSQL config bilgilerini settings'ten al
        db_user = settings.POSTGRES_USER
        db_password = settings.POSTGRES_PASSWORD
        db_host = settings.POSTGRES_SERVER
        db_port = settings.POSTGRES_PORT
        db_name = settings.POSTGRES_DB
        
        # Postgres root veritabanına bağlan
        conn = psycopg2.connect(
            dbname="postgres",  # root DB
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Veritabanı var mı kontrol et
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()
        
        # Yoksa oluştur
        if not exists:
            cur.execute(f"CREATE DATABASE {db_name};")
            logger.info(f"✔️ Veritabanı oluşturuldu: {db_name}")
        else:
            logger.info(f"✅ Veritabanı zaten var: {db_name}")
        
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ DB kontrol hatası: {e}")

# Veritabanının var olduğundan emin ol
ensure_database_exists()

# Veritabanı tablolarını oluştur
try:
    logger.info("Veritabanı tablolarını oluşturma başlıyor...")
    Base.metadata.create_all(bind=engine)
    logger.info("Veritabanı tabloları başarıyla oluşturuldu.")
except Exception as e:
    logger.error(f"Veritabanı tablolarını oluştururken hata: {e}")
    # Hata aldık ama çalıştırmaya devam et

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Solar Inverter Prediction API",
    description="Güneş enerjisi inverterleri için güç çıktısı tahmin API'si",
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
        # Önce veritabanı varlığını kontrol et
        ensure_database_exists()
        
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