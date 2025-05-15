import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, field_validator


# api link https://api.open-meteo.com/v1/forecast?latitude=37.8713&longitude=32.4846&hourly=temperature_2m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,global_tilted_irradiance,terrestrial_radiation,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,global_tilted_irradiance_instant,terrestrial_radiation_instant,wind_speed_10m,relative_humidity_2m,visibility&forecast_days=14
class Settings(BaseSettings):
    """
    Uygulama ayarları.
    Çevre değişkenleri veya .env dosyasından yüklenir.
    """

    # API ayarları
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Solar Inverter Prediction API"

    # PostgreSQL veritabanı ayarları
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "solar_db")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "1234"))
    DATABASE_URL: Optional[str] = None

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str):
            return v

        values = info.data
        dns = PostgresDsn.build(
            scheme="postgresql",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get(
                "POSTGRES_PORT"
            ),  # String'e dönüştürmeden direkt integer olarak geçirin
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
        return str(dns)
    # CORS ayarları
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # Model klasörü
    MODEL_DIR: str = os.path.join(Path(__file__).parents[2], "app", "ml", "models")

    # Open-meteo API ayarları
    DEFAULT_LATITUDE: float = 37.8713  # Konya için varsayılan enlem
    DEFAULT_LONGITUDE: float = 32.4846  # Konya için varsayılan boylam

    model_config = {"case_sensitive": True, "env_file": ".env"}


# Ayarları yükle
settings = Settings()
