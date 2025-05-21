import httpx
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

from app.models.weather import WeatherData, WeatherForecast
from app.schemas.weather import OpenMeteoResponse

# Logger yapılandırması
logger = logging.getLogger(__name__)

# Open-meteo API endpoint'leri
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"
SOLAR_RADIATION_ENDPOINT = f"{OPEN_METEO_BASE_URL}/forecast"
HISTORICAL_ENDPOINT = f"{OPEN_METEO_BASE_URL}/forecast"

# Solar radiation parametreleri
SOLAR_RADIATION_PARAMS = [
    "shortwave_radiation", 
    "direct_radiation", 
    "diffuse_radiation", 
    "direct_normal_irradiance", 
    "global_tilted_irradiance", 
    "terrestrial_radiation",
    "temperature_2m",
    "relative_humidity_2m", 
    "windspeed_10m",
    "visibility"
]

async def fetch_current_weather(
    latitude: float, 
    longitude: float, 
    db: Session,
    save_to_db: bool = True
) -> OpenMeteoResponse:
    """
    Open-meteo API'sinden güncel hava durumu verilerini çeker.
    
    Args:
        latitude: Enlem değeri
        longitude: Boylam değeri
        db: Veritabanı oturumu
        save_to_db: Verileri veritabanına kaydet
        
    Returns:
        API yanıtı ve işleme sonucu
    """
    # Bugünün tarihini al
    today = datetime.utcnow().date()
    
    # API parametreleri
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": today.isoformat(),
        "end_date": today.isoformat(),
        "hourly": ",".join(SOLAR_RADIATION_PARAMS),
        "timezone": "auto"
    }
    
    try:
        logger.info(f"OpenMeteo API'sine istek gönderiliyor: {params}")
        async with httpx.AsyncClient() as client:
            response = await client.get(SOLAR_RADIATION_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            
        if save_to_db:
            saved_rows = await _save_weather_data(data, db, is_forecast=False)
            logger.info(f"{saved_rows} adet hava durumu verisi kaydedildi")
        
        return OpenMeteoResponse(data=data)
    except Exception as e:
        logger.error(f"Hava durumu verisi çekilirken hata: {str(e)}")
        raise

async def fetch_weather_forecast(
    latitude: float, 
    longitude: float, 
    forecast_days: int,
    db: Session,
    save_to_db: bool = True
) -> OpenMeteoResponse:
    """
    Open-meteo API'sinden gelecek günler için hava durumu tahminlerini çeker.
    
    Args:
        latitude: Enlem değeri
        longitude: Boylam değeri
        forecast_days: Kaç gün sonrasına kadar tahmin isteniyor
        db: Veritabanı oturumu
        save_to_db: Verileri veritabanına kaydet
        
    Returns:
        API yanıtı ve işleme sonucu
    """
    # Bugünün ve son günün tarihini al
    start_date = datetime.utcnow().date()
    end_date = start_date + timedelta(days=forecast_days)
    
    # API parametreleri
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(SOLAR_RADIATION_PARAMS),
        "timezone": "auto"
    }
    
    try:
        logger.info(f"OpenMeteo API'sine tahmin isteği gönderiliyor: lat={latitude}, lon={longitude}, günler={forecast_days}")
        async with httpx.AsyncClient() as client:
            response = await client.get(SOLAR_RADIATION_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            
        if save_to_db:
            saved_rows = await _save_weather_forecast(data, db)
            logger.info(f"{saved_rows} adet hava durumu tahmini veritabanına kaydedildi")
        
        return OpenMeteoResponse(data=data)
    except Exception as e:
        logger.error(f"Hava durumu tahmini çekilirken hata: {str(e)}")
        raise

async def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    db: Session,
    save_to_db: bool = True
) -> OpenMeteoResponse:
    """
    Open-meteo API'sinden geçmiş tarihler için hava durumu verilerini çeker.
    
    Args:
        latitude: Enlem değeri
        longitude: Boylam değeri
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
        db: Veritabanı oturumu
        save_to_db: Verileri veritabanına kaydet
        
    Returns:
        API yanıtı ve işleme sonucu
    """
    # API parametreleri
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(SOLAR_RADIATION_PARAMS),
        "timezone": "auto"
    }
    
    try:
        logger.info(f"OpenMeteo API'sine geçmiş veri isteği gönderiliyor: {start_date} - {end_date}")
        async with httpx.AsyncClient() as client:
            response = await client.get(HISTORICAL_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            
        if save_to_db:
            saved_rows = await _save_weather_data(data, db, is_forecast=False)
            logger.info(f"{saved_rows} adet geçmiş hava durumu verisi kaydedildi")
        
        return OpenMeteoResponse(data=data)
    except Exception as e:
        logger.error(f"Geçmiş hava durumu verisi çekilirken hata: {str(e)}")
        raise

async def _save_weather_data(data: Dict[str, Any], db: Session, is_forecast: bool = False) -> int:
    """
    API'den alınan hava durumu verilerini veritabanına kaydeder.
    
    Args:
        data: API yanıtı
        db: Veritabanı oturumu
        is_forecast: Tahmin verisi mi?
        
    Returns:
        Kaydedilen satır sayısı
    """
    if "hourly" not in data:
        logger.warning("API yanıtında 'hourly' verisi bulunamadı")
        return 0
    
    hourly_data = data["hourly"]
    saved_count = 0
    
    # Tarih-saat verileri
    time_values = hourly_data.get("time", [])
    logger.info(f"Toplam {len(time_values)} adet hava durumu kaydı işlenecek")
    
    for i, time_str in enumerate(time_values):
        # ISO format tarihi datetime'a çevir
        timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        
        # Tüm değerleri topla
        weather_record = WeatherData(
            timestamp=timestamp,
            temperature=_get_value(hourly_data, "temperature_2m", i),
            shortwave_radiation=_get_value(hourly_data, "shortwave_radiation", i),
            direct_radiation=_get_value(hourly_data, "direct_radiation", i),
            diffuse_radiation=_get_value(hourly_data, "diffuse_radiation", i),
            direct_normal_irradiance=_get_value(hourly_data, "direct_normal_irradiance", i),
            global_tilted_irradiance=_get_value(hourly_data, "global_tilted_irradiance", i),
            terrestrial_radiation=_get_value(hourly_data, "terrestrial_radiation", i),
            relative_humidity=_get_value(hourly_data, "relative_humidity_2m", i),
            wind_speed=_get_value(hourly_data, "windspeed_10m", i),
            visibility=_get_value(hourly_data, "visibility", i),
            is_forecast=1 if is_forecast else 0,
            additional_data={}  # Ek verileri buraya ekleyebiliriz
        )
        
        db.add(weather_record)
        saved_count += 1
    
    try:
        db.commit()
        logger.info(f"{saved_count} adet hava durumu verisi veritabanına başarıyla kaydedildi")
    except Exception as e:
        db.rollback()
        logger.error(f"Hava durumu verileri kaydedilirken hata: {str(e)}")
        raise e
    
    return saved_count

async def _save_weather_forecast(data: Dict[str, Any], db: Session) -> int:
    """
    API'den alınan hava durumu tahmin verilerini veritabanına kaydeder.
    
    Args:
        data: API yanıtı
        db: Veritabanı oturumu
        
    Returns:
        Kaydedilen satır sayısı
    """
    if "hourly" not in data:
        logger.warning("API yanıtında 'hourly' verisi bulunamadı")
        return 0
    
    hourly_data = data["hourly"]
    saved_count = 0
    created_at = datetime.utcnow()
    
    # Tarih-saat verileri
    time_values = hourly_data.get("time", [])
    logger.info(f"Toplam {len(time_values)} adet hava durumu tahmini işlenecek")
    
    for i, time_str in enumerate(time_values):
        # ISO format tarihi datetime'a çevir
        forecast_timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        
        # Bugünden sonraki veriler için tahmin kaydı oluştur
        if forecast_timestamp > datetime.utcnow():
            # Tüm değerleri topla
            forecast_record = WeatherForecast(
                forecast_timestamp=forecast_timestamp,
                created_at=created_at,
                temperature=_get_value(hourly_data, "temperature_2m", i),
                shortwave_radiation=_get_value(hourly_data, "shortwave_radiation", i),
                direct_radiation=_get_value(hourly_data, "direct_radiation", i),
                diffuse_radiation=_get_value(hourly_data, "diffuse_radiation", i),
                direct_normal_irradiance=_get_value(hourly_data, "direct_normal_irradiance", i),
                global_tilted_irradiance=_get_value(hourly_data, "global_tilted_irradiance", i),
                terrestrial_radiation=_get_value(hourly_data, "terrestrial_radiation", i),
                relative_humidity=_get_value(hourly_data, "relative_humidity_2m", i),
                wind_speed=_get_value(hourly_data, "windspeed_10m", i),
                visibility=_get_value(hourly_data, "visibility", i),
                forecast_source="open-meteo",
                additional_data={}  # Ek verileri buraya ekleyebiliriz
            )
            
            db.add(forecast_record)
            saved_count += 1
    
    try:
        db.commit()
        logger.info(f"{saved_count} adet hava durumu tahmini veritabanına başarıyla kaydedildi")
    except Exception as e:
        db.rollback()
        logger.error(f"Hava durumu tahminleri kaydedilirken hata: {str(e)}")
        raise e
    
    return saved_count

def _get_value(data: Dict[str, Any], key: str, index: int) -> Optional[float]:
    """Verileri güvenli bir şekilde almak için yardımcı fonksiyon"""
    try:
        if key in data and index < len(data[key]):
            value = data[key][index]
            return float(value) if value is not None else None
        return None
    except (ValueError, TypeError):
        return None 