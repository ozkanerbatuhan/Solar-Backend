from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, date, timedelta

from app.db.database import get_db
from app.models.weather import WeatherData, WeatherForecast
from app.schemas.weather import (
    WeatherData as WeatherDataSchema,
    WeatherForecast as WeatherForecastSchema,
    WeatherDataFilter,
    ForecastRequest,
    OpenMeteoResponse
)
from app.services.weather_service import fetch_current_weather, fetch_weather_forecast

router = APIRouter()

@router.post("/fetch-current", response_model=OpenMeteoResponse)
async def fetch_current_weather_data(
    background_tasks: BackgroundTasks,
    latitude: float = Query(37.8713, description="Enlem değeri"),
    longitude: float = Query(32.4846, description="Boylam değeri"),
    save_to_db: bool = Query(True, description="Verileri veritabanına kaydet"),
    db: Session = Depends(get_db)
):
    """
    Open-meteo API'sinden güncel hava durumu verilerini çeker.
    Varsayılan olarak sonuçlar veritabanına kaydedilir.
    """
    try:
        result = await fetch_current_weather(latitude, longitude, db, save_to_db=save_to_db)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hava durumu verileri çekilirken hata oluştu: {str(e)}"
        )

@router.post("/fetch-forecast", response_model=OpenMeteoResponse)
async def fetch_weather_forecast_data(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    save_to_db: bool = Query(True, description="Tahmin verilerini veritabanına kaydet"),
    db: Session = Depends(get_db)
):
    """
    Open-meteo API'sinden hava durumu tahmin verilerini çeker.
    Varsayılan olarak sonuçlar veritabanına kaydedilir.
    """
    try:
        result = await fetch_weather_forecast(
            request.latitude, 
            request.longitude, 
            request.forecast_days, 
            db, 
            save_to_db=save_to_db
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hava durumu tahmin verileri çekilirken hata oluştu: {str(e)}"
        )

@router.get("/data", response_model=List[WeatherDataSchema])
async def get_weather_data(
    start_date: Optional[date] = Query(None, description="Başlangıç tarihi"),
    end_date: Optional[date] = Query(None, description="Bitiş tarihi"),
    limit: int = Query(100, description="Maksimum kayıt sayısı"),
    offset: int = Query(0, description="Başlangıç kaydı"),
    db: Session = Depends(get_db)
):
    """
    Belirli bir tarih aralığındaki hava durumu verilerini listeler.
    Tarih belirtilmezse en son veriler döndürülür.
    """
    query = db.query(WeatherData)
    
    if start_date:
        query = query.filter(WeatherData.timestamp >= start_date)
    if end_date:
        # Son tarihin sonuna kadar verileri almak için
        end_date = datetime.combine(end_date, datetime.max.time())
        query = query.filter(WeatherData.timestamp <= end_date)
    
    query = query.order_by(WeatherData.timestamp.desc())
    weather_data = query.offset(offset).limit(limit).all()
    
    return weather_data

@router.get("/forecast", response_model=List[WeatherForecastSchema])
async def get_weather_forecast(
    days_ahead: int = Query(7, description="Kaç gün sonrası için tahminler", ge=1, le=14),
    db: Session = Depends(get_db)
):
    """
    Gelecek zaman dilimi için en güncel hava durumu tahminlerini listeler.
    """
    now = datetime.utcnow()
    end_date = now + timedelta(days=days_ahead)
    
    # En güncel tahminleri almak için, her zaman dilimi için son tahmini seç
    subquery = db.query(
        WeatherForecast.forecast_timestamp,
        db.func.max(WeatherForecast.created_at).label("latest_created")
    ).group_by(WeatherForecast.forecast_timestamp).subquery()
    
    forecasts = db.query(WeatherForecast).join(
        subquery,
        db.and_(
            WeatherForecast.forecast_timestamp == subquery.c.forecast_timestamp,
            WeatherForecast.created_at == subquery.c.latest_created
        )
    ).filter(
        WeatherForecast.forecast_timestamp >= now,
        WeatherForecast.forecast_timestamp <= end_date
    ).order_by(WeatherForecast.forecast_timestamp).all()
    
    return forecasts 