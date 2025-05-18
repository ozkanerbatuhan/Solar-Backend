import io
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime, date, timedelta

from app.db.database import get_db
from app.models.inverter import Inverter, InverterData
from app.models.weather import WeatherData
from app.schemas.inverter import InverterData as InverterDataSchema
from app.schemas.weather import CSVUploadResponse
from app.schemas.data import DataUploadResponse, DataStatistics
from app.services.data_import_service import (
    process_csv_data, 
    validate_csv_data, 
    fetch_weather_data_for_dates
)

router = APIRouter()

@router.post("/upload-csv", response_model=DataUploadResponse)
async def upload_csv_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validate_only: bool = Form(False),
    fetch_weather: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    CSV dosyasından inverter verilerini yükler.
    
    Args:
        file: CSV dosyası
        validate_only: Sadece doğrulama yap, veritabanına kaydetme
        fetch_weather: İlgili hava durumu verilerini çek
        db: Veritabanı oturumu
    """
    try:
        # CSV içeriğini oku
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        
        # CSV verilerini doğrula
        validation_result = validate_csv_data(csv_file)
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_result["message"]
            )
        
        # Sadece doğrulama isteniyorsa, sonuçları döndür
        if validate_only:
            return {
                "success": True,
                "message": "CSV dosyası doğrulandı",
                "processed_rows": 0,
                "statistics": validation_result["statistics"]
            }
        
        # CSV verilerini işle
        processed_rows = await process_csv_data(csv_file, db)
        
        # Arka planda hava durumu verilerini çek
        if fetch_weather:
            background_tasks.add_task(fetch_weather_data_for_dates, db)
        
        return {
            "success": True,
            "message": f"{processed_rows} satır işlendi",
            "processed_rows": processed_rows,
            "statistics": validation_result["statistics"]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV yükleme hatası: {str(e)}"
        )

@router.post("/upload-weather-inverter-data", response_model=DataUploadResponse)
async def upload_weather_inverter_data(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    CSV dosyasından hava durumu ve inverter verilerini yükler.
    Bu endpoint sadece verileri yükler, model eğitimi yapmaz.
    
    CSV formatı:
    time,temperature_2m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,
    global_tilted_irradiance,terrestrial_radiation,relative_humidity_2m,wind_speed_10m,visibility,
    INV/1/DayEnergy,INV/2/DayEnergy,INV/3/DayEnergy,INV/4/DayEnergy,INV/5/DayEnergy,INV/6/DayEnergy,
    INV/7/DayEnergy,INV/8/DayEnergy
    
    Args:
        file: CSV dosyası
        db: Veritabanı oturumu
    """
    try:
        # CSV içeriğini oku
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        
        # CSV dosyasını pandas ile oku
        df = pd.read_csv(csv_file)
        
        # Tarih sütununun datetime formatına dönüştürülmesi
        df['time'] = pd.to_datetime(df['time'])
        
        # Kaç satır işlendiğini takip etmek için sayaç
        processed_rows = 0
        weather_rows = 0
        inverter_rows = 0
        
        # Mevcut inverterleri kontrol et, yoksa oluştur
        inverter_ids = list(range(1, 9))  # 1'den 8'e kadar inverter ID'leri
        for inv_id in inverter_ids:
            inverter = db.query(Inverter).filter(Inverter.id == inv_id).first()
            if not inverter:
                # Yeni inverter oluştur
                inverter = Inverter(
                    id=inv_id,
                    name=f"Inverter-{inv_id}",
                    capacity=10.0,  # Varsayılan kapasite
                    location=f"Location-{inv_id}",  # Varsayılan konum
                    is_active=True
                )
                db.add(inverter)
        
        # İnverterleri kaydetmek için commit
        db.commit()
        
        # DataFrame üzerinde iterate ederek verileri veritabanına kaydet
        for _, row in df.iterrows():
            # Hava durumu verisini ekle
            timestamp = row['time']
            
            # Mevcut hava durumu verisini kontrol et
            existing_weather = db.query(WeatherData).filter(
                WeatherData.timestamp == timestamp
            ).first()
            
            if not existing_weather:
                weather_data = WeatherData(
                    timestamp=timestamp,
                    temperature=row.get('temperature_2m'),
                    shortwave_radiation=row.get('shortwave_radiation'),
                    direct_radiation=row.get('direct_radiation'),
                    diffuse_radiation=row.get('diffuse_radiation'),
                    direct_normal_irradiance=row.get('direct_normal_irradiance'),
                    global_tilted_irradiance=row.get('global_tilted_irradiance'),
                    terrestrial_radiation=row.get('terrestrial_radiation'),
                    relative_humidity=row.get('relative_humidity_2m'),
                    wind_speed=row.get('wind_speed_10m'),
                    visibility=row.get('visibility'),
                    is_forecast=0  # Gerçek veri
                )
                db.add(weather_data)
                weather_rows += 1
            
            # İnverter verilerini ekle
            for inv_id in range(1, 9):
                inverter_column = f"INV/{inv_id}/DayEnergy"
                
                # İnverter verisi varsa ekle
                if inverter_column in row and not pd.isna(row[inverter_column]):
                    # Mevcut inverter verisini kontrol et
                    existing_inverter_data = db.query(InverterData).filter(
                        InverterData.inverter_id == inv_id,
                        InverterData.timestamp == timestamp
                    ).first()
                    
                    if not existing_inverter_data:
                        inverter_data = InverterData(
                            inverter_id=inv_id,
                            timestamp=timestamp,
                            power_output=row[inverter_column],
                            temperature=row.get('temperature_2m'),
                            irradiance=row.get('direct_normal_irradiance')
                        )
                        db.add(inverter_data)
                        inverter_rows += 1
            
            processed_rows += 1
            
            # Her 100 satırda bir commit yap (bellek optimizasyonu için)
            if processed_rows % 100 == 0:
                db.commit()
        
        # Kalan işlemler için son commit
        db.commit()
        
        return {
            "success": True,
            "message": f"Toplam {processed_rows} satır işlendi. {weather_rows} hava durumu ve {inverter_rows} inverter verisi eklendi.",
            "processed_rows": processed_rows,
            "statistics": {
                "total_rows": processed_rows,
                "weather_rows": weather_rows,
                "inverter_rows": inverter_rows,
                "date_range": {
                    "min_date": df['time'].min().isoformat() if not df.empty else None,
                    "max_date": df['time'].max().isoformat() if not df.empty else None
                }
            }
        }
    
    except Exception as e:
        # Hata durumunda rollback
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Veri yükleme hatası: {str(e)}"
        )

@router.get("/statistics", response_model=DataStatistics)
async def get_data_statistics(
    start_date: Optional[date] = Query(None, description="Başlangıç tarihi"),
    end_date: Optional[date] = Query(None, description="Bitiş tarihi"),
    inverter_id: Optional[int] = Query(None, description="İnverter ID'si"),
    db: Session = Depends(get_db)
):
    """
    Veritabanındaki inverter verileri için istatistikler döndürür.
    
    Args:
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
        inverter_id: İnverter ID'si
        db: Veritabanı oturumu
    """
    query = db.query(InverterData)
    
    # Filtreleri uygula
    if start_date:
        query = query.filter(InverterData.timestamp >= start_date)
    if end_date:
        # Son tarihin sonuna kadar verileri almak için
        end_date = datetime.combine(end_date, datetime.max.time())
        query = query.filter(InverterData.timestamp <= end_date)
    if inverter_id:
        query = query.filter(InverterData.inverter_id == inverter_id)
    
    # Toplam kayıt sayısı
    total_count = query.count()
    
    if total_count == 0:
        return {
            "total_records": 0,
            "date_range": None,
            "inverter_count": 0,
            "inverter_ids": []
        }
    
    # Tarih aralığı
    date_range = db.query(
        db.func.min(InverterData.timestamp).label("min_date"),
        db.func.max(InverterData.timestamp).label("max_date")
    ).filter(query.whereclause).first()
    
    # İnverter sayısı ve ID'leri
    inverter_data = db.query(
        InverterData.inverter_id
    ).filter(query.whereclause).distinct().all()
    
    inverter_ids = [item.inverter_id for item in inverter_data]
    
    return {
        "total_records": total_count,
        "date_range": {
            "min_date": date_range.min_date,
            "max_date": date_range.max_date,
            "days": (date_range.max_date.date() - date_range.min_date.date()).days + 1
        },
        "inverter_count": len(inverter_ids),
        "inverter_ids": inverter_ids
    }

@router.post("/fetch-weather-data")
async def fetch_weather_for_inverter_data(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Veritabanındaki inverter verileri için ilgili hava durumu verilerini çeker.
    Bu işlem arka planda çalışır.
    
    Args:
        db: Veritabanı oturumu
    """
    # Arka planda hava durumu verilerini çek
    background_tasks.add_task(fetch_weather_data_for_dates, db)
    
    return {
        "success": True,
        "message": "Hava durumu verileri arka planda çekiliyor"
    }

@router.get("/inverter-data/summary", response_model=dict)
async def get_inverter_data_summary(
    start_date: Optional[str] = Query(None, description="Başlangıç tarihi (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Bitiş tarihi (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    İnverter verilerinin özet istatistiklerini döndürür (tarih aralığına göre).
    """
    query = db.query(InverterData)
    
    if start_date:
        query = query.filter(InverterData.timestamp >= start_date)
    if end_date:
        query = query.filter(InverterData.timestamp <= end_date)
    
    data_count = query.count()
    
    if data_count == 0:
        return {
            "total_records": 0,
            "date_range": {"start": None, "end": None},
            "inverter_count": 0
        }
        
    # Tarih aralığı
    date_range = db.query(
        db.func.min(InverterData.timestamp),
        db.func.max(InverterData.timestamp)
    ).first()
    
    # İnverter sayısı
    inverter_count = db.query(Inverter).count()
    
    # Özet istatistikler
    return {
        "total_records": data_count,
        "date_range": {"start": date_range[0], "end": date_range[1]},
        "inverter_count": inverter_count
    } 