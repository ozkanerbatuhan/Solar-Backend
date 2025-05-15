import io
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime, date, timedelta

from app.db.database import get_db
from app.models.inverter import Inverter, InverterData
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