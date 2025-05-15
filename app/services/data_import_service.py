import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session

from app.models.inverter import Inverter, InverterData
from app.services.weather_service import fetch_historical_weather

async def process_csv_data(csv_file: io.StringIO, db: Session) -> int:
    """
    CSV dosyasından inverter verilerini işler ve veritabanına kaydeder.
    
    CSV format beklentisi:
    - timestamp: Zaman damgası (ISO format)
    - inverter_id: İnverter kimliği (1-8 arası)
    - power_output: Güç çıktısı (kW)
    - additional_columns: Ek sütunlar (opsiyonel)
    
    Args:
        csv_file: CSV dosyası içeriği
        db: Veritabanı oturumu
        
    Returns:
        İşlenen satır sayısı
    """
    try:
        # CSV dosyasını oku
        df = pd.read_csv(csv_file)
        
        # Gerekli sütunların varlığını kontrol et
        required_columns = ["timestamp", "inverter_id", "power_output"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Gerekli sütun bulunamadı: {col}")
        
        # Timestamp sütununu datetime'a çevir
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # İnverter kimliklerini tamsayıya çevir
        df["inverter_id"] = df["inverter_id"].astype(int)
        
        # Mevcut inverterleri kontrol et, yoksa oluştur
        inverter_ids = df["inverter_id"].unique()
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
        
        # İlk işlem için commit
        db.commit()
        
        # Verileri veritabanına kaydet
        processed_rows = 0
        for _, row in df.iterrows():
            # Varolan veriyi kontrol et
            existing = db.query(InverterData).filter(
                InverterData.inverter_id == row["inverter_id"],
                InverterData.timestamp == row["timestamp"]
            ).first()
            
            if existing:
                # Varolan veriyi güncelle
                existing.power_output = row["power_output"]
                # Diğer sütunları da ekleyebiliriz
                for col in df.columns:
                    if col not in required_columns and not pd.isna(row[col]):
                        if not existing.additional_data:
                            existing.additional_data = {}
                        existing.additional_data[col] = row[col]
            else:
                # Yeni veri oluştur
                additional_data = {}
                for col in df.columns:
                    if col not in required_columns and not pd.isna(row[col]):
                        additional_data[col] = row[col]
                
                # InverterData kaydı oluştur
                inverter_data = InverterData(
                    inverter_id=row["inverter_id"],
                    timestamp=row["timestamp"],
                    power_output=row["power_output"],
                    additional_data=additional_data if additional_data else None
                )
                db.add(inverter_data)
            
            processed_rows += 1
            
            # Belirli aralıklarla commit yap (bellek kullanımını optimize etmek için)
            if processed_rows % 1000 == 0:
                db.commit()
        
        # Kalan işlemler için commit
        db.commit()
        
        return processed_rows
    
    except Exception as e:
        # Hata durumunda rollback
        db.rollback()
        raise e

async def fetch_weather_data_for_dates(db: Session) -> int:
    """
    Veritabanındaki inverter verileri için ilgili hava durumu verilerini çeker.
    Inverter verilerinin bulunduğu tarih aralığı için Open-meteo API'den veri çeker.
    
    Args:
        db: Veritabanı oturumu
        
    Returns:
        İşlenen günlerin sayısı
    """
    try:
        # Tarih aralığını bul
        date_range = db.query(
            db.func.min(InverterData.timestamp).label("min_date"),
            db.func.max(InverterData.timestamp).label("max_date")
        ).first()
        
        if not date_range.min_date or not date_range.max_date:
            return 0
        
        start_date = date_range.min_date.date()
        end_date = date_range.max_date.date()
        
        # Varsayılan konum (Konya için)
        latitude = 37.8713
        longitude = 32.4846
        
        # Open-meteo API'sinden verileri çek
        response = await fetch_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            db=db,
            save_to_db=True
        )
        
        # Gün sayısını hesapla
        days_processed = (end_date - start_date).days + 1
        return days_processed
    
    except Exception as e:
        # Hata durumunda rollback
        db.rollback()
        raise e

def get_datetime_ranges(data: pd.DataFrame, timestamp_col: str = "timestamp") -> Dict[str, Any]:
    """
    Veri çerçevesindeki zaman aralığını belirler.
    
    Args:
        data: Veri çerçevesi
        timestamp_col: Zaman damgası sütunu adı
        
    Returns:
        Zaman aralığı bilgisi
    """
    if timestamp_col not in data.columns:
        return {"error": f"Belirtilen sütun bulunamadı: {timestamp_col}"}
    
    try:
        # Timestamp sütununu datetime'a çevir
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Zaman aralığı istatistikleri
        min_date = data[timestamp_col].min()
        max_date = data[timestamp_col].max()
        unique_dates = len(data[timestamp_col].dt.date.unique())
        
        return {
            "min_date": min_date,
            "max_date": max_date,
            "total_days": unique_dates,
            "total_records": len(data)
        }
    except Exception as e:
        return {"error": str(e)}

def validate_csv_data(csv_file: io.StringIO) -> Dict[str, Any]:
    """
    CSV verilerini doğrular ve temel istatistikleri döndürür.
    
    Args:
        csv_file: CSV dosyası içeriği
        
    Returns:
        Doğrulama sonuçları ve istatistikler
    """
    try:
        # CSV dosyasını oku
        df = pd.read_csv(csv_file)
        csv_file.seek(0)  # Dosyayı başa sar
        
        # Gerekli sütunları kontrol et
        required_columns = ["timestamp", "inverter_id", "power_output"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                "valid": False,
                "message": f"Eksik sütunlar: {', '.join(missing_columns)}",
                "columns": df.columns.tolist()
            }
        
        # Temel istatistikler
        stats = {
            "total_rows": len(df),
            "inverter_count": len(df["inverter_id"].unique()),
            "inverter_ids": df["inverter_id"].unique().tolist(),
            "date_range": get_datetime_ranges(df)
        }
        
        return {
            "valid": True,
            "message": "CSV verisi geçerli",
            "statistics": stats
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": f"CSV doğrulama hatası: {str(e)}"
        } 