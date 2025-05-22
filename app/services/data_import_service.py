import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import asyncio
import uuid
import time
import logging

from app.models.inverter import Inverter, InverterData
from app.models.weather import WeatherData
from app.utils.data_processing import InverterDataProcessor
from app.services.job_manager_service import get_job_manager
from app.services.weather_service import fetch_historical_weather, fetch_weather_forecast

# Logger yapılandırması
logger = logging.getLogger(__name__)

# İşlem durumunu takip etmek için global değişken
active_txt_upload_jobs = {}

# TXT yükleme job statü şablonu
def create_txt_upload_job_status(job_id, file_name=None):
    return {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "start_time": None,
        "end_time": None,
        "message": "Veri yükleme işlemi kuyruğa alındı",
        "processed_rows": 0,
        "conflict_count": 0,
        "updated_count": 0,
        "total_inverters": 0,
        "file_name": file_name
    }

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

async def process_txt_data_job(
    job_id: str,
    txt_content: str,
    db_connection_string: str,
    forced: bool = False,
    fetch_weather: bool = True,
    train_models: bool = False,
    fetch_future_weather: bool = True
) -> Dict[str, Any]:
    """
    TXT dosyasından inverter verilerini arka planda işler ve veritabanına kaydeder.
    
    Args:
        job_id: İş kimliği
        txt_content: TXT dosyası içeriği
        db_connection_string: Veritabanı bağlantı string'i
        forced: Aynı zaman dilimindeki verilerin üzerine yazılsın mı?
        fetch_weather: Hava durumu verisi çekilsin mi?
        train_models: Model eğitimi yapılsın mı?
        fetch_future_weather: Gelecek hava durumu tahminleri çekilsin mi?
        
    Returns:
        İşlenen satır sayısı ve diğer bilgiler içeren bir sözlük
    """
    global active_txt_upload_jobs
    job_manager = get_job_manager()
    
    # SqlAlchemy session oluştur
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.inverter import Inverter, InverterData
    
    engine = create_engine(db_connection_string)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    # İstatistik değişkenleri
    processed_rows = 0
    conflict_count = 0
    updated_count = 0
    
    try:
        # İş durumunu güncelle
        job_manager.update_job(
            job_id=job_id,
            status="running",
            progress=5,
            message="TXT dosyası işleniyor"
        )
        
        # TXT dosyasını oluştur
        from io import StringIO
        txt_file = StringIO(txt_content)
        
        # TXT dosyasını tab ayraçlı olarak oku
        df = pd.read_csv(txt_file, sep='\t')
        
        job_manager.update_job(
            job_id=job_id,
            progress=10,
            message="TXT dosyası okundu, veri hazırlanıyor"
        )
        
        # Boş sütunları kaldır
        df = df.dropna(axis=1, how='all')
        
        # İlk sütun adını "time" olarak değiştir (eğer farklıysa)
        df = df.rename(columns={df.columns[0]: "time"})
        
        # Zaman sütununu datetime formatına çevir
        df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        
        # Inverter sütunlarını tespit et
        inverter_columns = [col for col in df.columns if col.startswith("INV/") and "DayEnergy" in col]
        
        if not inverter_columns:
            raise ValueError("Hiç inverter veri sütunu (INV/#/DayEnergy) bulunamadı")
        
        job_manager.update_job(
            job_id=job_id,
            progress=20,
            message=f"Veri hazırlandı, {len(inverter_columns)} inverter bulundu"
        )
        
        # Eksik verileri 0 ile doldur
        df[inverter_columns] = df[inverter_columns].fillna(0)
        
        # Veriyi kümülatif değerlerden saatlik üretime dönüştür
        hourly_df, conversion_stats = InverterDataProcessor.cumulative_to_hourly(df, inverter_columns)
        
        job_manager.update_job(
            job_id=job_id,
            progress=30,
            message=f"Kümülatif veriler saatlik üretime dönüştürüldü: {conversion_stats['hourly_rows']} satır"
        )
        
        # İnverter kayıtlarını oluştur
        for inv_id in range(1, 9):
            inverter = db.query(Inverter).filter(Inverter.id == inv_id).first()
            if not inverter:
                inverter = Inverter(
                    id=inv_id,
                    name=f"Inverter-{inv_id}",
                    capacity=10.0,
                    location=f"Location-{inv_id}",
                    is_active=True
                )
                db.add(inverter)
        
        db.commit()
        
        # İnverter verilerini ekle
        for idx, row in enumerate(hourly_df.iterrows()):
            timestamp = row[1]['time']  # row[0] indeks, row[1] veri
            
            for col in inverter_columns:
                # "INV/1/DayEnergy" formatından inverter id'yi çıkar
                inv_id = int(col.split('/')[1])
                power_value = row[1][col]
                
                # Mevcut veriyi kontrol et
                existing = db.query(InverterData).filter(
                    InverterData.inverter_id == inv_id,
                    InverterData.timestamp == timestamp
                ).first()
                
                if existing:
                    if forced:
                        # Mevcut veriyi güncelle
                        existing.power_output = float(power_value)
                        updated_count += 1
                    else:
                        # Çakışma durumunda uyarı
                        conflict_count += 1
                else:
                    # Yeni veri oluştur
                    inverter_data = InverterData(
                        inverter_id=inv_id,
                        timestamp=timestamp,
                        power_output=float(power_value)
                    )
                    db.add(inverter_data)
                    processed_rows += 1
            
            # İlerlemeyi güncelle
            progress = 40 + ((idx + 1) / len(hourly_df)) * 30
            job_manager.update_job(
                job_id=job_id,
                progress=min(70, progress),
                message=f"Veriler yükleniyor: {idx + 1}/{len(hourly_df)} satır işlendi"
            )
            
            # Belirli aralıklarla commit yap
            if (idx + 1) % 50 == 0:
                db.commit()
                # Diğer işlemlerin çalışmasına izin ver
                await asyncio.sleep(0.001)
        
        # Kalan işlemler için commit
        db.commit()
        
        job_manager.update_job(
            job_id=job_id,
            progress=70,
            message=f"İnverter verileri yüklendi: {processed_rows} yeni, {conflict_count} çakışma, {updated_count} güncelleme"
        )
        
        # Hava durumu verilerini çek
        if fetch_weather:
            job_manager.update_job(
                job_id=job_id,
                progress=75,
                message="Hava durumu verileri çekiliyor..."
            )
            
            try:
                weather_result = await fetch_weather_data_for_dates(db)
                job_manager.update_job(
                    job_id=job_id,
                    progress=80,
                    message=f"Hava durumu verileri çekildi: {weather_result.get('weather_count', 0)} adet"
                )
            except Exception as e:
                job_manager.update_job(
                    job_id=job_id,
                    progress=80,
                    message=f"Hava durumu verileri çekilirken hata: {str(e)}",
                    level="WARNING"
                )
        
        # Model eğitimi yap
        if train_models:
            job_manager.update_job(
                job_id=job_id,
                progress=85,
                message="Model eğitimi başlatılıyor..."
            )
            
            try:
                # Model eğitimi başlat
                from app.services.model_training_service import start_all_models_training_job
                
                training_job_id = await start_all_models_training_job(
                    db_connection_string=db_connection_string,
                    test_split=True,
                    test_size=0.2
                )
                
                job_manager.update_job(
                    job_id=job_id,
                    progress=90,
                    message=f"Model eğitimi başlatıldı. Eğitim job ID: {training_job_id}"
                )
                
            except Exception as e:
                job_manager.update_job(
                    job_id=job_id,
                    progress=90,
                    message=f"Model eğitimi başlatılırken hata: {str(e)}",
                    level="WARNING"
                )
        
        # Gelecek hava durumu tahminlerini çek
        if fetch_future_weather:
            job_manager.update_job(
                job_id=job_id,
                progress=95,
                message="Gelecek bir haftalık hava durumu tahminleri çekiliyor..."
            )
            
            try:
                from app.services.weather_service import fetch_weather_forecast
                
                # Inverter lokasyon bilgilerini al (sabit değerler)
                latitude = 37.5704328  # Mersin
                longitude = 34.1371146
                forecast_days = 7
                
                # Hava durumu tahminlerini çek
                forecast_response = await fetch_weather_forecast(
                    latitude=latitude,
                    longitude=longitude,
                    forecast_days=forecast_days,
                    db=db,
                    save_to_db=True
                )
                
                job_manager.update_job(
                    job_id=job_id,
                    progress=98,
                    message=f"Gelecek hava durumu tahminleri çekildi"
                )
                
            except Exception as e:
                job_manager.update_job(
                    job_id=job_id,
                    progress=98,
                    message=f"Gelecek hava durumu tahminleri çekilirken hata: {str(e)}",
                    level="WARNING"
                )
        
        # İşlemi tamamla
        job_manager.update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message=f"Veri yükleme tamamlandı: {processed_rows} satır eklendi, {conflict_count} çakışma, {updated_count} güncelleme"
        )
        
        # Sonuç istatistiklerini döndür
        result = {
            "processed_rows": processed_rows,
            "conflict_count": conflict_count,
            "updated_count": updated_count,
            "total_inverters": len(inverter_columns),
            "conversion_stats": conversion_stats,
            "date_range": {
                "min_date": hourly_df['time'].min(),
                "max_date": hourly_df['time'].max()
            }
        }
        
        return result
    
    except Exception as e:
        # Hata durumunda rollback
        db.rollback()
        
        # Hata detaylarını job'a ekle
        import traceback
        error_trace = traceback.format_exc()
        
        job_manager.update_job(
            job_id=job_id,
            status="failed",
            message=f"Veri yükleme hatası: {str(e)}",
            level="ERROR"
        )
        
        logger.error(f"TXT veri yükleme hatası: {str(e)}")
        logger.error(error_trace)
        
        raise e
    finally:
        # Veritabanı oturumunu kapat
        db.close()

async def start_txt_upload_job(
    txt_content: str,
    db_connection_string: str,
    file_name: str = None,
    forced: bool = False,
    fetch_weather: bool = True,
    train_models: bool = False,
    fetch_future_weather: bool = True
) -> str:
    """
    TXT dosyası yükleme işlemi başlatır.
    
    Args:
        txt_content: TXT dosyası içeriği
        db_connection_string: Veritabanı bağlantı string'i
        file_name: Dosya adı
        forced: Aynı zaman dilimindeki verilerin üzerine yazılsın mı?
        fetch_weather: Hava durumu verisi çekilsin mi?
        train_models: Model eğitimi yapılsın mı?
        fetch_future_weather: Gelecek hava durumu tahminleri çekilsin mi?
        
    Returns:
        İş kimliği
    """
    # Job manager'ı al
    job_manager = get_job_manager()
    
    # Job parametrelerini oluştur
    params = {
        "file_name": file_name or "upload_via_api",
        "forced": forced,
        "fetch_weather": fetch_weather,
        "train_models": train_models,
        "fetch_future_weather": fetch_future_weather
    }
    
    # Job oluştur
    job_id = job_manager.create_job(
        job_type="txt_upload",
        params=params,
        description=f"TXT dosyası yükleme: {file_name or 'upload_via_api'}"
    )
    
    # Eski job yapısına kaydet (geçici olarak, eski kodun çalışması için)
    active_txt_upload_jobs[job_id] = create_txt_upload_job_status(job_id, file_name)
    
    # Job'ı başlat
    await job_manager.start_job(
        job_id=job_id,
        task_func=lambda job_id, update_job: process_txt_data_job(
            job_id=job_id,
            txt_content=txt_content,
            db_connection_string=db_connection_string,
            forced=forced,
            fetch_weather=fetch_weather,
            train_models=train_models,
            fetch_future_weather=fetch_future_weather
        )
    )
    
    return job_id

async def process_txt_data(txt_file: io.StringIO, db: Session, forced: bool = False, 
                          fetch_weather: bool = True, train_models: bool = False,
                          fetch_future_weather: bool = True) -> Dict[str, Any]:
    """
    TXT dosyasından inverter verilerini işler ve veritabanına kaydeder.
    Bu fonksiyon artık yeni asenkron job sistemini kullanır.
    
    TXT format beklentisi:
    - İlk sütun: Zaman damgası
    - Diğer sütunlar: INV/#/DayEnergy (kWh) formatında güç çıktısı değerleri
    
    Args:
        txt_file: TXT dosyası içeriği
        db: Veritabanı oturumu
        forced: Aynı zaman dilimindeki verilerin üzerine yazılsın mı?
        fetch_weather: Hava durumu verisi çekilsin mi?
        train_models: Model eğitimi yapılsın mı?
        fetch_future_weather: Gelecek hava durumu tahminleri çekilsin mi?
        
    Returns:
        İşlenen satır sayısı ve diğer bilgiler içeren bir sözlük
    """
    from app.core.config import settings
    
    # Dosya içeriğini string olarak al
    txt_content = txt_file.read()
    
    # Job başlat
    job_id = await start_txt_upload_job(
        txt_content=txt_content,
        db_connection_string=str(settings.DATABASE_URL),
        file_name="upload_via_api",
        forced=forced,
        fetch_weather=fetch_weather,
        train_models=train_models,
        fetch_future_weather=fetch_future_weather
    )
    
    # Job durumunu kontrol et
    job_status = get_job_manager().get_job_status(job_id)
    
    # Dummy sonuç döndür, gerçek işlem arka planda devam edecek
    return {
        "processed_rows": 0,
        "conflict_count": 0,
        "updated_count": 0,
        "total_inverters": 0,
        "job_id": job_id
    }

def get_txt_upload_job_status(job_id: str) -> Dict[str, Any]:
    """
    TXT dosyası yükleme işleminin durumunu döndürür.
    
    Args:
        job_id: İş kimliği
        
    Returns:
        İş durumu
    """
    # Eski job yapısı
    global active_txt_upload_jobs
    
    # Yeni job manager'ı kullan
    job_manager = get_job_manager()
    job_status = job_manager.get_job_status(job_id)
    
    # Eğer job bulunamadıysa
    if job_status.get("status") == "not_found":
        # Eski sistemde kontrol et
        if job_id in active_txt_upload_jobs:
            return active_txt_upload_jobs[job_id].copy()
        else:
            return {
                "job_id": job_id,
                "status": "not_found",
                "message": "Belirtilen ID'ye sahip bir yükleme işi bulunamadı"
            }
    
    # Job statüsünü TxtDataUploadJob formatına dönüştür
    return {
        "job_id": job_id,
        "status": job_status.get("status"),
        "progress": job_status.get("progress", 0),
        "start_time": job_status.get("started_at"),
        "end_time": job_status.get("completed_at"),
        "message": job_status.get("logs", [])[-1]["message"] if job_status.get("logs") else "",
        "processed_rows": job_status.get("result", {}).get("processed_rows", 0) if job_status.get("result") else 0,
        "conflict_count": job_status.get("result", {}).get("conflict_count", 0) if job_status.get("result") else 0,
        "updated_count": job_status.get("result", {}).get("updated_count", 0) if job_status.get("result") else 0,
        "total_inverters": job_status.get("result", {}).get("total_inverters", 0) if job_status.get("result") else 0,
        "file_name": job_status.get("params", {}).get("file_name") if job_status.get("params") else None
    }

def cleanup_old_txt_upload_jobs(max_age_hours: int = 24):
    """
    Belirli bir süreden daha eski TXT yükleme işlerini temizler.
    
    Args:
        max_age_hours: Maksimum saat cinsinden yaş
    """
    global active_txt_upload_jobs
    
    now = datetime.utcnow()
    to_remove = []
    
    for job_id, job_status in active_txt_upload_jobs.items():
        # Tamamlanmış veya başarısız olmuş ve sonlanma zamanı olan işleri kontrol et
        if job_status.get("end_time") and job_status["status"] in ["completed", "failed"]:
            age = now - job_status["end_time"]
            
            # Belirli bir süreden daha eski ise işaretleyelim
            if age.total_seconds() > max_age_hours * 3600:
                to_remove.append(job_id)
    
    # İşaretlenen işleri kaldır
    for job_id in to_remove:
        del active_txt_upload_jobs[job_id]
    
    return len(to_remove)

def convert_to_hourly(df: pd.DataFrame, inverter_columns: List[str]) -> pd.DataFrame:
    """
    5 dakikalık verileri saatlik verilere dönüştürür.
    
    Args:
        df: Orijinal veri çerçevesi
        inverter_columns: İnverter sütunları listesi
        
    Returns:
        Saatlik verilerden oluşan veri çerçevesi
    """
    # Yeni InverterDataProcessor sınıfını kullan
    return InverterDataProcessor.cumulative_to_hourly(df, inverter_columns)[0]

def validate_txt_data(txt_file: io.StringIO) -> Dict[str, Any]:
    """
    TXT verilerini doğrular ve temel istatistikleri döndürür.
    
    Args:
        txt_file: TXT dosyası içeriği
        
    Returns:
        Doğrulama sonuçları ve istatistikler
    """
    try:
        # TXT dosyasını tab ayraçlı olarak oku
        df = pd.read_csv(txt_file, sep='\t')
        txt_file.seek(0)  # Dosyayı başa sar
        
        # Boş sütunları kaldır
        df = df.dropna(axis=1, how='all')
        
        # İlk sütun adını "time" olarak değiştir (eğer farklıysa)
        df = df.rename(columns={df.columns[0]: "time"})
        
        # Zaman sütunu doğrulaması
        try:
            df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            invalid_dates = df['time'].isna().sum()
            if invalid_dates > 0:
                return {
                    "valid": False,
                    "message": f"{invalid_dates} adet geçersiz tarih formatı bulundu",
                    "columns": df.columns.tolist()
                }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Tarih sütunu doğrulanamadı: {str(e)}",
                "columns": df.columns.tolist()
            }
        
        # Inverter sütunlarını tespit et
        inverter_columns = [col for col in df.columns if col.startswith("INV/") and "DayEnergy" in col]
        
        if not inverter_columns:
            return {
                "valid": False,
                "message": "Hiç inverter veri sütunu (INV/#/DayEnergy) bulunamadı",
                "columns": df.columns.tolist()
            }
        
        # Temel istatistikler
        stats = {
            "total_rows": len(df),
            "inverter_count": len(inverter_columns),
            "inverter_ids": [int(col.split('/')[1]) for col in inverter_columns],
            "samples_per_hour": df.groupby(df['time'].dt.floor('H')).size().mean(),
            "date_range": {
                "min_date": df['time'].min(),
                "max_date": df['time'].max(),
                "days": (df['time'].max().date() - df['time'].min().date()).days + 1 if not df.empty else 0
            }
        }
        
        return {
            "valid": True,
            "message": "TXT verisi geçerli",
            "statistics": stats
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": f"TXT doğrulama hatası: {str(e)}"
        }

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
            func.min(InverterData.timestamp).label("min_date"),
            func.max(InverterData.timestamp).label("max_date")
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
        import traceback
        print(f"Hava durumu verisi çekilirken hata: {str(e)}")
        print(traceback.format_exc())
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