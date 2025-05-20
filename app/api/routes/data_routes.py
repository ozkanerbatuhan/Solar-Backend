import io
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, text
import pandas as pd
from datetime import datetime, date, timedelta
import asyncio

from app.db.database import get_db
from app.models.inverter import Inverter, InverterData
from app.models.weather import WeatherData
from app.schemas.inverter import InverterData as InverterDataSchema
from app.schemas.weather import CSVUploadResponse
from app.schemas.data import DataUploadResponse, DataStatistics, TxtDataUploadResponse, ModelTrainingStatus, ModelTrainingResponse, TxtDataUploadJob, ModelLogResponse
from app.services.data_import_service import (
    process_csv_data, 
    validate_csv_data, 
    fetch_weather_data_for_dates,
    process_txt_data,
    validate_txt_data
)
from app.core.config import settings

router = APIRouter()

# İşlem durumunu takip etmek için global değişkenler
current_job_status = {
    "job_id": None,
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,
    "total_rows": 0,
    "processed_rows": 0,
    "weather_rows": 0,
    "inverter_rows": 0,
    "start_time": None,
    "end_time": None,
    "message": ""
}

@router.post("/reset-database")
async def reset_database(
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Veritabanını tamamen sıfırlar. Bu tehlikeli bir işlemdir.
    
    Args:
        password: Güvenlik şifresi (123456)
        db: Veritabanı oturumu
    """
    # Güvenlik kontrolü
    if password != "123456":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz şifre. Veritabanı sıfırlaması yapılamaz."
        )
    
    try:
        # Tüm tabloları temizle (sırayla)
        db.execute(text("DELETE FROM inverter_predictions"))
        db.execute(text("DELETE FROM inverter_data"))
        db.execute(text("DELETE FROM weather_forecasts"))
        db.execute(text("DELETE FROM weather_data"))
        db.execute(text("DELETE FROM models"))
        db.execute(text("DELETE FROM inverters"))
        
        # İşlemleri kaydet
        db.commit()
        
        return {
            "success": True,
            "message": "Veritabanı başarıyla sıfırlandı."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Veritabanı sıfırlama sırasında hata: {str(e)}"
        )

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

async def process_weather_inverter_data_background(
    contents: bytes,
    start_date: datetime,
    end_date: datetime
):
    """
    Hava durumu ve inverter verilerini arka planda işler.
    
    Args:
        contents: CSV dosya içeriği
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
    """
    global current_job_status
    
    try:
        current_job_status["status"] = "running"
        current_job_status["start_time"] = datetime.now()
        current_job_status["message"] = "İşlem başladı"
        
        # CSV dosyasını pandas ile oku
        csv_file = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_file)
        
        # Toplam satır sayısını kaydet
        current_job_status["total_rows"] = len(df)
        
        # Tarih sütununun datetime formatına dönüştürülmesi
        df['time'] = pd.to_datetime(df['time'])
        
        # Filtrelemeyi uygula
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
        filtered_count = len(df)
        current_job_status["processed_rows"] = 0
        current_job_status["message"] = f"Tarih filtreleme: {len(df)} satır kaldı"
        
        # Sütun isimlerini eşleştir
        inverter_columns_map = {}
        for col in df.columns:
            if 'INV/' in col and 'DayEnergy' in col:
                simple_name = col.split(' ')[0]
                inverter_columns_map[simple_name] = col
        
        inverter_columns = list(inverter_columns_map.values())
        
        # Boş değerleri -1 ile doldur
        for col in inverter_columns:
            df[col] = df[col].fillna(-1)
        
        # Veritabanı bağlantısı
        from app.db.database import SessionLocal
        db = SessionLocal()
        
        try:
            # Mevcut inverterleri kontrol et, yoksa oluştur
            inverter_ids = list(range(1, 9))
            for inv_id in inverter_ids:
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
            
            # Sayaçlar
            processed_rows = 0
            weather_rows = 0
            inverter_rows = 0
            inverter_counts = {i: 0 for i in range(1, 9)}
            
            # İşleme başla
            batch_size = 100  # Toplu işleme için boyut
            weather_batch = []
            inverter_batch = []
            
            # DataFrame üzerinde iterate ederek verileri veritabanına kaydet
            for index, row in df.iterrows():
                timestamp = row['time']
                
                # Mevcut hava durumu verisini kontrol et
                existing_weather = db.query(WeatherData).filter(
                    WeatherData.timestamp == timestamp
                ).first()
                
                if not existing_weather:
                    try:
                        weather_data = WeatherData(
                            timestamp=timestamp,
                            temperature=row.get('temperature_2m (°C)'),
                            shortwave_radiation=row.get('shortwave_radiation (W/m²)'),
                            direct_radiation=row.get('direct_radiation (W/m²)'),
                            diffuse_radiation=row.get('diffuse_radiation (W/m²)'),
                            direct_normal_irradiance=row.get('direct_normal_irradiance (W/m²)'),
                            global_tilted_irradiance=row.get('global_tilted_irradiance (W/m²)'),
                            terrestrial_radiation=row.get('terrestrial_radiation (W/m²)'),
                            relative_humidity=row.get('relative_humidity_2m (%)'),
                            wind_speed=row.get('wind_speed_10m (km/h)'),
                            visibility=row.get('visibility (m)'),
                            is_forecast=0
                        )
                        weather_batch.append(weather_data)
                        weather_rows += 1
                    except Exception as e:
                        current_job_status["message"] = f"Hava durumu verisi eklenirken hata: {e}, Satır: {index}"
                
                # İnverter verilerini ekle
                for inv_id in range(1, 9):
                    simple_column = f"INV/{inv_id}/DayEnergy"
                    full_column = inverter_columns_map.get(simple_column)
                    
                    if full_column is not None and full_column in df.columns:
                        power_value = row[full_column]
                        
                        if pd.notna(power_value) or power_value == -1:
                            existing_inverter_data = db.query(InverterData).filter(
                                InverterData.inverter_id == inv_id,
                                InverterData.timestamp == timestamp
                            ).first()
                            
                            if not existing_inverter_data:
                                try:
                                    if power_value < 0:
                                        power_float = 0
                                    else:
                                        power_float = float(power_value)
                                    
                                    inverter_data = InverterData(
                                        inverter_id=inv_id,
                                        timestamp=timestamp,
                                        power_output=power_float,
                                        temperature=row.get('temperature_2m (°C)'),
                                        irradiance=row.get('direct_normal_irradiance (W/m²)')
                                    )
                                    inverter_batch.append(inverter_data)
                                    inverter_rows += 1
                                    inverter_counts[inv_id] += 1
                                except Exception as e:
                                    current_job_status["message"] = f"Değer dönüştürme hatası: {e}, Değer: {power_value}"
                
                processed_rows += 1
                current_job_status["processed_rows"] = processed_rows
                current_job_status["weather_rows"] = weather_rows
                current_job_status["inverter_rows"] = inverter_rows
                
                # İlerleme yüzdesini güncelle
                progress = int((processed_rows / filtered_count) * 100)
                current_job_status["progress"] = progress
                
                # Belirli aralıklarla commit yap
                if len(weather_batch) + len(inverter_batch) >= batch_size:
                    db.add_all(weather_batch)
                    db.add_all(inverter_batch)
                    db.commit()
                    
                    # Batchi temizle
                    weather_batch = []
                    inverter_batch = []
                    
                    # Belirli aralıklarla asyncio.sleep() çağrısı yaparak
                    # diğer işlemlerin çalışmasına izin ver
                    await asyncio.sleep(0.001)
            
            # Kalan batch'leri ekle
            if weather_batch or inverter_batch:
                db.add_all(weather_batch)
                db.add_all(inverter_batch)
                db.commit()
            
            # İşlem tamamlandı
            current_job_status["status"] = "completed"
            current_job_status["end_time"] = datetime.now()
            current_job_status["message"] = f"İşlem tamamlandı. {processed_rows} satır işlendi. {weather_rows} hava durumu ve {inverter_rows} inverter verisi eklendi."
            
        except Exception as e:
            db.rollback()
            current_job_status["status"] = "failed"
            current_job_status["end_time"] = datetime.now()
            current_job_status["message"] = f"Veri işleme hatası: {str(e)}"
            import traceback
            print(traceback.format_exc())
        finally:
            db.close()
    
    except Exception as e:
        current_job_status["status"] = "failed"
        current_job_status["end_time"] = datetime.now()
        current_job_status["message"] = f"Genel hata: {str(e)}"
        import traceback
        print(traceback.format_exc())

@router.post("/upload-weather-inverter-data")
async def upload_weather_inverter_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start_date: Optional[str] = Form("2023-04-30T23:00:00"),
    end_date: Optional[str] = Form("2024-11-18T12:00:00")
):
    """
    CSV dosyasından hava durumu ve inverter verilerini arka planda yükler.
    Bu endpoint sadece verileri yükler, model eğitimi yapmaz.
    
    Args:
        file: CSV dosyası
        start_date: Başlangıç tarihi (isteğe bağlı, varsayılan: 2023-04-30T23:00:00)
        end_date: Bitiş tarihi (isteğe bağlı, varsayılan: 2024-11-18T12:00:00)
    """
    try:
        global current_job_status
        
        # Eğer halihazırda bir işlem çalışıyorsa
        if current_job_status["status"] == "running":
            return {
                "success": False,
                "message": "Zaten devam eden bir veri yükleme işlemi mevcut.",
                "job_id": current_job_status["job_id"],
                "status": current_job_status["status"],
                "progress": current_job_status["progress"]
            }
        
        # Yeni bir iş başlat
        current_job_status["job_id"] = f"job_{int(time.time())}"
        current_job_status["status"] = "starting"
        current_job_status["progress"] = 0
        current_job_status["processed_rows"] = 0
        current_job_status["weather_rows"] = 0
        current_job_status["inverter_rows"] = 0
        current_job_status["start_time"] = datetime.now()
        current_job_status["end_time"] = None
        current_job_status["message"] = "Dosya yükleniyor..."
        
        # CSV içeriğini oku
        contents = await file.read()
        
        # Tarih formatlarını dönüştür
        try:
            start_date_dt = datetime.fromisoformat(start_date)
        except ValueError:
            print(f"Hata: {start_date}")
        
        try:
            end_date_dt = datetime.fromisoformat(end_date)
        except ValueError:
            print(f"Hata: {end_date}")
        
        # Arka planda işleme başlat
        background_tasks.add_task(
            process_weather_inverter_data_background,
            contents,
            start_date_dt,
            end_date_dt
        )
        
        return {
            "success": True,
            "message": "Veri yükleme işlemi başlatıldı. İlerlemeyi kontrol etmek için /api/data/job-status endpoint'ini kullanabilirsiniz.",
            "job_id": current_job_status["job_id"],
            "status": current_job_status["status"]
        }
    
    except Exception as e:
        current_job_status["status"] = "failed"
        current_job_status["end_time"] = datetime.now()
        current_job_status["message"] = f"Veri yükleme başlatma hatası: {str(e)}"
        
        return {
            "success": False,
            "message": f"Veri yükleme hatası: {str(e)}",
            "job_id": current_job_status["job_id"],
            "status": "failed"
        }

@router.get("/job-status")
async def get_job_status():
    """
    Mevcut işlemin durumunu döndürür.
    """
    global current_job_status
    
    elapsed_time = None
    if current_job_status["start_time"]:
        if current_job_status["end_time"]:
            elapsed_time = (current_job_status["end_time"] - current_job_status["start_time"]).total_seconds()
        else:
            elapsed_time = (datetime.now() - current_job_status["start_time"]).total_seconds()
    
    return {
        "job_id": current_job_status["job_id"],
        "status": current_job_status["status"],
        "progress": current_job_status["progress"],
        "total_rows": current_job_status["total_rows"],
        "processed_rows": current_job_status["processed_rows"],
        "weather_rows": current_job_status["weather_rows"],
        "inverter_rows": current_job_status["inverter_rows"],
        "elapsed_time": elapsed_time,
        "message": current_job_status["message"]
    }

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
    try:
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
            func.min(InverterData.timestamp).label("min_date"),
            func.max(InverterData.timestamp).label("max_date")
        ).filter(query.whereclause).first()
        
        # İnverter sayısı ve ID'leri
        inverter_data = db.query(
            InverterData.inverter_id
        ).filter(query.whereclause).distinct().all()
        
        inverter_ids = [item.inverter_id for item in inverter_data]
        
        # Tarih aralığı için None kontrolü
        date_range_info = None
        if date_range and date_range.min_date is not None and date_range.max_date is not None:
            date_range_info = {
                "min_date": date_range.min_date,
                "max_date": date_range.max_date,
                "days": (date_range.max_date.date() - date_range.min_date.date()).days + 1
            }
        else:
            date_range_info = {
                "min_date": None,
                "max_date": None,
                "days": 0
            }
        
        return {
            "total_records": total_count,
            "date_range": date_range_info,
            "inverter_count": len(inverter_ids),
            "inverter_ids": inverter_ids
        }
    except Exception as e:
        return {
            "total_records": 0,
            "date_range": None,
            "inverter_count": 0,
            "inverter_ids": []
        }

@router.get("/detailed-statistics")
async def get_detailed_statistics(
    start_date: Optional[date] = Query(None, description="Başlangıç tarihi"),
    end_date: Optional[date] = Query(None, description="Bitiş tarihi"),
    db: Session = Depends(get_db)
):
    """
    Veritabanındaki hava durumu ve inverter verileri için detaylı istatistikler döndürür.
    
    Args:
        start_date: Başlangıç tarihi (isteğe bağlı)
        end_date: Bitiş tarihi (isteğe bağlı)
        db: Veritabanı oturumu
    """
    try:
        # Tarih aralığı filtrelerini oluştur
        start_filter = None
        end_filter = None
        
        if start_date:
            start_filter = datetime.combine(start_date, datetime.min.time())
        if end_date:
            end_filter = datetime.combine(end_date, datetime.max.time())
        
        # WeatherData istatistikleri
        weather_query = db.query(WeatherData)
        if start_filter:
            weather_query = weather_query.filter(WeatherData.timestamp >= start_filter)
        if end_filter:
            weather_query = weather_query.filter(WeatherData.timestamp <= end_filter)
        
        weather_count = weather_query.count()
        
        # Hava durumu tarih aralığı
        weather_date_range = None
        if weather_count > 0:
            try:
                # Daha güvenli bir yaklaşım - doğrudan SQL ile sorgu yapalım
                min_max_date = db.execute(
                    "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM weather_data"
                ).fetchone()
                
                if min_max_date and min_max_date[0] and min_max_date[1]:
                    min_date = min_max_date[0]
                    max_date = min_max_date[1]
                    weather_date_range = {
                        "min_date": min_date,
                        "max_date": max_date,
                        "days": (max_date.date() - min_date.date()).days + 1
                    }
                else:
                    weather_date_range = {
                        "min_date": None,
                        "max_date": None,
                        "days": 0
                    }
            except Exception as e:
                print(f"Hava durumu tarih aralığı hesaplanırken hata: {e}")
                weather_date_range = {
                    "min_date": None,
                    "max_date": None,
                    "days": 0
                }
        
        # İnverter verileri istatistikleri
        inverter_query = db.query(InverterData)
        if start_filter:
            inverter_query = inverter_query.filter(InverterData.timestamp >= start_filter)
        if end_filter:
            inverter_query = inverter_query.filter(InverterData.timestamp <= end_filter)
        
        inverter_count = inverter_query.count()
        
        # İnverter tarih aralığı
        inverter_date_range = None
        if inverter_count > 0:
            inverter_range = db.query(
                func.min(InverterData.timestamp).label("min_date"),
                func.max(InverterData.timestamp).label("max_date")
            ).filter(inverter_query.whereclause).first()
            
            if inverter_range and inverter_range.min_date is not None and inverter_range.max_date is not None:
                inverter_date_range = {
                    "min_date": inverter_range.min_date,
                    "max_date": inverter_range.max_date,
                    "days": (inverter_range.max_date.date() - inverter_range.min_date.date()).days + 1
                }
            else:
                inverter_date_range = {
                    "min_date": None,
                    "max_date": None,
                    "days": 0
                }
        
        # İnverter bazında istatistikler
        inverter_stats = {}
        if inverter_count > 0:
            # Tüm inverter ID'lerini al
            inverter_ids = db.query(InverterData.inverter_id).distinct().all()
            inverter_ids = [item.inverter_id for item in inverter_ids]
            
            for inv_id in inverter_ids:
                # Bu inverter için verileri filtrele
                inv_query = inverter_query.filter(InverterData.inverter_id == inv_id)
                inv_count = inv_query.count()
                
                if inv_count > 0:
                    # Tarih aralığı
                    inv_range = db.query(
                        func.min(InverterData.timestamp).label("min_date"),
                        func.max(InverterData.timestamp).label("max_date")
                    ).filter(inv_query.whereclause).first()
                    
                    # Güç çıktısı istatistikleri
                    power_stats = db.query(
                        func.avg(InverterData.power_output).label("avg_power"),
                        func.min(InverterData.power_output).label("min_power"),
                        func.max(InverterData.power_output).label("max_power")
                    ).filter(inv_query.whereclause).first()
                    
                    # Kayıp veri analizi
                    # Tarih aralığındaki tüm saatlerin sayısı
                    total_hours = 0
                    date_range_info = {}

                    if inv_range and inv_range.min_date is not None and inv_range.max_date is not None:
                        total_hours = (inv_range.max_date - inv_range.min_date).total_seconds() / 3600
                        total_hours = int(total_hours) + 1
                        date_range_info = {
                            "min_date": inv_range.min_date,
                            "max_date": inv_range.max_date,
                            "days": (inv_range.max_date.date() - inv_range.min_date.date()).days + 1
                        }
                    else:
                        date_range_info = {
                            "min_date": None,
                            "max_date": None,
                            "days": 0
                        }
                    
                    missing_hours = total_hours - inv_count
                    
                    inverter_stats[f"inverter_{inv_id}"] = {
                        "total_records": inv_count,
                        "date_range": date_range_info,
                        "power_output": {
                            "avg": float(power_stats.avg_power) if power_stats and power_stats.avg_power else 0,
                            "min": float(power_stats.min_power) if power_stats and power_stats.min_power else 0,
                            "max": float(power_stats.max_power) if power_stats and power_stats.max_power else 0
                        },
                        "data_completeness": {
                            "total_hours": total_hours,
                            "recorded_hours": inv_count,
                            "missing_hours": missing_hours,
                            "completeness_percentage": round((inv_count / total_hours) * 100, 2) if total_hours > 0 else 0
                        }
                    }
        
        # Hava durumu veri analizi
        weather_stats = {}
        if weather_count > 0:
            try:
                # Sıcaklık istatistikleri
                temp_stats = db.query(
                    func.avg(WeatherData.temperature).label("avg_temp"),
                    func.min(WeatherData.temperature).label("min_temp"),
                    func.max(WeatherData.temperature).label("max_temp")
                ).filter(weather_query.whereclause).first()
                
                # Radyasyon istatistikleri
                radiation_stats = db.query(
                    func.avg(WeatherData.direct_normal_irradiance).label("avg_dni"),
                    func.max(WeatherData.direct_normal_irradiance).label("max_dni")
                ).filter(weather_query.whereclause).first()
                
                # Tarih aralığındaki tüm saatlerin sayısı
                total_hours = 0
                missing_hours = 0
                
                # Yukarda elde ettiğimiz min_date ve max_date değerlerini kullanarak hesaplama yapalım
                if weather_date_range and weather_date_range["min_date"] and weather_date_range["max_date"]:
                    min_date = weather_date_range["min_date"]
                    max_date = weather_date_range["max_date"]
                    
                    total_hours = (max_date - min_date).total_seconds() / 3600
                    total_hours = int(total_hours) + 1
                    missing_hours = total_hours - weather_count
                
                weather_stats = {
                    "temperature": {
                        "avg": float(temp_stats.avg_temp) if temp_stats and temp_stats.avg_temp else 0,
                        "min": float(temp_stats.min_temp) if temp_stats and temp_stats.min_temp else 0,
                        "max": float(temp_stats.max_temp) if temp_stats and temp_stats.max_temp else 0
                    },
                    "radiation": {
                        "avg_dni": float(radiation_stats.avg_dni) if radiation_stats and radiation_stats.avg_dni else 0,
                        "max_dni": float(radiation_stats.max_dni) if radiation_stats and radiation_stats.max_dni else 0
                    },
                    "data_completeness": {
                        "total_hours": total_hours,
                        "recorded_hours": weather_count,
                        "missing_hours": missing_hours,
                        "completeness_percentage": round((weather_count / total_hours) * 100, 2) if total_hours > 0 else 0
                    }
                }
            except Exception as e:
                print(f"Hava durumu istatistikleri hesaplanırken hata: {e}")
                weather_stats = {
                    "temperature": {"avg": 0, "min": 0, "max": 0},
                    "radiation": {"avg_dni": 0, "max_dni": 0},
                    "data_completeness": {
                        "total_hours": 0, 
                        "recorded_hours": weather_count,
                        "missing_hours": 0,
                        "completeness_percentage": 0
                    }
                }
        
        # Sonuç
        return {
            "weather_data": {
                "total_records": weather_count,
                "date_range": weather_date_range,
                "statistics": weather_stats
            },
            "inverter_data": {
                "total_records": inverter_count,
                "date_range": inverter_date_range,
                "inverter_count": len(inverter_stats),
                "inverter_stats": inverter_stats
            },
            "query_parameters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }
    except Exception as e:
        print(f"Detaylı istatistikler hesaplanırken hata: {str(e)}")
        return {
            "error": True,
            "message": f"İstatistik hesaplanırken hata oluştu: {str(e)}",
            "weather_data": {
                "total_records": 0,
                "date_range": None,
                "statistics": {}
            },
            "inverter_data": {
                "total_records": 0,
                "date_range": None,
                "inverter_count": 0,
                "inverter_stats": {}
            },
            "query_parameters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
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
    try:
        # Arka planda hava durumu verilerini çek
        background_tasks.add_task(fetch_weather_data_for_dates, db)
        
        return {
            "success": True,
            "message": "Hava durumu verileri arka planda çekiliyor"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Hava durumu verileri çekilirken hata oluştu: {str(e)}"
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
    try:
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
            func.min(InverterData.timestamp),
            func.max(InverterData.timestamp)
        ).first()
        
        # İnverter sayısı
        inverter_count = db.query(Inverter).count()
        
        # Özet istatistikler
        return {
            "total_records": data_count,
            "date_range": {"start": date_range[0], "end": date_range[1]},
            "inverter_count": inverter_count
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"İnverter veri özeti alınırken hata oluştu: {str(e)}",
            "total_records": 0,
            "date_range": {"start": None, "end": None},
            "inverter_count": 0
        }

@router.post("/upload-txt", response_model=TxtDataUploadResponse)
async def upload_txt_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validate_only: bool = Form(False),
    fetch_weather: bool = Form(True),
    forced: bool = Form(False),
    train_models: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    TXT dosyasından inverter verilerini yükler.
    
    Args:
        file: TXT dosyası (tab-separated)
        validate_only: Sadece doğrulama yap, veritabanına kaydetme
        fetch_weather: İlgili hava durumu verilerini çek
        forced: Aynı zaman dilimindeki verilerin üzerine yazılsın mı?
        train_models: Veriler yüklendikten sonra model eğitimi yapılsın mı?
        db: Veritabanı oturumu
    """
    try:
        # TXT içeriğini oku
        contents = await file.read()
        txt_file = io.StringIO(contents.decode('utf-8'))
        
        # TXT verilerini doğrula
        validation_result = validate_txt_data(txt_file)
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_result["message"]
            )
        
        # Sadece doğrulama isteniyorsa, sonuçları döndür
        if validate_only:
            return {
                "success": True,
                "message": "TXT dosyası doğrulandı",
                "processed_rows": 0,
                "statistics": validation_result["statistics"]
            }
        
        # Dosyayı başa sar
        txt_file.seek(0)
        
        # TXT verilerini işle
        result = await process_txt_data(txt_file, db, forced)
        
        job_id = result.get("job_id")
        
        # Arka planda hava durumu verilerini çek
        if fetch_weather and result["processed_rows"] > 0:
            background_tasks.add_task(fetch_weather_data_for_dates, db)
        
        # Model eğitimi yapılsın mı?
        train_job_id = None
        if train_models and job_id:
            from app.services.model_training_service import start_all_models_training_job
            
            # Model eğitim işlemini başlat
            train_job_id = await start_all_models_training_job(
                db_connection_string=str(settings.DATABASE_URL),
                test_split=True,
                test_size=0.2
            )
        
        # Sonuç döndür
        return {
            "success": True,
            "message": f"Veri yükleme işlemi başlatıldı, durum için /api/data/txt-job-status/{job_id} endpointi kullanılabilir",
            "processed_rows": result["processed_rows"],
            "conflict_count": result["conflict_count"],
            "updated_count": result["updated_count"],
            "total_inverters": result["total_inverters"],
            "statistics": validation_result["statistics"],
            "job_id": job_id
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TXT yükleme hatası: {str(e)}"
        )

@router.get("/txt-job-status/{job_id}", response_model=TxtDataUploadJob)
async def get_txt_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    TXT veri yükleme işleminin durumunu döndürür.
    
    Args:
        job_id: İş kimliği
        db: Veritabanı oturumu
    """
    try:
        from app.services.data_import_service import get_txt_upload_job_status
        
        # İş durumunu al
        job_status = get_txt_upload_job_status(job_id)
        
        if job_status["status"] == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ID: {job_id} olan TXT yükleme işi bulunamadı"
            )
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TXT yükleme durumu alınırken hata: {str(e)}"
        )

@router.get("/model-logs/{model_version}", response_model=ModelLogResponse)
async def get_model_logs(
    model_version: str,
    db: Session = Depends(get_db)
):
    """
    Belirli bir model versiyonu için eğitim loglarını döndürür.
    
    Args:
        model_version: Model versiyonu
        db: Veritabanı oturumu
    """
    try:
        from app.models.model import Model
        import os
        import json
        
        # Modeli veritabanında ara
        model = db.query(Model).filter(Model.version == model_version).first()
        
        if not model:
            return {
                "success": False,
                "message": f"Model versiyonu bulunamadı: {model_version}",
                "log": None
            }
        
        # Model meta dosyasını kontrol et
        meta_path = os.path.join(os.path.dirname(model.model_path), f"{model_version}_meta.json")
        
        if not os.path.exists(meta_path):
            return {
                "success": False,
                "message": f"Model meta dosyası bulunamadı: {meta_path}",
                "log": None
            }
        
        # Meta dosyasını oku
        with open(meta_path, "r") as f:
            model_meta = json.load(f)
        
        # Log nesnesi oluştur
        from app.schemas.data import ModelLog, ModelLogEntry
        
        # Örnek log kayıtları (gerçek loglar yoksa)
        log_entries = []
        
        # Mevcut log dosyası varsa oku
        log_path = os.path.join(os.path.dirname(model.model_path), f"{model_version}.log")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_lines = f.readlines()
                for line in log_lines:
                    parts = line.strip().split(" - ", 3)
                    if len(parts) >= 3:
                        try:
                            timestamp = datetime.fromisoformat(parts[0])
                            level = parts[2]
                            message = parts[3] if len(parts) > 3 else ""
                            log_entries.append(ModelLogEntry(
                                timestamp=timestamp,
                                level=level,
                                message=message
                            ))
                        except Exception as e:
                            print(f"Log satırı ayrıştırılamadı: {line}, Hata: {str(e)}")
        
        # Eğer log kayıtları yoksa en azından temel bilgileri içeren bir log ekle
        if not log_entries:
            log_entries.append(ModelLogEntry(
                timestamp=model.created_at,
                level="INFO",
                message=f"Model eğitimi tamamlandı. Metrikler: {json.dumps(model.metrics, ensure_ascii=False)}"
            ))
        
        # Veri özeti (data_details)
        data_summary = model_meta.get("data_details", {})
        if not data_summary:
            data_summary = {
                "total_rows": model_meta.get("data_size", 0),
                "used_rows": model_meta.get("data_size", 0),
                "dropped_rows_ratio": 0
            }
        
        # ModelLog nesnesi oluştur
        model_log = ModelLog(
            model_version=model_version,
            inverter_id=model.inverter_id,
            created_at=model.created_at,
            data_summary=data_summary,
            feature_importance=model.feature_importance or {},
            metrics=model.metrics or {},
            logs=log_entries
        )
        
        return {
            "success": True,
            "message": "Model eğitim logları başarıyla alındı",
            "log": model_log
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
        return {
            "success": False,
            "message": f"Model logları alınırken hata oluştu: {str(e)}",
            "log": None
        }

@router.get("/cleanup-jobs")
async def cleanup_jobs(
    hours: int = Query(24, description="Temizlenecek saatlerin sayısı"),
    db: Session = Depends(get_db)
):
    """
    Belirli bir süreden daha eski işleri temizler.
    
    Args:
        hours: Maksimum saat cinsinden yaş
        db: Veritabanı oturumu
    """
    try:
        from app.services.model_training_service import cleanup_old_jobs
        from app.services.data_import_service import cleanup_old_txt_upload_jobs
        
        # Eski model eğitim işlerini temizle
        removed_model_jobs = cleanup_old_jobs(hours)
        
        # Eski TXT yükleme işlerini temizle
        removed_txt_jobs = cleanup_old_txt_upload_jobs(hours)
        
        return {
            "success": True,
            "message": f"{removed_model_jobs} model eğitim işi ve {removed_txt_jobs} TXT yükleme işi temizlendi",
            "removed_model_jobs": removed_model_jobs,
            "removed_txt_jobs": removed_txt_jobs
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"İşler temizlenirken hata: {str(e)}"
        )

@router.post("/train-models", response_model=ModelTrainingResponse)
async def train_models(
    background_tasks: BackgroundTasks,
    train_type: str = Form(..., description="Eğitim tipi: 'all' veya 'single'"),
    inverter_id: Optional[int] = Form(None, description="Tek bir inverter için eğitim yapılacaksa, ID'si"),
    test_split: bool = Form(True, description="Eğitim/test verisi ayrılsın mı?"),
    test_size: float = Form(0.2, description="Test verisi oranı (0-1 arası)"),
    db: Session = Depends(get_db)
):
    """
    İnverterler için model eğitimi başlatır.
    Bu işlem arka planda çalışır ve durumu job_id ile takip edilebilir.
    
    Args:
        train_type: Eğitim tipi ('all': tüm inverterler, 'single': tek inverter)
        inverter_id: Tek bir inverter için eğitim yapılacaksa ID'si
        test_split: Eğitim/test verisi ayrılsın mı?
        test_size: Test verisi oranı (0-1 arası) - Sabit 0.2 kullanılacak
        db: Veritabanı oturumu
    """
    try:
        from app.services.model_training_service import start_model_training_job, start_all_models_training_job
        
        job_id = None
        status = "queued"
        message = ""
        
        # Eğitim tipine göre işlem yap
        if train_type == "all":
            # Tüm inverterler için eğitim başlat
            job_id = await start_all_models_training_job(
                db_connection_string=str(settings.DATABASE_URL),
                test_split=test_split,
                test_size=0.2  # Sabit 0.2 değeri kullan
            )
            message = "Tüm inverterler için model eğitimi başlatıldı"
            
        elif train_type == "single":
            # Tek bir inverter için eğitim kontrolü
            if not inverter_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="'single' eğitim tipi için inverter_id gereklidir"
                )
            
            # İnverter'in var olup olmadığını kontrol et
            inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
            if not inverter:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"ID: {inverter_id} olan inverter bulunamadı"
                )
            
            # Tek inverter için eğitim başlat
            job_id = await start_model_training_job(
                inverter_id=inverter_id,
                db_connection_string=str(settings.DATABASE_URL),
                test_split=test_split,
                test_size=0.2  # Sabit 0.2 değeri kullan
            )
            message = f"Inverter {inverter_id} için model eğitimi başlatıldı"
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz eğitim tipi. 'all' veya 'single' olmalıdır"
            )
        
        return {
            "success": True,
            "message": message,
            "job_id": job_id,
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model eğitimi başlatma hatası: {str(e)}"
        )

@router.get("/training-status/{job_id}", response_model=ModelTrainingStatus)
async def get_training_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Model eğitim işleminin durumunu döndürür.
    
    Args:
        job_id: İş kimliği
        db: Veritabanı oturumu
    """
    try:
        from app.services.model_training_service import get_training_job_status
        
        # İş durumunu al
        job_status = get_training_job_status(job_id)
        
        if job_status["status"] == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ID: {job_id} olan eğitim işi bulunamadı"
            )
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Eğitim durumu alınırken hata: {str(e)}"
        ) 