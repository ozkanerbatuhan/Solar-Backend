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
        
        # DEBUG: CSV sütunlarını yazdır
        print(f"CSV Sütunları: {df.columns.tolist()}")
        print(f"CSV İlk satır: {df.iloc[0].to_dict()}")
        
        # Tarih sütununun datetime formatına dönüştürülmesi
        df['time'] = pd.to_datetime(df['time'])
        
        # Kaç satır işlendiğini takip etmek için sayaç
        processed_rows = 0
        weather_rows = 0
        inverter_rows = 0
        
        # DEBUG: Inverter sütunları var mı kontrol et
        inverter_columns = [col for col in df.columns if 'INV' in col and 'DayEnergy' in col]
        print(f"Bulunan inverter sütunları: {inverter_columns}")
        
        # Mevcut inverterleri kontrol et, yoksa oluştur
        inverter_ids = list(range(1, 9))  # 1'den 8'e kadar inverter ID'leri
        for inv_id in inverter_ids:
            inverter = db.query(Inverter).filter(Inverter.id == inv_id).first()
            if not inverter:
                print(f"Inverter-{inv_id} bulunamadı, oluşturuluyor...")
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
        print(f"Inverterler veritabanına kaydedildi. İşlenecek satır sayısı: {len(df)}")
        
        # DataFrame üzerinde iterate ederek verileri veritabanına kaydet
        for index, row in df.iterrows():
            # Hava durumu verisini ekle
            timestamp = row['time']
            
            # Mevcut hava durumu verisini kontrol et
            existing_weather = db.query(WeatherData).filter(
                WeatherData.timestamp == timestamp
            ).first()
            
            if not existing_weather:
                try:
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
                except Exception as e:
                    print(f"Hava durumu verisi eklenirken hata: {e}, Satır: {index}")
            
            # İnverter verilerini ekle
            for inv_id in range(1, 9):
                inverter_column = f"INV/{inv_id}/DayEnergy"
                
                # DEBUG: Her 100. satırda inverter verilerini kontrol et
                if index % 100 == 0:
                    print(f"Satır {index}, Inverter {inv_id}, Sütun: {inverter_column}, Değer: {row.get(inverter_column)}")
                
                # İnverter verisi varsa ekle
                if inverter_column in row and pd.notna(row[inverter_column]):
                    # Mevcut inverter verisini kontrol et
                    existing_inverter_data = db.query(InverterData).filter(
                        InverterData.inverter_id == inv_id,
                        InverterData.timestamp == timestamp
                    ).first()
                    
                    if not existing_inverter_data:
                        try:
                            inverter_data = InverterData(
                                inverter_id=inv_id,
                                timestamp=timestamp,
                                power_output=float(row[inverter_column]),
                                temperature=row.get('temperature_2m'),
                                irradiance=row.get('direct_normal_irradiance')
                            )
                            db.add(inverter_data)
                            inverter_rows += 1
                            
                            # Her 100. satırda commit durumunu bildir
                            if inverter_rows % 100 == 0:
                                print(f"Toplam {inverter_rows} inverter verisi eklendi")
                        except Exception as e:
                            print(f"Inverter verisi eklenirken hata: {e}, Inverter: {inv_id}, Satır: {index}, Değer: {row.get(inverter_column)}")
                    else:
                        if index % 1000 == 0:
                            print(f"Inverter {inv_id} için zaten veri var: {timestamp}")
            
            processed_rows += 1
            
            # Her 100 satırda bir commit yap (bellek optimizasyonu için)
            if processed_rows % 100 == 0:
                print(f"İşlenen satır: {processed_rows}, Weather: {weather_rows}, Inverter: {inverter_rows}")
                db.commit()
        
        # Kalan işlemler için son commit
        db.commit()
        print(f"İşlem tamamlandı. Toplam satır: {processed_rows}, Weather: {weather_rows}, Inverter: {inverter_rows}")
        
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
        # Hata durumunda rollback ve debug mesajı
        db.rollback()
        print(f"Kritik hata: {str(e)}")
        import traceback
        print(traceback.format_exc())
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
        weather_range = db.query(
            db.func.min(WeatherData.timestamp).label("min_date"),
            db.func.max(WeatherData.timestamp).label("max_date")
        ).filter(weather_query.whereclause).first()
        
        weather_date_range = {
            "min_date": weather_range.min_date,
            "max_date": weather_range.max_date,
            "days": (weather_range.max_date.date() - weather_range.min_date.date()).days + 1
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
            db.func.min(InverterData.timestamp).label("min_date"),
            db.func.max(InverterData.timestamp).label("max_date")
        ).filter(inverter_query.whereclause).first()
        
        inverter_date_range = {
            "min_date": inverter_range.min_date,
            "max_date": inverter_range.max_date,
            "days": (inverter_range.max_date.date() - inverter_range.min_date.date()).days + 1
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
                    db.func.min(InverterData.timestamp).label("min_date"),
                    db.func.max(InverterData.timestamp).label("max_date")
                ).filter(inv_query.whereclause).first()
                
                # Güç çıktısı istatistikleri
                power_stats = db.query(
                    db.func.avg(InverterData.power_output).label("avg_power"),
                    db.func.min(InverterData.power_output).label("min_power"),
                    db.func.max(InverterData.power_output).label("max_power")
                ).filter(inv_query.whereclause).first()
                
                # Kayıp veri analizi
                # Tarih aralığındaki tüm saatlerin sayısı
                total_hours = (inv_range.max_date - inv_range.min_date).total_seconds() / 3600
                total_hours = int(total_hours) + 1
                missing_hours = total_hours - inv_count
                
                inverter_stats[f"inverter_{inv_id}"] = {
                    "total_records": inv_count,
                    "date_range": {
                        "min_date": inv_range.min_date,
                        "max_date": inv_range.max_date,
                        "days": (inv_range.max_date.date() - inv_range.min_date.date()).days + 1
                    },
                    "power_output": {
                        "avg": float(power_stats.avg_power) if power_stats.avg_power else 0,
                        "min": float(power_stats.min_power) if power_stats.min_power else 0,
                        "max": float(power_stats.max_power) if power_stats.max_power else 0
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
        # Sıcaklık istatistikleri
        temp_stats = db.query(
            db.func.avg(WeatherData.temperature).label("avg_temp"),
            db.func.min(WeatherData.temperature).label("min_temp"),
            db.func.max(WeatherData.temperature).label("max_temp")
        ).filter(weather_query.whereclause).first()
        
        # Radyasyon istatistikleri
        radiation_stats = db.query(
            db.func.avg(WeatherData.direct_normal_irradiance).label("avg_dni"),
            db.func.max(WeatherData.direct_normal_irradiance).label("max_dni")
        ).filter(weather_query.whereclause).first()
        
        # Tarih aralığındaki tüm saatlerin sayısı
        total_hours = (weather_range.max_date - weather_range.min_date).total_seconds() / 3600
        total_hours = int(total_hours) + 1
        missing_hours = total_hours - weather_count
        
        weather_stats = {
            "temperature": {
                "avg": float(temp_stats.avg_temp) if temp_stats.avg_temp else 0,
                "min": float(temp_stats.min_temp) if temp_stats.min_temp else 0,
                "max": float(temp_stats.max_temp) if temp_stats.max_temp else 0
            },
            "radiation": {
                "avg_dni": float(radiation_stats.avg_dni) if radiation_stats.avg_dni else 0,
                "max_dni": float(radiation_stats.max_dni) if radiation_stats.max_dni else 0
            },
            "data_completeness": {
                "total_hours": total_hours,
                "recorded_hours": weather_count,
                "missing_hours": missing_hours,
                "completeness_percentage": round((weather_count / total_hours) * 100, 2) if total_hours > 0 else 0
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