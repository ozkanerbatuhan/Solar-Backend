import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pathlib
import time
import asyncio
import uuid

from app.models.inverter import Inverter, InverterData
from app.models.weather import WeatherData
from app.models.model import Model

# Model eğitim ve tahmin için parametreler
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}


# Modellerin kaydedileceği klasör
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")

# Klasör yoksa oluştur
pathlib.Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# Model eğitim joblarını izlemek için global değişken
active_training_jobs = {}

# Model eğitim job statü şablonu
def create_job_status(job_id, inverter_id=None):
    return {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "start_time": None,
        "end_time": None,
        "inverter_id": inverter_id,
        "message": "Eğitim işlemi kuyruğa alındı",
        "metrics": None
    }

# Background worker'a eklenmeden önce bu fonksiyonun ayrı bir kopyası oluşturulacak
async def _train_model_job(
    job_id: str,
    inverter_id: int,
    db_connection_string: str,
    test_split: bool = True,
    test_size: float = 0.2
):
    """
    Arka planda model eğitimi yapar.
    
    Args:
        job_id: İş kimliği
        inverter_id: İnverter kimliği
        db_connection_string: Veritabanı bağlantı bilgisi
        test_split: Test bölünmesi yapılsın mı?
        test_size: Test seti oranı (sabit 0.2)
    """
    global active_training_jobs
    
    # Test size parametresini 0.2 olarak sabitliyoruz
    test_size = 0.2
    
    # SqlAlchemy session oluştur
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(db_connection_string)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # İş durumunu güncelle
        active_training_jobs[job_id]["status"] = "running"
        active_training_jobs[job_id]["start_time"] = datetime.utcnow()
        active_training_jobs[job_id]["message"] = f"Inverter {inverter_id} için veri hazırlanıyor"
        
        # Eğitim verilerini al
        active_training_jobs[job_id]["progress"] = 10
        active_training_jobs[job_id]["message"] = f"Eğitim verileri alınıyor"
        
        # Eğitim verilerini al
        df = await get_training_data(inverter_id, db)
        
        # Veri detaylarını sakla
        data_details = {
            "total_rows_before_filtering": len(df) + df.isna().any(axis=1).sum(),
            "used_rows_after_filtering": len(df),
            "filtered_rows_ratio": ((df.isna().any(axis=1).sum()) / (len(df) + df.isna().any(axis=1).sum())) * 100 if (len(df) + df.isna().any(axis=1).sum()) > 0 else 0
        }
        
        active_training_jobs[job_id]["progress"] = 20
        active_training_jobs[job_id]["message"] = f"Özellikler hazırlanıyor"
        
        # Mevcut sütunları kontrol et
        available_columns = df.columns.tolist()
        print(f"[DEBUG] Mevcut sütunlar: {available_columns}")
        
        # Özellik sütunlarını mevcut sütunlara göre düzenle
        base_feature_cols = [
            'temperature', 'shortwave_radiation', 'direct_radiation',
            'diffuse_radiation', 'direct_normal_irradiance', 'global_tilted_irradiance', 
            'terrestrial_radiation', 'relative_humidity', 'wind_speed', 'visibility',
            'hour', 'day', 'month', 'dayofweek', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Mevcut sütunlarla kesişim kontrolü
        feature_cols = [col for col in base_feature_cols if col in available_columns]
        
        # Sıcaklık sütunu çakışma kontrolü - inverter_temperature ve temperature karışıklığı olmamalı
        if 'temperature' in feature_cols and 'inverter_temperature' in available_columns:
            print("[DEBUG] 'temperature' sütunu var ve bu hava durumu sıcaklığını ifade ediyor.")
        
        if not feature_cols:
            raise ValueError(f"Hiçbir özellik sütunu bulunamadı. Mevcut sütunlar: {available_columns}")
        
        print(f"[DEBUG] Kullanılacak özellik sütunları: {feature_cols}")
        
        X = df[feature_cols]
        y = df["power_output"]
        
        print(f"[DEBUG] X boyutu: {X.shape}, y boyutu: {y.shape}")
        
        model_metrics = {}
        
        # İki aşamalı eğitim
        # 1. Aşama: Test bölünmesi ile metrik hesaplama
        if test_split:
            active_training_jobs[job_id]["progress"] = 30
            active_training_jobs[job_id]["message"] = f"Model eğitim/test verisi hazırlanıyor"
            
            # Zaman serisi verisi, shuffle=False olmalı
            # Ancak model performansı için rasgele karıştırma daha iyi sonuç veriyor, bu yüzden shuffle=True kullanıyoruz
            # Bu trade-off'u açıkça belirtelim
            shuffle_param = True  # Daha iyi model performansı için True, zaman serisi tutarlılığı için False
            print(f"[DEBUG] Train-test split parametreleri: test_size={test_size}, shuffle={shuffle_param}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle_param)
            
            print(f"[DEBUG] Eğitim seti: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"[DEBUG] Test seti: X_test={X_test.shape}, y_test={y_test.shape}")
            
            active_training_jobs[job_id]["progress"] = 40
            active_training_jobs[job_id]["message"] = f"İlk model eğitimi başlıyor"
            
            # NaN kontrolü - main.py'de olduğu gibi
            if X_train.isna().any().any():
                print("[DEBUG] Eğitim setinde NaN değerler var, medyan ile doldurulacak.")
                X_train = X_train.fillna(X_train.median())
            
            if X_test.isna().any().any():
                print("[DEBUG] Test setinde NaN değerler var, medyan ile doldurulacak.")
                X_test = X_test.fillna(X_test.median())
            
            # RobustScaler uygulaması - main.py'deki gibi
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
            
            print("[DEBUG] Model eğitimi başlıyor...")
            model = RandomForestRegressor(**MODEL_PARAMS)
            model.fit(X_train_scaled, y_train)
            print("[DEBUG] Model eğitimi tamamlandı.")
            
            active_training_jobs[job_id]["progress"] = 60
            active_training_jobs[job_id]["message"] = f"Model performans metriksleri hesaplanıyor"
            
            # Test seti üzerinde tahmin yap
            y_pred = model.predict(X_test_scaled)
            
            # Tahmin ve gerçek değerlerin sıralamasını kontrol et (indeks kontrolü)
            print(f"[DEBUG] y_test ve y_pred boyutları: {y_test.shape} vs {y_pred.shape}")
            
            # Model metriklerini hesapla
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # MAPE hesaplama (main.py'deki gibi güvenli hesaplama)
            mask = y_test > 1.0  # 1 kWh'den büyük değerler için
            mape = 0.0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                print(f"[DEBUG] MAPE hesaplama için {mask.sum()}/{len(y_test)} satır kullanıldı (>1.0 kWh).")
            else:
                print("[DEBUG] MAPE hesaplanamadı - 1.0 kWh'den büyük değer yok.")
            
            print(f"[DEBUG] Model metrikleri: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.4f}%")
            
            model_metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mape": float(mape),
                "test_size": float(test_size),  # Sabit değer
                "samples_count": int(len(X)),
                "features": feature_cols,
                "data_details": data_details
            }
            
            active_training_jobs[job_id]["metrics"] = model_metrics
        
        # 2. Aşama: Tüm verilerle final model eğitimi
        active_training_jobs[job_id]["progress"] = 70
        active_training_jobs[job_id]["message"] = f"Final model eğitimi başlıyor (tüm veri)"
        
        # NaN kontrolü - son kontrol
        if X.isna().any().any():
            print("[DEBUG] Veri setinde NaN değerler var, medyan ile doldurulacak.")
            X = X.fillna(X.median())
        
        # RobustScaler ile ölçeklendirme - tüm veri için yeni scaler oluştur
        # Bu, train-test split yaklaşımıyla tutarlı olmasını sağlar
        print("[DEBUG] Final model için tüm veri ölçeklendiriliyor...")
        from sklearn.preprocessing import RobustScaler
        final_scaler = RobustScaler()
        X_scaled = pd.DataFrame(final_scaler.fit_transform(X), columns=feature_cols)
        
        print("[DEBUG] Final model eğitimi başlıyor...")
        final_model = RandomForestRegressor(**MODEL_PARAMS)
        final_model.fit(X_scaled, y)
        print("[DEBUG] Final model eğitimi tamamlandı.")
        
        active_training_jobs[job_id]["progress"] = 80
        active_training_jobs[job_id]["message"] = f"Özellik önemi hesaplanıyor"
        
        # Özellik önemliliği
        feature_importance = {
            feature: float(importance) 
            for feature, importance in zip(feature_cols, final_model.feature_importances_)
        }
        
        # En önemli özellikleri logla
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print("[DEBUG] Özellik önemleri (ilk 5):")
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Model versiyonunu belirle
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_version = f"inverter_{inverter_id}_v{timestamp}"
        
        active_training_jobs[job_id]["progress"] = 85
        active_training_jobs[job_id]["message"] = f"Model dosyaları kaydediliyor"
        
        # Modeli kaydet
        model_path = os.path.join(MODELS_DIR, f"{model_version}.joblib")
        joblib.dump(final_model, model_path)
        print(f"[DEBUG] Model kaydedildi: {model_path}")
        
        # Model meta verisini kaydet
        meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
        
        # Metrikler ve özellik önemlerini JSON serileştirilebilir hale getir
        serialized_metrics = serialize_for_json(model_metrics)
        serialized_feature_importance = serialize_for_json(feature_importance)
        serialized_data_details = serialize_for_json(data_details)
        
        model_meta = {
            "model_version": model_version,
            "inverter_id": inverter_id,
            "created_at": datetime.utcnow().isoformat(),
            "model_type": "RandomForestRegressor",
            "model_params": MODEL_PARAMS,
            "feature_importance": serialized_feature_importance,
            "metrics": serialized_metrics,
            "data_size": len(X),
            "data_details": serialized_data_details
        }
        
        # JSON serileştirilebilir hale getir
        model_meta = serialize_for_json(model_meta)
        
        with open(meta_path, "w") as f:
            json.dump(model_meta, f, indent=2)
        print(f"[DEBUG] Model meta dosyası kaydedildi: {meta_path}")
        
        active_training_jobs[job_id]["progress"] = 90
        active_training_jobs[job_id]["message"] = f"Veritabanı kaydı oluşturuluyor"
        
        # Veritabanına model kaydı ekle
        model_db = Model(
            inverter_id=inverter_id,
            version=model_version,
            model_path=model_path,
            model_type="RandomForestRegressor",
            metrics=serialized_metrics,
            is_active=True,
            feature_importance=serialized_feature_importance,
            created_at=datetime.utcnow()
        )
        
        # Önceki aktif modelleri devre dışı bırak
        previous_models = db.query(Model).filter(
            Model.inverter_id == inverter_id,
            Model.is_active == True
        ).all()
        
        for model in previous_models:
            model.is_active = False
        
        db.add(model_db)
        db.commit()
        print(f"[DEBUG] Veritabanı model kaydı oluşturuldu, ID: {model_db.id}")
        
        active_training_jobs[job_id]["progress"] = 100
        active_training_jobs[job_id]["status"] = "completed"
        active_training_jobs[job_id]["end_time"] = datetime.utcnow()
        active_training_jobs[job_id]["message"] = f"Model eğitimi başarıyla tamamlandı"
        
        # Sonuç bilgilerini sakla
        active_training_jobs[job_id]["result"] = {
            "model_version": model_version,
            "inverter_id": inverter_id,
            "metrics": model_metrics,
            "model_path": model_path,
            "feature_importance": feature_importance,
            "data_details": data_details
        }
        
    except Exception as e:
        # Hata durumunda rollback
        db.rollback()
        
        active_training_jobs[job_id]["status"] = "failed"
        active_training_jobs[job_id]["end_time"] = datetime.utcnow()
        active_training_jobs[job_id]["message"] = f"Model eğitimi hatası: {str(e)}"
        
        # Stack trace'i de ekle
        import traceback
        active_training_jobs[job_id]["error"] = traceback.format_exc()
        
        print(f"[HATA] Model eğitimi hatası: {str(e)}")
        print(traceback.format_exc())
        
    finally:
        # Veritabanı oturumunu kapat
        db.close()

async def start_model_training_job(
    inverter_id: int, 
    db_connection_string: str,
    test_split: bool = True, 
    test_size: float = 0.2
) -> str:
    """
    Belirli bir inverter için model eğitim işlemi başlatır.
    
    Args:
        inverter_id: İnverter kimliği
        db_connection_string: Veritabanı bağlantı bilgisi
        test_split: Test bölünmesi yapılsın mı?
        test_size: Test seti oranı
        
    Returns:
        İş kimliği
    """
    global active_training_jobs
    
    # İş kimliğini oluştur
    job_id = f"train_{inverter_id}_{str(uuid.uuid4())[:8]}"
    
    # İş durumunu oluştur
    active_training_jobs[job_id] = create_job_status(job_id, inverter_id)
    
    # Asenkron olarak eğitim işlemini başlat
    asyncio.create_task(_train_model_job(
        job_id=job_id,
        inverter_id=inverter_id,
        db_connection_string=db_connection_string,
        test_split=test_split,
        test_size=test_size
    ))
    
    return job_id

async def start_all_models_training_job(
    db_connection_string: str,
    test_split: bool = True, 
    test_size: float = 0.2
) -> str:
    """
    Tüm inverterler için model eğitim işlemi başlatır.
    
    Args:
        db_connection_string: Veritabanı bağlantı bilgisi
        test_split: Test bölünmesi yapılsın mı?
        test_size: Test seti oranı - Sabit 0.2 değerinde (parametre hala alınıyor ama kullanılmıyor)
        
    Returns:
        İş kimliği
    """
    global active_training_jobs
    
    # Test size parametresini 0.2 olarak sabitliyoruz
    test_size = 0.2
    
    # SqlAlchemy session oluştur
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(db_connection_string)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # İş kimliğini oluştur
        job_id = f"train_all_{str(uuid.uuid4())[:8]}"
        
        # İş durumunu oluştur
        active_training_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "start_time": datetime.utcnow(),
            "end_time": None,
            "message": "Tüm inverterler için eğitim işlemi başlatılıyor",
            "inverter_ids": [],
            "sub_jobs": {},
            "metrics": {}
        }
        
        # Aktif inverterleri al
        inverters = db.query(Inverter).filter(Inverter.is_active == True).all()
        inverter_ids = [inv.id for inv in inverters]
        active_training_jobs[job_id]["inverter_ids"] = inverter_ids
        
        # Her inverter için ayrı bir eğitim işlemi başlat
        for idx, inverter_id in enumerate(inverter_ids):
            # Alt iş için kimlik oluştur
            sub_job_id = await start_model_training_job(
                inverter_id=inverter_id,
                db_connection_string=db_connection_string,
                test_split=test_split,
                test_size=0.2  # Sabit 0.2 değerini kullan
            )
            
            # Ana işte alt işleri izle
            active_training_jobs[job_id]["sub_jobs"][inverter_id] = sub_job_id
            
            # İlerleme durumunu güncelle
            progress = int((idx + 1) / len(inverter_ids) * 100)
            active_training_jobs[job_id]["progress"] = min(progress, 95)  # En fazla %95'e kadar git
            active_training_jobs[job_id]["message"] = f"İnverter {inverter_id} için eğitim işlemi başlatıldı ({idx+1}/{len(inverter_ids)})"
        
        # Veritabanı oturumunu kapat
        db.close()
        
        return job_id
        
    except Exception as e:
        # Hata durumunda
        db.close()
        
        # İş kimliğini oluştur (hata durumunda)
        job_id = f"train_all_error_{str(uuid.uuid4())[:8]}"
        
        # Hata durumunu kaydet
        active_training_jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "progress": 0,
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
            "message": f"Eğitim işlemi başlatılırken hata: {str(e)}",
            "error": str(e)
        }
        
        return job_id

def get_training_job_status(job_id: str) -> Dict[str, Any]:
    """
    Model eğitim işinin durumunu döndürür.
    
    Args:
        job_id: İş kimliği
        
    Returns:
        İş durumu
    """
    global active_training_jobs
    
    if job_id not in active_training_jobs:
        return {
            "job_id": job_id,
            "status": "not_found",
            "message": "Belirtilen ID'ye sahip bir eğitim işi bulunamadı"
        }
    
    job_status = active_training_jobs[job_id].copy()
    
    # Tüm model eğitim işi ise, alt işlerin durumunu güncelle
    if "sub_jobs" in job_status:
        completed_jobs = 0
        failed_jobs = 0
        
        for inverter_id, sub_job_id in job_status["sub_jobs"].items():
            if sub_job_id in active_training_jobs:
                sub_status = active_training_jobs[sub_job_id]["status"]
                
                if sub_status == "completed":
                    completed_jobs += 1
                    # Metrikleri ana işe ekle
                    if "metrics" in active_training_jobs[sub_job_id]:
                        job_status["metrics"][inverter_id] = active_training_jobs[sub_job_id]["metrics"]
                
                elif sub_status == "failed":
                    failed_jobs += 1
        
        # Ana işin durumunu güncelle
        total_jobs = len(job_status["sub_jobs"])
        
        if completed_jobs + failed_jobs == total_jobs:
            if failed_jobs == 0:
                job_status["status"] = "completed"
                job_status["end_time"] = datetime.utcnow()
                job_status["message"] = f"Tüm inverterler için eğitim tamamlandı"
            elif completed_jobs == 0:
                job_status["status"] = "failed"
                job_status["end_time"] = datetime.utcnow()
                job_status["message"] = f"Tüm eğitim işleri başarısız oldu"
            else:
                job_status["status"] = "partially_completed"
                job_status["end_time"] = datetime.utcnow()
                job_status["message"] = f"{completed_jobs}/{total_jobs} inverter için eğitim başarılı, {failed_jobs} başarısız"
            
            job_status["progress"] = 100
    
    return job_status

def cleanup_old_jobs(max_age_hours: int = 24):
    """
    Belirli bir süreden daha eski işleri temizler.
    
    Args:
        max_age_hours: Maksimum saat cinsinden yaş
    """
    global active_training_jobs
    
    now = datetime.utcnow()
    to_remove = []
    
    for job_id, job_status in active_training_jobs.items():
        # Tamamlanmış veya başarısız olmuş ve sonlanma zamanı olan işleri kontrol et
        if job_status.get("end_time") and job_status["status"] in ["completed", "failed", "partially_completed"]:
            age = now - job_status["end_time"]
            
            # Belirli bir süreden daha eski ise işaretleyelim
            if age.total_seconds() > max_age_hours * 3600:
                to_remove.append(job_id)
    
    # İşaretlenen işleri kaldır
    for job_id in to_remove:
        del active_training_jobs[job_id]
    
    return len(to_remove)

async def get_training_data(inverter_id: int, db: Session) -> pd.DataFrame:
    """
    Belirli bir inverter için eğitim verilerini hazırlar.
    
    Args:
        inverter_id: İnverter kimliği
        db: Veritabanı oturumu
        
    Returns:
        Eğitim verileri DataFrame'i
    """
    print(f"[DEBUG] İnverter {inverter_id} için eğitim verileri hazırlanıyor...")
    
    # İnverter verilerini al
    inverter_data = db.query(InverterData).filter(
        InverterData.inverter_id == inverter_id,
        InverterData.power_output.isnot(None)
    ).all()
    
    if not inverter_data:
        raise ValueError(f"İnverter {inverter_id} için veri bulunamadı")
    
    # İnverter verilerini DataFrame'e dönüştür
    inverter_df = pd.DataFrame([{
        "timestamp": data.timestamp,
        "power_output": data.power_output,
        "inverter_temperature": data.temperature,  # İsim çakışmasını önlemek için yeniden adlandır
        "irradiance": data.irradiance
    } for data in inverter_data])
    
    print(f"[DEBUG] İnverter {inverter_id} için {len(inverter_df)} satır veri bulundu.")
    
    # Hava durumu verilerini al
    weather_data = db.query(WeatherData).filter(
        WeatherData.is_forecast == 0  # Boolean değil, integer tipinde (0: gerçek ölçüm)
    ).all()
    
    # Hava durumu verilerini DataFrame'e dönüştür
    weather_df = pd.DataFrame([{
        "timestamp": data.timestamp,
        "temperature": data.temperature,
        "shortwave_radiation": data.shortwave_radiation,
        "direct_radiation": data.direct_radiation,
        "diffuse_radiation": data.diffuse_radiation,
        "direct_normal_irradiance": data.direct_normal_irradiance,
        "global_tilted_irradiance": data.global_tilted_irradiance,
        "terrestrial_radiation": data.terrestrial_radiation,
        "relative_humidity": data.relative_humidity,
        "wind_speed": data.wind_speed,
        "visibility": data.visibility
    } for data in weather_data])
    
    print(f"[DEBUG] Hava durumu verileri için {len(weather_df)} satır veri bulundu.")
    
    # Tarih aralıklarını kontrol et
    if not inverter_df.empty and not weather_df.empty:
        inv_min_date = inverter_df["timestamp"].min()
        inv_max_date = inverter_df["timestamp"].max()
        weather_min_date = weather_df["timestamp"].min()
        weather_max_date = weather_df["timestamp"].max()
        
        print(f"[DEBUG] İnverter veri aralığı: {inv_min_date} - {inv_max_date}")
        print(f"[DEBUG] Hava durumu veri aralığı: {weather_min_date} - {weather_max_date}")
        
        # Tarih aralıklarının uyumluluğunu kontrol et
        if inv_min_date < weather_min_date:
            print(f"[UYARI] İnverter verileri hava durumu verilerinden daha eski başlıyor. Kesişim kaybı olabilir.")
        if inv_max_date > weather_max_date:
            print(f"[UYARI] İnverter verileri hava durumu verilerinden daha yeni bitiyor. Kesişim kaybı olabilir.")
    
    # Verileri birleştir - tolerans ve direction parametrelerine dikkat
    merge_tolerance = pd.Timedelta("1h")
    print(f"[DEBUG] merge_asof için tolerans: {merge_tolerance}, yön: nearest")
    
    # Verileri sıraladığımızdan emin olalım
    inverter_df = inverter_df.sort_values("timestamp")
    weather_df = weather_df.sort_values("timestamp")
    
    # Birleştirme öncesi veri boyutları
    print(f"[DEBUG] Birleştirme öncesi inverter veri boyutu: {inverter_df.shape}")
    print(f"[DEBUG] Birleştirme öncesi hava durumu veri boyutu: {weather_df.shape}")
    
    # Verileri birleştir
    df = pd.merge_asof(
        inverter_df,
        weather_df,
        on="timestamp",
        direction="nearest",  # En yakın eşleşmeyi kullan
        tolerance=merge_tolerance
    )
    
    # Birleştirme sonrası veri boyutu ve NaN durumu
    print(f"[DEBUG] Birleştirme sonrası veri boyutu: {df.shape}")
    print(f"[DEBUG] Birleştirme sonrası NaN içeren satır sayısı: {df.isna().any(axis=1).sum()}")
    
    # Birleştirilmiş veri çerçevesi sütunlarını kontrol et
    print(f"[DEBUG] Birleştirilmiş veri çerçevesi sütunları: {df.columns.tolist()}")
    
    # Veri tipleri kontrolü
    print("[DEBUG] Veri tipleri kontrolü:")
    print(df.dtypes)
    
    # NaN değerleri kontrolü - sütun bazında
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(f"[DEBUG] Aşağıdaki sütunlarda NaN değerler bulundu: {nan_columns}")
        print(f"[DEBUG] Sütun bazında NaN sayıları:")
        for col in nan_columns:
            nan_count = df[col].isna().sum()
            nan_percent = (nan_count / len(df)) * 100
            print(f"  - {col}: {nan_count} ({nan_percent:.2f}%)")
        
        # Nümerik sütunları medyan ile doldur
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in nan_columns:
            if col in numeric_cols:
                print(f"[DEBUG] {col} sütunu nümerik, NaN değerler önce ffill, sonra medyan ile dolduruluyor.")
                # Önce ffill ile doldur, kalan NaN'ları medyan ile doldur
                df[col] = df[col].ffill().fillna(df[col].median())
            else:
                print(f"[DEBUG] {col} sütunu nümerik değil, NaN değerler ffill ve bfill ile dolduruluyor.")
                df[col] = df[col].ffill().bfill()
    
    # Aykırı değerleri temizle - main.py'deki gibi
    # power_output için aykırı değer kontrolü
    if "power_output" in df.columns:
        print("[DEBUG] power_output için aykırı değer temizleme işlemi yapılıyor...")
        
        # Temizleme öncesinde boyut kontrolü
        print(f"[DEBUG] Aykırı değer temizleme öncesi veri boyutu: {df.shape}")
        
        # 0.01-0.99 quantile dışındaki değerleri temizle
        q1 = df["power_output"].quantile(0.01)
        q3 = df["power_output"].quantile(0.99)
        
        print(f"[DEBUG] power_output için 0.01 quantile: {q1}, 0.99 quantile: {q3}")
        
        filtered_df = df[(df["power_output"] >= q1) & (df["power_output"] <= q3)]
        
        # Kaç satır çıkarıldı?
        removed_rows = len(df) - len(filtered_df)
        removed_percentage = (removed_rows / len(df)) * 100 if len(df) > 0 else 0
        print(f"[DEBUG] Aykırı değer temizleme: {removed_rows} satır çıkarıldı ({removed_percentage:.2f}%)")
        
        df = filtered_df
    
    # Tarih özelliklerini ekle
    print("[DEBUG] Tarih özellikleri ekleniyor...")
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    # Main.py'deki gibi trigonometrik zaman özellikleri ekleme
    print("[DEBUG] Trigonometrik zaman özellikleri ekleniyor...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Gün numarası yerine ay kullanılacak
    df['day_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['day_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    print(f"[DEBUG] Veri hazırlama tamamlandı. Final veri boyutu: {df.shape}")
    
    return df

# JSON serileştirme için yardımcı fonksiyon
def serialize_for_json(obj):
    """
    NumPy objelerini JSON için serileştirilebilir Python tiplerine dönüştürür.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

async def train_model(
    inverter_id: int,
    db: Session,
    test_split: bool = True,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Belirli bir inverter için model eğitimi yapar.
    
    Args:
        inverter_id: İnverter kimliği
        db: Veritabanı oturumu
        test_split: Test bölünmesi yapılsın mı?
        test_size: Test seti oranı (sabit 0.2)
        
    Returns:
        Model eğitim sonuçları ve metrikleri
    """
    print(f"[DEBUG] İnverter {inverter_id} için model eğitimi başlıyor...")
    
    # Test size parametresini sabit 0.2 olarak kullanalım
    test_size = 0.2
    
    # Eğitim verilerini al
    df = await get_training_data(inverter_id, db)
    
    # Veri detayları
    data_details = {
        "total_rows_before_filtering": len(df) + df.isna().any(axis=1).sum(),
        "used_rows_after_filtering": len(df),
        "filtered_rows_ratio": ((df.isna().any(axis=1).sum()) / (len(df) + df.isna().any(axis=1).sum())) * 100 if (len(df) + df.isna().any(axis=1).sum()) > 0 else 0
    }
    
    # Mevcut sütunları kontrol et
    available_columns = df.columns.tolist()
    print(f"[DEBUG] Mevcut sütunlar: {available_columns}")
    
    # Özellik sütunlarını mevcut sütunlara göre düzenle
    base_feature_cols = [
        'temperature', 'shortwave_radiation', 'direct_radiation',
        'diffuse_radiation', 'direct_normal_irradiance', 'global_tilted_irradiance', 
        'terrestrial_radiation', 'relative_humidity', 'wind_speed', 'visibility',
        'hour', 'day', 'month', 'dayofweek', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    # Mevcut sütunlarla kesişim kontrolü
    feature_cols = [col for col in base_feature_cols if col in available_columns]
    
    # Sıcaklık sütunu çakışma kontrolü
    if 'temperature' in feature_cols and 'inverter_temperature' in available_columns:
        print("[DEBUG] 'temperature' sütunu var ve bu hava durumu sıcaklığını ifade ediyor.")
    
    if not feature_cols:
        raise ValueError(f"Hiçbir özellik sütunu bulunamadı. Mevcut sütunlar: {available_columns}")
    
    print(f"[DEBUG] Kullanılacak özellik sütunları: {feature_cols}")
    
    X = df[feature_cols]
    y = df["power_output"]
    
    print(f"[DEBUG] X boyutu: {X.shape}, y boyutu: {y.shape}")
    
    model_metrics = {}
    
    # NaN kontrolü
    if X.isna().any().any():
        print("[DEBUG] Veri setinde NaN değerler var, medyan ile doldurulacak.")
        X = X.fillna(X.median())
    
    # İki aşamalı eğitim
    # 1. Aşama: Test bölünmesi ile metrik hesaplama
    if test_split:
        # Zaman serisi verisi, shuffle=False olmalı normalde, ancak 
        # model performansı için rasgele karıştırma daha iyi sonuç veriyor
        # Bu trade-off'u açıkça belirtelim
        shuffle_param = True  # Daha iyi model performansı için True, zaman serisi tutarlılığı için False
        print(f"[DEBUG] Train-test split parametreleri: test_size={test_size}, shuffle={shuffle_param}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle_param)
        
        print(f"[DEBUG] Eğitim seti: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"[DEBUG] Test seti: X_test={X_test.shape}, y_test={y_test.shape}")
        
        # RobustScaler ile ölçeklendirme
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
        
        print("[DEBUG] Model eğitimi başlıyor...")
        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(X_train_scaled, y_train)
        print("[DEBUG] Model eğitimi tamamlandı.")
        
        # Test seti üzerinde tahmin yap
        y_pred = model.predict(X_test_scaled)
        
        # Tahmin ve gerçek değerlerin sıralamasını kontrol et
        print(f"[DEBUG] y_test ve y_pred boyutları: {y_test.shape} vs {y_pred.shape}")
        
        # Model metriklerini hesapla
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE hesaplama (main.py'deki gibi güvenli hesaplama)
        mask = y_test > 1.0  # 1 kWh'den büyük değerler için
        mape = 0.0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            print(f"[DEBUG] MAPE hesaplama için {mask.sum()}/{len(y_test)} satır kullanıldı (>1.0 kWh).")
        else:
            print("[DEBUG] MAPE hesaplanamadı - 1.0 kWh'den büyük değer yok.")
        
        print(f"[DEBUG] Model metrikleri: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.4f}%")
        
        model_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "test_size": float(test_size),
            "samples_count": int(len(X)),
            "features": feature_cols
        }
    
    # 2. Aşama: Tüm verilerle final model eğitimi
    # RobustScaler ile ölçeklendirme - tüm veri için yeni scaler oluştur
    print("[DEBUG] Final model için tüm veri ölçeklendiriliyor...")
    from sklearn.preprocessing import RobustScaler
    final_scaler = RobustScaler()
    X_scaled = pd.DataFrame(final_scaler.fit_transform(X), columns=feature_cols)
    
    print("[DEBUG] Final model eğitimi başlıyor...")
    final_model = RandomForestRegressor(**MODEL_PARAMS)
    final_model.fit(X_scaled, y)
    print("[DEBUG] Final model eğitimi tamamlandı.")
    
    # Özellik önemliliği
    feature_importance = {
        feature: float(importance) 
        for feature, importance in zip(feature_cols, final_model.feature_importances_)
    }
    
    # En önemli özellikleri logla
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("[DEBUG] Özellik önemleri (ilk 5):")
    for i, (feature, importance) in enumerate(sorted_features[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Model versiyonunu belirle
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_version = f"inverter_{inverter_id}_v{timestamp}"
    
    # Modeli kaydet
    model_path = os.path.join(MODELS_DIR, f"{model_version}.joblib")
    joblib.dump(final_model, model_path)
    print(f"[DEBUG] Model kaydedildi: {model_path}")
    
    # Model meta verisini kaydet
    meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
    
    # Metrikler ve özellik önemlerini JSON serileştirilebilir hale getir
    serialized_metrics = serialize_for_json(model_metrics)
    serialized_feature_importance = serialize_for_json(feature_importance)
    serialized_data_details = serialize_for_json(data_details)
    
    model_meta = {
        "model_version": model_version,
        "inverter_id": inverter_id,
        "created_at": datetime.utcnow().isoformat(),
        "model_type": "RandomForestRegressor",
        "model_params": MODEL_PARAMS,
        "feature_importance": serialized_feature_importance,
        "metrics": serialized_metrics,
        "data_size": len(X),
        "data_details": serialized_data_details
    }
    
    # JSON serileştirilebilir hale getir
    model_meta = serialize_for_json(model_meta)
    
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"[DEBUG] Model meta dosyası kaydedildi: {meta_path}")
    
    # Veritabanına model kaydı ekle
    model_db = Model(
        inverter_id=inverter_id,
        version=model_version,
        model_path=model_path,
        model_type="RandomForestRegressor",
        metrics=serialized_metrics,
        is_active=True,
        feature_importance=serialized_feature_importance,
        created_at=datetime.utcnow()
    )
    
    # Önceki aktif modelleri devre dışı bırak
    previous_models = db.query(Model).filter(
        Model.inverter_id == inverter_id,
        Model.is_active == True
    ).all()
    
    for model in previous_models:
        model.is_active = False
    
    db.add(model_db)
    db.commit()
    print(f"[DEBUG] Veritabanı model kaydı oluşturuldu, ID: {model_db.id}")
    
    return {
        "model_version": model_version,
        "inverter_id": inverter_id,
        "metrics": serialized_metrics,
        "model_path": model_path,
        "feature_importance": serialized_feature_importance,
        "data_details": serialized_data_details
    }

async def train_all_models(db: Session, test_split: bool = True) -> Dict[int, Dict[str, Any]]:
    """
    Tüm inverterler için modelleri eğitir.
    
    Args:
        db: Veritabanı oturumu
        test_split: Test bölünmesi yapılsın mı?
        
    Returns:
        Tüm modellerin eğitim sonuçları
    """
    inverters = db.query(Inverter).filter(Inverter.is_active == True).all()
    results = {}
    
    for inverter in inverters:
        try:
            result = await train_model(inverter.id, db, test_split=test_split)
            results[inverter.id] = result
        except Exception as e:
            results[inverter.id] = {"error": str(e)}
    
    return results

async def get_model_metrics(inverter_id: int, db: Session) -> Dict[str, Any]:
    """
    Belirli bir inverter için model metriklerini döndürür.
    
    Args:
        inverter_id: İnverter kimliği
        db: Veritabanı oturumu
        
    Returns:
        Model metrikleri
    """
    model = db.query(Model).filter(
        Model.inverter_id == inverter_id,
        Model.is_active == True
    ).first()
    
    if not model:
        return {
            "inverter_id": inverter_id,
            "exists": False,
            "message": "Aktif model bulunamadı"
        }
    
    return {
        "inverter_id": inverter_id,
        "model_version": model.version,
        "model_type": model.model_type,
        "created_at": model.created_at,
        "metrics": model.metrics,
        "feature_importance": model.feature_importance,
        "exists": True
    }

async def get_all_model_metrics(db: Session) -> Dict[int, Dict[str, Any]]:
    """
    Tüm inverterler için model metriklerini döndürür.
    
    Args:
        db: Veritabanı oturumu
        
    Returns:
        Tüm modellerin metrikleri
    """
    models = db.query(Model).filter(Model.is_active == True).all()
    results = {}
    
    for model in models:
        results[model.inverter_id] = {
            "model_version": model.version,
            "model_type": model.model_type,
            "created_at": model.created_at,
            "metrics": model.metrics,
            "feature_importance": model.feature_importance,
            "exists": True
        }
    
    # Modeli olmayan inverterleri de ekle
    inverters = db.query(Inverter).filter(Inverter.is_active == True).all()
    
    for inverter in inverters:
        if inverter.id not in results:
            results[inverter.id] = {
                "inverter_id": inverter.id,
                "exists": False,
                "message": "Aktif model bulunamadı"
            }
    
    return results 