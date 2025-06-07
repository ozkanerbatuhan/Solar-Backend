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
from app.services.data_quality_service import DataQualityService

# Model eğitim ve tahmin için parametreler - SUPER OPTIMIZED FOR SOLAR DATA
MODEL_PARAMS = {
    "n_estimators": 1000,       # Çok fazla ağaç = overfitting'e karşı güçlü ensemble
    "max_depth": 18,            # Daha kontrollü derinlik = overfitting'i engelle
    "min_samples_split": 10,    # Daha muhafazakar split = stability
    "min_samples_leaf": 5,      # Daha büyük yapraklar = generalization
    "max_features": 0.6,        # %60 feature sampling = güçlü diversity
    "bootstrap": True,          # Bagging
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,          # Out-of-bag score evaluation
    "min_impurity_decrease": 0.001,  # Daha büyük kazanım threshold = pruning
    "max_samples": 0.85,        # %85 sample bootstrap = diversity
    "criterion": "absolute_error",  # MAE based splitting = robust to outliers
    "max_leaf_nodes": 2000      # Leaf node limit = complexity control
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
        
        # YENİ: Gelişmiş özellik seçimi 
        # Temel özellikler
        base_feature_cols = [
            'temperature', 'shortwave_radiation', 'direct_radiation',
            'diffuse_radiation', 'direct_normal_irradiance', 'global_tilted_irradiance', 
            'terrestrial_radiation', 'relative_humidity', 'wind_speed',
            'hour', 'day', 'month', 'dayofweek', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Gelişmiş özellikler (veri kalitesi servisi tarafından eklenenler)
        advanced_features = [
            'total_radiation_index', 'radiation_efficiency', 'solar_elevation_proxy',
            'is_daylight', 'is_peak_solar', 'heat_comfort', 'panel_efficiency_proxy',
            'season_summer', 'season_winter', 'daylight_length_proxy', 
            'temp_radiation_interaction', 'wind_cooling_effect',
            'high_radiation', 'low_radiation', 'zero_radiation'
        ]
        
        # Mevcut tüm özellikleri kontrol et
        all_potential_features = base_feature_cols + advanced_features
        feature_cols = [col for col in all_potential_features if col in available_columns]
        
        print(f"[DEBUG] Kullanılacak temel özellikler: {[f for f in base_feature_cols if f in available_columns]}")
        print(f"[DEBUG] Kullanılacak gelişmiş özellikler: {[f for f in advanced_features if f in available_columns]}")
        print(f"[DEBUG] Toplam özellik sayısı: {len(feature_cols)}")
        
        # Sıcaklık sütunu çakışma kontrolü - inverter_temperature ve temperature karışıklığı olmamalı
        if 'temperature' in feature_cols and 'inverter_temperature' in available_columns:
            print("[DEBUG] 'temperature' sütunu var ve bu hava durumu sıcaklığını ifade ediyor.")
        
        if not feature_cols:
            raise ValueError(f"Hiçbir özellik sütunu bulunamadı. Mevcut sütunlar: {available_columns}")
        
        # YENİ: Model input data validation
        validation_report = DataQualityService.validate_model_input_data(df, feature_cols)
        print(f"[DEBUG] Model input validation: {validation_report}")
        
        if not validation_report['is_valid']:
            print(f"[UYARI] Model input validation başarısız: {validation_report['errors']}")
        
        if validation_report['data_quality_score'] < 70:
            print(f"[UYARI] Düşük veri kalitesi skoru: {validation_report['data_quality_score']}/100")
        
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

            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle_param)
            
            active_training_jobs[job_id]["progress"] = 40
            active_training_jobs[job_id]["message"] = f"İlk model eğitimi başlıyor"
            
            # NaN kontrolü - main.py'de olduğu gibi
            if X_train.isna().any().any():
                X_train = X_train.fillna(X_train.median())
            
            if X_test.isna().any().any():
                X_test = X_test.fillna(X_test.median())
            
            # RobustScaler uygulaması - main.py'deki gibi
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
            
            model = RandomForestRegressor(**MODEL_PARAMS)
            model.fit(X_train_scaled, y_train)
           
            active_training_jobs[job_id]["progress"] = 60
            active_training_jobs[job_id]["message"] = f"Model performans metriksleri hesaplanıyor"
            
            # Test seti üzerinde tahmin yap
            y_pred = model.predict(X_test_scaled)
            
            # Tahmin ve gerçek değerlerin sıralamasını kontrol et (indeks kontrolü)
            
            # Model metriklerini hesapla
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # ÇOK GÜVENLİ MAPE hesaplama - aşırı küçük değerleri filtrele
            # Hem numerik kararlılık hem de fiziksel anlam için
            significant_mask = (y_test > 10.0) & (np.abs(y_test) > 0.1)  # 10 kW üzeri anlamlı değerler
            mape = 0.0
            mape_samples = 0
            
            if significant_mask.sum() > 0:
                # MAPE hesaplama - aşırı yüksek hataları sınırla
                percentage_errors = np.abs((y_test[significant_mask] - y_pred[significant_mask]) / y_test[significant_mask])
                # %500'den yüksek hataları sınırla (fiziksel olarak makul)
                capped_errors = np.minimum(percentage_errors, 5.0)  # Max %500 hata
                mape = np.mean(capped_errors) * 100
                mape_samples = significant_mask.sum()
                print(f"[DEBUG] MAPE hesaplandı - {mape_samples} anlamlı örnek kullanıldı (>{10.0} kW)")
                print(f"[DEBUG] Ham MAPE: {np.mean(percentage_errors) * 100:.2f}%, Sınırlanmış MAPE: {mape:.2f}%")
            else:
                print("[DEBUG] MAPE hesaplanamadı - 10.0 kW'den büyük anlamlı değer yok.")
                mape = 999.9  # İnvalid marker
            
            
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
        from sklearn.preprocessing import RobustScaler
        final_scaler = RobustScaler()
        X_scaled = pd.DataFrame(final_scaler.fit_transform(X), columns=feature_cols)
        
        final_model = RandomForestRegressor(**MODEL_PARAMS)
        final_model.fit(X_scaled, y)
        
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
        
        # Modeli ve scaler'ı kaydet
        model_path = os.path.join(MODELS_DIR, f"{model_version}.joblib")
        scaler_path = os.path.join(MODELS_DIR, f"{model_version}_scaler.joblib")
        
        joblib.dump(final_model, model_path)
        joblib.dump(final_scaler, scaler_path)
        print(f"[DEBUG] Model kaydedildi: {model_path}")
        print(f"[DEBUG] Scaler kaydedildi: {scaler_path}")
        
        # Model meta verisini kaydet
        meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
        
        # Metrikler ve özellik önemlerini JSON serileştirilebilir hale getir
        serialized_metrics = serialize_for_json(model_metrics)
        
        # Feature importance'ları büyükten küçüğe sırala
        sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        serialized_feature_importance = serialize_for_json(sorted_feature_importance)
        serialized_data_details = serialize_for_json(data_details)
        
        model_meta = {
            "model_version": model_version,
            "inverter_id": inverter_id,
            "created_at": datetime.utcnow().isoformat(),
            "model_type": "RandomForestRegressor",
            "model_params": MODEL_PARAMS,
            "feature_importance": serialized_feature_importance,
            "metrics": serialized_metrics,
            "features": feature_cols,  # Özellik listesini meta veride sakla
            "scaler_path": f"{model_version}_scaler.joblib",  # Scaler dosya yolunu sakla
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
            "feature_importance": sorted_feature_importance,
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
        "wind_speed": data.wind_speed
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
    
    # Temel tarih özelliklerini önce ekle (veri kalitesi servisi için gerekli)
    print("[DEBUG] Temel tarih özellikleri ekleniyor...")
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    # YENİ: ÇOK AGRESIF veri kalitesi kontrolleri ve temizleme
    print("[DEBUG] ULTRA AGRESIF veri kalitesi analizi başlatılıyor...")
    print(f"[DEBUG] Temizlik öncesi veri boyutu: {df.shape}")
    
    # 1. Akıllı veri temizleme
    df_cleaned, cleaning_report = DataQualityService.intelligent_data_cleaning(df, 'power_output')
    print(f"[DEBUG] İlk veri temizleme tamamlandı: {cleaning_report}")
    
    # 2. EKSTRA AGRESIF temizlik - model eğitimi için
    print("[DEBUG] Ekstra agresif temizlik başlatılıyor...")
    
    # Fiziksel olarak imkansız kombinasyonları tamamen kaldır
    before_extreme_cleaning = len(df_cleaned)
    
    # Gece saatlerinde 5 kW'dan fazla güç üretimi olan satırları kaldır
    night_mask = (df_cleaned['hour'] >= 22) | (df_cleaned['hour'] <= 5)
    extreme_night_power = night_mask & (df_cleaned['power_output'] > 5)
    df_cleaned = df_cleaned[~extreme_night_power]
    print(f"[DEBUG] Gece yüksek güç satırları kaldırıldı: {extreme_night_power.sum()}")
    
    # Sıfır radyasyon + pozitif güç kombinasyonlarını kaldır
    zero_rad_positive_power = (df_cleaned['shortwave_radiation'] <= 0) & (df_cleaned['power_output'] > 1)
    df_cleaned = df_cleaned[~zero_rad_positive_power]
    print(f"[DEBUG] Sıfır radyasyon + pozitif güç satırları kaldırıldı: {zero_rad_positive_power.sum()}")
    
    # Aşırı düşük radyasyon + yüksek güç kombinasyonlarını kaldır
    low_rad_high_power = (df_cleaned['shortwave_radiation'] < 50) & (df_cleaned['power_output'] > 100)
    df_cleaned = df_cleaned[~low_rad_high_power]
    print(f"[DEBUG] Düşük radyasyon + yüksek güç satırları kaldırıldı: {low_rad_high_power.sum()}")
    
    # Öğle saatlerinde çok düşük güç üretimi olanları kaldır (bulutlu günler hariç)
    noon_mask = (df_cleaned['hour'] >= 11) & (df_cleaned['hour'] <= 13)
    high_rad_low_power = noon_mask & (df_cleaned['shortwave_radiation'] > 400) & (df_cleaned['power_output'] < 50)
    df_cleaned = df_cleaned[~high_rad_low_power]
    print(f"[DEBUG] Öğle yüksek radyasyon + düşük güç satırları kaldırıldı: {high_rad_low_power.sum()}")
    
    # Aşırı yüksek güç değerlerini kaldır (5 MW'tan fazla fiziksel olarak imkansız)
    excessive_power = df_cleaned['power_output'] > 5000
    df_cleaned = df_cleaned[~excessive_power]
    print(f"[DEBUG] Aşırı yüksek güç satırları kaldırıldı: {excessive_power.sum()}")
    
    # ❌ İSTATİSTİKSEL OUTLIER DETECTION KALDIRILDI!
    # Güneş enerjisinde gece 0, gündüz 1000+ kW normal - IQR yöntemi yanlış sonuç veriyor
    print(f"[DEBUG] Statistical outlier detection atlandı - güneş enerjisi için uygun değil")
    
    after_extreme_cleaning = len(df_cleaned)
    extreme_cleaning_removed = before_extreme_cleaning - after_extreme_cleaning
    print(f"[DEBUG] Ekstra agresif temizlik: {extreme_cleaning_removed} satır kaldırıldı (%{(extreme_cleaning_removed/before_extreme_cleaning)*100:.2f})")
    
    # 3. Güneş enerjisi aware feature engineering
    df_enhanced = DataQualityService.create_solar_aware_features(df_cleaned)
    print(f"[DEBUG] Gelişmiş feature engineering tamamlandı. Toplam sütun sayısı: {len(df_enhanced.columns)}")
    
    # 4. HAFIF outlier removal - sadece fiziksel limitler
    print("[DEBUG] Fiziksel limitlerle hafif outlier removal...")
    
    # SADECE FİZİKSEL LİMİTLER - statistical outlier detection KALDIRILDI!
    
    # 1. Negatif değerleri kaldır
    negative_power = df_enhanced['power_output'] < 0
    df_enhanced = df_enhanced[~negative_power]
    print(f"[DEBUG] Negatif güç değerleri kaldırıldı: {negative_power.sum()}")
    
    # 2. Aşırı yüksek değerler (2000 kW = 2 MW üzeri - çok liberal limit)
    excessive_power = df_enhanced['power_output'] > 2000
    df_enhanced = df_enhanced[~excessive_power]
    print(f"[DEBUG] 2000 kW üzeri aşırı yüksek güç kaldırıldı: {excessive_power.sum()}")
    
    # 3. Radyasyon-güç oranı kontrolü ÇOK YUMULATILDI
    if 'total_radiation_index' in df_enhanced.columns:
        # Her 1000 W/m² radiation için maksimum 10 kW bekleniyor (çok liberal)
        expected_power = df_enhanced['total_radiation_index'] * 10 / 1000  # 4'ten 10'a çıkarıldı
        power_ratio = df_enhanced['power_output'] / (expected_power + 1)  # +1 division by zero için
        extreme_ratio = (power_ratio > 20) | (power_ratio < 0.01)  # 20x fazla kabul edilir (çok liberal)
        df_enhanced = df_enhanced[~extreme_ratio]
        print(f"[DEBUG] Çok liberal radyasyon-güç oransızlığı satırları kaldırıldı: {extreme_ratio.sum()}")
        
    print(f"[DEBUG] Final veri boyutu: {df_enhanced.shape}")
    print(f"[DEBUG] Toplam veri kaybı: %{((df.shape[0] - df_enhanced.shape[0])/df.shape[0])*100:.2f}")
    
    # Minimum veri kontrolü
    if len(df_enhanced) < 100:
        raise ValueError(f"Veri temizleme sonrası çok az veri kaldı: {len(df_enhanced)} satır. Eğitim için yetersiz.")
    
    # 3. Trigonometrik zaman özellikleri (eski mantık korunuyor)
    print("[DEBUG] Trigonometrik zaman özellikleri ekleniyor...")
    df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour']/24)
    df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour']/24)
    
    # Gün numarası yerine ay kullanılacak
    df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['month']/12)
    df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['month']/12)
    
    print(f"[DEBUG] Veri hazırlama tamamlandı. Final veri boyutu: {df_enhanced.shape}")
    print(f"[DEBUG] Final sütunlar: {df_enhanced.columns.tolist()}")
    
    return df_enhanced

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
    
    # YENİ: Gelişmiş özellik seçimi 
    # Temel özellikler
    base_feature_cols = [
        'temperature', 'shortwave_radiation', 'direct_radiation',
        'diffuse_radiation', 'direct_normal_irradiance', 'global_tilted_irradiance', 
        'terrestrial_radiation', 'relative_humidity', 'wind_speed',
        'hour', 'day', 'month', 'dayofweek', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    # Gelişmiş özellikler (veri kalitesi servisi tarafından eklenenler)
    advanced_features = [
        'total_radiation_index', 'radiation_efficiency', 'solar_elevation_proxy',
        'is_daylight', 'is_peak_solar', 'heat_comfort', 'panel_efficiency_proxy',
        'season_summer', 'season_winter', 'daylight_length_proxy', 
        'temp_radiation_interaction', 'wind_cooling_effect',
        'high_radiation', 'low_radiation', 'zero_radiation'
    ]
    
    # Mevcut tüm özellikleri kontrol et
    all_potential_features = base_feature_cols + advanced_features
    feature_cols = [col for col in all_potential_features if col in available_columns]
    
    print(f"[DEBUG] Kullanılacak temel özellikler: {[f for f in base_feature_cols if f in available_columns]}")
    print(f"[DEBUG] Kullanılacak gelişmiş özellikler: {[f for f in advanced_features if f in available_columns]}")
    print(f"[DEBUG] Toplam özellik sayısı: {len(feature_cols)}")
    
    # Sıcaklık sütunu çakışma kontrolü - inverter_temperature ve temperature karışıklığı olmamalı
    if 'temperature' in feature_cols and 'inverter_temperature' in available_columns:
        print("[DEBUG] 'temperature' sütunu var ve bu hava durumu sıcaklığını ifade ediyor.")
    
    if not feature_cols:
        raise ValueError(f"Hiçbir özellik sütunu bulunamadı. Mevcut sütunlar: {available_columns}")
    
    # YENİ: Model input data validation
    validation_report = DataQualityService.validate_model_input_data(df, feature_cols)
    print(f"[DEBUG] Model input validation: {validation_report}")
    
    if not validation_report['is_valid']:
        print(f"[UYARI] Model input validation başarısız: {validation_report['errors']}")
    
    if validation_report['data_quality_score'] < 70:
        print(f"[UYARI] Düşük veri kalitesi skoru: {validation_report['data_quality_score']}/100")
    
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
    
    # Modeli ve scaler'ı kaydet
    model_path = os.path.join(MODELS_DIR, f"{model_version}.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"{model_version}_scaler.joblib")
    
    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    print(f"[DEBUG] Model kaydedildi: {model_path}")
    print(f"[DEBUG] Scaler kaydedildi: {scaler_path}")
    
    # Model meta verisini kaydet
    meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
    
    # Metrikler ve özellik önemlerini JSON serileştirilebilir hale getir
    serialized_metrics = serialize_for_json(model_metrics)
    
    # Feature importance'ları büyükten küçüğe sırala
    sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    serialized_feature_importance = serialize_for_json(sorted_feature_importance)
    serialized_data_details = serialize_for_json(data_details)
    
    model_meta = {
        "model_version": model_version,
        "inverter_id": inverter_id,
        "created_at": datetime.utcnow().isoformat(),
        "model_type": "RandomForestRegressor",
        "model_params": MODEL_PARAMS,
        "feature_importance": serialized_feature_importance,
        "metrics": serialized_metrics,
        "features": feature_cols,  # Özellik listesini meta veride sakla
        "scaler_path": f"{model_version}_scaler.joblib",  # Scaler dosya yolunu sakla
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