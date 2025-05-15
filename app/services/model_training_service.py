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

from app.models.inverter import Inverter, InverterData
from app.models.weather import WeatherData
from app.models.model import Model

# Model eğitim ve tahmin için parametreler
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1
}

# Modellerin kaydedileceği klasör
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")

# Klasör yoksa oluştur
pathlib.Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

async def get_training_data(inverter_id: int, db: Session) -> pd.DataFrame:
    """
    Belirli bir inverter için eğitim verilerini hazırlar.
    Inverter verileri ile hava durumu verilerini birleştirir.
    
    Args:
        inverter_id: İnverter kimliği
        db: Veritabanı oturumu
        
    Returns:
        Eğitim veri çerçevesi
    """
    # İnverter verilerini çek
    inverter_data = db.query(InverterData).filter(
        InverterData.inverter_id == inverter_id
    ).order_by(InverterData.timestamp).all()
    
    if not inverter_data:
        raise ValueError(f"Inverter ID {inverter_id} için veri bulunamadı")
    
    # Inverter verilerini DataFrame'e dönüştür
    inverter_df = pd.DataFrame([
        {
            "timestamp": data.timestamp,
            "power_output": data.power_output,
            "inverter_id": data.inverter_id,
        } for data in inverter_data
    ])
    
    # Tüm zaman etiketlerini al
    timestamps = [data.timestamp for data in inverter_data]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    
    # Hava durumu verilerini çek
    weather_data = db.query(WeatherData).filter(
        WeatherData.timestamp >= min_timestamp,
        WeatherData.timestamp <= max_timestamp,
        WeatherData.is_forecast == 0  # Sadece gerçek ölçüm verileri
    ).order_by(WeatherData.timestamp).all()
    
    if not weather_data:
        raise ValueError(f"Belirtilen tarih aralığında hava durumu verisi bulunamadı")
    
    # Hava durumu verilerini DataFrame'e dönüştür
    weather_df = pd.DataFrame([
        {
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
        } for data in weather_data
    ])
    
    # Yakın zaman damgalarını eşleştirmek için yardımcı fonksiyon
    def find_closest_timestamp(target_timestamp, timestamps, max_diff=timedelta(hours=1)):
        closest = min(timestamps, key=lambda x: abs(x - target_timestamp))
        if abs(closest - target_timestamp) <= max_diff:
            return closest
        return None
    
    # Inverter verilerini en yakın hava durumu verileriyle birleştir
    weather_timestamps = weather_df["timestamp"].tolist()
    matched_data = []
    
    for idx, row in inverter_df.iterrows():
        closest_ts = find_closest_timestamp(row["timestamp"], weather_timestamps)
        if closest_ts:
            weather_row = weather_df[weather_df["timestamp"] == closest_ts].iloc[0]
            matched_row = {
                "timestamp": row["timestamp"],
                "power_output": row["power_output"],
                "inverter_id": row["inverter_id"],
                "temperature": weather_row["temperature"],
                "shortwave_radiation": weather_row["shortwave_radiation"],
                "direct_radiation": weather_row["direct_radiation"],
                "diffuse_radiation": weather_row["diffuse_radiation"],
                "direct_normal_irradiance": weather_row["direct_normal_irradiance"],
                "global_tilted_irradiance": weather_row["global_tilted_irradiance"],
                "terrestrial_radiation": weather_row["terrestrial_radiation"],
                "relative_humidity": weather_row["relative_humidity"],
                "wind_speed": weather_row["wind_speed"],
                "visibility": weather_row["visibility"]
            }
            matched_data.append(matched_row)
    
    if not matched_data:
        raise ValueError("İnverter ve hava durumu verilerini eşleştirilemedi")
    
    # Eğitim veri çerçevesini oluştur
    df = pd.DataFrame(matched_data)
    
    # Zaman özelliklerini ekle
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    # Eksik değerleri doldur
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
    
    return df

async def train_model(
    inverter_id: int,
    db: Session,
    test_split: bool = True,
    test_size: float = 0.3
) -> Dict[str, Any]:
    """
    Belirli bir inverter için model eğitimi yapar.
    
    Args:
        inverter_id: İnverter kimliği
        db: Veritabanı oturumu
        test_split: Test bölünmesi yapılsın mı?
        test_size: Test seti oranı
        
    Returns:
        Model eğitim sonuçları ve metrikleri
    """
    # Eğitim verilerini al
    df = await get_training_data(inverter_id, db)
    
    # Özellik ve hedef değişkenleri ayır
    feature_cols = [
        'temperature', 'shortwave_radiation', 'direct_radiation',
        'diffuse_radiation', 'direct_normal_irradiance', 'global_tilted_irradiance', 
        'terrestrial_radiation', 'relative_humidity', 'wind_speed', 'visibility',
        'hour', 'day', 'month', 'dayofweek'
    ]
    
    X = df[feature_cols]
    y = df["power_output"]
    
    model_metrics = {}
    
    # İki aşamalı eğitim
    # 1. Aşama: Test bölünmesi ile metrik hesaplama
    if test_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        
        # Test seti üzerinde tahmin yap
        y_pred = model.predict(X_test)
        
        # Model metriklerini hesapla
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "test_size": test_size,
            "samples_count": len(X),
            "features": feature_cols
        }
    
    # 2. Aşama: Tüm verilerle final model eğitimi
    final_model = RandomForestRegressor(**MODEL_PARAMS)
    final_model.fit(X, y)
    
    # Özellik önemliliği
    feature_importance = {
        feature: float(importance) 
        for feature, importance in zip(feature_cols, final_model.feature_importances_)
    }
    
    # Model versiyonunu belirle
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_version = f"inverter_{inverter_id}_v{timestamp}"
    
    # Modeli kaydet
    model_path = os.path.join(MODELS_DIR, f"{model_version}.joblib")
    joblib.dump(final_model, model_path)
    
    # Model meta verisini kaydet
    meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
    
    model_meta = {
        "model_version": model_version,
        "inverter_id": inverter_id,
        "created_at": datetime.utcnow().isoformat(),
        "model_type": "RandomForestRegressor",
        "model_params": MODEL_PARAMS,
        "feature_importance": feature_importance,
        "metrics": model_metrics,
        "data_size": len(X)
    }
    
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    
    # Veritabanına model kaydı ekle
    model_db = Model(
        inverter_id=inverter_id,
        version=model_version,
        model_path=model_path,
        model_type="RandomForestRegressor",
        metrics=model_metrics,
        is_active=True,
        feature_importance=feature_importance,
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
    
    return {
        "model_version": model_version,
        "inverter_id": inverter_id,
        "metrics": model_metrics,
        "model_path": model_path,
        "feature_importance": feature_importance
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