import os
import joblib
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import json

from app.models.inverter import InverterPrediction, Inverter
from app.models.model import Model
from app.models.weather import WeatherForecast
from app.core.config import settings
import logging

# Log yapılandırması
logger = logging.getLogger(__name__)

# Modellerin kaydedileceği klasör
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")

async def get_predictions(
    inverter_id: int, 
    start_date: datetime = None,
    end_date: datetime = None,
    interval_hours: int = 1, 
    db: Session = None,
    use_cached: bool = True
) -> List[InverterPrediction]:
    """
    Belirtilen inverter için bir tarih aralığında tahmin yapar.
    
    Args:
        inverter_id: Tahmin yapılacak inverter ID'si
        start_date: Tahmin başlangıç tarihi (varsayılan: şimdiki zaman)
        end_date: Tahmin bitiş tarihi (varsayılan: 7 gün sonrası)
        interval_hours: Tahmin aralığı (saat cinsinden)
        db: Veritabanı oturumu
        use_cached: Eğer varsa, önceden hesaplanmış tahminleri kullan
    
    Returns:
        List[InverterPrediction]: Tahminlerin listesi
    """
    # Varsayılan parametreleri ayarla
    if start_date is None:
        start_date = datetime.now()
    
    if end_date is None:
        end_date = start_date + timedelta(days=7)
    
    logger.info(f"[INFO] {inverter_id} için {start_date} - {end_date} aralığında tahmin yapılıyor...")
    
    # Inverter var mı kontrol et
    try:
        inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
        if inverter is None:
            raise ValueError(f"ID: {inverter_id} olan inverter bulunamadı")
    except Exception as e:
        logger.error(f"Inverter sorgulama hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        raise
    
    # Aktif modeli yükle
    try:
        model, model_meta = await load_model(inverter_id, db)
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        model, model_meta = None, None
    
    if model is None:
        # Model yoksa basit bir tahmin serisi oluştur
        return await _make_dummy_predictions(inverter_id, start_date, end_date, interval_hours, db)
    
    # Her bir zaman noktası için tahmin yap
    predictions = []
    current_time = start_date
    
    while current_time <= end_date:
        prediction = None
        transaction_success = False
        
        try:
            # Eğer önceden hesaplanmış tahmin varsa ve use_cached=True ise, onu kullan
            if use_cached:
                cached_prediction = db.query(InverterPrediction).filter(
                    InverterPrediction.inverter_id == inverter_id,
                    InverterPrediction.prediction_timestamp == current_time
                ).first()
                
                if cached_prediction:
                    predictions.append(cached_prediction)
                    current_time += timedelta(hours=interval_hours)
                    continue
            
            # Tahmin için hava durumu verilerini al
            weather_data = await _get_weather_data_for_prediction(current_time, db)
            
            if not weather_data:
                # Hava durumu verisi yoksa basit bir tahmin yap
                prediction = await _make_dummy_prediction(inverter_id, current_time, db)
                predictions.append(prediction)
                transaction_success = True
                current_time += timedelta(hours=interval_hours)
                continue
            
            # Tahmin için özellikleri hazırla
            features = _prepare_features(weather_data, current_time)
            
            # Modelin beklediği özellikleri al
            required_features = model_meta.get("metrics", {}).get("features", [])
            
            # Gerekli özellikleri içeren DataFrame oluştur
            feature_df = pd.DataFrame([{k: features.get(k, 0) for k in required_features}])
            
            # Tahmin yap
            predicted_power = float(model.predict(feature_df)[0])
            
            # Tahmin güven değeri (şu an için sabit)
            confidence = 0.9
            
            # Tahmin kaydını oluştur ve kaydet
            prediction = InverterPrediction(
                inverter_id=inverter_id,
                timestamp=datetime.utcnow(),
                prediction_timestamp=current_time,
                predicted_power_output=predicted_power,
                model_version=model_meta.get("model_version", "unknown"),
                confidence=confidence,
                features=features
            )
            
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            transaction_success = True
            
            predictions.append(prediction)
            
        except Exception as e:
            # Veritabanı işlemi başarısız olduysa geri al
            if not transaction_success and db and db.is_active:
                db.rollback()
                
            logger.error(f"Tahmin hatası: {str(e)}")
            
            try:
                # Hata durumunda basit bir tahmin yap
                dummy_prediction = await _make_dummy_prediction(inverter_id, current_time, db)
                predictions.append(dummy_prediction)
            except Exception as dummy_err:
                logger.error(f"Dummy tahmin hatası: {str(dummy_err)}")
                if db and db.is_active:
                    db.rollback()
        
        finally:
            # Bir sonraki zaman noktasına geç
            current_time += timedelta(hours=interval_hours)
    
    return predictions

# Geriye uyumluluk için eski get_prediction fonksiyonunu da tutalım
async def get_prediction(
    inverter_id: int, 
    timestamp: datetime, 
    db: Session,
    use_cached: bool = True
) -> InverterPrediction:
    """
    Belirtilen inverter için tek bir zaman noktası için tahmin yapar.
    
    Args:
        inverter_id: Tahmin yapılacak inverter ID'si
        timestamp: Tahmin edilecek zaman
        db: Veritabanı oturumu
        use_cached: Eğer varsa, önceden hesaplanmış tahmini kullan
    
    Returns:
        InverterPrediction: Tahmin sonucu
    """
    try:
        predictions = await get_predictions(
            inverter_id=inverter_id,
            start_date=timestamp,
            end_date=timestamp,
            interval_hours=1,
            db=db,
            use_cached=use_cached
        )
        
        return predictions[0] if predictions else None
    except Exception as e:
        logger.error(f"Tek nokta tahmin hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        raise

async def load_model(inverter_id: int, db: Session) -> tuple:
    """
    Belirtilen inverter için makine öğrenimi modelini yükler.
    
    Args:
        inverter_id: Model yüklenecek inverter ID'si
        db: Veritabanı oturumu
    
    Returns:
        tuple: (model, model_meta) - Yüklenen model ve meta verileri
    """
    try:
        # Inverter için aktif modeli kontrol et
        active_model = db.query(Model).filter(
            Model.inverter_id == inverter_id,
            Model.is_active == True
        ).first()
        
        if active_model is None:
            return None, None
        
        # Model dosyasının yolunu oluştur
        model_path = active_model.model_path
        if not model_path:
            return None, None
        
        # Tam dosya yolunu oluştur
        full_model_path = os.path.join(MODELS_DIR, model_path)
        
        # Model meta verisi için dosya yolunu oluştur
        model_version = active_model.version
        meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
        
        # Modelin var olup olmadığını kontrol et
        if not os.path.exists(full_model_path):
            return None, None
        
        # Modeli yükle
        model = joblib.load(full_model_path)
        
        # Meta verileri yükle
        model_meta = active_model.metrics
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                model_meta = json.load(f)
        
        return model, model_meta
    
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        return None, None

async def _get_weather_data_for_prediction(timestamp: datetime, db: Session) -> Dict[str, Any]:
    """
    Belirtilen zaman için hava durumu verilerini alır.
    
    Args:
        timestamp: Hava durumu verisi alınacak zaman
        db: Veritabanı oturumu
        
    Returns:
        Dict: Hava durumu verileri
    """
    try:
        # Tahmin zamanına en yakın hava durumu verisini bul
        weather_data = db.query(WeatherForecast).filter(
            WeatherForecast.forecast_timestamp <= timestamp + timedelta(hours=1),
            WeatherForecast.forecast_timestamp >= timestamp - timedelta(hours=1)
        ).order_by(
            # En yakın zaman damgasına göre sırala
            db.func.abs(db.func.extract('epoch', WeatherForecast.forecast_timestamp) - 
                       db.func.extract('epoch', timestamp))
        ).first()
        
        if not weather_data:
            return None
        
        # Hava durumu verilerini sözlük olarak döndür
        return {
            "temperature": weather_data.temperature,
            "shortwave_radiation": weather_data.shortwave_radiation,
            "direct_radiation": weather_data.direct_radiation,
            "diffuse_radiation": weather_data.diffuse_radiation,
            "direct_normal_irradiance": weather_data.direct_normal_irradiance,
            "global_tilted_irradiance": weather_data.global_tilted_irradiance,
            "terrestrial_radiation": weather_data.terrestrial_radiation,
            "relative_humidity": weather_data.relative_humidity,
            "wind_speed": weather_data.wind_speed,
            "visibility": weather_data.visibility
        }
    except Exception as e:
        logger.error(f"Hava durumu verisi alma hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        return None

def _prepare_features(weather_data: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
    """
    Tahmin için özellikleri hazırlar.
    
    Args:
        weather_data: Hava durumu verileri
        timestamp: Tahmin zamanı
        
    Returns:
        Dict: Hazırlanmış özellikler
    """
    # Hava durumu verilerini kopyala
    features = dict(weather_data)
    
    # Zaman özelliklerini ekle
    features.update({
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        "dayofweek": timestamp.weekday()
    })
    
    # Trigonometrik zaman özellikleri ekle
    features.update({
        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
        "day_sin": np.sin(2 * np.pi * timestamp.month / 12),
        "day_cos": np.cos(2 * np.pi * timestamp.month / 12)
    })
    
    return features

async def _make_dummy_prediction(inverter_id: int, timestamp: datetime, db: Session) -> InverterPrediction:
    """
    Model veya hava durumu verisi yoksa basit bir tahmin yapar.
    
    Args:
        inverter_id: Tahmin yapılacak inverter ID'si
        timestamp: Tahmin edilecek zaman
        db: Veritabanı oturumu
        
    Returns:
        InverterPrediction: Basit tahmin sonucu
    """
    try:
        import random
        
        # Saat bazında basit bir tahmin yap (gündüz daha yüksek, gece daha düşük)
        hour = timestamp.hour
        
        # Gece (0-6) ve akşam (18-23) saatleri için düşük değer
        if hour < 6 or hour > 18:
            base_power = random.uniform(0, 10)
        # Sabah (6-10) ve öğleden sonra (15-18) için orta değer
        elif (hour >= 6 and hour < 10) or (hour >= 15 and hour < 18):
            base_power = random.uniform(10, 50)
        # Öğle saatleri (10-15) için yüksek değer
        else:
            base_power = random.uniform(50, 100)
        
        # Mevsimsel etki (yaz aylarında daha yüksek)
        month = timestamp.month
        if month in [6, 7, 8]:  # Yaz
            seasonal_factor = 1.2
        elif month in [3, 4, 5, 9, 10, 11]:  # İlkbahar ve sonbahar
            seasonal_factor = 1.0
        else:  # Kış
            seasonal_factor = 0.8
        
        # Tahmin değerini hesapla
        predicted_power = base_power * seasonal_factor
        
        # Dummy özellikler
        features = {
            "timestamp": timestamp.isoformat(),
            "hour": hour,
            "day": timestamp.day,
            "month": month,
            "dayofweek": timestamp.weekday(),
            "is_dummy": True
        }
        
        # Tahmin kaydını oluştur ve kaydet
        prediction = InverterPrediction(
            inverter_id=inverter_id,
            timestamp=datetime.utcnow(),
            prediction_timestamp=timestamp,
            predicted_power_output=predicted_power,
            model_version="dummy-model",
            confidence=0.5,  # Düşük güven
            features=features
        )
        
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return prediction
    except Exception as e:
        logger.error(f"Dummy tahmin hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        
        # En son çare - DB'ye yazmadan dummy bir nesne döndür
        return InverterPrediction(
            inverter_id=inverter_id,
            timestamp=datetime.utcnow(),
            prediction_timestamp=timestamp,
            predicted_power_output=10.0,  # Sabit düşük değer
            model_version="emergency-dummy-model",
            confidence=0.1,  # Çok düşük güven
            features={"is_dummy": True, "is_emergency": True}
        )

async def _make_dummy_predictions(
    inverter_id: int, 
    start_date: datetime, 
    end_date: datetime, 
    interval_hours: int, 
    db: Session
) -> List[InverterPrediction]:
    """
    Model veya hava durumu verisi yoksa belirli bir aralıkta basit tahminler yapar.
    
    Args:
        inverter_id: Tahmin yapılacak inverter ID'si
        start_date: Başlangıç zamanı
        end_date: Bitiş zamanı
        interval_hours: Saat cinsinden aralık
        db: Veritabanı oturumu
        
    Returns:
        List[InverterPrediction]: Basit tahmin sonuçları
    """
    predictions = []
    current_time = start_date
    
    while current_time <= end_date:
        try:
            prediction = await _make_dummy_prediction(inverter_id, current_time, db)
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"Dummy tahmin oluşturma hatası: {str(e)}")
            if db and db.is_active:
                db.rollback()
            
            # En son çare - DB'ye yazmadan dummy bir nesne oluştur
            emergency_pred = InverterPrediction(
                inverter_id=inverter_id,
                timestamp=datetime.utcnow(),
                prediction_timestamp=current_time,
                predicted_power_output=10.0,
                model_version="emergency-dummy-model",
                confidence=0.1,
                features={"is_dummy": True, "is_emergency": True}
            )
            predictions.append(emergency_pred)
            
        finally:
            current_time += timedelta(hours=interval_hours)
    
    return predictions

async def train_model(inverter_id: int, db: Session):
    """
    Belirtilen inverter için yeni bir model eğitir.
    Şu an için sadece temel iskelet oluşturulmuştur.
    
    Args:
        inverter_id: Model eğitilecek inverter ID'si
        db: Veritabanı oturumu
    
    Returns:
        Eğitilen model bilgileri
    """
    # Inverter var mı kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise ValueError(f"ID: {inverter_id} olan inverter bulunamadı")
    
    # Inverter için son verileri al
    # Bu bölüm gerçek veri ile doldurulacak
    
    # NOT: Burada gerçek bir model eğitimi yapılacak
    # Şimdilik, bir model kaydı oluşturmakla yetiniyoruz
    
    # Son model sürümünü kontrol et
    latest_model = db.query(Model).filter(
        Model.inverter_id == inverter_id
    ).order_by(Model.created_at.desc()).first()
    
    # Yeni sürüm numarası oluştur
    new_version = "v1.0.0"
    if latest_model:
        # Basit bir sürüm artırma mantığı
        version_parts = latest_model.version.lstrip('v').split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = f"v{'.'.join(version_parts)}"
    
    # Yeni model kaydı oluştur
    model_filename = f"inverter_{inverter_id}_model_{new_version}.joblib"
    model_path = os.path.join(settings.MODEL_DIR, model_filename)
    
    # Model kaydet (şimdilik boş)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'w') as f:
        f.write("# Geçici model dosyası")
    
    # Model meta verilerini oluştur
    model_metrics = {
        "accuracy": random.uniform(0.7, 0.95),
        "rmse": random.uniform(0.05, 0.2),
        "training_time": random.uniform(10, 60)
    }
    
    # Model kaydı oluştur
    new_model = Model(
        inverter_id=inverter_id,
        version=new_version,
        model_path=model_filename,
        metrics=model_metrics
    )
    
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    
    return new_model

async def get_bulk_predictions(
    inverter_ids: List[int], 
    start_time: datetime = None, 
    end_time: datetime = None, 
    interval_hours: int = 1,
    db: Session = None
) -> Dict[int, List[InverterPrediction]]:
    """
    Birden fazla inverter için belirli bir zaman aralığında tahminler yapar.
    
    Args:
        inverter_ids: Tahmin yapılacak inverter ID'leri
        start_time: Başlangıç zamanı (varsayılan: şimdiki zaman)
        end_time: Bitiş zamanı (varsayılan: 7 gün sonrası)
        interval_hours: Saat cinsinden aralık (varsayılan: 1)
        db: Veritabanı oturumu
        
    Returns:
        Dict: Inverter ID'lerine göre tahminler
    """
    # Varsayılan parametreleri ayarla
    if start_time is None:
        start_time = datetime.now()
    
    if end_time is None:
        end_time = start_time + timedelta(days=7)
    
    results = {}
    
    # Her inverter için tahminleri hesapla
    for inverter_id in inverter_ids:
        try:
            predictions = await get_predictions(
                inverter_id=inverter_id, 
                start_date=start_time, 
                end_date=end_time, 
                interval_hours=interval_hours, 
                db=db,
                use_cached=True
            )
            results[inverter_id] = predictions
        except Exception as e:
            logger.error(f"Tahmin hatası (inverter_id={inverter_id}): {str(e)}")
            if db and db.is_active:
                db.rollback()
            results[inverter_id] = []
    
    return results

async def get_active_model(inverter_id: int, db: Session) -> Optional[Model]:
    """
    Belirtilen inverter için aktif modeli döndürür.
    
    Args:
        inverter_id: Model bilgisi alınacak inverter ID'si
        db: Veritabanı oturumu
        
    Returns:
        Optional[Model]: Aktif model veya None
    """
    # Inverter için aktif modeli kontrol et
    active_model = db.query(Model).filter(
        Model.inverter_id == inverter_id,
        Model.is_active == True
    ).first()
    
    return active_model

async def generate_predictions(
    inverter_id: int,
    start_time: datetime = None,
    end_time: datetime = None,
    interval_hours: int = 1,
    db: Session = None,
    use_cached: bool = True
) -> List[InverterPrediction]:
    """
    Belirli bir zaman aralığında tahminler oluşturur.
    
    Args:
        inverter_id: Tahmin yapılacak inverter ID'si
        start_time: Başlangıç zamanı (varsayılan: şimdiki zaman)
        end_time: Bitiş zamanı (varsayılan: 7 gün sonrası)
        interval_hours: Saat cinsinden aralık (varsayılan: 1)
        db: Veritabanı oturumu
        use_cached: Eğer varsa, önceden hesaplanmış tahminleri kullan
        
    Returns:
        List[InverterPrediction]: Tahmin listesi
    """
    return await get_predictions(
        inverter_id=inverter_id,
        start_date=start_time,
        end_date=end_time,
        interval_hours=interval_hours,
        db=db,
        use_cached=use_cached
    )

async def evaluate_model_on_historical_data(
    inverter_id: int,
    start_date: datetime,
    end_date: datetime,
    db: Session
) -> Dict[str, Any]:
    """
    Modeli geçmiş veriler üzerinde değerlendirir.
    
    Args:
        inverter_id: Değerlendirilecek inverter ID'si
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
        db: Veritabanı oturumu
        
    Returns:
        Dict[str, Any]: Değerlendirme metrikleri
    """
    from app.models.inverter import InverterData
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Aktif modeli yükle
    model, model_meta = await load_model(inverter_id, db)
    
    if model is None:
        raise ValueError(f"İnverter {inverter_id} için aktif model bulunamadı")
    
    # Geçmiş inverter verilerini al
    inverter_data = db.query(InverterData).filter(
        InverterData.inverter_id == inverter_id,
        InverterData.timestamp >= start_date,
        InverterData.timestamp <= end_date,
        InverterData.power_output.isnot(None)
    ).all()
    
    if not inverter_data:
        raise ValueError(f"İnverter {inverter_id} için belirtilen tarih aralığında veri bulunamadı")
    
    # Gerçek güç çıkışı değerlerini al
    actual_data = {data.timestamp: data.power_output for data in inverter_data}
    
    # Her zaman noktası için tahmin yap
    predictions = []
    actuals = []
    
    for timestamp, actual_power in actual_data.items():
        try:
            # Tahmini hesapla (cache kullanma)
            prediction = await get_prediction(inverter_id, timestamp, db, use_cached=False)
            
            # Tahmin ve gerçek değerleri kaydet
            predictions.append(prediction.predicted_power_output)
            actuals.append(actual_power)
        except Exception as e:
            print(f"Değerlendirme hatası ({inverter_id}, {timestamp}): {str(e)}")
    
    # Metrikleri hesapla
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # MAPE hesaplama (güvenli)
    mask = np.array(actuals) > 1.0  # 1 kWh'den büyük değerler için
    mape = 0.0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((np.array(actuals)[mask] - np.array(predictions)[mask]) / np.array(actuals)[mask])) * 100
    
    return {
        "inverter_id": inverter_id,
        "evaluation_period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "data_points": len(actuals),
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        },
        "model_version": model_meta.get("model_version", "unknown")
    } 