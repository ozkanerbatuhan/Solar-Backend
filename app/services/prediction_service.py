import os
import joblib
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
import json

from app.models.inverter import InverterPrediction, Inverter
from app.models.model import Model
from app.models.weather import WeatherForecast
from app.core.config import settings
from app.services.weather_service import fetch_weather_forecast
from app.services.data_quality_service import DataQualityService
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
    
    # Tarihleri tam saat olarak normalize et
    start_date = start_date.replace(minute=0, second=0, microsecond=0)
    end_date = end_date.replace(minute=0, second=0, microsecond=0)
    
    logger.info(f"Normalize edilmiş tarih aralığı: {start_date} - {end_date}")
    
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
        model, scaler, model_meta = await load_model(inverter_id, db)
        
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        model, scaler, model_meta = None, None, None
    
    if model is None:
        # Model yoksa basit bir tahmin serisi oluştur
        logger.info(f"Model yok, basit tahmin serisi oluşturuluyor...")
        return await _make_dummy_predictions(inverter_id, start_date, end_date, interval_hours, db)
    
    # Her bir zaman noktası için tahmin yap
    predictions = []
    current_time = start_date
    dummy_count = 0
    real_count = 0
    
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
                # Hava durumu verisi bulunamadı, dummy tahmin yap
                logger.warning(f"Hava durumu verisi bulunamadı, {current_time} için dummy tahmin yapılıyor...")
                prediction = await _make_dummy_prediction(inverter_id, current_time, db)
                predictions.append(prediction)
                dummy_count += 1
                transaction_success = True
            else:
                # Tahmin için özellikleri hazırla
                features = _prepare_features(weather_data, current_time)
                
                # Veri kalitesi kontrolü
                quality_issues = _check_data_quality(weather_data, current_time)
                if quality_issues:
                    logger.warning(f"Veri kalitesi sorunları tespit edildi ({current_time}): {quality_issues}")
                
                # Modelin beklediği özellikleri al
                required_features = model_meta.get("features", [])
                
                # Gerekli özellikleri içeren DataFrame oluştur
                if required_features:
                    feature_df = pd.DataFrame([{k: features.get(k, 0) for k in required_features}])
                    logger.info(f"Gerekli özellikler: {required_features}")
                    logger.info(f"Özellikler: {features}")
                    logger.info(f"Feature DataFrame: {feature_df}")
                else:
                    # Özellikler belirtilmemişse, tüm özellikleri kullan
                    feature_df = pd.DataFrame([features])
                
                # Scaler kullanarak özellikleri ölçekle
                if scaler is not None:
                    try:
                        feature_df_scaled = pd.DataFrame(
                            scaler.transform(feature_df),
                            columns=feature_df.columns
                        )
                        logger.info("Özellikler scaler ile ölçeklendirildi")
                    except Exception as scale_error:
                        logger.warning(f"Scaler hatası: {str(scale_error)}, ham özellikler kullanılacak")
                        feature_df_scaled = feature_df
                else:
                    logger.warning("Scaler bulunamadı, ham özellikler kullanılacak")
                    feature_df_scaled = feature_df
                
                # Tahmin yap
                predicted_power = float(model.predict(feature_df_scaled)[0])
                
                # YENİ: Fiziksel kısıtlar ve post-processing
                predicted_power = _apply_prediction_constraints(predicted_power, features, current_time)
                
                # Tahmin güven değeri (veri kalitesine göre ayarlanabilir)
                confidence = 0.9
                if quality_issues:
                    confidence = max(0.5, confidence - len(quality_issues) * 0.1)
                
                # Model kalitesi kontrolü
                if 'data_quality_score' in locals():
                    quality_factor = min(1.0, locals()['data_quality_score'] / 100)
                    confidence *= quality_factor
                
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
                real_count += 1
                
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
                dummy_count += 1
            except Exception as dummy_err:
                logger.error(f"Dummy tahmin hatası: {str(dummy_err)}")
                if db and db.is_active:
                    db.rollback()
        
        finally:
            # Bir sonraki zaman noktasına geç
            current_time += timedelta(hours=interval_hours)
    
    logger.info(f"{inverter_id} için tahmin tamamlandı: {len(predictions)} tahmin")
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
    Belirtilen inverter için makine öğrenimi modelini ve scaler'ını yükler.
    
    Args:
        inverter_id: Model yüklenecek inverter ID'si
        db: Veritabanı oturumu
    
    Returns:
        tuple: (model, scaler, model_meta) - Yüklenen model, scaler ve meta verileri
    """
    try:
        # Inverter için aktif modeli kontrol et
        active_model = db.query(Model).filter(
            Model.inverter_id == inverter_id,
            Model.is_active == True
        ).first()
        
        if active_model is None:
            return None, None, None
        
        # Model dosyasının yolunu oluştur
        model_path = active_model.model_path
        if not model_path:
            return None, None, None
        
        # Tam dosya yolunu oluştur
        full_model_path = os.path.join(MODELS_DIR, model_path)
        
        # Model meta verisi için dosya yolunu oluştur
        model_version = active_model.version
        meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
        
        # Scaler dosya yolunu oluştur
        scaler_path = os.path.join(MODELS_DIR, f"{model_version}_scaler.joblib")
        
        # Modelin var olup olmadığını kontrol et
        if not os.path.exists(full_model_path):
            logger.warning(f"Model dosyası bulunamadı: {full_model_path}")
            return None, None, None
        
        # Modeli yükle
        model = joblib.load(full_model_path)
        
        # Scaler'ı yükle
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler başarıyla yüklendi: {scaler_path}")
        else:
            logger.warning(f"Scaler dosyası bulunamadı: {scaler_path}")
        
        # Meta verileri yükle
        model_meta = active_model.metrics
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                model_meta = json.load(f)
        
        return model, scaler, model_meta
    
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        if db and db.is_active:
            db.rollback()
        return None, None, None

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
        # Timestamp'i tam saat olarak yuvarla
        normalized_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        logger.info(f"Hava durumu verisi için normalize edilmiş zaman: {normalized_timestamp}")
        
        # Tahmin zamanına en yakın hava durumu verisini bul
        weather_data = db.query(WeatherForecast).filter(
            WeatherForecast.forecast_timestamp <= normalized_timestamp + timedelta(hours=1),
            WeatherForecast.forecast_timestamp >= normalized_timestamp - timedelta(hours=1)
        ).order_by(
            # En yakın zaman damgasına göre sırala
            func.abs(func.extract('epoch', WeatherForecast.forecast_timestamp) - 
                   func.extract('epoch', normalized_timestamp))
        ).first()
        
        # Eğer veritabanında hava durumu verisi yoksa, API'den çek
        if not weather_data:
            # Önce o gün için veritabanında hiç veri var mı kontrol et
            # Bu şekilde API'ye her eksik saat için değil, sadece günlük bazda istek yapılır
            start_of_day = datetime(normalized_timestamp.year, normalized_timestamp.month, normalized_timestamp.day)
            end_of_day = start_of_day + timedelta(days=1)
            
            existing_data_count = db.query(WeatherForecast).filter(
                WeatherForecast.forecast_timestamp >= start_of_day,
                WeatherForecast.forecast_timestamp < end_of_day
            ).count()
            
            if existing_data_count == 0:
                logger.info(f"{normalized_timestamp.date()} günü için hava durumu verisi bulunamadı. API'den çekiliyor...")
                
                # Inverter lokasyon bilgilerini al (burada örnek değerler, gerçek proje için ayarlanmalı)
                latitude = 37.5704328 # Mersin
                longitude = 34.1371146
                
                # Sadece gerekli gün için tahmin çek
                forecast_days = 7  # Bugün ve sonraki 7 gün için tahmin çek
                
                # Hava durumu verisini API'den çek
                try:
                    # API'den veri çek
                    weather_response = await fetch_weather_forecast(
                        latitude=latitude,
                        longitude=longitude,
                        forecast_days=forecast_days,
                        db=db,
                        save_to_db=True
                    )
                    
                    logger.info(f"API'den hava durumu verisi başarıyla çekildi ve veritabanına kaydedildi.")
                    
                    # Veritabanına kaydedilen veriyi tekrar sorgula
                    weather_data = db.query(WeatherForecast).filter(
                        WeatherForecast.forecast_timestamp <= normalized_timestamp + timedelta(hours=1),
                        WeatherForecast.forecast_timestamp >= normalized_timestamp - timedelta(hours=1)
                    ).order_by(
                        func.abs(func.extract('epoch', WeatherForecast.forecast_timestamp) - 
                               func.extract('epoch', normalized_timestamp))
                    ).first()
                    
                    if not weather_data:
                        logger.warning(f"API'den veri çekildi ancak istenen zaman ({normalized_timestamp}) için veri bulunamadı.")
                        return None
                except Exception as api_error:
                    logger.error(f"API'den hava durumu verisi çekilirken hata: {str(api_error)}")
                    return None
            else:
                logger.info(f"{normalized_timestamp.date()} günü için DB'de {existing_data_count} kayıt var ama tam saat için eşleşme bulunamadı.")
                return None
        
        # Hava durumu verilerini sözlük olarak döndür
        weather_dict = {
            "temperature": weather_data.temperature,
            "shortwave_radiation": weather_data.shortwave_radiation,
            "direct_radiation": weather_data.direct_radiation,
            "diffuse_radiation": weather_data.diffuse_radiation,
            "direct_normal_irradiance": weather_data.direct_normal_irradiance,
            "global_tilted_irradiance": weather_data.global_tilted_irradiance,
            "terrestrial_radiation": weather_data.terrestrial_radiation,
            "relative_humidity": weather_data.relative_humidity,
            "wind_speed": weather_data.wind_speed,
        }
        
        # Veri kalitesi ön kontrolü
        quality_issues = _check_data_quality(weather_dict, normalized_timestamp)
        if quality_issues:
            logger.warning(f"Hava durumu verisi kalite sorunları ({normalized_timestamp}): {quality_issues}")
        
        return weather_dict
        
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
    # Timestamp'i tam saat olarak yuvarla
    normalized_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
    
    # Hava durumu verilerini kopyala
    features = dict(weather_data)
    
    # Zaman özelliklerini ekle
    features.update({
        "hour": normalized_timestamp.hour,
        "day": normalized_timestamp.day,
        "month": normalized_timestamp.month,
        "dayofweek": normalized_timestamp.weekday()
    })
    
    # Trigonometrik zaman özellikleri ekle (model eğitimle tutarlı)
    features.update({
        "hour_sin": np.sin(2 * np.pi * normalized_timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * normalized_timestamp.hour / 24),
        "day_sin": np.sin(2 * np.pi * normalized_timestamp.month / 12),  # Model eğitimde month kullanılıyor
        "day_cos": np.cos(2 * np.pi * normalized_timestamp.month / 12)
    })
    
    # YENİ: Gelişmiş feature engineering (data quality service kullanarak)
    # Önce DataFrame'e dönüştür
    temp_df = pd.DataFrame([features])
    
    # Gelişmiş özellikler ekle
    enhanced_df = DataQualityService.create_solar_aware_features(temp_df)
    
    # Geri sözlük formatına dönüştür
    enhanced_features = enhanced_df.iloc[0].to_dict()
    
    # YENİ: Fiziksel kısıtlar ve validation
    enhanced_features = _apply_physics_constraints(enhanced_features, normalized_timestamp)
    
    return enhanced_features

def _apply_physics_constraints(features: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
    """
    Fiziksel kısıtları uygular ve mantıksız değerleri düzeltir.
    
    Args:
        features: Özellikler sözlüğü
        timestamp: Zaman damgası
        
    Returns:
        Düzeltilmiş özellikler
    """
    corrected_features = features.copy()
    hour = timestamp.hour
    
    # 1. Gece saatleri (22:00-05:59) için güneş radyasyonu sıfır olmalı
    if hour >= 22 or hour <= 5:
        radiation_fields = [
            'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
            'direct_normal_irradiance', 'global_tilted_irradiance', 'terrestrial_radiation'
        ]
        
        for field in radiation_fields:
            if field in corrected_features and corrected_features[field] > 0:
                logger.warning(f"Gece saatinde ({hour}:00) {field} > 0 ({corrected_features[field]}) düzeltiliyor")
                corrected_features[field] = 0
        
        # İlgili composite features da güncellenmeli
        if 'total_radiation_index' in corrected_features:
            corrected_features['total_radiation_index'] = 0
        if 'is_daylight' in corrected_features:
            corrected_features['is_daylight'] = 0
        if 'is_peak_solar' in corrected_features:
            corrected_features['is_peak_solar'] = 0
        if 'zero_radiation' in corrected_features:
            corrected_features['zero_radiation'] = 1
        if 'high_radiation' in corrected_features:
            corrected_features['high_radiation'] = 0
        if 'low_radiation' in corrected_features:
            corrected_features['low_radiation'] = 1
    
    # 2. Sıcaklık makul aralıkta olmalı (-50°C ile +60°C arası)
    if 'temperature' in corrected_features:
        temp = corrected_features['temperature']
        if temp < -50:
            logger.warning(f"Çok düşük sıcaklık ({temp}°C) -50°C'ye ayarlanıyor")
            corrected_features['temperature'] = -50
        elif temp > 60:
            logger.warning(f"Çok yüksek sıcaklık ({temp}°C) 60°C'ye ayarlanıyor")
            corrected_features['temperature'] = 60
    
    # 3. Nem %0-100 arasında olmalı
    if 'relative_humidity' in corrected_features:
        humidity = corrected_features['relative_humidity']
        if humidity < 0:
            corrected_features['relative_humidity'] = 0
        elif humidity > 100:
            corrected_features['relative_humidity'] = 100
    
    # 4. Rüzgar hızı negatif olamaz ve makul üst sınır
    if 'wind_speed' in corrected_features:
        wind = corrected_features['wind_speed']
        if wind < 0:
            corrected_features['wind_speed'] = 0
        elif wind > 200:  # 200 km/h üzeri çok yüksek
            logger.warning(f"Çok yüksek rüzgar hızı ({wind} km/h) 200 km/h'ye ayarlanıyor")
            corrected_features['wind_speed'] = 200
    
    return corrected_features

def _check_data_quality(weather_data: Dict[str, Any], timestamp: datetime) -> List[str]:
    """
    Hava durumu verilerinin kalitesini kontrol eder.
    
    Args:
        weather_data: Hava durumu verileri
        timestamp: Zaman damgası
        
    Returns:
        List[str]: Tespit edilen kalite sorunlarının listesi
    """
    issues = []
    
    # Gece saatlerinde güneş radyasyonu kontrolü
    hour = timestamp.hour
    if 22 <= hour or hour <= 5:  # Gece saatleri
        if weather_data.get("shortwave_radiation", 0) > 50:
            issues.append(f"Gece saatinde yüksek shortwave_radiation: {weather_data.get('shortwave_radiation')}")
        if weather_data.get("direct_radiation", 0) > 30:
            issues.append(f"Gece saatinde yüksek direct_radiation: {weather_data.get('direct_radiation')}")
        if weather_data.get("global_tilted_irradiance", 0) > 50:
            issues.append(f"Gece saatinde yüksek global_tilted_irradiance: {weather_data.get('global_tilted_irradiance')}")
    
    # Gündüz saatlerinde çok düşük değer kontrolü
    elif 10 <= hour <= 15:  # Öğle saatleri
        if weather_data.get("shortwave_radiation", 0) < 100 and weather_data.get("global_tilted_irradiance", 0) < 100:
            issues.append("Öğle saatlerinde beklenenden düşük güneş radyasyonu")
    
    # Aşırı yüksek değer kontrolleri
    if weather_data.get("shortwave_radiation", 0) > 1200:
        issues.append(f"Aşırı yüksek shortwave_radiation: {weather_data.get('shortwave_radiation')}")
    
    if weather_data.get("temperature", 0) > 60 or weather_data.get("temperature", 0) < -40:
        issues.append(f"Anormal sıcaklık değeri: {weather_data.get('temperature')}°C")
    
    if weather_data.get("wind_speed", 0) > 200:  # 200 km/h üzeri anormal
        issues.append(f"Aşırı yüksek rüzgar hızı: {weather_data.get('wind_speed')} km/h")
    
    if weather_data.get("relative_humidity", 0) > 100 or weather_data.get("relative_humidity", 0) < 0:
        issues.append(f"Anormal nem oranı: {weather_data.get('relative_humidity')}%")
    
    # NaN veya None değer kontrolleri
    critical_fields = ["temperature", "shortwave_radiation", "relative_humidity"]
    for field in critical_fields:
        value = weather_data.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            issues.append(f"Kritik alan eksik: {field}")
    
    return issues

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
    interval_minutes: int = None,
    db: Session = None
) -> Dict[int, List[InverterPrediction]]:
    """
    Birden fazla inverter için belirli bir zaman aralığında tahminler yapar.
    
    Args:
        inverter_ids: Tahmin yapılacak inverter ID'leri
        start_time: Başlangıç zamanı (varsayılan: şimdiki zaman)
        end_time: Bitiş zamanı (varsayılan: 7 gün sonrası)
        interval_hours: Saat cinsinden aralık (varsayılan: 1)
        interval_minutes: Dakika cinsinden aralık (varsa, interval_hours yerine kullanılır)
        db: Veritabanı oturumu
        
    Returns:
        Dict: Inverter ID'lerine göre tahminler
    """
    # Varsayılan parametreleri ayarla
    if start_time is None:
        start_time = datetime.now()
    
    if end_time is None:
        end_time = start_time + timedelta(days=7)
    
    # Dakika parametresi varsa, saate dönüştür
    if interval_minutes is not None:
        interval_hours = interval_minutes / 60
        logger.info(f"interval_minutes={interval_minutes} değeri interval_hours={interval_hours} olarak dönüştürüldü")
    
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
    model, scaler, model_meta = await load_model(inverter_id, db)
    
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
        "model_version": model_meta.get("model_version", "unknown"),
        "scaler_used": scaler is not None
    }

def _apply_prediction_constraints(predicted_power: float, features: Dict[str, Any], timestamp: datetime) -> float:
    """
    Çok daha sıkı fiziksel kısıtları uygular - güneş enerjisi fizik kuralları.
    
    Args:
        predicted_power: Ham tahmin değeri
        features: Kullanılan özellikler
        timestamp: Tahmin zamanı
        
    Returns:
        Düzeltilmiş tahmin değeri
    """
    hour = timestamp.hour
    
    # 1. Negatif değerleri sıfırla
    if predicted_power < 0:
        logger.warning(f"Negatif tahmin değeri ({predicted_power:.2f}) sıfıra ayarlandı")
        return 0.0
    
    # 2. Gece saatleri kontrolü (22:00-05:59) - KATIYECI
    if hour >= 22 or hour <= 5:
        if predicted_power > 0:
            logger.warning(f"Gece saatinde ({hour}:00) pozitif tahmin ({predicted_power:.2f}) sıfıra ayarlandı")
        return 0.0
    
    # 3. Güneş radyasyonu kontrolü - ÇOK SIKI
    total_radiation = features.get('total_radiation_index', 0) or features.get('shortwave_radiation', 0)
    shortwave_radiation = features.get('shortwave_radiation', 0)
    
    # Sıfır radyasyon = sıfır güç
    if total_radiation == 0 or shortwave_radiation == 0:
        if predicted_power > 0:
            logger.warning(f"Sıfır radyasyonda tahmin ({predicted_power:.2f}) sıfıra ayarlandı")
        return 0.0
    
    # 4. Radyasyon-güç ilişkisi - GÜNCELLENMIŞ SINIRLARI (1.2 MW hedefi)
    # Teorik maksimum: ~1.5-2.0 kW per 1000 W/m² (büyük sistem verimliliği)
    max_efficiency_ratio = 2.0  # kW per 1000 W/m² (büyük sistemler için daha gerçekçi)
    
    # Mevsimsel faktör (yaz aylarında daha yüksek)
    month = timestamp.month
    seasonal_efficiency_boost = 1.0
    if month in [6, 7, 8]:  # Yaz ayları - yüksek radyasyon
        seasonal_efficiency_boost = 1.5
    elif month in [4, 5, 9, 10]:  # İlkbahar/sonbahar - orta
        seasonal_efficiency_boost = 1.2
    
    # Düşük radyasyon durumları için daha esnek sınırlar
    if shortwave_radiation < 50:  # 50 W/m² altında
        max_allowed_power = max(50, shortwave_radiation * 1.0)  # Min 50 kW
        if predicted_power > max_allowed_power:
            logger.info(f"Çok düşük radyasyon ({shortwave_radiation}) tahmin sınırı: {predicted_power:.2f} -> {max_allowed_power:.2f}")
            return max_allowed_power
    
    elif shortwave_radiation < 200:  # 200 W/m² altında
        max_allowed_power = shortwave_radiation * 2.5 * seasonal_efficiency_boost
        if predicted_power > max_allowed_power:
            logger.info(f"Düşük radyasyon ({shortwave_radiation}) tahmin sınırı: {predicted_power:.2f} -> {max_allowed_power:.2f}")
            return max_allowed_power
    
    elif shortwave_radiation < 500:  # Orta seviye radyasyon
        max_allowed_power = shortwave_radiation * 2.0 * seasonal_efficiency_boost
        if predicted_power > max_allowed_power:
            logger.info(f"Orta radyasyon ({shortwave_radiation}) tahmin sınırı: {predicted_power:.2f} -> {max_allowed_power:.2f}")
            return max_allowed_power
    
    else:  # Yüksek radyasyon (>500 W/m²)
        max_allowed_power = shortwave_radiation * max_efficiency_ratio * seasonal_efficiency_boost
        # Yaz aylarında çok yüksek radyasyonda 1.2 MW'a kadar izin ver
        if month in [6, 7, 8] and shortwave_radiation > 800:
            max_allowed_power = min(1200, max_allowed_power)  # 1.2 MW maksimum
        
        if predicted_power > max_allowed_power:
            logger.info(f"Yüksek radyasyon ({shortwave_radiation}) tahmin sınırı: {predicted_power:.2f} -> {max_allowed_power:.2f}")
            return max_allowed_power
    
    # 5. Aşırı yüksek değer kontrolü (inverter kapasitesi)
    max_inverter_capacity = 5000  # 5 MW
    if predicted_power > max_inverter_capacity:
        logger.warning(f"Kapasiteyi aşan tahmin ({predicted_power:.2f}) {max_inverter_capacity}'ye sınırlandı")
        return max_inverter_capacity
    
    # 6. Sabah/akşam saatleri için GÜNCELLENMIŞ esnek sınırlar (1.2 MW hedefi)
    if 6 <= hour <= 8:  # Sabah saatleri - daha esnek
        hour_factor = max(0.3, (hour - 5) / 3)  # 0.3-1.0 arası (daha yüksek başlangıç)
        # Yaz aylarında sabah saatleri için daha yüksek sınır
        max_morning = 800 if month in [6, 7, 8] else 600  # Yaz: 800kW, diğer: 600kW
        max_allowed = min(max_morning, predicted_power / hour_factor)  # Ters orantı yerine daha esnek
        if predicted_power > max_allowed and max_allowed < predicted_power * 0.7:  # Sadece çok büyük farklarda müdahale
            logger.info(f"Sabah saati ({hour}:00) tahmin düzeltmesi: {predicted_power:.2f} -> {max_allowed:.2f}")
            return max_allowed
    
    elif 17 <= hour <= 20:  # Akşam saatleri - daha esnek aralık ve sınır
        hour_factor = max(0.4, (21 - hour) / 4)  # 1.0-0.4 arası (daha yüksek minimum)
        # Yaz aylarında akşam saatleri için daha yüksek sınır  
        max_evening = 900 if month in [6, 7, 8] else 700  # Yaz: 900kW, diğer: 700kW
        max_allowed = min(max_evening, predicted_power / hour_factor)  # Daha esnek hesaplama
        if predicted_power > max_allowed and max_allowed < predicted_power * 0.7:  # Sadık çok büyük farklarda müdahale
            logger.info(f"Akşam saati ({hour}:00) tahmin düzeltmesi: {predicted_power:.2f} -> {max_allowed:.2f}")
            return max_allowed
    
    # 7. Hava durumu tabanlı AGRESIF düzeltmeler
    if 'relative_humidity' in features and features['relative_humidity'] > 85:
        # Yüksek nemde panel verimliliği ciddi şekilde düşer
        humidity_factor = max(0.5, 1 - (features['relative_humidity'] - 50) / 100)
        corrected_power = predicted_power * humidity_factor
        if abs(corrected_power - predicted_power) > 10:
            logger.info(f"Yüksek nem düzeltmesi (%{features['relative_humidity']}): {predicted_power:.2f} -> {corrected_power:.2f}")
        predicted_power = corrected_power
    
    if 'temperature' in features and features['temperature'] > 40:
        # Yüksek sıcaklıkta panel verimi ciddi şekilde düşer
        temp_factor = max(0.6, 1 - (features['temperature'] - 25) / 50)
        corrected_power = predicted_power * temp_factor
        if abs(corrected_power - predicted_power) > 10:
            logger.info(f"Yüksek sıcaklık düzeltmesi ({features['temperature']}°C): {predicted_power:.2f} -> {corrected_power:.2f}")
        predicted_power = corrected_power
    
    # 8. Bulutlu hava durumu kontrolü (diffuse radiation yüksek, direct düşük)
    if 'direct_radiation' in features and 'diffuse_radiation' in features:
        direct_rad = features['direct_radiation']
        diffuse_rad = features['diffuse_radiation']
        
        if direct_rad > 0 and diffuse_rad > 0:
            diffuse_ratio = diffuse_rad / (direct_rad + diffuse_rad)
            if diffuse_ratio > 0.7:  # %70'den fazla diffuse = bulutlu
                cloud_factor = max(0.7, 1 - diffuse_ratio * 0.5)
                corrected_power = predicted_power * cloud_factor
                if abs(corrected_power - predicted_power) > 10:
                    logger.info(f"Bulutlu hava düzeltmesi (diffuse ratio: %{diffuse_ratio*100:.1f}): {predicted_power:.2f} -> {corrected_power:.2f}")
                predicted_power = corrected_power
    
    # 9. Final güvenlik kontrolü - ESNEK hale getirildi (1.2 MW hedefi)
    # Sadece çok extreme durumlarda müdahale et
    if shortwave_radiation < 50 and predicted_power > 300:  # Çok düşük radyasyon + çok yüksek tahmin
        final_power = min(predicted_power, shortwave_radiation * 3)  # 3x faktör ile daha esnek
        if final_power != predicted_power:
            logger.info(f"Final güvenlik kontrolü: radyasyon {shortwave_radiation}, tahmin {predicted_power:.2f} -> {final_power:.2f}")
        return final_power
    
    # Çok aşırı durumlar için ek kontrol (çok düşük radyasyon + MW seviyesi tahmin)
    elif shortwave_radiation < 20 and predicted_power > 500:
        final_power = 100  # Minimum makul değer
        logger.warning(f"Aşırı düşük radyasyon kontrolü: radyasyon {shortwave_radiation}, tahmin {predicted_power:.2f} -> {final_power:.2f}")
        return final_power
    
    return predicted_power 