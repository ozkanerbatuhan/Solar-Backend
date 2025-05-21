from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, date
import json
import os
import pandas as pd
import numpy as np

from app.db.database import get_db
from app.models.model import Model
from app.models.inverter import Inverter, InverterData
from app.schemas.model import (
    ModelInfo,
    ModelTrainingResponse,
    ModelMetrics
)
from app.schemas.data import ModelLogResponse
from app.services.model_training_service import train_model, train_all_models, get_model_metrics, get_all_model_metrics, MODELS_DIR
from app.services.prediction_service import get_prediction, get_bulk_predictions

router = APIRouter()

@router.post("/models/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(model: ModelInfo, db: Session = Depends(get_db)):
    """Yeni bir model kaydı oluşturur."""
    try:
        # Inverter'ın varlığını kontrol et
        inverter = db.query(Inverter).filter(Inverter.id == model.inverter_id).first()
        if inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        
        db_model = Model(**model.model_dump())
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model oluşturulurken hata oluştu: {str(e)}"
        )

@router.get("/models", response_model=List[ModelInfo])
def list_models(
    inverter_id: int = Query(None, description="Filtrelenecek inverter ID'si"),
    active_only: bool = Query(False, description="Sadece aktif modeller"),
    skip: int = Query(0, description="Atlanacak kayıt sayısı"),
    limit: int = Query(100, description="Alınacak maksimum kayıt sayısı"),
    db: Session = Depends(get_db)
):
    """
    Modelleri listeler. Opsiyonel olarak belirli bir inverter için filtrelenebilir.
    
    Args:
        inverter_id: Filtrelenecek inverter ID'si
        active_only: Sadece aktif modeller
        skip: Atlanacak kayıt sayısı
        limit: Alınacak maksimum kayıt sayısı
        db: Veritabanı oturumu
    """
    try:
        query = db.query(Model)
        
        if inverter_id is not None:
            query = query.filter(Model.inverter_id == inverter_id)
        
        if active_only:
            query = query.filter(Model.is_active == True)
        
        models = query.order_by(Model.created_at.desc()).offset(skip).limit(limit).all()
        return models
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Modeller listelenirken hata oluştu: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=ModelInfo)
def read_model(model_id: int, db: Session = Depends(get_db)):
    """Belirli bir modeli ID'sine göre getirir."""
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
        if model is None:
            raise HTTPException(status_code=404, detail="Model bulunamadı")
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model bilgisi alınırken hata oluştu: {str(e)}"
        )

@router.post("/train/{inverter_id}", response_model=ModelTrainingResponse)
async def train_inverter_model(
    inverter_id: int,
    background_tasks: BackgroundTasks,
    test_split: bool = Query(True, description="Test bölünmesi yapılsın mı?"),
    test_size: float = Query(0.3, description="Test seti oranı", ge=0.1, le=0.5),
    db: Session = Depends(get_db)
):
    """
    Belirtilen inverter için yeni bir model eğitir.
    
    Args:
        inverter_id: Model eğitilecek inverter ID'si
        test_split: Test bölünmesi yapılsın mı?
        test_size: Test seti oranı
        db: Veritabanı oturumu
    """
    # Inverter'ın varlığını kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    try:
        # Model eğitimini başlat (arka planda)
        if background_tasks:
            background_tasks.add_task(train_model, inverter_id, db, test_split, test_size)
            return {
                "success": True,
                "message": f"Inverter {inverter_id} için model eğitimi arka planda başlatıldı",
                "inverter_id": inverter_id,
                "status": "training_started"
            }
        else:
            # Senkron eğitim
            result = await train_model(inverter_id, db, test_split, test_size)
            return {
                "success": True,
                "message": f"Inverter {inverter_id} için model eğitimi tamamlandı",
                "inverter_id": inverter_id,
                "status": "training_completed",
                "model_info": result
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model eğitimi sırasında bir hata oluştu: {str(e)}"
        )

@router.post("/train-all", response_model=Dict[str, Any])
async def train_all_inverter_models(
    background_tasks: BackgroundTasks,
    test_split: bool = Query(True, description="Test bölünmesi yapılsın mı?"),
    db: Session = Depends(get_db)
):
    """
    Tüm aktif inverterler için modelleri eğitir.
    
    Args:
        test_split: Test bölünmesi yapılsın mı?
        db: Veritabanı oturumu
    """
    try:
        # Arka planda tüm modelleri eğit
        background_tasks.add_task(train_all_models, db, test_split)
        
        # Aktif inverter sayısını al
        inverter_count = db.query(Inverter).filter(Inverter.is_active == True).count()
        
        return {
            "success": True,
            "message": f"{inverter_count} inverter için model eğitimi arka planda başlatıldı",
            "inverter_count": inverter_count,
            "status": "training_started"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model eğitimi sırasında bir hata oluştu: {str(e)}"
        )

@router.get("/metrics/{inverter_id}", response_model=ModelMetrics)
async def get_inverter_model_metrics(
    inverter_id: int,
    db: Session = Depends(get_db)
):
    """
    Belirtilen inverter için model metriklerini döndürür.
    
    Args:
        inverter_id: Metrikleri alınacak inverter ID'si
        db: Veritabanı oturumu
    """
    try:
        # Inverter'ın varlığını kontrol et
        inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
        if inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        
        metrics = await get_model_metrics(inverter_id, db)
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model metrikleri alınırken hata oluştu: {str(e)}"
        )

@router.get("/metrics", response_model=Dict[str, Any])
async def get_all_model_metrics_endpoint(
    db: Session = Depends(get_db)
):
    """
    Tüm inverterler için model metriklerini döndürür.
    
    Args:
        db: Veritabanı oturumu
    """
    try:
        metrics = await get_all_model_metrics(db)
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tüm model metrikleri alınırken hata oluştu: {str(e)}"
        )

@router.get("/predict/{inverter_id}")
async def predict_inverter_output(
    inverter_id: int,
    timestamp: datetime = Query(..., description="Tahmin edilecek zaman"),
    use_cached: bool = Query(True, description="Eğer varsa, önceden hesaplanmış tahmini kullan"),
    db: Session = Depends(get_db)
):
    """
    Belirtilen inverter için güç çıktısı tahmini yapar.
    
    Args:
        inverter_id: Tahmin yapılacak inverter ID'si
        timestamp: Tahmin edilecek zaman
        use_cached: Eğer varsa, önceden hesaplanmış tahmini kullan
        db: Veritabanı oturumu
    """
    try:
        prediction = await get_prediction(inverter_id, timestamp, db, use_cached)
        
        return {
            "inverter_id": inverter_id,
            "timestamp": timestamp,
            "predicted_power_output": prediction.predicted_power_output,
            "confidence": prediction.confidence,
            "model_version": prediction.model_version
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin yapılırken hata oluştu: {str(e)}"
        )

@router.post("/predict-bulk")
async def predict_bulk_inverter_output(
    inverter_ids: List[int] = Query(..., description="Tahmin yapılacak inverter ID'leri"),
    start_time: datetime = Query(datetime.now(), description="Başlangıç zamanı"),
    end_time: datetime = Query(datetime.now() + timedelta(days=7), description="Bitiş zamanı"),
    interval_minutes: int = Query(60, description="Tahmin aralığı (dakika)", ge=15, le=1440),
    db: Session = Depends(get_db)
):
    """
    Birden fazla inverter için toplu tahmin yapar
    """
    try:
        # Prediction service'e interval_minutes parametresini doğrudan geçir
        predictions = await get_bulk_predictions(
            inverter_ids=inverter_ids,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,  # Dakika cinsinden değeri doğrudan gönder
            db=db
        )
        
        # Tahminleri JSON serileştirilebilir hale getir
        formatted_predictions = {}
        for inverter_id, inverter_predictions in predictions.items():
            formatted_predictions[inverter_id] = [
                {
                    "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
                    "prediction_timestamp": pred.prediction_timestamp.isoformat() if pred.prediction_timestamp else None,
                    "predicted_power": pred.predicted_power_output,
                    "model_version": pred.model_version,
                    "confidence": pred.confidence,
                    "features": pred.features
                }
                for pred in inverter_predictions
            ]
        
        return formatted_predictions
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin hatası: {str(e)}"
        )

@router.get("/logs/{model_version}", response_model=ModelLogResponse)
async def get_model_log_file(
    model_version: str,
    db: Session = Depends(get_db)
):
    """
    Model eğitimi sırasında oluşturulan log dosyasını dönderir.
    
    Args:
        model_version: Model versiyonu
        
    Returns:
        Log dosyası içeriği
    """
    try:
        # Önce modelin var olup olmadığını kontrol et
        model_parts = model_version.split('_')
        if len(model_parts) < 2 or not model_parts[1].isdigit():
            raise HTTPException(status_code=400, detail="Geçersiz model versiyonu formatı")
        
        inverter_id = int(model_parts[1])
        
        # Modeli veritabanında bul
        model = db.query(Model).filter(
            Model.inverter_id == inverter_id,
            Model.version == model_version
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model bulunamadı")
        
        # Meta dosya yolunu oluştur
        meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
        
        if not os.path.exists(meta_path):
            raise HTTPException(status_code=404, detail="Model meta dosyası bulunamadı")
        
        # Meta dosyasını oku
        with open(meta_path, "r") as f:
            model_meta = json.load(f)
        
        # Log dosyasının var olup olmadığını kontrol et
        inverter_name = str(inverter_id)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(MODELS_DIR))), "logs")
        log_file = os.path.join(log_dir, f"model_{inverter_name}.log")
        log_details_file = os.path.join(log_dir, f"model_{inverter_name}_details.txt")
        
        logs = []
        
        # Log dosyası varsa içeriğini oku
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" - ", 3)
                    if len(parts) >= 4:
                        ts_str, name, level, msg = parts
                        try:
                            timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                            logs.append({
                                "timestamp": timestamp,
                                "level": level,
                                "message": msg
                            })
                        except ValueError:
                            # Tarih ayrıştırılamadı, satırı atla
                            continue
        
        # Log dosyası yoksa model_meta bilgisini kullan
        response_data = {
            "model_version": model_version,
            "inverter_id": inverter_id,
            "created_at": datetime.fromisoformat(model_meta["created_at"]) if isinstance(model_meta["created_at"], str) else model_meta["created_at"],
            "data_summary": model_meta.get("data_details", {}),
            "feature_importance": model_meta.get("feature_importance", {}),
            "metrics": model_meta.get("metrics", {}),
            "logs": logs
        }
        
        return {
            "success": True,
            "message": "Model log bilgileri başarıyla alındı",
            "log": response_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log dosyası alınırken hata oluştu: {str(e)}")

@router.get("/logs/download/{model_version}")
async def download_model_log_file(
    model_version: str,
    db: Session = Depends(get_db)
):
    """
    Model eğitimi sırasında oluşturulan log dosyasını indirir.
    
    Args:
        model_version: Model versiyonu
        
    Returns:
        Log dosyası
    """
    try:
        # Önce modelin var olup olmadığını kontrol et
        model_parts = model_version.split('_')
        if len(model_parts) < 2 or not model_parts[1].isdigit():
            raise HTTPException(status_code=400, detail="Geçersiz model versiyonu formatı")
        
        inverter_id = int(model_parts[1])
        
        # Modeli veritabanında bul
        model = db.query(Model).filter(
            Model.inverter_id == inverter_id,
            Model.version == model_version
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model bulunamadı")
        
        # Log dosyasının var olup olmadığını kontrol et
        inverter_name = str(inverter_id)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(MODELS_DIR))), "logs")
        log_file = os.path.join(log_dir, f"model_{inverter_name}.log")
        log_details_file = os.path.join(log_dir, f"model_{inverter_name}_details.txt")
        
        # Detaylı log dosyası varsa onu tercih et
        if os.path.exists(log_details_file):
            return FileResponse(
                path=log_details_file,
                filename=f"model_{inverter_name}_details.txt",
                media_type="text/plain"
            )
        # Normal log dosyası varsa onu kullan
        elif os.path.exists(log_file):
            return FileResponse(
                path=log_file,
                filename=f"model_{inverter_name}.log",
                media_type="text/plain"
            )
        else:
            # Log dosyası yoksa, model meta dosyasından bir log oluştur
            meta_path = os.path.join(MODELS_DIR, f"{model_version}_meta.json")
            
            if not os.path.exists(meta_path):
                raise HTTPException(status_code=404, detail="Model meta dosyası bulunamadı")
            
            # Meta dosyasını oku
            with open(meta_path, "r") as f:
                model_meta = json.load(f)
            
            # Geçici bir log dosyası oluştur
            temp_log_file = os.path.join(MODELS_DIR, f"temp_{model_version}_log.txt")
            
            with open(temp_log_file, "w", encoding="utf-8") as f:
                f.write(f"{'='*80}\n")
                f.write(f"{' '*30}İNVERTER {inverter_name} MODEL ÖZETİ\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Model Versiyonu: {model_version}\n")
                f.write(f"Oluşturulma Tarihi: {model_meta.get('created_at', 'Bilinmiyor')}\n\n")
                
                f.write(f"1. VERİ ÖZETİ\n")
                f.write(f"{'-'*80}\n\n")
                
                data_details = model_meta.get("data_details", {})
                f.write(f"    Toplam Satır: {data_details.get('total_rows', 'Bilinmiyor')}\n")
                f.write(f"    Kullanılan Satır: {data_details.get('used_rows', 'Bilinmiyor')}\n\n")
                
                f.write(f"2. MODEL METRİKLERİ\n")
                f.write(f"{'-'*80}\n\n")
                
                metrics = model_meta.get("metrics", {})
                f.write(f"    R² Skoru    : {metrics.get('r2', 'Bilinmiyor')}\n")
                f.write(f"    MAE         : {metrics.get('mae', 'Bilinmiyor')}\n")
                f.write(f"    RMSE        : {metrics.get('rmse', 'Bilinmiyor')}\n")
                f.write(f"    MAPE        : {metrics.get('mape', 'Bilinmiyor')}%\n\n")
                
                f.write(f"3. KULLANILAN ÖZELLİKLER\n")
                f.write(f"{'-'*80}\n\n")
                
                feature_cols = metrics.get("features", [])
                for feat in feature_cols:
                    f.write(f"    - {feat}\n")
                
                f.write(f"\n4. ÖZELLİK ÖNEMLERİ\n")
                f.write(f"{'-'*80}\n\n")
                
                f.write(f"{'Özellik':<30}{'Önem':>15}\n")
                f.write(f"{'-'*45}\n")
                
                feature_importance = model_meta.get("feature_importance", {})
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                for feature, importance in sorted_features:
                    f.write(f"{feature:<30}{importance:>15.4f}\n")
            
            return FileResponse(
                path=temp_log_file,
                filename=f"model_{inverter_name}_summary.txt",
                media_type="text/plain",
                background=lambda: os.remove(temp_log_file) if os.path.exists(temp_log_file) else None
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log dosyası indirilirken hata oluştu: {str(e)}") 