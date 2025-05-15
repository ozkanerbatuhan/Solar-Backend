from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.model import Model as ModelDB
from app.models.inverter import Inverter
from app.schemas.model import ModelInfo, ModelMetrics, ModelTrainingResponse
from app.services.model_training_service import train_model, train_all_models, get_model_metrics, get_all_model_metrics
from app.services.prediction_service import get_prediction, get_bulk_predictions

router = APIRouter()

@router.post("/models/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(model: ModelInfo, db: Session = Depends(get_db)):
    """Yeni bir model kaydı oluşturur."""
    # Inverter'ın varlığını kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == model.inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    db_model = ModelDB(**model.model_dump())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

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
    query = db.query(ModelDB)
    
    if inverter_id is not None:
        query = query.filter(ModelDB.inverter_id == inverter_id)
    
    if active_only:
        query = query.filter(ModelDB.is_active == True)
    
    models = query.order_by(ModelDB.created_at.desc()).offset(skip).limit(limit).all()
    return models

@router.get("/models/{model_id}", response_model=ModelInfo)
def read_model(model_id: int, db: Session = Depends(get_db)):
    """Belirli bir modeli ID'sine göre getirir."""
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Model bulunamadı")
    return model

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
    # Inverter'ın varlığını kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    metrics = await get_model_metrics(inverter_id, db)
    return metrics

@router.get("/metrics", response_model=Dict[str, Any])
async def get_all_model_metrics_endpoint(
    db: Session = Depends(get_db)
):
    """
    Tüm inverterler için model metriklerini döndürür.
    
    Args:
        db: Veritabanı oturumu
    """
    metrics = await get_all_model_metrics(db)
    return metrics

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
            "prediction_timestamp": prediction.prediction_timestamp,
            "predicted_power_output": prediction.predicted_power_output,
            "confidence": prediction.confidence,
            "model_version": prediction.model_version
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin sırasında bir hata oluştu: {str(e)}"
        )

@router.post("/predict-bulk")
async def predict_bulk_inverter_output(
    inverter_ids: List[int] = Query(..., description="Tahmin yapılacak inverter ID'leri"),
    start_time: datetime = Query(..., description="Başlangıç zamanı"),
    end_time: datetime = Query(..., description="Bitiş zamanı"),
    interval_minutes: int = Query(60, description="Tahmin aralığı (dakika)", ge=15, le=1440),
    db: Session = Depends(get_db)
):
    """
    Birden fazla inverter için belirli bir zaman aralığında tahminler yapar.
    
    Args:
        inverter_ids: Tahmin yapılacak inverter ID'leri
        start_time: Başlangıç zamanı
        end_time: Bitiş zamanı
        interval_minutes: Tahmin aralığı (dakika)
        db: Veritabanı oturumu
    """
    try:
        # Inverter'ların varlığını kontrol et
        for inverter_id in inverter_ids:
            inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
            if inverter is None:
                raise HTTPException(status_code=404, detail=f"Inverter {inverter_id} bulunamadı")
        
        # Zaman aralığını kontrol et
        if end_time <= start_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bitiş zamanı başlangıç zamanından sonra olmalıdır"
            )
        
        # Tahminleri yap
        predictions = await get_bulk_predictions(
            inverter_ids, start_time, end_time, interval_minutes, db
        )
        
        # Sonuçları formatla
        results = {}
        for inverter_id, inverter_predictions in predictions.items():
            results[inverter_id] = [
                {
                    "timestamp": p.prediction_timestamp,
                    "predicted_power_output": p.predicted_power_output,
                    "confidence": p.confidence
                }
                for p in inverter_predictions
            ]
        
        return {
            "success": True,
            "predictions": results,
            "metadata": {
                "start_time": start_time,
                "end_time": end_time,
                "interval_minutes": interval_minutes,
                "inverter_count": len(inverter_ids)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Toplu tahmin sırasında bir hata oluştu: {str(e)}"
        ) 