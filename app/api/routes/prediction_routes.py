from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.inverter import Inverter, InverterPrediction
from app.schemas.inverter import (
    InverterPredictionCreate, 
    InverterPrediction as InverterPredictionSchema,
    InverterWithPredictions
)
from app.services.prediction_service import get_prediction  # Bu servisi daha sonra oluşturacağız

router = APIRouter()

@router.post("/predictions/", response_model=InverterPredictionSchema, status_code=status.HTTP_201_CREATED)
def create_prediction(prediction: InverterPredictionCreate, db: Session = Depends(get_db)):
    """Yeni bir tahmin kaydı oluştur."""
    # Inverter'ın varlığını kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == prediction.inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    db_prediction = InverterPrediction(**prediction.model_dump())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

@router.get("/predictions/", response_model=List[InverterPredictionSchema])
def read_predictions(
    skip: int = Query(0, description="Atlanacak kayıt sayısı"),
    limit: int = Query(100, description="Alınacak maksimum kayıt sayısı"),
    inverter_id: Optional[int] = Query(None, description="Filtrelenecek inverter ID'si"),
    db: Session = Depends(get_db)
):
    """Tahminleri listeler. Opsiyonel olarak belirli bir inverter için filtrelenebilir."""
    query = db.query(InverterPrediction)
    
    if inverter_id is not None:
        query = query.filter(InverterPrediction.inverter_id == inverter_id)
    
    predictions = query.order_by(InverterPrediction.prediction_timestamp.desc()).offset(skip).limit(limit).all()
    return predictions

@router.get("/predictions/{prediction_id}", response_model=InverterPredictionSchema)
def read_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Belirli bir tahmini ID'sine göre getirir."""
    prediction = db.query(InverterPrediction).filter(InverterPrediction.id == prediction_id).first()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Tahmin bulunamadı")
    return prediction

@router.get("/inverters/{inverter_id}/predictions", response_model=InverterWithPredictions)
def read_inverter_with_predictions(
    inverter_id: int, 
    limit: int = Query(100, description="Alınacak maksimum tahmin sayısı"),
    db: Session = Depends(get_db)
):
    """Inverter ve ilişkili tahmin verilerini getirir."""
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    # Son tahminleri al
    inverter.predictions = db.query(InverterPrediction).filter(
        InverterPrediction.inverter_id == inverter_id
    ).order_by(InverterPrediction.prediction_timestamp.desc()).limit(limit).all()
    
    return inverter

@router.get("/inverters/{inverter_id}/predict", response_model=InverterPredictionSchema)
async def predict_inverter_output(
    inverter_id: int,
    timestamp: Optional[datetime] = Query(None, description="Tahmin yapılacak zaman, belirtilmezse şu anki zaman + 1 saat kullanılır"),
    db: Session = Depends(get_db)
):
    """Belirtilen inverter için güç çıktısı tahmini yapar."""
    # Inverter'ın varlığını kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    # Tahmin zamanı belirtilmemişse, şu anki zamandan 1 saat sonrasını kullan
    if timestamp is None:
        timestamp = datetime.utcnow() + timedelta(hours=1)
    
    # Tahmin servisi çağrısı (Bu servis daha sonra oluşturulacak)
    try:
        prediction_result = await get_prediction(inverter_id, timestamp, db)
        return prediction_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin oluşturulurken bir hata oluştu: {str(e)}"
        ) 