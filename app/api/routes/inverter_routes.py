from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.inverter import Inverter, InverterData, InverterPrediction
from app.schemas.inverter import (
    InverterCreate, Inverter as InverterSchema,
    InverterUpdate, InverterWithData, InverterWithPredictions,
    InverterDataCreate, InverterData as InverterDataSchema
)

router = APIRouter()

@router.post("/inverters/", response_model=InverterSchema, status_code=status.HTTP_201_CREATED)
def create_inverter(inverter: InverterCreate, db: Session = Depends(get_db)):
    """Yeni bir inverter oluştur."""
    db_inverter = Inverter(**inverter.model_dump())
    db.add(db_inverter)
    db.commit()
    db.refresh(db_inverter)
    return db_inverter

@router.get("/inverters/", response_model=List[InverterSchema])
def read_inverters(
    skip: int = Query(0, description="Atlanacak kayıt sayısı"),
    limit: int = Query(100, description="Alınacak maksimum kayıt sayısı"),
    db: Session = Depends(get_db)
):
    """Tüm inverterleri listeler."""
    inverters = db.query(Inverter).offset(skip).limit(limit).all()
    return inverters

@router.get("/inverters/{inverter_id}", response_model=InverterSchema)
def read_inverter(inverter_id: int, db: Session = Depends(get_db)):
    """Belirli bir inverteri ID'sine göre getirir."""
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    return inverter

@router.put("/inverters/{inverter_id}", response_model=InverterSchema)
def update_inverter(
    inverter_id: int, 
    inverter: InverterUpdate, 
    db: Session = Depends(get_db)
):
    """Inverter bilgilerini günceller."""
    db_inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if db_inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    update_data = inverter.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_inverter, key, value)
    
    db.commit()
    db.refresh(db_inverter)
    return db_inverter

@router.delete("/inverters/{inverter_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inverter(inverter_id: int, db: Session = Depends(get_db)):
    """Inverter kaydını siler."""
    db_inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if db_inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    db.delete(db_inverter)
    db.commit()
    return {"ok": True}

@router.get("/inverters/{inverter_id}/data", response_model=InverterWithData)
def read_inverter_with_data(inverter_id: int, db: Session = Depends(get_db)):
    """Inverter ve ilişkili ölçüm verilerini getirir."""
    inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    return inverter

@router.post("/inverter-data/", response_model=InverterDataSchema, status_code=status.HTTP_201_CREATED)
def create_inverter_data(inverter_data: InverterDataCreate, db: Session = Depends(get_db)):
    """Yeni inverter ölçüm verisi ekler."""
    # Inverter'ın varlığını kontrol et
    inverter = db.query(Inverter).filter(Inverter.id == inverter_data.inverter_id).first()
    if inverter is None:
        raise HTTPException(status_code=404, detail="Inverter bulunamadı")
    
    db_inverter_data = InverterData(**inverter_data.model_dump())
    db.add(db_inverter_data)
    db.commit()
    db.refresh(db_inverter_data)
    return db_inverter_data 