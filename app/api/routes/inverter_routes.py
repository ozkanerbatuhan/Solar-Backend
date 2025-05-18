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
    try:
        db_inverter = Inverter(**inverter.model_dump())
        db.add(db_inverter)
        db.commit()
        db.refresh(db_inverter)
        return db_inverter
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverter oluşturulurken hata oluştu: {str(e)}"
        )

@router.get("/inverters/", response_model=List[InverterSchema])
def read_inverters(
    skip: int = Query(0, description="Atlanacak kayıt sayısı"),
    limit: int = Query(100, description="Alınacak maksimum kayıt sayısı"),
    db: Session = Depends(get_db)
):
    """Tüm inverterleri listeler."""
    try:
        inverters = db.query(Inverter).offset(skip).limit(limit).all()
        return inverters
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverterler listelenirken hata oluştu: {str(e)}"
        )

@router.get("/inverters/{inverter_id}", response_model=InverterSchema)
def read_inverter(inverter_id: int, db: Session = Depends(get_db)):
    """Belirli bir inverteri ID'sine göre getirir."""
    try:
        inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
        if inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        return inverter
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverter bilgisi alınırken hata oluştu: {str(e)}"
        )

@router.put("/inverters/{inverter_id}", response_model=InverterSchema)
def update_inverter(
    inverter_id: int, 
    inverter: InverterUpdate, 
    db: Session = Depends(get_db)
):
    """Inverter bilgilerini günceller."""
    try:
        db_inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
        if db_inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        
        update_data = inverter.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_inverter, key, value)
        
        db.commit()
        db.refresh(db_inverter)
        return db_inverter
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverter güncellenirken hata oluştu: {str(e)}"
        )

@router.delete("/inverters/{inverter_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inverter(inverter_id: int, db: Session = Depends(get_db)):
    """Inverter kaydını siler."""
    try:
        db_inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
        if db_inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        
        db.delete(db_inverter)
        db.commit()
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverter silinirken hata oluştu: {str(e)}"
        )

@router.get("/inverters/{inverter_id}/data", response_model=InverterWithData)
def read_inverter_with_data(
    inverter_id: int, 
    skip: int = Query(0, description="Atlanacak kayıt sayısı"),
    limit: int = Query(100, description="Alınacak maksimum kayıt sayısı"),
    db: Session = Depends(get_db)
):
    """Inverter ve ilişkili ölçüm verilerini sayfalı şekilde getirir."""
    try:
        inverter = db.query(Inverter).filter(Inverter.id == inverter_id).first()
        if inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        
        # Toplam veri sayısını al
        total_records = db.query(InverterData).filter(InverterData.inverter_id == inverter_id).count()
        
        # Sayfalı verileri al
        inverter_data = db.query(InverterData).filter(
            InverterData.inverter_id == inverter_id
        ).order_by(
            InverterData.timestamp.desc()
        ).offset(skip).limit(limit).all()
        
        # Şemaya uygun şekilde verileri ata
        result = InverterWithData.model_validate(inverter)
        result.data = inverter_data
        result.pagination = {
            "total_records": total_records,
            "skip": skip,
            "limit": limit,
            "has_more": total_records > (skip + limit)
        }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverter verileri alınırken hata oluştu: {str(e)}"
        )

@router.post("/inverter-data/", response_model=InverterDataSchema, status_code=status.HTTP_201_CREATED)
def create_inverter_data(inverter_data: InverterDataCreate, db: Session = Depends(get_db)):
    """Yeni inverter ölçüm verisi ekler."""
    try:
        # Inverter'ın varlığını kontrol et
        inverter = db.query(Inverter).filter(Inverter.id == inverter_data.inverter_id).first()
        if inverter is None:
            raise HTTPException(status_code=404, detail="Inverter bulunamadı")
        
        db_inverter_data = InverterData(**inverter_data.model_dump())
        db.add(db_inverter_data)
        db.commit()
        db.refresh(db_inverter_data)
        return db_inverter_data
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inverter verisi eklenirken hata oluştu: {str(e)}"
        ) 