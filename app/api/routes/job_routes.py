from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.database import get_db
from app.services.job_manager_service import get_job_manager, JobManager
from app.schemas.job import JobStatus, JobList, JobDetail, JobSummary

router = APIRouter()

@router.get("/status/{job_id}", response_model=JobDetail)
async def get_job_status(
    job_id: str,
    include_logs: bool = Query(False, description="Log kayıtlarını da döndür"),
    db: Session = Depends(get_db)
):
    """
    Belirli bir job'ın durumunu döndürür.
    """
    job_manager = get_job_manager()
    job_status = job_manager.get_job_status(job_id)
    
    if job_status.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID: {job_id} olan job bulunamadı"
        )
    
    # Log kayıtları istenmediyse, yalnızca son log mesajını dahil et
    if not include_logs and "logs" in job_status:
        # Son log mesajını sakla
        logs = job_status.get("logs", [])
        last_log = logs[-1] if logs else None
        job_status["logs"] = [last_log] if last_log else []
    
    return job_status

@router.get("/active", response_model=JobList)
async def get_active_jobs(
    job_type: Optional[str] = Query(None, description="Job tipi filtresi"),
    db: Session = Depends(get_db)
):
    """
    Aktif job'ları listeler.
    """
    job_manager = get_job_manager()
    active_jobs = job_manager.get_active_jobs(job_type)
    
    # Özet bilgileri oluştur
    job_summaries = []
    for job in active_jobs:
        # Özet için log kayıtlarını ve detayları hariç tut
        summary = {k: v for k, v in job.items() if k != "logs" and k != "error_trace"}
        
        # Son log mesajını ekle
        logs = job.get("logs", [])
        if logs:
            summary["last_message"] = logs[-1].get("message", "")
        else:
            summary["last_message"] = ""
        
        job_summaries.append(summary)
    
    return {
        "total": len(job_summaries),
        "jobs": job_summaries
    }

@router.get("/history", response_model=JobList)
async def get_job_history(
    job_type: Optional[str] = Query(None, description="Job tipi filtresi"),
    limit: int = Query(20, description="Maksimum job sayısı", ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Tamamlanmış job'ların geçmişini listeler.
    """
    job_manager = get_job_manager()
    job_history = job_manager.get_job_history(job_type, limit)
    
    # Özet bilgileri oluştur
    job_summaries = []
    for job in job_history:
        # Özet için log kayıtlarını ve detayları hariç tut
        summary = {k: v for k, v in job.items() if k != "logs" and k != "error_trace"}
        
        # Son log mesajını ekle
        logs = job.get("logs", [])
        if logs:
            summary["last_message"] = logs[-1].get("message", "")
        else:
            summary["last_message"] = ""
        
        job_summaries.append(summary)
    
    return {
        "total": len(job_summaries),
        "jobs": job_summaries
    }

@router.post("/cancel/{job_id}", response_model=JobStatus)
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Çalışan bir job'ı iptal etmeye çalışır.
    """
    job_manager = get_job_manager()
    job_status = job_manager.get_job_status(job_id)
    
    if job_status.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID: {job_id} olan job bulunamadı"
        )
    
    if job_status.get("status") != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job şu anda çalışmıyor. Mevcut durum: {job_status.get('status')}"
        )
    
    success = job_manager.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job iptal edilemedi"
        )
    
    return {
        "id": job_id,
        "status": "cancelling",
        "message": "Job iptal isteği alındı. İşlem kısa süre içinde durdurulacak."
    }

@router.post("/cleanup", response_model=dict)
async def cleanup_old_jobs(
    hours: int = Query(24, description="Temizlenecek saatlerin sayısı", ge=1),
    db: Session = Depends(get_db)
):
    """
    Belirli bir süreden daha eski job'ları temizler.
    """
    job_manager = get_job_manager()
    removed_jobs = job_manager.cleanup_old_jobs(hours)
    
    return {
        "success": True,
        "message": f"{removed_jobs} job temizlendi",
        "removed_jobs": removed_jobs
    } 