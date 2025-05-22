import uuid
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Coroutine
from datetime import datetime, timedelta
import json

# Logger yapılandırması
logger = logging.getLogger(__name__)

class JobManager:
    """
    Tüm job'ları merkezi olarak yöneten servis.
    Farklı tipteki job'lar (veri yükleme, model eğitimi, tahmin) için ortak bir arayüz sağlar.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton pattern uygulaması"""
        if cls._instance is None:
            cls._instance = super(JobManager, cls).__new__(cls)
            cls._instance.jobs = {}
            cls._instance.job_history = {}  # Tamamlanan işleri geçici olarak saklar
            cls._instance.max_history = 100  # Saklanacak maksimum tamamlanmış iş sayısı
        return cls._instance
    
    def create_job(self, job_type: str, params: Dict[str, Any], description: str = "") -> str:
        """
        Yeni bir job oluşturur.
        
        Args:
            job_type: Job tipi ("data_upload", "model_training", "prediction", vb.)
            params: Job parametreleri
            description: Job açıklaması
            
        Returns:
            job_id: Oluşturulan job'ın ID'si
        """
        job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow()
        
        self.jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "created",
            "progress": 0,
            "params": params,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "logs": [],
            "result": None,
            "error": None
        }
        
        logger.info(f"Job oluşturuldu: {job_id} ({job_type})")
        self._add_log(job_id, "Job oluşturuldu")
        
        return job_id
    
    async def start_job(self, job_id: str, task_func: Callable[[str, Callable], Coroutine]) -> str:
        """
        Bir job'ı başlatır.
        
        Args:
            job_id: Job ID
            task_func: Asenkron job fonksiyonu, (job_id, update_callback) parametrelerini alır
            
        Returns:
            job_id: Başlatılan job'ın ID'si
        """
        if job_id not in self.jobs:
            logger.error(f"Job bulunamadı: {job_id}")
            return None
        
        if self.jobs[job_id]["status"] in ["running", "completed"]:
            logger.warning(f"Job zaten {self.jobs[job_id]['status']} durumunda: {job_id}")
            return job_id
        
        self.jobs[job_id]["status"] = "running"
        self.jobs[job_id]["started_at"] = datetime.utcnow()
        self.jobs[job_id]["updated_at"] = datetime.utcnow()
        
        logger.info(f"Job başlatıldı: {job_id}")
        self._add_log(job_id, "Job başlatıldı")
        
        # Job'ı arka planda çalıştır
        asyncio.create_task(self._run_job(job_id, task_func))
        
        return job_id
    
    async def _run_job(self, job_id: str, task_func: Callable[[str, Callable], Coroutine]) -> None:
        """
        Job'ı arka planda çalıştırır.
        
        Args:
            job_id: Job ID
            task_func: Asenkron job fonksiyonu
        """
        try:
            # Job fonksiyonunu çalıştır
            result = await task_func(job_id, self.update_job)
            
            # Başarılı tamamlanma
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            self.jobs[job_id]["updated_at"] = datetime.utcnow()
            
            logger.info(f"Job tamamlandı: {job_id}")
            self._add_log(job_id, "Job başarıyla tamamlandı")
            
        except Exception as e:
            # Hata durumu
            import traceback
            error_trace = traceback.format_exc()
            
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["error_trace"] = error_trace
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            self.jobs[job_id]["updated_at"] = datetime.utcnow()
            
            logger.error(f"Job başarısız oldu: {job_id}, Hata: {str(e)}")
            self._add_log(job_id, f"Job başarısız oldu: {str(e)}", level="ERROR")
        
        finally:
            # Tamamlanan job'ı geçmişe taşı
            self._move_to_history(job_id)
    
    def update_job(self, job_id: str, status: Optional[str] = None, 
                  progress: Optional[int] = None, message: Optional[str] = None,
                  result: Optional[Any] = None, level: str = "INFO") -> bool:
        """
        Job durumunu günceller.
        
        Args:
            job_id: Job ID
            status: Job durumu
            progress: İlerleme yüzdesi (0-100)
            message: Güncelleme mesajı
            result: Job sonucu
            level: Log seviyesi
            
        Returns:
            bool: Güncelleme başarılı mı?
        """
        if job_id not in self.jobs:
            logger.warning(f"Güncelleme için job bulunamadı: {job_id}")
            return False
        
        if status:
            self.jobs[job_id]["status"] = status
        
        if progress is not None:
            self.jobs[job_id]["progress"] = min(100, max(0, progress))
        
        if result is not None:
            self.jobs[job_id]["result"] = result
        
        if message:
            self._add_log(job_id, message, level)
        
        self.jobs[job_id]["updated_at"] = datetime.utcnow()
        return True
    
    def _add_log(self, job_id: str, message: str, level: str = "INFO") -> None:
        """
        Job'a log ekler.
        
        Args:
            job_id: Job ID
            message: Log mesajı
            level: Log seviyesi
        """
        if job_id not in self.jobs:
            return
        
        log_entry = {
            "timestamp": datetime.utcnow(),
            "level": level,
            "message": message
        }
        
        self.jobs[job_id]["logs"].append(log_entry)
        
        # Log seviyesine göre logger'a da ekle
        if level == "ERROR":
            logger.error(f"Job {job_id}: {message}")
        elif level == "WARNING":
            logger.warning(f"Job {job_id}: {message}")
        else:
            logger.info(f"Job {job_id}: {message}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Job durumunu döndürür.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict: Job durumu
        """
        # Aktif job'larda ara
        if job_id in self.jobs:
            return self.jobs[job_id].copy()
        
        # Geçmiş job'larda ara
        if job_id in self.job_history:
            return self.job_history[job_id].copy()
        
        # Bulunamadı
        return {
            "id": job_id,
            "status": "not_found",
            "message": "Belirtilen ID'ye sahip bir job bulunamadı"
        }
    
    def get_active_jobs(self, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Aktif job'ları listeler.
        
        Args:
            job_type: Filtrelenecek job tipi (isteğe bağlı)
            
        Returns:
            List[Dict]: Aktif job'ların listesi
        """
        if job_type:
            return [job.copy() for job_id, job in self.jobs.items() 
                   if job["type"] == job_type]
        else:
            return [job.copy() for job_id, job in self.jobs.items()]
    
    def get_job_history(self, job_type: Optional[str] = None, 
                       limit: int = 20) -> List[Dict[str, Any]]:
        """
        Tamamlanmış job geçmişini listeler.
        
        Args:
            job_type: Filtrelenecek job tipi (isteğe bağlı)
            limit: Maksimum dönecek job sayısı
            
        Returns:
            List[Dict]: Tamamlanmış job'ların listesi
        """
        history = []
        
        for job_id, job in self.job_history.items():
            if job_type is None or job["type"] == job_type:
                history.append(job.copy())
        
        # Tarihe göre sıralama (en yeni önce)
        history.sort(key=lambda x: x.get("completed_at", datetime.min), reverse=True)
        
        return history[:limit]
    
    def _move_to_history(self, job_id: str) -> None:
        """
        Tamamlanmış job'ı geçmişe taşır.
        
        Args:
            job_id: Job ID
        """
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        if job["status"] in ["completed", "failed"]:
            # Job'ı geçmişe taşı
            self.job_history[job_id] = job
            
            # Aktif job'lardan kaldır
            del self.jobs[job_id]
            
            # Geçmiş limit kontrolü
            if len(self.job_history) > self.max_history:
                # En eski job'ı bul ve sil
                oldest_job_id = min(self.job_history.keys(), 
                                  key=lambda k: self.job_history[k].get("completed_at", datetime.max))
                del self.job_history[oldest_job_id]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Çalışan bir job'ı iptal eder.
        NOT: Gerçek iptal işlemi job fonksiyonu tarafından kontrol edilmelidir.
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: İptal işlemi başarılı mı?
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job["status"] == "running":
            job["status"] = "cancelling"
            job["updated_at"] = datetime.utcnow()
            self._add_log(job_id, "Job iptal isteği alındı", "WARNING")
            logger.warning(f"Job iptal isteği: {job_id}")
            return True
        
        return False
    
    def cleanup_old_jobs(self, hours: int = 24) -> int:
        """
        Belirli bir süreden daha eski tamamlanmış job'ları temizler.
        
        Args:
            hours: Maksimum saat cinsinden yaş
            
        Returns:
            int: Temizlenen job sayısı
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        jobs_to_remove = []
        
        for job_id, job in self.job_history.items():
            completed_at = job.get("completed_at")
            if completed_at and completed_at < cutoff_time:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.job_history[job_id]
        
        logger.info(f"{len(jobs_to_remove)} eski job temizlendi")
        return len(jobs_to_remove)

# Global job manager instance
job_manager = JobManager()

def get_job_manager() -> JobManager:
    """
    Global job manager instance'ını döndürür.
    """
    return job_manager 