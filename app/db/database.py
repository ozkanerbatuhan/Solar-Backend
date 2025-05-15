from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from app.core.config import settings

# PostgreSQL veritabanı bağlantı URL'si
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# SQLAlchemy engine oluştur - veritabanı olmadığında SQLite kullan
try:
    # PostgreSQL bağlantısını deneyecek
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    print("PostgreSQL bağlantısı kullanılıyor.")
except Exception as e:
    # Hata durumunda geçici çözüm
    print(f"PostgreSQL bağlantısı yapılamadı: {e}")
    print("Geçici olarak bellek içi SQLite kullanılıyor.")
    
    # Bellek içi SQLite kullan
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    
    # In-memory veritabanını oluşturmak için Base.metadata.create_all() main.py'de çağrılacak

# Oturum fabrikası oluştur
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Temel model sınıfı
Base = declarative_base()

# Bağımlılık olarak veritabanı oturumu
def get_db():
    """
    FastAPI bağımlılığı olarak veritabanı oturumu sağlar.
    Her istek için yeni bir oturum oluşturur ve işlem tamamlandığında kapatır.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 