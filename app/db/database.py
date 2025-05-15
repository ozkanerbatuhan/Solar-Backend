from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# PostgreSQL veritabanı bağlantı URL'si
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# SQLAlchemy engine oluştur
engine = create_engine(SQLALCHEMY_DATABASE_URL)

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