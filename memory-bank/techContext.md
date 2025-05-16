# Teknik Bağlam

## Kullanılan Teknolojiler

### Backend
- **Python**: Ana programlama dili
- **FastAPI**: API framework'ü
  - Asenkron işleme
  - Otomatik API dokümantasyonu (Swagger/OpenAPI)
  - Pydantic ile veri doğrulama
  - Path ve query parametreler
  - Dependency Injection sistemi
- **SQLAlchemy**: ORM (Object-Relational Mapping)
  - Deklaratif model tanımlamaları
  - İlişkisel veritabanı işlemleri
  - Session yönetimi
- **Alembic**: Veritabanı migrasyonları (henüz yapılandırılmadı)
- **Pydantic**: Veri doğrulama ve serileştirme
  - Şema tanımlamaları
  - Validasyon kuralları
  - Model dönüşümleri
- **Uvicorn/Gunicorn**: ASGI sunucu
- **Docker/Docker Compose**: Konteynerizasyon
  - Uygulama ve veritabanı konteynerleri
  - Ortam bağımsız çalışma imkanı
  - Kolay dağıtım ve ölçeklendirme

### Veritabanı
- **PostgreSQL**: Ana veritabanı sistemi
  - JSON veri tipi desteği (inverter ölçümleri ve özellikler için)
  - Zaman serisi veri depolama
  - İlişkisel tablo yapıları
  - Docker konteyneri içinde çalıştırma

### Makine Öğrenimi
- Tahmin servisi için temel yapı oluşturuldu (placeholder)
- Modellerin versiyonlanması için veritabanı şeması hazırlandı
- Jolib özelliği modellerin disk üzerinde saklanması için düşünülmüş
- Gerçek model entegrasyonu ileri aşamada yapılacak

### İmplementasyon Detayları
- API için toplam 3 ana rota grubu oluşturuldu: 
  - `inverter_routes.py`: İnverterlerle ilgili CRUD işlemleri
  - `prediction_routes.py`: Tahminlerle ilgili işlemler
  - `model_routes.py`: ML modellerinin yönetimi
- Veritabanı için 4 ana model tanımlandı:
  - `Inverter`: İnverter bilgileri
  - `InverterData`: İnverter ölçüm verileri
  - `InverterPrediction`: Tahmin sonuçları
  - `Model`: ML model metadataları
- Config yapısı ile uygulama ayarları merkezileştirildi
- Veri validasyonu için Pydantic şemaları kullanıldı
- Containerizasyon için Docker ve Docker Compose yapılandırması eklendi

## Teknik Kısıtlamalar
- PostgreSQL veritabanı bağlantısı Docker konteynerinde 5432 portu üzerinden yapılıyor
- Makine öğrenimi modellerinin disk üzerinde saklanabilmesi için ML klasörüne disk erişimi
- API'nin performansı veri boyutuyla orantılı şekilde optimize edilmeli
- Büyük veri aktarımında bellek yönetimi önemli

## Geliştirme Ortamı Kurulumu
### Docker ile Kurulum (Önerilen)
```bash
# Docker Compose ile projeyi başlatma
docker-compose up -d

# API'ye erişim
# http://localhost:8000
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Manuel Kurulum
```bash
# Virtual environment oluşturma
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükleme
pip install -r requirements.txt

# Geliştirme sunucusunu başlatma
uvicorn app.main:app --reload
```

## Uygulama Bağımlılıkları
Aşağıdaki paketler requirements.txt dosyasında tanımlanmıştır:

```
fastapi==0.103.1
uvicorn==0.23.2
sqlalchemy==2.0.21
psycopg2-binary==2.9.7
pydantic==2.3.0
pydantic-settings==2.0.3
python-dotenv==1.0.0
httpx==0.24.1
pandas==2.1.0
numpy==1.25.2
scikit-learn==1.3.0
joblib==1.3.2
python-multipart==0.0.6
```

## Veritabanı Bağlantı Bilgileri
Veritabanı bağlantısı Docker Compose ile yönetilen bir PostgreSQL instance'ı üzerinde sağlanmaktadır:
```
POSTGRES_SERVER=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=solar_db
POSTGRES_PORT=5432
```

## Dış Bağımlılıklar
- PostgreSQL Veritabanı (Docker ile yapılandırıldı)
- Python 3.8+
- Docker ve Docker Compose
- Makine öğrenimi modelleri (gelecekte entegre edilecek) 