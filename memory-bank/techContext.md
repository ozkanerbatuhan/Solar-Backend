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
- **Uvicorn/Gunicorn**: ASGI sunucu (henüz yapılandırılmadı)

### Veritabanı
- **PostgreSQL**: Ana veritabanı sistemi
  - JSON veri tipi desteği (inverter ölçümleri ve özellikler için)
  - Zaman serisi veri depolama
  - İlişkisel tablo yapıları

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

## Teknik Kısıtlamalar
- PostgreSQL veritabanı bağlantısı (şu an 1234 portu kullanılıyor)
- Makine öğrenimi modellerinin disk üzerinde saklanabilmesi için ML klasörüne disk erişimi
- API'nin performansı veri boyutuyla orantılı şekilde optimize edilmeli
- Büyük veri aktarımında bellek yönetimi önemli

## Geliştirme Ortamı Kurulumu
```bash
# Virtual environment oluşturma
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükleme
pip install -r requirements.txt

# Geliştirme sunucusunu başlatma (henüz yapılandırılmadı)
# uvicorn app.main:app --reload
```

## Uygulama Bağımlılıkları
Aşağıdaki paketler requirements.txt dosyasında tanımlanmıştır:

```
fastapi>=0.104.0
uvicorn>=0.23.2
sqlalchemy>=2.0.23
psycopg2-binary>=2.9.9
alembic>=1.12.1
pydantic>=2.4.2
python-dotenv>=1.0.0
scikit-learn>=1.3.2
pandas>=2.1.2
numpy>=1.26.1
joblib>=1.3.2
```

## Veritabanı Bağlantı Bilgileri
Veritabanı bağlantısı için bir PostgreSQL instance'ı gereklidir. Bağlantı, app/core/config.py içinde yapılandırılmıştır:
```python
DB_USERNAME: str = "postgres"
DB_PASSWORD: str = "postgres"
DB_HOST: str = "localhost"
DB_PORT: str = "1234"  # Doğru port yapılandırılmalı
DB_NAME: str = "solar_db"
```

## Dış Bağımlılıklar
- PostgreSQL Veritabanı (bağlantı doğrulanmalı)
- Python 3.8+
- Makine öğrenimi modelleri (gelecekte entegre edilecek) 