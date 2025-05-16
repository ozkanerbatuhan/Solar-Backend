# Solar Inverter Prediction API

Bu proje, güneş enerjisi sistemindeki 8 farklı inverterin güç çıktılarını tahmin eden bir API geliştirmeyi amaçlar. FastAPI ve PostgreSQL kullanılarak geliştirilmiştir.

## Özellikler

- 8 farklı inverter için güç çıktısı tahmini
- Hava durumu verileri ile entegrasyon (Open-meteo API)
- CSV dosyası ile veri yükleme
- Model eğitimi ve değerlendirme
- Metrik hesaplama (R², MAE, RMSE)

## Kurulum

### Gereksinimler

- Python 3.8+
- PostgreSQL
- Docker ve Docker Compose (Docker ile çalıştırmak için)

### Docker ile Kurulum (Önerilen)

1. Projeyi klonlayın:
   ```
   git clone https://github.com/yourusername/solar-inverter-prediction.git
   cd solar-inverter-prediction
   ```

2. Docker Compose ile uygulamayı başlatın:
   ```
   docker-compose up -d
   ```

3. API'ye http://localhost:8000 adresinden erişebilirsiniz
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Manuel Kurulum

1. Projeyi klonlayın:
   ```
   git clone https://github.com/yourusername/solar-inverter-prediction.git
   cd solar-inverter-prediction
   ```

2. Sanal ortam oluşturun ve etkinleştirin:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. Gereksinimleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

4. `.env` dosyasını oluşturun:
   ```
   POSTGRES_SERVER=localhost
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=solar_db
   POSTGRES_PORT=5432
   ```

5. Uygulamayı çalıştırın:
   ```
   uvicorn app.main:app --reload
   ```

## API Endpoints

### Inverter Verileri

- `GET /api/inverters/`: Tüm inverterleri listeler
- `GET /api/inverters/{inverter_id}`: Belirli bir inverteri getirir
- `GET /api/inverters/{inverter_id}/data`: Belirli bir inverterin verilerini listeler

### Veri Yükleme

- `POST /api/data/upload-csv`: CSV dosyasından veri yükler
- `GET /api/data/statistics`: Veritabanındaki veriler için istatistikler döndürür
- `POST /api/data/fetch-weather-data`: İnverter verileri için hava durumu verilerini çeker

### Hava Durumu

- `POST /api/weather/fetch-current`: Güncel hava durumu verilerini çeker
- `POST /api/weather/fetch-forecast`: Hava durumu tahminlerini çeker
- `GET /api/weather/data`: Hava durumu verilerini listeler
- `GET /api/weather/forecast`: Hava durumu tahminlerini listeler

### Model Eğitimi ve Tahmin

- `POST /api/models/train/{inverter_id}`: Belirli bir inverter için model eğitir
- `POST /api/models/train-all`: Tüm inverterler için model eğitir
- `GET /api/models/metrics/{inverter_id}`: Belirli bir inverter için model metriklerini döndürür
- `GET /api/models/metrics`: Tüm inverterler için model metriklerini döndürür
- `GET /api/models/predict/{inverter_id}`: Belirli bir inverter için güç çıktısı tahmini yapar
- `POST /api/models/predict-bulk`: Birden fazla inverter için tahmin yapar

## Proje Yapısı

```
solar-inverter-prediction/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── data_routes.py
│   │   │   ├── inverter_routes.py
│   │   │   ├── model_routes.py
│   │   │   └── weather_routes.py
│   │   └── __init__.py
│   ├── core/
│   │   ├── config.py
│   │   └── __init__.py
│   ├── db/
│   │   ├── database.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── inverter.py
│   │   ├── model.py
│   │   ├── weather.py
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── data.py
│   │   ├── inverter.py
│   │   ├── model.py
│   │   ├── weather.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── data_import_service.py
│   │   ├── model_training_service.py
│   │   ├── prediction_service.py
│   │   ├── weather_service.py
│   │   └── __init__.py
│   ├── ml/
│   │   └── models/
│   └── main.py
├── .env
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Lisans

MIT 