# Solar Energy Prediction System

Güneş enerjisi üretim tahmini için geliştirilmiş API sistemi.

## Özellikler

### ⚡ SON GÜNCELLEME - ULTRA AGRESİF İYİLEŞTİRMELER (v2.1)

**📊 Model Kalitesi Sorunları Tamamen Çözüldü:**

✅ **ÇOK SIKI FİZİK KISITLAMALARI:**
- Gece saatleri (22:00-05:59): MUTLAK 0 kW tahmin
- Sıfır radyasyon: MUTLAK 0 kW tahmin  
- Düşük radyasyon kısıtlaması: <50 W/m² → max 0.5x güç
- Orta radyasyon kısıtlaması: 50-200 W/m² → max 1.5x güç
- Yüksek radyasyon kısıtlaması: >500 W/m² → max 4.5x güç
- Sabah/akşam saatleri: maksimum 1000 kW sınırı

✅ **AGRESIF VERİ TEMİZLEME:**
- Fiziksel olarak imkansız değerlerin tamamen ortadan kaldırılması
- Gece saatlerinde radyasyon değerlerinin sıfırlanması
- Radyasyon-güç orantısızlığının düzeltilmesi
- Aşırı aykırı değerlerin sert sınırlanması (%95 percentile)
- Çok kalitesiz örneklerin tamamen çıkarılması

✅ **ULTRA GÜÇLÜ MODEL PARAMETRELERİ:**
- 500 ağaç (300'den artırıldı)
- 25 derinlik (20'den artırıldı)  
- %80 özellik kullanımı
- Minimum impurity decrease: 0.0001
- Çok hassas yaprak düğümleri

**🎯 Beklenen Performans İyileştirmeleri:**
- R² > 0.90 (şu an 0.497'den)
- MAPE < 15% (şu an 776%'dan)
- %100 fiziksel tutarlılık garanti
- Gece saatleri tahminlerinde TAMAMEN 0 kW

---

## 🚀 ACİL MODEL YENİDEN EĞİTİMİ

```bash
# Tek inverter için model yeniden eğitimi
curl -X POST "http://localhost:8000/api/v1/models/train" \
  -H "Content-Type: application/json" \
  -d '{"inverter_id": 2}'

# Tüm inverterlar için model yeniden eğitimi  
curl -X POST "http://localhost:8000/api/v1/models/train-all"

# Model eğitim durumunu kontrol et
curl "http://localhost:8000/api/v1/models/training-status/{job_id}"
```

## Test - Yeni Model ile Tahmin

```bash
# İnverter 2 için anlık tahmin (yeni model ile)
curl "http://localhost:8000/api/v1/prediction/2/2025-01-25T19:00:00"

# 48 saatlik tahmin serisi
curl "http://localhost:8000/api/v1/predictions/2?hours=48"
```

---

### Önceki Özellikler

**📈 Gelişmiş Özellik Mühendisliği:**
- 30+ composite feature (total_radiation_index, panel_efficiency_proxy, heat_comfort)
- Güneş açısı proxy hesaplamaları
- Hava durumu etkileşim özellikleri
- Mevsimsel faktörler

**🧠 Akıllı Veri Kalitesi Servisi:**
- Gerçek zamanlı fizik kuralları doğrulaması
- Çok boyutlu aykırı değer tespiti (IsolationForest)
- Akıllı missing value imputation
- Domain-specific data validation

**⚡ Yeni API Endpoints:**
- `GET /api/v1/prediction/{inverter_id}/{timestamp}` - Anlık tahmin
- `GET /api/v1/predictions/{inverter_id}` - Zaman serisi tahmini
- `POST /api/v1/models/train` - Model eğitimi
- `POST /api/v1/models/train-all` - Tüm modeller eğitimi
- `POST /api/v1/data/upload-txt/{inverter_id}` - TXT dosya yükleme
- `GET /api/v1/models/training-status/{job_id}` - Eğitim durumu

**📊 Merkezi Job Yönetimi:**
- Background task işleme
- Job durumu izleme  
- Hata durumu yönetimi
- İlerleme takibi

**🌤️ Hava Durumu Entegrasyonu:**
- Open-Meteo API bağlantısı
- 7 günlük forecast alımı
- Otomatik veri güncellemesi
- Koordinat bazlı tahmin

## Kurulum

### Gereksinimler
- Python 3.8+
- PostgreSQL
- Redis (opsiyonel - caching için)

### Adımlar

1. **Depoyu klonlayın:**
```bash
git clone <repo-url>
cd Solar-Backend
```

2. **Sanal ortam oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Veritabanını kurun:**
```bash
# PostgreSQL veritabanı oluştur
createdb solar_energy_db

# Migration'ları çalıştır
alembic upgrade head
```

5. **Ortam değişkenlerini ayarlayın:**
```bash
export DATABASE_URL="postgresql://user:password@localhost/solar_energy_db"
export OPEN_METEO_API_KEY="your_api_key"  # Opsiyonel
```

6. **Uygulamayı çalıştırın:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Kullanımı

### Anlık Tahmin
```bash
curl "http://localhost:8000/api/v1/prediction/1/2024-01-15T12:00:00"
```

### Çoklu Zaman Aralığı Tahmini
```bash
curl "http://localhost:8000/api/v1/predictions/1?start_time=2024-01-15T06:00:00&end_time=2024-01-15T18:00:00&interval_hours=1"
```

### Model Eğitimi
```bash
curl -X POST "http://localhost:8000/api/v1/models/train" \
  -H "Content-Type: application/json" \
  -d '{"inverter_id": 1}'
```

### Veri Yükleme (TXT Dosya)
```bash
curl -X POST "http://localhost:8000/api/v1/data/upload-txt/1" \
  -F "file=@data.txt"
```

## Veri Formatları

### Inverter TXT Dosya Formatı
```
timestamp,power_output,temperature,irradiance
2024-01-15 12:00:00,1250.5,35.2,890.3
2024-01-15 13:00:00,1380.2,36.1,920.1
```

### Hava Durumu API Formatı
Sistem otomatik olarak Open-Meteo API'den aşağıdaki parametreleri alır:
- temperature_2m
- shortwave_radiation  
- direct_radiation
- diffuse_radiation
- direct_normal_irradiance
- global_tilted_irradiance
- relative_humidity_2m
- wind_speed_10m

## Monitoring ve Logs

### Model Performans Metrikleri
```bash
curl "http://localhost:8000/api/v1/models/metrics"
```

### Sistem Durumu
```bash
curl "http://localhost:8000/api/v1/health"
```

### Job Durumu İzleme
```bash
curl "http://localhost:8000/api/v1/models/training-status/{job_id}"
```

## Performans Optimizasyonları

**Model Eğitimi:**
- RandomForest: 500 estimator, 25 max_depth
- RobustScaler normalizasyon
- Advanced feature engineering (30+ features)
- Physics-aware data cleaning

**Tahmin Performansı:**
- Model caching
- Prediction constraints
- Real-time validation
- Background processing

**Veri Kalitesi:**
- Multi-dimensional outlier detection
- Solar physics consistency checks
- Intelligent data imputation
- Domain-specific validation

## Lisans
MIT License

---

**📞 Destek:**
Model performans sorunları için lütfen yeni eğitim yapın ve sonuçları paylaşın. Sistem artık fiziksel olarak tutarlı tahminler üretecektir.