# Güneş Enerjisi İnverter Güç Çıkışı Tahmin API Projesi

## Proje Özeti
Bu proje, 8 farklı güneş enerjisi inverterının güç çıktısını tahmin eden makine öğrenimi modellerini içeren bir API geliştirmeyi amaçlamaktadır. API, FastAPI framework'ü kullanılarak geliştirilecek ve PostgreSQL veritabanı ile entegre edilecektir. Proje, Docker ve Docker Compose ile containerize edilmiş olup, farklı ortamlarda tutarlı bir şekilde çalıştırılabilmektedir.

## Ana Hedefler
- 8 farklı inverterin güç çıktısı verilerini tahmin eden 8 farklı modelin sonuçlarını servis eden bir API oluşturmak
- Web sitesinden gelen yeni verileri veritabanına kaydetmek
- Veritabanındaki mevcut ve yeni verilerle modelleri periyodik olarak yeniden eğitmek
- Web uygulaması için gerekli veri akışını sağlamak
- Containerize edilmiş bir ortam sunarak kolay dağıtım ve ölçeklendirme imkanı sağlamak

## Teknoloji Yığını
- **Backend**: Python FastAPI
  - SQLAlchemy ORM
  - Pydantic veri doğrulama
  - Asenkron API işlemleri
- **Veritabanı**: PostgreSQL
  - SQLAlchemy ile bağlantı
  - Alembic ile migrasyon yönetimi (planlama aşamasında)
- **Makine Öğrenimi**:
  - Scikit-learn, pandas, numpy, joblib
  - Model versiyonlama
  - Dinamik model yükleme
- **Containerization**:
  - Docker
  - Docker Compose
  - Multi-container yapılandırma (API ve veritabanı)

## Proje İlerleme Durumu
- ✅ Proje yapısı oluşturuldu
- ✅ Veritabanı modelleri tasarlandı
- ✅ API endpoint'leri oluşturuldu
- ✅ Temel tahmin servisi yapısı kuruldu
- ✅ PostgreSQL veritabanı bağlantısı yapılandırıldı (Docker)
- ✅ Docker containerization tamamlandı
- 🔜 Test verileriyle veritabanının doldurulması
- 🔜 Docker ortamında makine öğrenimi modeli entegrasyonu
- 🔜 Docker imajlarının optimizasyonu

## Proje Kapsamı
1. API geliştirme ve endpoint tasarımı ✅
2. Veritabanı yapısı ve veri modellerinin oluşturulması ✅
3. Docker containerization ✅
4. Mevcut model eğitim kodlarının entegrasyonu 🔜
5. Otomatik model yeniden eğitim mekanizmalarının oluşturulması 🔜
6. Web uygulaması ile API arasındaki veri alışverişinin sağlanması 🔜
7. Docker imajlarının optimizasyonu 🔜

## Teknik Ayrıntılar
- API, `/api/v1` prefix'i ile servis edilecektir
- Veritabanı modelleri ilişkisel yapıda tasarlanmıştır
- Tahmin servisi asenkron yapıda çalışacaktır
- Model eğitimi API üzerinden tetiklenebilecektir
- PostgreSQL bağlantısı Docker Compose aracılığıyla yapılandırılmıştır
- API http://localhost:8000 üzerinden erişilebilir
- Swagger dokümantasyonu http://localhost:8000/docs adresinde mevcuttur 