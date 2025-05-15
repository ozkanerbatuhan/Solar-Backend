# Güneş Enerjisi İnverter Güç Çıkışı Tahmin API Projesi

## Proje Özeti
Bu proje, 8 farklı güneş enerjisi inverterının güç çıktısını tahmin eden makine öğrenimi modellerini içeren bir API geliştirmeyi amaçlamaktadır. API, FastAPI framework'ü kullanılarak geliştirilecek ve PostgreSQL veritabanı ile entegre edilecektir.

## Ana Hedefler
- 8 farklı inverterin güç çıktısı verilerini tahmin eden 8 farklı modelin sonuçlarını servis eden bir API oluşturmak
- Web sitesinden gelen yeni verileri veritabanına kaydetmek
- Veritabanındaki mevcut ve yeni verilerle modelleri periyodik olarak yeniden eğitmek
- Web uygulaması için gerekli veri akışını sağlamak

## Teknoloji Yığını
- **Backend**: Python FastAPI
  - SQLAlchemy ORM
  - Pydantic veri doğrulama
  - Asenkron API işlemleri
- **Veritabanı**: PostgreSQL
  - SQLAlchemy ile bağlantı
  - Alembic ile migrasyon yönetimi
- **Makine Öğrenimi**:
  - Scikit-learn, pandas, numpy, joblib
  - Model versiyonlama
  - Dinamik model yükleme

## Proje İlerleme Durumu
- ✅ Proje yapısı oluşturuldu
- ✅ Veritabanı modelleri tasarlandı
- ✅ API endpoint'leri oluşturuldu
- ✅ Temel tahmin servisi yapısı kuruldu
- ⏳ Uygulama ana dosyası hazırlanması gerekiyor
- ⏳ Veritabanı migrasyon yönetimi yapılandırılmalı
- ⏳ PostgreSQL bağlantısı test edilmeli
- 🔜 Makine öğrenimi model entegrasyonu yapılacak

## Proje Kapsamı
1. API geliştirme ve endpoint tasarımı ✅
2. Veritabanı yapısı ve veri modellerinin oluşturulması ✅
3. Mevcut model eğitim kodlarının entegrasyonu 🔜
4. Otomatik model yeniden eğitim mekanizmalarının oluşturulması
5. Web uygulaması ile API arasındaki veri alışverişinin sağlanması

## Teknik Ayrıntılar
- API, `/api/v1` prefix'i ile servis edilecektir
- Veritabanı modelleri ilişkisel yapıda tasarlanmıştır
- Tahmin servisi asenkron yapıda çalışacaktır
- Model eğitimi API üzerinden tetiklenebilecektir
- PostgreSQL bağlantısı environment veya ayar dosyasından yapılandırılabilecektir 