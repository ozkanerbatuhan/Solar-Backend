# Proje İlerleme Durumu

## Tamamlananlar
- Hafıza bankası başlatıldı
- Proje gereksinimleri belirlendi
- Mimari yapı ve teknoloji yığını kararlaştırıldı
- Proje dosya yapısı oluşturuldu
- FastAPI uygulaması için gerekli paket yapısı hazırlandı
- PostgreSQL veritabanı şemaları tasarlandı ve modeller oluşturuldu
- Veri validasyonu için Pydantic şemaları tanımlandı
- API endpoint'leri (inverter, tahmin, model) oluşturuldu
- Tahmin servisi için temel yapı hazırlandı
- Veritabanı bağlantısı konfigürasyonu tamamlandı

## Devam Edenler
- API rotalarının yeni gereksinimlere göre düzenlenmesi

## Yeni Gereksinimler
- CSV veri yükleme endpoint'i
- Open-meteo API entegrasyonu
- İki aşamalı model eğitim süreci
- Tahmin sonuçlarının veritabanında saklanması
- Gerçek ve tahmin değerlerinin API üzerinden sunulması

## Yapılacaklar
- [x] Proje dosya yapısının kurulması
- [x] FastAPI uygulamasının temel kurulumu
- [x] PostgreSQL veritabanı şemalarının tasarlanması
- [x] Temel API endpoint'lerinin oluşturulması
- [x] Veritabanı bağlantısının kurulması
- [x] Model yükleme ve tahmin mekanizmasının oluşturulması (temel yapı)
- [x] Veri alma ve kaydetme endpoint'lerinin geliştirilmesi
- [ ] CSV dosyası yükleme endpoint'inin geliştirilmesi
- [ ] Yüklenen CSV verilerini veritabanına işleme mekanizması
- [ ] Open-meteo API ile entegrasyon için servis oluşturulması
- [ ] Hava durumu verilerini veritabanına kaydetme işlevselliği
- [ ] Model eğitimi için servis oluşturulması (main.py adaptasyonu)
- [ ] %70/%30 bölünmüş veri ile model metriklerini hesaplama
- [ ] Tüm veri ile final modellerin eğitilmesi
- [ ] Modellerin disk üzerinde saklanması için yapı oluşturulması
- [ ] Gelecek hava durumu verileri ile tahminleme işlevselliği
- [ ] Tahmin sonuçlarını veritabanında saklama mekanizması
- [ ] 8 inverterın gerçek ve tahmin verilerini sunan API endpoint'leri
- [ ] Ana uygulama dosyasının (main.py) oluşturulması
- [ ] API rotalarının ana uygulamaya bağlanması
- [ ] Veritabanı migrasyon işlemlerinin yapılandırılması (Alembic)
- [ ] Hata yönetimi ve loglama
- [ ] API dokümantasyonu

## Engeller
- PostgreSQL veritabanı bağlantısı kurulmalı ve test edilmeli (DB_PORT değerinin doğru olduğundan emin olunmalı)
- main.py dosyası veritabanı entegrasyonu için adapte edilmeli
- Makine öğrenimi modellerinin disk üzerinde tutarlı bir şekilde saklanması ve yüklenmesi sağlanmalı
- Open-meteo API'nin rate limit'lerine dikkat edilmeli

## Proje Zaman Çizelgesi
- **Aşama 1**: Temel API kurulumu ve veritabanı bağlantısı ✅
- **Aşama 2**: Veri yükleme ve hava durumu API entegrasyonu (devam ediyor)
- **Aşama 3**: Model eğitim sürecinin entegrasyonu 
- **Aşama 4**: Tahminleme işlevselliğinin oluşturulması
- **Aşama 5**: API endpoint'lerinin son haline getirilmesi
- **Aşama 6**: Test ve optimizasyon

## Öncelikli Görevler
1. CSV dosyası yükleme endpoint'i ve veritabanı işleme mekanizması
2. Open-meteo API entegrasyonu
3. Model eğitim sürecinin adaptasyonu
4. Tahminleme ve sonuçları saklama işlevselliği
5. API endpoint'lerinin düzenlenmesi

## Bilinen Sorunlar
- Henüz main.py dosyası oluşturulmadığından API çalışmıyor
- ML modelleri için klasör yapısı henüz oluşturulmadı
- Tahmin servisi şu an sadece rastgele değerler üretiyor, gerçek makine öğrenimi modelleri entegre edilmeli
- Open-meteo API entegrasyonu henüz yapılmadı 