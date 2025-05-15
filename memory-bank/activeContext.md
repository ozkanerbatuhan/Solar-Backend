# Aktif Bağlam

## Mevcut Odak
API rotaları ve model eğitimi süreçlerinin yapılandırılması ve entegrasyonu. Veritabanına veri yükleme, model eğitimi, tahminleme ve sonuçların API üzerinden sunulması için gerekli bileşenlerin oluşturulması.

## Son Değişiklikler
- Proje dizin yapısı oluşturuldu
- Veritabanı modelleri (Inverter, InverterData, InverterPrediction, Model) tanımlandı
- Pydantic şemaları ile veri doğrulama yapısı kuruldu
- API endpoint'leri (inverter, tahmin ve model rotaları) oluşturuldu
- Tahmin servisi için temel yapı hazırlandı

## Yeni Gereksinimler
1. CSV dosyası ile inverter çıktı verilerini sisteme yükleme endpoint'i
2. Open-meteo API'sinden hava durumu verilerini çekme ve veritabanına kaydetme işlevselliği
3. Model eğitim sürecinin yapılandırılması (2 aşamalı):
   - İlk eğitim: %70 eğitim, %30 test ile model metrikleri hesaplama
   - İkinci eğitim: Tüm veri ile final model oluşturma
4. Gelecek hava durumu verileri ile tahmin yapma ve veritabanına kaydetme
5. 8 inverterın gerçek ve tahmin değerlerini API üzerinden sunma

## Bir Sonraki Adımlar
1. CSV veri yükleme endpoint'inin geliştirilmesi
2. Hava durumu API entegrasyonunun yapılması
3. Model eğitim servisinin oluşturulması
4. Tahmin pipeline'ının kurulması
5. Ana uygulama giriş noktasının (main.py) oluşturulması
6. API rotalarının düzenlenmesi ve ana uygulamaya entegrasyonu
7. Veritabanı migrasyon scriptlerinin hazırlanması

## Açık Sorular
- Hava durumu verileri ne sıklıkla çekilecek?
- Model eğitim işlemi ne sıklıkla ve nasıl tetiklenecek (manuel/otomatik)?
- Tahmin sonuçları için ne kadar ileriye tahmin yapılacak?
- Hava durumu API'sinin limitleri nedir? Çok fazla istek göndermemek için önlemler alınmalı mı?
- main.py dosyasından ne kadarı yeniden kullanılacak ve ne kadarı değiştirilecek?

## Güncel Durum
Temel API ve veritabanı yapısı kuruldu, şimdi yeni gereksinimler doğrultusunda API rotalarının düzenlenmesi ve model eğitimi/tahmin süreçlerinin entegrasyonu gerekiyor. 