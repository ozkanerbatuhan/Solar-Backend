import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime
import json
from pvlib import solarposition
import pytz
import sys

# Konum bilgileri
LATITUDE = 37.56
LONGITUDE = 34.12
TIMEZONE = 'Europe/Istanbul'

# Dosya yolları
DATA_PATH = "open-meteo-37.56N34.12E1050m.csv"

# Çıktı dosyalarının saklanacağı klasörler
MODEL_DIR = "trained_models"
LOG_DIR = "logs"
PLOT_DIR = "plots"

# Gerekli klasörleri oluştur
for directory in [MODEL_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Windows konsolunda Türkçe karakter sorununu çözmek için
# UTF-8 encoding ayarla
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Log yapılandırması
def setup_logger(name, log_file, level=logging.INFO):
    # File handler ekleme (UTF-8 encoding ile)
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Konsola da çıktı ver
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

# Ana günlük oluştur
main_logger = setup_logger('main', os.path.join(LOG_DIR, 'main.log'))

def calculate_solar_positions(data, lat=LATITUDE, lon=LONGITUDE, tz=TIMEZONE):
    """Güneş pozisyon bilgilerini hesaplar ve veri çerçevesine ekler."""
    main_logger.info("Güneş pozisyonları hesaplanıyor...")
    timezone = pytz.timezone(tz)
    
    # Datetime sütununu timezone ile lokalize et
    try:
        times = pd.DatetimeIndex(data['datetime']).tz_localize(timezone, ambiguous='raise', nonexistent='raise')
        
        solar_position = solarposition.get_solarposition(times, lat, lon)
        data['solar_elevation'] = solar_position['elevation']
        data['solar_azimuth'] = solar_position['azimuth']
        data['solar_zenith'] = solar_position['zenith']
        main_logger.info("Güneş pozisyonları başarıyla hesaplandı.")
    except Exception as e:
        main_logger.error(f"Güneş pozisyonları hesaplanırken hata oluştu: {e}")
        # Timezone hatası durumunda basitleştirilmiş yaklaşım
        main_logger.info("Basitleştirilmiş güneş pozisyonları hesaplanıyor...")
        for time in data['datetime'].dt.hour.unique():
            data.loc[data['datetime'].dt.hour == time, 'hour_angle'] = (time - 12) * 15
        
        # Basit güneş yüksekliği hesaplama (Kaba tahmin)
        data['solar_elevation'] = 90 - abs(data['hour_angle']) * 90/180
        data['solar_elevation'] = data['solar_elevation'].clip(0, 90)
        data['solar_zenith'] = 90 - data['solar_elevation']
        data['solar_azimuth'] = 180  # Basit varsayım
    
    return data

def load_data():
    """CSV dosyasından hava durumu ve inverter verilerini yükler."""
    main_logger.info("Veriler yükleniyor...")
    
    try:
        
        data = pd.read_csv(DATA_PATH, header=0)
        main_logger.info(f"Veri dosyası yüklendi, şekil: {data.shape}")
        
        # Zaman sütununu datetime formatına çevirme
        data['datetime'] = pd.to_datetime(data['time'])
        
        # 2023-05-01 tarihinden itibaren verileri filtrele (inverter verileri burada başlıyor)
        start_date = '2023-05-01'
        filtered_data = data[data['datetime'] >= start_date].copy()
        main_logger.info(f"2023-05-01 tarihinden itibaren veriler filtrelendi, şekil: {filtered_data.shape}")
        
        # NaN değerleri kontrol etme
        nan_columns = filtered_data.columns[filtered_data.isna().any()].tolist()
        if nan_columns:
            main_logger.warning(f"Aşağıdaki sütunlarda NaN değerler bulundu: {nan_columns}")
            # NaN değerleri doldurma
            filtered_data = filtered_data.fillna(method='ffill').fillna(method='bfill')
            main_logger.info("NaN değerler dolduruldu.")
        
        return filtered_data
    
    except Exception as e:
        main_logger.error(f"Veri yükleme sırasında hata oluştu: {e}")
        raise

def preprocess_data(data):
    """Veri ön işleme adımlarını gerçekleştirir."""
    main_logger.info("Veri ön işleme başlıyor...")
    
    # Tarih-zaman özellikleri oluşturma
    data['hour'] = data['datetime'].dt.hour
    data['month'] = data['datetime'].dt.month
    data['day_of_year'] = data['datetime'].dt.dayofyear
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_year']/365)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_year']/365)
    
    # Güneş pozisyonlarını hesaplama
    data = calculate_solar_positions(data)
    
    # Gündüz saatlerini filtrele (güneş yüksekliği > 0)
    day_data = data[data['solar_elevation'] > 0].copy()
    data_count = len(day_data)
    
    # Eğer gündüz verisi çok az veya hiç yoksa, saatlere göre filtrele
    if data_count < 100:
        main_logger.warning(f"Gündüz verisi çok az ({data_count} satır), saat bazlı filtre kullanılıyor...")
        day_data = data[(data['hour'] >= 6) & (data['hour'] <= 18)].copy()
    
    main_logger.info(f"Gündüz saatleri filtrelendi: {data.shape} -> {day_data.shape}")
    
    # Eğer hala veri yoksa, tüm veriyi kullan
    if len(day_data) == 0:
        main_logger.warning("Gündüz verisi bulunamadı, tüm veri kullanılacak!")
        day_data = data.copy()
    
    # CSV dosyasından inverter sütunlarını otomatik olarak bul
    inverter_columns = [col for col in data.columns if 'INV' in col and 'DayEnergy' in col]
    main_logger.info(f"CSV dosyasından bulunan inverter sütunları: {inverter_columns}")
    
    if not inverter_columns:
        main_logger.error("Inverter sütunları bulunamadı! CSV dosyasını kontrol edin.")
        raise ValueError("Inverter sütunları bulunamadı")
    
    # Her inverter için saatlik enerji üretimini hesapla
    for col in inverter_columns:
        day_data[f'hourly_{col}'] = day_data[col].diff().fillna(0)
        
        # Negatif değerleri sıfıra ayarla (günün başlangıcında veya veri hatasında)
        day_data.loc[day_data[f'hourly_{col}'] < 0, f'hourly_{col}'] = 0
    
    main_logger.info("Veri ön işleme tamamlandı.")
    return day_data, inverter_columns

def train_model(data, inverter_col, features, model_min_r2=0.3):
    """Belirtilen inverter için model eğitir."""
    inverter_name = inverter_col.split('/')[1]
    model_logger = setup_logger(f'model_{inverter_name}', 
                               os.path.join(LOG_DIR, f'model_{inverter_name}.log'))
    
    # Ayrıca bir txt günlük dosyası oluştur
    txt_log_path = os.path.join(LOG_DIR, f'model_{inverter_name}_details.txt')
    with open(txt_log_path, 'w', encoding='utf-8') as txt_log:
        txt_log.write(f"{'='*80}\n")
        txt_log.write(f"{' '*30}İNVERTER {inverter_name} MODELİ DETAYLARI\n")
        txt_log.write(f"{'='*80}\n\n")
    
    target_col = f'hourly_{inverter_col}'
    model_logger.info(f"'{inverter_col}' için model eğitimi başlıyor...")
    model_logger.info(f"Hedef değişken: {target_col}")
    model_logger.info(f"Özellikler: {features}")
    
    # Veri boyutu kontrolü
    if len(data) < 10:
        model_logger.error(f"Veri seti çok küçük: {len(data)} satır. En az 10 satır gerekli.")
        return None, None, None, 0, 0, 0, 0
    
    # Kullanılacak özelliklerin veri setinde olup olmadığını kontrol et
    available_features = [f for f in features if f in data.columns]
    missing_features = [f for f in features if f not in data.columns]
    
    if missing_features:
        model_logger.warning(f"Aşağıdaki özellikler veri setinde bulunamadı: {missing_features}")
        model_logger.info(f"Kullanılabilir özelliklerle devam edilecek: {available_features}")
        features = available_features
    
    # Aykırı değerleri temizle
    q1 = data[target_col].quantile(0.01)
    q3 = data[target_col].quantile(0.99)
    filtered_data = data[(data[target_col] >= q1) & (data[target_col] <= q3)]
    model_logger.info(f"Aykırı değer temizlemeden sonra: {data.shape} -> {filtered_data.shape}")
    
    # Veri analizi ve özellik-hedef incelemesi
    with open(txt_log_path, 'a', encoding='utf-8') as txt_log:
        txt_log.write(f"1. VERİ ANALİZİ\n")
        txt_log.write(f"{'-'*80}\n\n")
        
        # Hedef değişkenin istatistikleri
        txt_log.write(f"1.1 Hedef Değişken: {target_col}\n\n")
        target_stats = filtered_data[target_col].describe()
        txt_log.write(f"    Ortalama   : {target_stats['mean']:.2f}\n")
        txt_log.write(f"    Std Sapma  : {target_stats['std']:.2f}\n")
        txt_log.write(f"    Minimum    : {target_stats['min']:.2f}\n")
        txt_log.write(f"    25%        : {target_stats['25%']:.2f}\n")
        txt_log.write(f"    Medyan     : {target_stats['50%']:.2f}\n")
        txt_log.write(f"    75%        : {target_stats['75%']:.2f}\n")
        txt_log.write(f"    Maksimum   : {target_stats['max']:.2f}\n\n")
        
        # Özellikler hakkında bilgi
        txt_log.write(f"1.2 Kullanılan Özellikler ve NaN Değer Analizi\n\n")
        txt_log.write(f"{'Özellik':<30}{'Ortalama':>12}{'Std':>12}{'Min':>12}{'Maks':>12}{'NaN Sayısı':>12}{'NaN %':>12}\n")
        txt_log.write(f"{'-'*90}\n")
        
        feature_nan_counts = {}
        
        for feat in features:
            if feat in filtered_data.columns:
                feat_stats = filtered_data[feat].describe()
                nan_count = filtered_data[feat].isna().sum()
                nan_percentage = (nan_count / len(filtered_data)) * 100
                feature_nan_counts[feat] = nan_count
                
                txt_log.write(f"{feat:<30}{feat_stats['mean']:>12.2f}{feat_stats['std']:>12.2f}"
                             f"{feat_stats['min']:>12.2f}{feat_stats['max']:>12.2f}"
                             f"{nan_count:>12}{nan_percentage:>12.2f}%\n")
        
        # NaN değer analizi özeti
        txt_log.write(f"\n1.3 NaN Değer Özeti\n\n")
        total_nan = sum(feature_nan_counts.values())
        txt_log.write(f"    Toplam NaN Değer Sayısı: {total_nan}\n")
        
        # NaN değerleri en çok olan özellikler
        sorted_nan_features = sorted(feature_nan_counts.items(), key=lambda x: x[1], reverse=True)
        txt_log.write(f"    En Çok NaN İçeren Özellikler:\n")
        for feat, count in sorted_nan_features[:5]:  # İlk 5 özellik
            if count > 0:
                txt_log.write(f"        {feat:<30}: {count} ({(count/len(filtered_data))*100:.2f}%)\n")
        
        txt_log.write(f"\n2. ÖZELLİK KORELASYONLARI\n")
        txt_log.write(f"{'-'*80}\n\n")
        
        # Hedef değişken ile özellikler arasındaki korelasyonlar
        corr_with_target = {}
        for feat in features:
            if feat in filtered_data.columns:
                corr = filtered_data[feat].corr(filtered_data[target_col])
                corr_with_target[feat] = corr
        
        # Korelasyonları sıralama
        sorted_corr = sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True)
        
        txt_log.write(f"{'Özellik':<30}{'Korelasyon':>15}\n")
        txt_log.write(f"{'-'*45}\n")
        for feat, corr in sorted_corr:
            txt_log.write(f"{feat:<30}{corr:>15.4f}\n")
        
    # Ölçeklendirici
    scaler = RobustScaler()
    
    # Özellikler ve hedef
    X = filtered_data[features]
    y = filtered_data[target_col]
    
    # NaN değerleri kontrol et ve temizle
    if X.isna().any().any():
        model_logger.warning("Özelliklerde NaN değerler bulundu. Medyan ile doldurulacak.")
        X = X.fillna(X.median())
    
    # Bilgilendirmeyi model_logger ile de yap
    model_logger.info("Eğitim öncesi veri analizi tamamlandı, detaylar için %s dosyasını inceleyiniz.", txt_log_path)
    
    plt.figure(figsize=(12, 10))
    
    # Özellik dağılımları ve hedef ile ilişkileri
    selected_features = min(6, len(features))  # En çok 6 özellik göster
    
    for i, feat in enumerate(features[:selected_features]):
        if feat in X.columns:
            plt.subplot(selected_features, 2, i*2+1)
            sns.histplot(X[feat].dropna(), kde=True)
            plt.title(f'{feat} Dağılımı')
            
            plt.subplot(selected_features, 2, i*2+2)
            sns.scatterplot(x=X[feat], y=y)
            plt.title(f'{feat} vs {target_col}')
    
    plt.tight_layout()
    data_viz_path = os.path.join(PLOT_DIR, f'data_analysis_{inverter_name}.png')
    plt.savefig(data_viz_path)
    model_logger.info(f"Veri analiz grafiği kaydedildi: {data_viz_path}")
    
    try:
        # Verileri eğitim ve test kümelerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Eğitim setinin özelliklerini logla
        model_logger.info(f"Eğitim seti boyutu: {X_train.shape}")
        model_logger.info(f"Test seti boyutu: {X_test.shape}")
        
        # Özellikleri ölçeklendirme
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=features
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=features
        )
        
        # Model oluşturma ve eğitme
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model_logger.info("Model eğitimi başlıyor...")
        model.fit(X_train_scaled, y_train)
        model_logger.info("Model eğitimi tamamlandı.")
        
        # Tahmin ve değerlendirme
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # MAPE hesaplamasını güvenli hale getir
        # Sadece sıfırdan büyük değerler için MAPE hesapla
        mask = y_test > 1.0  # 1 kWh'den büyük değerler için
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan
            model_logger.warning("MAPE hesaplanamadı - sıfır olmayan değer yok")
        
        model_logger.info(f"Model Metrikleri:")
        model_logger.info(f"R² Skoru: {r2:.4f}")
        model_logger.info(f"MAE: {mae:.4f}")
        model_logger.info(f"RMSE: {rmse:.4f}")
        model_logger.info(f"MAPE: {mape:.4f}%" if not np.isnan(mape) else "MAPE: Hesaplanamadı")
        
        # R² değeri çok düşükse uyarı ver
        if r2 < model_min_r2:
            model_logger.warning(f"R² değeri çok düşük ({r2:.4f}). Model yeterince iyi değil!")
            return None, None, None, r2, mae, rmse, mape
        
        # YENİ: Artık analizi
        residuals = y_test - y_pred
        
        # Artık analizi grafikleri
        plt.figure(figsize=(15, 10))
        
        # Artık histogramı
        plt.subplot(2, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title('Artık Dağılımı')
        plt.xlabel('Artık Değeri')
        plt.ylabel('Frekans')
        
        # Tahmin vs Artık
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Tahmin Değerleri vs Artıklar')
        plt.xlabel('Tahmin Değerleri')
        plt.ylabel('Artıklar')
        
        # Q-Q Plot (Normal dağılım kontrolü)
        plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Artıklar için Q-Q Plot')
        
        # Zaman serisi olarak artıklar (indeks bazlı)
        plt.subplot(2, 2, 4)
        plt.plot(residuals.values)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Artıklar Zaman Serisi')
        plt.xlabel('Örnekler')
        plt.ylabel('Artık Değeri')
        
        plt.tight_layout()
        residuals_path = os.path.join(PLOT_DIR, f'residuals_{inverter_name}.png')
        plt.savefig(residuals_path)
        model_logger.info(f"Artık analizi grafiği kaydedildi: {residuals_path}")
        
        # YENİ: Overfitting/Underfitting Kontrolü
        from sklearn.model_selection import learning_curve
        
        try:
            # Öğrenme eğrisi hesaplama
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train_scaled, y_train, 
                cv=min(5, len(X_train_scaled)//10),  # Veri boyutuna uygun CV değeri
                train_sizes=np.linspace(0.1, 1.0, min(10, len(X_train_scaled)//10)),
                scoring='r2',
                n_jobs=-1  # Çoklu işlemci kullan
            )
            
            # Öğrenme eğrisi ortalama ve standart sapma
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Eğitim/test performans eğrisi grafiği
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Eğitim skoru')
            plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
            plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Doğrulama skoru')
            plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
            plt.grid()
            plt.xlabel('Eğitim Seti Boyutu')
            plt.ylabel('R² Skoru')
            plt.title(f'Öğrenme Eğrisi - {inverter_name}')
            plt.legend(loc='lower right')
            
            learning_curve_path = os.path.join(PLOT_DIR, f'learning_curve_{inverter_name}.png')
            plt.savefig(learning_curve_path)
            model_logger.info(f"Öğrenme eğrisi grafiği kaydedildi: {learning_curve_path}")
            
            final_train_score = train_mean[-1]
            final_test_score = test_mean[-1]
            score_diff = final_train_score - final_test_score
            
            if score_diff > 0.2:
                fitting_status = "Overfitting (Aşırı Öğrenme)"
                model_logger.warning(f"Model aşırı öğrenme gösteriyor: Eğitim/Test skor farkı {score_diff:.4f}")
            elif final_test_score < 0.5:
                fitting_status = "Underfitting (Yetersiz Öğrenme)"
                model_logger.warning(f"Model yetersiz öğrenme gösteriyor: Test skoru {final_test_score:.4f}")
            else:
                fitting_status = "İyi Uyum"
                model_logger.info(f"Model iyi uyum gösteriyor: Eğitim/Test skor farkı {score_diff:.4f}")
        except Exception as e:
            model_logger.warning(f"Öğrenme eğrisi hesaplanamadı: {e}")
            fitting_status = "Belirlenemedi"
            final_train_score = r2
            final_test_score = r2
            score_diff = 0
        
        # Modeli kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f'model_{inverter_name}_{timestamp}_r2_{r2:.4f}.joblib')
        joblib.dump(model, model_path)
        model_logger.info(f"Model kaydedildi: {model_path}")
        
        # Özellik önemlerini görselleştir
        plt.figure(figsize=(12, 6))
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        plt.title(f'{inverter_name} İçin Özellik Önemleri')
        plt.bar(range(len(features)), feature_importances[indices])
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f'feature_importance_{inverter_name}.png')
        plt.savefig(plot_path)
        model_logger.info(f"Özellik önemleri grafiği kaydedildi: {plot_path}")
        
        # Gerçek vs Tahmin grafiği
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.title(f'{inverter_name} Gerçek vs Tahmin')
        pred_plot_path = os.path.join(PLOT_DIR, f'predictions_{inverter_name}.png')
        plt.savefig(pred_plot_path)
        model_logger.info(f"Tahmin grafiği kaydedildi: {pred_plot_path}")
        feature_importance_dict = {features[i]: feature_importances[i] for i in range(len(features))}
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        model_logger.info("Özellik Önemleri:")
        for feature, importance in sorted_features:
            model_logger.info(f"{feature}: {importance:.4f}")

        with open(txt_log_path, 'a', encoding='utf-8') as txt_log:
            txt_log.write(f"\n4. MODEL SONUÇLARI\n")
            txt_log.write(f"{'-'*80}\n\n")
            txt_log.write(f"R² Skoru    : {r2:.4f}\n")
            txt_log.write(f"MAE         : {mae:.4f}\n")
            txt_log.write(f"RMSE        : {rmse:.4f}\n")
            txt_log.write(f"MAPE        : {mape:.4f}%\n\n")
            txt_log.write(f"Fitting Durumu: {fitting_status}\n")
            txt_log.write(f"Eğitim R² Skoru: {final_train_score:.4f}\n")
            txt_log.write(f"Test R² Skoru  : {final_test_score:.4f}\n")
            txt_log.write(f"Fark           : {score_diff:.4f}\n\n")
            txt_log.write(f"Özellik Önemleri:\n")
            txt_log.write(f"{'-'*45}\n")
            txt_log.write(f"{'Özellik':<30}{'Önem':>15}\n")
            txt_log.write(f"{'-'*45}\n")
            for feature, importance in sorted_features:
                txt_log.write(f"{feature:<30}{importance:>15.4f}\n")
            txt_log.write(f"\n5. YORUMLAR\n")
            txt_log.write(f"{'-'*80}\n\n")
            
           
            txt_log.write(f"- Model fitting durumu: {fitting_status}\n")
            if fitting_status == "Overfitting (Aşırı Öğrenme)":
                txt_log.write("  Model eğitim verilerini çok iyi öğrenmiş ancak genelleme yeteneği daha düşük.\n")
                txt_log.write("  Regularizasyon, daha az karmaşık model veya daha fazla veri yardımcı olabilir.\n\n")
            elif fitting_status == "Underfitting (Yetersiz Öğrenme)":
                txt_log.write("  Model yeterince karmaşık değil veya önemli özellikler eksik olabilir.\n")
                txt_log.write("  Daha karmaşık model, daha fazla özellik veya hiperparametre optimizasyonu denenebilir.\n\n")
            else:
                txt_log.write("  Model iyi bir denge gösteriyor, eğitim ve test performansı uyumlu.\n\n")
        
        return model, model_path, scaler, r2, mae, rmse, mape
    
    except Exception as e:
        model_logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
        return None, None, None, 0, 0, 0, 0

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_logger.info(f"Uygulama başladı: {timestamp}")
    
    try:
        # Verileri yükle
        data = load_data()
        
        # Mevcut sütunları görüntüle
        main_logger.info(f"Veri seti sütunları: {data.columns.tolist()}")
        
        # Verileri ön işle
        processed_data, inverter_columns = preprocess_data(data)
        
        # Hava durumu özelliklerini otomatik olarak belirle
        weather_features_patterns = [
            'temperature', 'radiation', 'irradiance', 'humidity', 'wind', 'visibility'
        ]
        
        # Mevcut sütunlardan hava durumu özelliklerini belirle
        weather_features = []
        for col in data.columns:
            if any(pattern in str(col).lower() for pattern in weather_features_patterns):
                weather_features.append(col)
        
        main_logger.info(f"CSV'den tespit edilen hava durumu özellikleri: {weather_features}")
        
        # Güneş pozisyonu ve zaman özellikleri
        solar_features = []
        
        # Zaman özellikleri
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        # Tüm özellikleri birleştir
        all_features = weather_features + solar_features + time_features
        
        # YENİ: Günlük, haftalık ve yıllık elektrik üretimi görselleştirmeleri
        create_time_series_visualizations(processed_data, inverter_columns)
        
        # YENİ: Hava durumu verilerinin görselleştirilmesi
        visualize_weather_data(data, weather_features)
        
        # Her inverter için model oluştur
        results = []
        
        for inv_col in inverter_columns:
            main_logger.info(f"\n{'='*50}")
            main_logger.info(f"{inv_col} için model eğitimi başlıyor...")
            
            model, model_path, scaler, r2, mae, rmse, mape = train_model(
                processed_data, 
                inv_col, 
                all_features,
                model_min_r2=0.3  # Minimum kabul edilebilir R² değeri
            )
            
            # Model sonuçlarını kaydet
            result = {
                'inverter': inv_col,
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'model_path': model_path if model is not None else None
            }
            results.append(result)
            
            if model is None:
                main_logger.warning(f"{inv_col} için model eğitimi başarısız oldu (düşük R²).")
            else:
                main_logger.info(f"{inv_col} için model eğitimi başarıyla tamamlandı.")
        
        # Sonuçları özet olarak kaydet
        results_path = os.path.join(MODEL_DIR, f'model_results_{timestamp}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        main_logger.info(f"Model sonuçları kaydedildi: {results_path}")
        
        # Özet tablosu oluştur ve görselleştir
        results_df = pd.DataFrame(results)
        
        # R² skorları
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        sns.barplot(x='inverter', y='r2_score', data=results_df)
        plt.title('İnverter R² Skorları')
        plt.xticks(rotation=45)
        
        # MAE skorları
        plt.subplot(2, 2, 2)
        sns.barplot(x='inverter', y='mae', data=results_df)
        plt.title('İnverter MAE Değerleri')
        plt.xticks(rotation=45)
        
        # RMSE skorları
        plt.subplot(2, 2, 3)
        sns.barplot(x='inverter', y='rmse', data=results_df)
        plt.title('İnverter RMSE Değerleri')
        plt.xticks(rotation=45)
        
        # MAPE skorları
        plt.subplot(2, 2, 4)
        sns.barplot(x='inverter', y='mape', data=results_df)
        plt.title('İnverter MAPE Değerleri (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        summary_plot_path = os.path.join(PLOT_DIR, f'model_summary_{timestamp}.png')
        plt.savefig(summary_plot_path)
        main_logger.info(f"Özet grafiği kaydedildi: {summary_plot_path}")
        
        main_logger.info("Uygulama başarıyla tamamlandı!")
        
    except Exception as e:
        main_logger.error(f"Uygulama sırasında hata oluştu: {e}", exc_info=True)
        raise

# YENİ: Zaman serisi görselleştirme fonksiyonu
def create_time_series_visualizations(data, inverter_columns):
    """Günlük, haftalık ve yıllık enerji üretimini görselleştirir."""
    main_logger.info("Zaman serisi görselleştirmeleri oluşturuluyor...")
    
    # Her inverter için saatlik enerji verilerini alıp veri çerçevesine dönüştür
    hourly_data = pd.DataFrame({'datetime': data['datetime']})
    
    for col in inverter_columns:
        hourly_col = f'hourly_{col}'
        if hourly_col in data.columns:
            hourly_data[col] = data[hourly_col]
    
    # Datetime'ı indeks olarak ayarla
    hourly_data.set_index('datetime', inplace=True)
    
    try:
        # 1. Günlük Toplam Üretim
        daily_data = hourly_data.resample('D').sum()
        
        plt.figure(figsize=(15, 6))
        for col in inverter_columns:
            plt.plot(daily_data.index, daily_data[col], label=col)
        
        plt.title('Günlük Toplam Enerji Üretimi')
        plt.xlabel('Tarih')
        plt.ylabel('Enerji (kWh)')
        plt.legend()
        plt.grid(True)
        
        daily_plot_path = os.path.join(PLOT_DIR, 'daily_energy_production.png')
        plt.savefig(daily_plot_path)
        main_logger.info(f"Günlük enerji üretim grafiği kaydedildi: {daily_plot_path}")
        
        # 2. Haftalık Toplam Üretim
        weekly_data = hourly_data.resample('W').sum()
        
        plt.figure(figsize=(15, 6))
        for col in inverter_columns:
            plt.plot(weekly_data.index, weekly_data[col], label=col, marker='o')
        
        plt.title('Haftalık Toplam Enerji Üretimi')
        plt.xlabel('Tarih')
        plt.ylabel('Enerji (kWh)')
        plt.legend()
        plt.grid(True)
        
        weekly_plot_path = os.path.join(PLOT_DIR, 'weekly_energy_production.png')
        plt.savefig(weekly_plot_path)
        main_logger.info(f"Haftalık enerji üretim grafiği kaydedildi: {weekly_plot_path}")
        
        # 3. Aylık Toplam Üretim
        monthly_data = hourly_data.resample('M').sum()
        
        plt.figure(figsize=(15, 6))
        for col in inverter_columns:
            plt.plot(monthly_data.index, monthly_data[col], label=col, marker='s')
        
        plt.title('Aylık Toplam Enerji Üretimi')
        plt.xlabel('Tarih')
        plt.ylabel('Enerji (kWh)')
        plt.legend()
        plt.grid(True)
        
        monthly_plot_path = os.path.join(PLOT_DIR, 'monthly_energy_production.png')
        plt.savefig(monthly_plot_path)
        main_logger.info(f"Aylık enerji üretim grafiği kaydedildi: {monthly_plot_path}")
        
        # 4. Saat bazlı ortalama üretim (günün saatine göre)
        hourly_avg = hourly_data.groupby(hourly_data.index.hour).mean()
        
        plt.figure(figsize=(15, 6))
        for col in inverter_columns:
            plt.plot(hourly_avg.index, hourly_avg[col], label=col, marker='o')
        
        plt.title('Saat Bazlı Ortalama Enerji Üretimi')
        plt.xlabel('Saat')
        plt.ylabel('Ortalama Enerji (kWh)')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True)
        
        hourly_avg_path = os.path.join(PLOT_DIR, 'hourly_avg_energy_production.png')
        plt.savefig(hourly_avg_path)
        main_logger.info(f"Saatlik ortalama enerji üretim grafiği kaydedildi: {hourly_avg_path}")
        
        # 5. Hafta içi vs Hafta sonu karşılaştırması
        weekday_data = hourly_data.copy()
        weekday_data['weekday'] = weekday_data.index.dayofweek
        weekday_data['is_weekend'] = weekday_data['weekday'].isin([5, 6])  # 5=Cumartesi, 6=Pazar
        
        weekday_avg = weekday_data.groupby(['is_weekend', weekday_data.index.hour]).mean()
        
        plt.figure(figsize=(15, 6))
        for col in inverter_columns[:4]:  # Grafik kalabalık olmasın diye sadece ilk 4 inverter
            plt.plot(weekday_avg.loc[False].index, weekday_avg.loc[False, col], 
                    label=f'{col} - Hafta içi', linestyle='-')
            plt.plot(weekday_avg.loc[True].index, weekday_avg.loc[True, col], 
                    label=f'{col} - Hafta sonu', linestyle='--')
        
        plt.title('Hafta içi vs Hafta sonu Ortalama Enerji Üretimi')
        plt.xlabel('Saat')
        plt.ylabel('Ortalama Enerji (kWh)')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True)
        
        weekday_path = os.path.join(PLOT_DIR, 'weekday_vs_weekend_energy.png')
        plt.savefig(weekday_path)
        main_logger.info(f"Hafta içi/sonu enerji üretim grafiği kaydedildi: {weekday_path}")
        
        # 6. Isı haritası: Gün ve saat bazlı ortalama üretim
        hourly_data_for_heatmap = hourly_data.copy()
        hourly_data_for_heatmap['hour'] = hourly_data_for_heatmap.index.hour
        hourly_data_for_heatmap['dayofweek'] = hourly_data_for_heatmap.index.dayofweek
        
        # Örnek olarak ilk inverteri kullanarak ısı haritası
        sample_col = inverter_columns[0]
        pivot_data = hourly_data_for_heatmap.pivot_table(
            values=sample_col, 
            index='hour', 
            columns='dayofweek',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='viridis', annot=False, fmt=".1f", cbar_kws={'label': 'Ortalama Enerji (kWh)'})
        plt.title(f'Gün ve Saat Bazlı Ortalama Enerji Üretimi - {sample_col}')
        plt.xlabel('Haftanın Günü (0=Pazartesi, 6=Pazar)')
        plt.ylabel('Saat')
        plt.tight_layout()
        
        heatmap_path = os.path.join(PLOT_DIR, 'hour_day_energy_heatmap.png')
        plt.savefig(heatmap_path)
        main_logger.info(f"Gün-saat bazlı enerji üretim ısı haritası kaydedildi: {heatmap_path}")
        
    except Exception as e:
        main_logger.error(f"Zaman serisi görselleştirmeleri oluşturulurken hata oluştu: {e}")

# YENİ: Hava durumu verilerini görselleştirme fonksiyonu
def visualize_weather_data(data, weather_features):
    """Hava durumu verilerini görselleştirir."""
    main_logger.info("Hava durumu görselleştirmeleri oluşturuluyor...")
    
    try:
        # Datetime sütununu indeks olarak ayarla
        weather_data = data.copy()
        weather_data.set_index('datetime', inplace=True)
        
        # 1. Hava durumu özelliklerinin zaman içindeki değişimi
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(weather_features):
            if feature in weather_data.columns:
                # Veriyi örnekle (tüm noktaları çizmek grafiği ağırlaştırabilir)
                sampled_data = weather_data.sample(min(1000, len(weather_data)))
                sampled_data = sampled_data.sort_index()
                
                plt.subplot(len(weather_features), 1, i+1)
                plt.plot(sampled_data.index, sampled_data[feature])
                plt.title(f'{feature} Değişimi')
                plt.ylabel(feature)
                if i == len(weather_features) - 1:
                    plt.xlabel('Tarih')
                plt.grid(True)
        
        plt.tight_layout()
        weather_time_path = os.path.join(PLOT_DIR, 'weather_time_series.png')
        plt.savefig(weather_time_path)
        main_logger.info(f"Hava durumu zaman serisi grafiği kaydedildi: {weather_time_path}")
        
        # 2. Hava durumu özelliklerinin korelasyon matrisi
        weather_corr = weather_data[weather_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(weather_corr, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0.5, cbar_kws={'label': 'Korelasyon Katsayısı'})
        plt.title('Hava Durumu Özellikleri Korelasyon Matrisi')
        plt.tight_layout()
        
        corr_path = os.path.join(PLOT_DIR, 'weather_correlation.png')
        plt.savefig(corr_path)
        main_logger.info(f"Hava durumu korelasyon matrisi kaydedildi: {corr_path}")
        
        # 3. Sıcaklık ve Işınım ilişkisi
        if 'temperature_2m' in weather_features and any('radiation' in f for f in weather_features):
            radiation_feature = next(f for f in weather_features if 'radiation' in f)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(weather_data['temperature_2m'], weather_data[radiation_feature], alpha=0.3)
            plt.title(f'Sıcaklık vs {radiation_feature}')
            plt.xlabel('Sıcaklık (°C)')
            plt.ylabel(f'{radiation_feature}')
            plt.grid(True)
            
            temp_rad_path = os.path.join(PLOT_DIR, 'temperature_radiation_scatter.png')
            plt.savefig(temp_rad_path)
            main_logger.info(f"Sıcaklık-ışınım ilişki grafiği kaydedildi: {temp_rad_path}")
            
        # 4. Gün içindeki saat bazlı ışınım ve sıcaklık değişimi
        weather_hourly = weather_data.copy()
        weather_hourly['hour'] = weather_hourly.index.hour
        
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(weather_features[:4]):  # İlk 4 özellik için
            if feature in weather_hourly.columns:
                hourly_avg = weather_hourly.groupby('hour')[feature].mean()
                
                plt.subplot(2, 2, i+1)
                plt.plot(hourly_avg.index, hourly_avg.values, marker='o')
                plt.title(f'Saat Bazlı Ortalama {feature}')
                plt.xlabel('Saat')
                plt.ylabel(feature)
                plt.xticks(range(0, 24, 2))
                plt.grid(True)
        
        plt.tight_layout()
        hourly_weather_path = os.path.join(PLOT_DIR, 'hourly_weather_averages.png')
        plt.savefig(hourly_weather_path)
        main_logger.info(f"Saatlik hava durumu ortalamaları grafiği kaydedildi: {hourly_weather_path}")
        
        # 5. Mevsimsel değişim (ay bazlı) 
        weather_monthly = weather_data.copy()
        weather_monthly['month'] = weather_monthly.index.month
        
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(weather_features[:4]):  # İlk 4 özellik için
            if feature in weather_monthly.columns:
                monthly_avg = weather_monthly.groupby('month')[feature].mean()
                
                plt.subplot(2, 2, i+1)
                plt.plot(monthly_avg.index, monthly_avg.values, marker='s')
                plt.title(f'Ay Bazlı Ortalama {feature}')
                plt.xlabel('Ay')
                plt.ylabel(feature)
                plt.xticks(range(1, 13))
                plt.grid(True)
        
        plt.tight_layout()
        monthly_weather_path = os.path.join(PLOT_DIR, 'monthly_weather_averages.png')
        plt.savefig(monthly_weather_path)
        main_logger.info(f"Aylık hava durumu ortalamaları grafiği kaydedildi: {monthly_weather_path}")
        
    except Exception as e:
        main_logger.error(f"Hava durumu görselleştirmeleri oluşturulurken hata oluştu: {e}")

if __name__ == "__main__":
    main()
