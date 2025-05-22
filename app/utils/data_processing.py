import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

# Logger yapılandırması
logger = logging.getLogger(__name__)

class InverterDataProcessor:
    """
    İnverter verilerini işlemek için utility sınıfı.
    """
    
    @staticmethod
    def cumulative_to_hourly(
        df: pd.DataFrame, 
        inverter_columns: List[str],
        time_column: str = 'time',
        log_anomalies: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Kümülatif inverter verilerini saatlik üretim verilerine dönüştürür.
        
        Args:
            df: Kümülatif verileri içeren DataFrame
            inverter_columns: İnverter verilerini içeren sütunların listesi
            time_column: Zaman sütununun adı
            log_anomalies: Anormallikleri loglamak için bayrak
            
        Returns:
            hourly_df: Saatlik üretim verilerini içeren DataFrame
            stats: İşlem istatistikleri
        """
        logger.info(f"Kümülatif inverter verileri saatlik üretime dönüştürülüyor. {len(df)} satır, {len(inverter_columns)} inverter.")
        
        # Zaman sütununun datetime tipinde olduğundan emin ol
        if df[time_column].dtype != 'datetime64[ns]':
            df[time_column] = pd.to_datetime(df[time_column])
            logger.info(f"{time_column} sütunu datetime formatına dönüştürüldü.")
        
        # Saatlik veri oluşturmak için önce verileri saate göre sırala
        df = df.sort_values(by=[time_column])
        
        # İstatistik bilgilerini tutacak sözlük
        stats = {
            "original_rows": len(df),
            "inverter_columns": len(inverter_columns),
            "anomalies": {col: 0 for col in inverter_columns},
            "negative_values": {col: 0 for col in inverter_columns},
            "large_jumps": {col: 0 for col in inverter_columns},
            "hourly_rows": 0
        }
        
        # Her inverter için saatlik farkları hesapla
        hourly_data = []
        
        # Her saat için grupla
        df['hour'] = df[time_column].dt.floor('H')
        
        for name, group in df.groupby('hour'):
            # Her saat için son değerleri al (saatin son kayıtları)
            if not group.empty:
                row_dict = {time_column: name}  # Saat başlangıcını kullan
                
                for col in inverter_columns:
                    # Bu saat için mevcut verileri al
                    values = group[col].dropna().tolist()
                    
                    if not values:
                        row_dict[col] = 0
                        continue
                    
                    # En son değer ile bir önceki saatin son değeri arasındaki farkı al
                    current_value = values[-1]  # Bu saatin son değeri
                    
                    # Eğer bu ilk kayıt değilse fark hesapla
                    if name in [h for h in hourly_data]:
                        prev_hourly_data = [h for h in hourly_data if h[time_column] < name]
                        if prev_hourly_data:
                            prev_hourly_row = prev_hourly_data[-1]
                            prev_value = prev_hourly_row.get(col, 0)
                            
                            # Farkı hesapla
                            diff = current_value - prev_value
                            
                            # Negatif fark kontrolü (gün başlangıcı veya reset durumu)
                            if diff < 0:
                                stats["negative_values"][col] += 1
                                if log_anomalies:
                                    logger.warning(f"Negatif değer tespit edildi: {col}, saat: {name}, önceki: {prev_value}, şimdiki: {current_value}, fark: {diff}")
                                
                                # Negatif değer durumunda mevcut değeri kullan (reset varsayımı)
                                diff = current_value
                            
                            # Çok büyük artış kontrolü (anormal durum)
                            if diff > 10:  # 10 kWh'den fazla artış anormal kabul edilebilir
                                stats["large_jumps"][col] += 1
                                if log_anomalies:
                                    logger.warning(f"Büyük artış tespit edildi: {col}, saat: {name}, önceki: {prev_value}, şimdiki: {current_value}, fark: {diff}")
                            
                            row_dict[col] = max(0, diff)  # Negatif değerleri sıfırla
                        else:
                            # İlk değer için fark hesaplanamaz, direkt değeri kullan
                            row_dict[col] = current_value
                    else:
                        # İlk kayıt için direkt değeri kullan
                        row_dict[col] = current_value
                
                hourly_data.append(row_dict)
        
        # Saatlik verileri DataFrame'e dönüştür
        hourly_df = pd.DataFrame(hourly_data)
        
        # İstatistikleri güncelle
        stats["hourly_rows"] = len(hourly_df)
        
        # Anormallikleri topla
        for col in inverter_columns:
            stats["anomalies"][col] = stats["negative_values"][col] + stats["large_jumps"][col]
        
        logger.info(f"Dönüştürme tamamlandı. {stats['hourly_rows']} saatlik veri oluşturuldu.")
        logger.info(f"Anormallikler: {sum(stats['anomalies'].values())} adet")
        
        return hourly_df, stats
    
    @staticmethod
    def convert_to_hourly(df: pd.DataFrame, inverter_columns: List[str], time_column: str = 'time') -> pd.DataFrame:
        """
        5 dakikalık verileri saatlik verilere dönüştürür.
        Her saat için son 3 değerin ortalamasını alır.
        
        Args:
            df: Orijinal veri çerçevesi
            inverter_columns: İnverter sütunları listesi
            time_column: Zaman sütununun adı
            
        Returns:
            Saatlik verilerden oluşan veri çerçevesi
        """
        logger.info(f"Veriler saatlik formata dönüştürülüyor. {len(df)} satır, {len(inverter_columns)} inverter.")
        
        # Tarih sütununu saat başı olarak yuvarlama
        df['hour'] = df[time_column].dt.floor('H')
        
        # Saatlik gruplar oluşturulması
        result_rows = []
        
        # Benzersiz saatleri al
        unique_hours = df['hour'].unique()
        
        for hour in unique_hours:
            # Bu saat için verileri filtrele
            hour_data = df[df['hour'] == hour]
            
            # Son 3 değerin (varsa) alınması
            last_rows = hour_data.tail(3)
            
            row_dict = {'time': hour}
            
            for col in inverter_columns:
                # İnverter sütunu için son 3 değerin ortalaması
                if not last_rows.empty:
                    avg_value = last_rows[col].mean()
                    row_dict[col] = avg_value
                else:
                    row_dict[col] = 0
            
            result_rows.append(row_dict)
        
        # Yeni veri çerçevesi oluşturma
        hourly_df = pd.DataFrame(result_rows)
        
        logger.info(f"Saatlik formata dönüştürme tamamlandı. {len(hourly_df)} saatlik veri oluşturuldu.")
        
        return hourly_df
    
    @staticmethod
    def calculate_hourly_power(df: pd.DataFrame, inverter_columns: List[str]) -> pd.DataFrame:
        """
        Kümülatif enerji verilerinden saatlik güç değerlerini hesaplar.
        Ardışık ölçümler arasındaki farkı alarak saatlik üretimi bulur.
        Absürt değerleri filtreleyerek veri kalitesini artırır.
        
        Args:
            df: Kümülatif enerji verilerini içeren DataFrame
            inverter_columns: İnverter enerji sütunları
            
        Returns:
            Saatlik güç verilerini içeren DataFrame
        """
        logger.info(f"Saatlik güç değerleri hesaplanıyor. {len(df)} satır, {len(inverter_columns)} inverter.")
        
        # Veriyi kopyala
        result_df = df.copy()
        
        # Her inverter için absürt değerleri filtrele ve saatlik güç hesapla
        for col in inverter_columns:
            hourly_col = f'hourly_{col}'
            
            # 1. Negatif kümülatif değerleri düzelt (hatalı okuma olabilir)
            negative_values = result_df[col] < 0
            negative_count = negative_values.sum()
            if negative_count > 0:
                logger.warning(f"{col} için {negative_count} adet negatif kümülatif değer tespit edildi ve NaN olarak işaretlendi.")
                result_df.loc[negative_values, col] = float('nan')
            
            # 2. Aşırı yüksek değerleri tespit et (tipik bir inverter için makul üst sınır 10.000 kWh/gün)
            max_daily_energy = 9000  # kWh - bu değer inverter kapasitesine göre ayarlanabilir
            outliers = result_df[col] > max_daily_energy
            outlier_count = outliers.sum()
            if outlier_count > 0:
                logger.warning(f"{col} için {outlier_count} adet aşırı yüksek değer tespit edildi ve NaN olarak işaretlendi.")
                result_df.loc[outliers, col] = float('nan')
            
            # 3. Ani sıçramaları tespit et (bir önceki değere göre imkansız artışlar)
            # Önce sütunu sırala ve geçici değişkene kopyala
            sorted_values = result_df[col].sort_index()
            
            # Ardışık değerler arasındaki farkı hesapla
            diffs = sorted_values.diff()
            
            # Tipik bir inverter için saatlik maksimum üretim (örn. 50 kWh/saat)
            max_hourly_production = 1500  # kWh/saat - bu değer inverter kapasitesine göre ayarlanabilir
            
            # Zaman farkını saat cinsinden hesapla
            time_diffs = df['time'].diff().dt.total_seconds() / 3600
            
            # Saatlik üretim limitini aşan değerleri tespit et
            # Saat başına maksimum üretim * saat sayısı
            # NaN değerlerden kaçınmak için time_diffs'in ilk değeri NaN olacağından bunları filtrele
            valid_indices = (~time_diffs.isna()) & (~diffs.isna())
            
            # Hızlı artışları hesapla (saatlik üretim limitini aşanlar)
            hourly_rate = diffs[valid_indices] / time_diffs[valid_indices]
            sudden_jumps = hourly_rate > max_hourly_production
            jump_count = sudden_jumps.sum()
            
            if jump_count > 0:
                logger.warning(f"{col} için {jump_count} adet ani sıçrama tespit edildi.")
                # Ani sıçrama olan değerleri işaretle
                jump_indices = sudden_jumps[sudden_jumps].index
                result_df.loc[jump_indices, col] = float('nan')
            
            # 4. NaN değerleri enterpolasyon ile doldur
            if result_df[col].isna().any():
                logger.info(f"{col} için filtrelenen değerler enterpolasyon ile doldurulacak.")
                result_df[col] = result_df[col].interpolate(method='linear', limit_direction='both')
            
            # 5. Diff fonksiyonu ile ardışık değerler arasındaki farkı hesapla
            result_df[hourly_col] = result_df[col].diff().fillna(0)
            
            # 6. Yine de negatif değerler varsa düzelt
            negative_mask = result_df[hourly_col] < 0
            negative_count = negative_mask.sum()
            
            if negative_count > 0:
                logger.warning(f"{col} için {negative_count} adet negatif saatlik değer tespit edildi ve sıfıra ayarlandı.")
                result_df.loc[negative_mask, hourly_col] = 0
        
        logger.info(f"Saatlik güç hesaplaması tamamlandı. Her inverter için hourly_ önekli sütunlar eklendi.")
        
        return result_df 