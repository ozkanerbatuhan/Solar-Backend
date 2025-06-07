import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class DataQualityService:
    """
    Gelişmiş veri kalitesi kontrolü ve iyileştirme servisi.
    """
    
    @staticmethod
    def analyze_solar_physics_consistency(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Güneş fizik kurallarına uygunluk analizi yapar.
        
        Args:
            df: Analiz edilecek DataFrame
            
        Returns:
            Fizik uygunluk raporu
        """
        logger.info("Güneş fizik uygunluk analizi başlatılıyor...")
        
        issues = []
        fixes_applied = 0
        
        # Saat bazında grup analizi
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            
            if len(hour_data) == 0:
                continue
                
            # Gece saatleri kontrolü (22:00-05:59)
            if hour >= 22 or hour <= 5:
                # Güneş radyasyonu > 0 olanlar anormal
                solar_radiation_fields = [
                    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
                    'direct_normal_irradiance', 'global_tilted_irradiance', 'terrestrial_radiation'
                ]
                
                for field in solar_radiation_fields:
                    if field in hour_data.columns:
                        anomalies = hour_data[hour_data[field] > 10]  # 10 W/m² threshold
                        if len(anomalies) > 0:
                            issues.append({
                                'type': 'night_solar_radiation',
                                'hour': hour,
                                'field': field,
                                'count': len(anomalies),
                                'max_value': anomalies[field].max(),
                                'mean_value': anomalies[field].mean()
                            })
                            
                            # Düzeltme uygula
                            df.loc[(df['hour'] == hour) & (df[field] > 10), field] = 0
                            fixes_applied += len(anomalies)
                            
            # Öğle saatleri kontrolü (10:00-15:00)
            elif 10 <= hour <= 15:
                # Çok düşük radyasyon değerleri şüpheli
                if 'shortwave_radiation' in hour_data.columns:
                    low_radiation = hour_data[hour_data['shortwave_radiation'] < 50]
                    if len(low_radiation) > len(hour_data) * 0.2:  # %20'den fazlası düşükse
                        issues.append({
                            'type': 'low_noon_radiation',
                            'hour': hour,
                            'count': len(low_radiation),
                            'percentage': (len(low_radiation) / len(hour_data)) * 100
                        })
        
        logger.info(f"Fizik uygunluk analizi tamamlandı. {len(issues)} sorun tespit edildi, {fixes_applied} düzeltme uygulandı.")
        
        return {
            'issues': issues,
            'fixes_applied': fixes_applied,
            'total_issues': len(issues)
        }
    
    @staticmethod
    def advanced_outlier_detection(df: pd.DataFrame, target_column: str = 'power_output') -> Dict[str, Any]:
        """
        Gelişmiş aykırı değer tespiti - çok boyutlu analiz.
        
        Args:
            df: Analiz edilecek DataFrame
            target_column: Hedef sütun
            
        Returns:
            Aykırı değer raporu
        """
        logger.info("Gelişmiş aykırı değer tespiti başlatılıyor...")
        
        if target_column not in df.columns:
            return {'error': f'{target_column} sütunu bulunamadı'}
        
        outlier_info = {
            'statistical_outliers': 0,
            'isolation_forest_outliers': 0,
            'domain_outliers': 0,
            'total_outliers': 0,
            'outlier_indices': set()
        }
        
        # 1. İstatistiksel aykırı değerler (IQR method)
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        statistical_outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)].index
        outlier_info['statistical_outliers'] = len(statistical_outliers)
        outlier_info['outlier_indices'].update(statistical_outliers)
        
        # 2. Isolation Forest - çok boyutlu aykırı değer tespiti
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 1:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            isolation_outliers = iso_forest.fit_predict(df[numeric_columns].fillna(0))
            isolation_outlier_indices = df.index[isolation_outliers == -1]
            
            outlier_info['isolation_forest_outliers'] = len(isolation_outlier_indices)
            outlier_info['outlier_indices'].update(isolation_outlier_indices)
        
        # 3. Domain-specific aykırı değerler (güneş enerjisi kuralları)
        # Fiziksel olarak imkansız değerler
        domain_outliers = []
        
        # Negatif güç çıkışı
        if (df[target_column] < 0).any():
            negative_indices = df[df[target_column] < 0].index
            domain_outliers.extend(negative_indices)
        
        # Gece saatlerinde yüksek güç çıkışı
        if 'hour' in df.columns:
            night_hours_mask = (df['hour'] >= 22) | (df['hour'] <= 5)
            night_high_power = df[night_hours_mask & (df[target_column] > 50)].index
            domain_outliers.extend(night_high_power)
        
        # Çok yüksek güç çıkışı (inverter kapasitesini aşan)
        max_theoretical_power = 5000  # 5 MW - inverter kapasitesine göre ayarlanabilir
        excessive_power = df[df[target_column] > max_theoretical_power].index
        domain_outliers.extend(excessive_power)
        
        outlier_info['domain_outliers'] = len(domain_outliers)
        outlier_info['outlier_indices'].update(domain_outliers)
        
        outlier_info['total_outliers'] = len(outlier_info['outlier_indices'])
        outlier_info['outlier_percentage'] = (outlier_info['total_outliers'] / len(df)) * 100
        
        logger.info(f"Aykırı değer tespiti tamamlandı. {outlier_info['total_outliers']} aykırı değer tespit edildi (%{outlier_info['outlier_percentage']:.2f})")
        
        return outlier_info
    
    @staticmethod
    def intelligent_data_cleaning(df: pd.DataFrame, target_column: str = 'power_output') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        ÇOK AGRESIF akıllı veri temizleme - fiziksel imkansızlıkları tamamen ortadan kaldırır.
        
        Args:
            df: Temizlenecek DataFrame
            target_column: Hedef sütun
            
        Returns:
            Temizlenmiş DataFrame ve rapor
        """
        logger.info("AGRESIF akıllı veri temizleme başlatılıyor...")
        
        df_cleaned = df.copy()
        cleaning_report = {
            'original_rows': len(df),
            'actions': [],
            'rows_removed': 0,
            'values_corrected': 0
        }
        
        # 1. Fizik kuralları ile düzeltme
        physics_report = DataQualityService.analyze_solar_physics_consistency(df_cleaned)
        cleaning_report['actions'].append(f"Fizik kuralları: {physics_report['fixes_applied']} düzeltme")
        cleaning_report['values_corrected'] += physics_report['fixes_applied']
        
        # 2. Negatif güç değerlerini sıfırla
        if target_column in df_cleaned.columns:
            negative_count = (df_cleaned[target_column] < 0).sum()
            if negative_count > 0:
                df_cleaned.loc[df_cleaned[target_column] < 0, target_column] = 0
                cleaning_report['actions'].append(f"Negatif güç değerleri: {negative_count} düzeltme")
                cleaning_report['values_corrected'] += negative_count
        
        # 3. Gece saatlerinde güç çıkışını KATIYECI olarak sıfırla
        if 'hour' in df_cleaned.columns and target_column in df_cleaned.columns:
            night_mask = (df_cleaned['hour'] >= 22) | (df_cleaned['hour'] <= 5)
            night_power_count = (df_cleaned[night_mask][target_column] > 0).sum()
            if night_power_count > 0:
                df_cleaned.loc[night_mask, target_column] = 0
                cleaning_report['actions'].append(f"Gece güç çıkışı: {night_power_count} düzeltme")
                cleaning_report['values_corrected'] += night_power_count
                
            # Gece saatlerinde radyasyon değerlerini de sıfırla
            radiation_cols = ['shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 
                              'direct_normal_irradiance', 'global_tilted_irradiance', 'terrestrial_radiation']
            for rad_col in radiation_cols:
                if rad_col in df_cleaned.columns:
                    night_rad_count = (df_cleaned[night_mask][rad_col] > 10).sum()
                    if night_rad_count > 0:
                        df_cleaned.loc[night_mask & (df_cleaned[rad_col] > 10), rad_col] = 0
                        cleaning_report['actions'].append(f"Gece {rad_col}: {night_rad_count} düzeltme")
                        cleaning_report['values_corrected'] += night_rad_count
        
        # 4. KATIYECI radyasyon-güç ilişkisi kontrolü
        if target_column in df_cleaned.columns and 'shortwave_radiation' in df_cleaned.columns:
            # Sıfır radyasyon, pozitif güç olamaz
            zero_rad_positive_power = (df_cleaned['shortwave_radiation'] == 0) & (df_cleaned[target_column] > 0)
            zero_rad_count = zero_rad_positive_power.sum()
            if zero_rad_count > 0:
                df_cleaned.loc[zero_rad_positive_power, target_column] = 0
                cleaning_report['actions'].append(f"Sıfır radyasyon, pozitif güç: {zero_rad_count} düzeltme")
                cleaning_report['values_corrected'] += zero_rad_count
            
            # Düşük radyasyon, yüksek güç imkansızlığı
            low_rad_high_power = (df_cleaned['shortwave_radiation'] < 100) & (df_cleaned[target_column] > 200)
            low_rad_count = low_rad_high_power.sum()
            if low_rad_count > 0:
                # Radyasyon orantılı düzeltme
                df_cleaned.loc[low_rad_high_power, target_column] = (
                    df_cleaned.loc[low_rad_high_power, 'shortwave_radiation'] * 1.5
                )
                cleaning_report['actions'].append(f"Düşük radyasyon, yüksek güç: {low_rad_count} düzeltme")
                cleaning_report['values_corrected'] += low_rad_count
                
            # Fiziksel maksimum kontrolü - 4.5 kW per 1000 W/m²
            max_efficiency = 4.5
            for idx, row in df_cleaned.iterrows():
                if row['shortwave_radiation'] > 0:
                    max_possible_power = row['shortwave_radiation'] * max_efficiency
                    if row[target_column] > max_possible_power:
                        df_cleaned.at[idx, target_column] = max_possible_power
                        cleaning_report['values_corrected'] += 1
        
        # 5. Sabah/akşam saatleri KATIYECI kontrolü
        if target_column in df_cleaned.columns and 'hour' in df_cleaned.columns:
            # Sabah erken saatler (6-8)
            early_morning = (df_cleaned['hour'] >= 6) & (df_cleaned['hour'] <= 8)
            high_early_power = early_morning & (df_cleaned[target_column] > 1000)
            early_count = high_early_power.sum()
            if early_count > 0:
                df_cleaned.loc[high_early_power, target_column] = np.minimum(
                    df_cleaned.loc[high_early_power, target_column],
                    1000 * (df_cleaned.loc[high_early_power, 'hour'] - 5) / 3
                )
                cleaning_report['actions'].append(f"Sabah erken yüksek güç: {early_count} düzeltme")
                cleaning_report['values_corrected'] += early_count
                
            # Akşam geç saatler (18-21)
            late_evening = (df_cleaned['hour'] >= 18) & (df_cleaned['hour'] <= 21)
            high_late_power = late_evening & (df_cleaned[target_column] > 1000)
            late_count = high_late_power.sum()
            if late_count > 0:
                df_cleaned.loc[high_late_power, target_column] = np.minimum(
                    df_cleaned.loc[high_late_power, target_column],
                    1000 * (22 - df_cleaned.loc[high_late_power, 'hour']) / 4
                )
                cleaning_report['actions'].append(f"Akşam geç yüksek güç: {late_count} düzeltme")
                cleaning_report['values_corrected'] += late_count
        
        # 6. Fiziksel olarak imkansız yüksek değerleri SERT sınırla
        if target_column in df_cleaned.columns:
            max_physical_power = 5000  # 5 MW inverter kapasitesi
            excessive_power = df_cleaned[target_column] > max_physical_power
            excessive_count = excessive_power.sum()
            if excessive_count > 0:
                df_cleaned.loc[excessive_power, target_column] = max_physical_power
                cleaning_report['actions'].append(f"Kapasiteyi aşan güç: {excessive_count} düzeltme")
                cleaning_report['values_corrected'] += excessive_count
            
            # Aşırı aykırı değerleri daha agresif sınırla (99% percentile yerine 95%)
            Q95 = df_cleaned[target_column].quantile(0.95)
            if Q95 > 0:
                high_outliers = df_cleaned[target_column] > Q95 * 1.5  # %95'in 1.5 katından fazla
                outlier_count = high_outliers.sum()
                if outlier_count > 0:
                    df_cleaned.loc[high_outliers, target_column] = Q95 * 1.2  # %95'in 1.2 katına sınırla
                    cleaning_report['actions'].append(f"Aşırı aykırı değerler: {outlier_count} düzeltme (>{Q95*1.5:.0f} -> {Q95*1.2:.0f})")
                    cleaning_report['values_corrected'] += outlier_count
        
        # 7. Missing value imputation - çok daha akıllı doldurma
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype in ['float64', 'int64']:
                missing_count = df_cleaned[column].isna().sum()
                if missing_count > 0:
                    if 'hour' in df_cleaned.columns and 'month' in df_cleaned.columns:
                        # Saat ve ay kombinasyonu ile doldur
                        hourly_monthly_means = df_cleaned.groupby(['hour', 'month'])[column].transform('mean')
                        df_cleaned[column] = df_cleaned[column].fillna(hourly_monthly_means)
                        
                        # Hala missing varsa saatlik ortalama
                        remaining_missing = df_cleaned[column].isna().sum()
                        if remaining_missing > 0:
                            hourly_means = df_cleaned.groupby('hour')[column].transform('mean')
                            df_cleaned[column] = df_cleaned[column].fillna(hourly_means)
                    elif 'hour' in df_cleaned.columns:
                        # Saatlik ortalama ile doldur
                        hourly_means = df_cleaned.groupby('hour')[column].transform('mean')
                        df_cleaned[column] = df_cleaned[column].fillna(hourly_means)
                    else:
                        # Genel medyan ile doldur
                        df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                    
                    cleaning_report['actions'].append(f"{column}: {missing_count} missing value dolduruldu")
                    cleaning_report['values_corrected'] += missing_count
        
        # 8. Çok kalitesiz örnekleri tamamen kaldır (sadece çok kötü durumda)
        if target_column in df_cleaned.columns and 'shortwave_radiation' in df_cleaned.columns:
            # Aynı anda hem sıfır radyasyon hem yüksek güç veren satırları kaldır
            impossible_rows = (
                (df_cleaned['shortwave_radiation'] == 0) & 
                (df_cleaned[target_column] > 100)
            )
            impossible_count = impossible_rows.sum()
            
            if impossible_count > 0 and impossible_count < len(df_cleaned) * 0.05:  # %5'ten az ise kaldır
                df_cleaned = df_cleaned[~impossible_rows]
                cleaning_report['actions'].append(f"Fiziksel olarak imkansız satırlar kaldırıldı: {impossible_count}")
                cleaning_report['rows_removed'] += impossible_count
        
        cleaning_report['final_rows'] = len(df_cleaned)
        cleaning_report['data_retention'] = (cleaning_report['final_rows'] / cleaning_report['original_rows']) * 100
        
        logger.info(f"AGRESIF veri temizleme tamamlandı. {cleaning_report['values_corrected']} düzeltme, {cleaning_report['rows_removed']} satır kaldırıldı, %{cleaning_report['data_retention']:.1f} veri korundu.")
        
        return df_cleaned, cleaning_report
    
    @staticmethod
    def create_solar_aware_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Güneş enerjisi domain bilgisi ile gelişmiş feature engineering.
        
        Args:
            df: Feature engineering yapılacak DataFrame
            
        Returns:
            Gelişmiş özelliklerle DataFrame
        """
        logger.info("Güneş enerjisi aware feature engineering başlatılıyor...")
        
        df_enhanced = df.copy()
        
        # 1. Güneş radyasyonu composite features
        radiation_columns = [
            'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
            'global_tilted_irradiance', 'direct_normal_irradiance'
        ]
        
        available_radiation_cols = [col for col in radiation_columns if col in df_enhanced.columns]
        
        if available_radiation_cols:
            # Total radiation index
            df_enhanced['total_radiation_index'] = df_enhanced[available_radiation_cols].sum(axis=1)
            
            # Radiation efficiency (direct vs diffuse ratio)
            if 'direct_radiation' in df_enhanced.columns and 'diffuse_radiation' in df_enhanced.columns:
                df_enhanced['radiation_efficiency'] = (
                    df_enhanced['direct_radiation'] / 
                    (df_enhanced['diffuse_radiation'] + 1)  # +1 to avoid division by zero
                )
        
        # 2. Güneş açısı ve gün uzunluğu proxy'leri
        if 'hour' in df_enhanced.columns and 'month' in df_enhanced.columns:
            # Solar elevation proxy (simplified)
            df_enhanced['solar_elevation_proxy'] = np.cos(
                2 * np.pi * (df_enhanced['hour'] - 12) / 24
            ) * np.cos(2 * np.pi * (df_enhanced['month'] - 6) / 12)
            
            # Daylight indicator (approximate)
            df_enhanced['is_daylight'] = (
                (df_enhanced['hour'] >= 6) & (df_enhanced['hour'] <= 18)
            ).astype(int)
            
            # Peak solar hours
            df_enhanced['is_peak_solar'] = (
                (df_enhanced['hour'] >= 10) & (df_enhanced['hour'] <= 14)
            ).astype(int)
        
        # 3. Hava durumu composite features
        if 'temperature' in df_enhanced.columns and 'relative_humidity' in df_enhanced.columns:
            # Heat index proxy
            df_enhanced['heat_comfort'] = df_enhanced['temperature'] * (1 - df_enhanced['relative_humidity'] / 100)
            
            # Panel efficiency proxy (cooler is better for solar panels)
            df_enhanced['panel_efficiency_proxy'] = np.exp(-((df_enhanced['temperature'] - 25) / 20) ** 2)
        
        # 4. Mevsimsel faktörler
        if 'month' in df_enhanced.columns:
            # Summer/winter classification
            df_enhanced['season_summer'] = ((df_enhanced['month'] >= 6) & (df_enhanced['month'] <= 8)).astype(int)
            df_enhanced['season_winter'] = ((df_enhanced['month'] <= 2) | (df_enhanced['month'] >= 12)).astype(int)
            
            # Daylight length proxy
            df_enhanced['daylight_length_proxy'] = 12 + 4 * np.sin(2 * np.pi * (df_enhanced['month'] - 3) / 12)
        
        # 5. Etkileşim özellikleri (interaction features)
        if 'temperature' in df_enhanced.columns and 'total_radiation_index' in df_enhanced.columns:
            df_enhanced['temp_radiation_interaction'] = (
                df_enhanced['temperature'] * df_enhanced['total_radiation_index'] / 1000
            )
        
        if 'wind_speed' in df_enhanced.columns and 'temperature' in df_enhanced.columns:
            # Wind cooling effect on panels
            df_enhanced['wind_cooling_effect'] = df_enhanced['wind_speed'] * np.exp(-df_enhanced['temperature'] / 30)
        
        # 6. Güneş radyasyonu threshold features
        if 'shortwave_radiation' in df_enhanced.columns:
            df_enhanced['high_radiation'] = (df_enhanced['shortwave_radiation'] > 500).astype(int)
            df_enhanced['low_radiation'] = (df_enhanced['shortwave_radiation'] < 100).astype(int)
            df_enhanced['zero_radiation'] = (df_enhanced['shortwave_radiation'] == 0).astype(int)
        
        new_features = [col for col in df_enhanced.columns if col not in df.columns]
        logger.info(f"Feature engineering tamamlandı. {len(new_features)} yeni özellik eklendi: {new_features}")
        
        return df_enhanced
    
    @staticmethod
    def validate_model_input_data(df: pd.DataFrame, required_features: List[str]) -> Dict[str, Any]:
        """
        Model input verilerinin kalitesini doğrular.
        
        Args:
            df: Doğrulanacak DataFrame
            required_features: Gerekli özellikler listesi
            
        Returns:
            Doğrulama raporu
        """
        logger.info("Model input data doğrulaması başlatılıyor...")
        
        validation_report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'feature_coverage': {},
            'data_quality_score': 100.0
        }
        
        # 1. Gerekli özellikler kontrolü
        missing_features = [feat for feat in required_features if feat not in df.columns]
        if missing_features:
            validation_report['errors'].append(f"Eksik özellikler: {missing_features}")
            validation_report['is_valid'] = False
            validation_report['data_quality_score'] -= len(missing_features) * 10
        
        # 2. Her özellik için kalite analizi
        for feature in required_features:
            if feature in df.columns:
                feature_quality = {
                    'missing_percentage': (df[feature].isna().sum() / len(df)) * 100,
                    'zero_percentage': (df[feature] == 0).sum() / len(df) * 100 if df[feature].dtype in ['float64', 'int64'] else 0,
                    'unique_values': df[feature].nunique(),
                    'data_type': str(df[feature].dtype)
                }
                
                validation_report['feature_coverage'][feature] = feature_quality
                
                # Kalite skorunu düşür
                if feature_quality['missing_percentage'] > 10:
                    validation_report['warnings'].append(f"{feature}: %{feature_quality['missing_percentage']:.1f} missing")
                    validation_report['data_quality_score'] -= feature_quality['missing_percentage'] / 2
                
                if feature_quality['unique_values'] == 1:
                    validation_report['warnings'].append(f"{feature}: tek değer (constant)")
                    validation_report['data_quality_score'] -= 5
        
        # 3. Güneş enerjisi specific validations
        if 'hour' in df.columns:
            night_hours = df[df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])]
            radiation_cols = ['shortwave_radiation', 'direct_radiation', 'global_tilted_irradiance']
            
            for rad_col in radiation_cols:
                if rad_col in df.columns:
                    night_radiation = night_hours[night_hours[rad_col] > 0]
                    if len(night_radiation) > 0:
                        validation_report['warnings'].append(
                            f"Gece saatlerinde {rad_col} > 0: {len(night_radiation)} örnek"
                        )
                        validation_report['data_quality_score'] -= 2
        
        validation_report['data_quality_score'] = max(0, validation_report['data_quality_score'])
        
        logger.info(f"Data doğrulaması tamamlandı. Kalite skoru: {validation_report['data_quality_score']:.1f}/100")
        
        return validation_report 