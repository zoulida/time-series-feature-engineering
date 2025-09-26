#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版时间序列特征提取器
提供类似tsfresh的功能，但不依赖GPU/CUDA
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureExtractor:
    """增强版时间序列特征提取器"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, time_series_df, column_id='id', column_sort='time', column_value='value'):
        """
        提取时间序列特征
        
        参数:
        time_series_df: 时间序列数据框
        column_id: ID列名
        column_sort: 时间列名
        column_value: 值列名
        
        返回:
        features_df: 特征数据框
        """
        print("正在使用增强版特征提取器...")
        
        # 按ID分组处理
        features_list = []
        
        for group_id, group_data in time_series_df.groupby(column_id):
            # 按时间排序
            group_data = group_data.sort_values(column_sort)
            values = group_data[column_value].values
            
            # 提取特征
            features = self._extract_single_series_features(values)
            features[column_id] = group_id
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        print(f"提取了 {features_df.shape[1]-1} 个特征")
        
        return features_df
    
    def _extract_single_series_features(self, values):
        """为单个时间序列提取特征"""
        features = {}
        
        # 基础统计特征
        features.update(self._basic_statistics(values))
        
        # 趋势特征
        features.update(self._trend_features(values))
        
        # 周期性特征
        features.update(self._periodicity_features(values))
        
        # 波动性特征
        features.update(self._volatility_features(values))
        
        # 分布特征
        features.update(self._distribution_features(values))
        
        # 自相关特征
        features.update(self._autocorrelation_features(values))
        
        # 频域特征
        features.update(self._frequency_features(values))
        
        # 峰值特征
        features.update(self._peak_features(values))
        
        # 变化率特征
        features.update(self._change_rate_features(values))
        
        return features
    
    def _basic_statistics(self, values):
        """基础统计特征"""
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'range': np.max(values) - np.min(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'sum': np.sum(values),
            'count': len(values)
        }
    
    def _trend_features(self, values):
        """趋势特征"""
        n = len(values)
        x = np.arange(n)
        
        # 线性趋势
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        return {
            'linear_trend_slope': slope,
            'linear_trend_intercept': intercept,
            'linear_trend_r_squared': r_value ** 2,
            'linear_trend_p_value': p_value,
            'trend_strength': abs(r_value),
            'trend_direction': np.sign(slope)
        }
    
    def _periodicity_features(self, values):
        """周期性特征"""
        n = len(values)
        if n < 4:
            return {'seasonal_strength': 0, 'periodicity_score': 0}
        
        # 自相关分析
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]
        
        # 寻找周期性
        peaks, _ = find_peaks(autocorr[1:min(n//2, 50)], height=0.1)
        
        if len(peaks) > 0:
            period = peaks[0] + 1
            seasonal_strength = autocorr[period] if period < len(autocorr) else 0
        else:
            period = 0
            seasonal_strength = 0
        
        return {
            'seasonal_strength': seasonal_strength,
            'periodicity_score': len(peaks) / max(1, n//10),
            'dominant_period': period
        }
    
    def _volatility_features(self, values):
        """波动性特征"""
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return {'volatility': 0, 'volatility_of_volatility': 0}
        
        volatility = np.std(returns)
        vol_of_vol = np.std(np.abs(returns)) if len(returns) > 1 else 0
        
        return {
            'volatility': volatility,
            'volatility_of_volatility': vol_of_vol,
            'mean_absolute_change': np.mean(np.abs(np.diff(values))),
            'max_absolute_change': np.max(np.abs(np.diff(values))),
            'change_std': np.std(np.diff(values))
        }
    
    def _distribution_features(self, values):
        """分布特征"""
        return {
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'normality_test_p': stats.normaltest(values)[1],
            'entropy': stats.entropy(np.histogram(values, bins=min(20, len(values)//5))[0] + 1e-10)
        }
    
    def _autocorrelation_features(self, values):
        """自相关特征"""
        n = len(values)
        if n < 2:
            return {'autocorr_lag1': 0, 'autocorr_lag2': 0}
        
        # 滞后1和2的自相关
        lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1] if n > 1 else 0
        lag2_corr = np.corrcoef(values[:-2], values[2:])[0, 1] if n > 2 else 0
        
        return {
            'autocorr_lag1': lag1_corr if not np.isnan(lag1_corr) else 0,
            'autocorr_lag2': lag2_corr if not np.isnan(lag2_corr) else 0,
            'autocorr_mean': np.mean([lag1_corr, lag2_corr]) if not np.isnan(lag1_corr) and not np.isnan(lag2_corr) else 0
        }
    
    def _frequency_features(self, values):
        """频域特征"""
        n = len(values)
        if n < 4:
            return {'dominant_frequency': 0, 'spectral_entropy': 0}
        
        # FFT分析
        fft_vals = fft(values)
        power_spectrum = np.abs(fft_vals[:n//2]) ** 2
        
        # 主导频率
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1
        dominant_frequency = dominant_freq_idx / n
        
        # 频谱熵
        normalized_spectrum = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-10))
        
        return {
            'dominant_frequency': dominant_frequency,
            'spectral_entropy': spectral_entropy,
            'spectral_centroid': np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum),
            'spectral_bandwidth': np.sqrt(np.sum(((np.arange(len(power_spectrum)) - np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)) ** 2) * power_spectrum) / np.sum(power_spectrum))
        }
    
    def _peak_features(self, values):
        """峰值特征"""
        peaks, properties = find_peaks(values, height=np.mean(values))
        troughs, _ = find_peaks(-values, height=-np.mean(values))
        
        return {
            'peak_count': len(peaks),
            'trough_count': len(troughs),
            'peak_trough_ratio': len(peaks) / max(1, len(troughs)),
            'peak_height_mean': np.mean(properties['peak_heights']) if len(peaks) > 0 else 0,
            'peak_height_std': np.std(properties['peak_heights']) if len(peaks) > 1 else 0
        }
    
    def _change_rate_features(self, values):
        """变化率特征"""
        if len(values) < 2:
            return {'mean_change_rate': 0, 'change_rate_std': 0}
        
        change_rates = np.diff(values) / values[:-1]
        change_rates = change_rates[~np.isnan(change_rates)]
        
        return {
            'mean_change_rate': np.mean(change_rates),
            'change_rate_std': np.std(change_rates),
            'positive_change_ratio': np.sum(change_rates > 0) / len(change_rates),
            'negative_change_ratio': np.sum(change_rates < 0) / len(change_rates)
        }

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    n_timesteps = 50
    
    data = []
    for i in range(n_samples):
        for t in range(n_timesteps):
            data.append({
                'id': i,
                'time': pd.Timestamp('2023-01-01') + pd.Timedelta(days=t),
                'value': np.random.normal(0, 1) + 0.1 * t
            })
    
    df = pd.DataFrame(data)
    
    # 提取特征
    extractor = EnhancedFeatureExtractor()
    features = extractor.extract_features(df)
    
    print(f"提取了 {features.shape[1]-1} 个特征")
    print("特征列名:", list(features.columns)[:-1]) 