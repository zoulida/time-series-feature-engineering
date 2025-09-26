
# -*- coding: utf-8 -*-
"""
tsfresh因子计算模板
提供tsfresh特征计算的基本模板和示例
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_tsfresh_factors(data: pd.DataFrame, 
                            feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算tsfresh因子
    
    Parameters:
    data: DataFrame, 包含时间序列数据的DataFrame
          需要包含时间列和数值列
    feature_names: List[str], 要计算的特征名称列表
                  如果为None，则计算所有可用特征
    
    Returns:
    DataFrame: 包含所有tsfresh特征值的DataFrame
    """
    result = data.copy()
    
    # 示例：计算基本统计特征
    if feature_names is None or 'mean' in feature_names:
        result['tsfresh_mean'] = data['close'].rolling(window=5).mean()
    
    if feature_names is None or 'std' in feature_names:
        result['tsfresh_std'] = data['close'].rolling(window=5).std()
    
    if feature_names is None or 'max' in feature_names:
        result['tsfresh_max'] = data['close'].rolling(window=5).max()
    
    if feature_names is None or 'min' in feature_names:
        result['tsfresh_min'] = data['close'].rolling(window=5).min()
    
    # 示例：计算变化率特征
    if feature_names is None or 'mean_abs_change' in feature_names:
        result['tsfresh_mean_abs_change'] = data['close'].diff().abs().rolling(window=5).mean()
    
    # 示例：计算复杂度特征
    if feature_names is None or 'approximate_entropy' in feature_names:
        result['tsfresh_approx_entropy'] = calculate_approximate_entropy(data['close'], m=2, r=0.2)
    
    return result


def calculate_approximate_entropy(series: pd.Series, m: int = 2, r: float = 0.2) -> pd.Series:
    """
    计算近似熵
    
    Parameters:
    series: 时间序列数据
    m: 模式长度
    r: 容差参数
    
    Returns:
    Series: 近似熵值
    """
    def _maxdist(xi, xj, N):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _approximate_entropy(U, m, r):
        N = len(U)
        C = np.zeros(N - m + 1)
        
        for i in range(N - m + 1):
            template_i = U[i:i + m]
            for j in range(N - m + 1):
                template_j = U[j:j + m]
                if _maxdist(template_i, template_j, m) <= r:
                    C[i] += 1.0
        
        phi = np.mean(np.log(C / float(N - m + 1.0)))
        return phi
    
    # 计算滚动近似熵
    result = pd.Series(index=series.index, dtype=float)
    window_size = 20  # 使用20个数据点计算近似熵
    
    for i in range(window_size, len(series)):
        window_data = series.iloc[i-window_size:i].values
        if len(window_data) >= window_size:
            result.iloc[i] = _approximate_entropy(window_data, m, r)
    
    return result


def get_tsfresh_feature_categories() -> Dict[str, List[str]]:
    """
    获取tsfresh特征分类
    
    Returns:
    Dict: 特征分类字典
    """
    return {
        '统计特征': [
            'mean', 'std', 'var', 'skewness', 'kurtosis', 'median',
            'min', 'max', 'sum', 'abs_energy', 'absolute_sum_of_changes'
        ],
        '频域特征': [
            'fft_aggregated', 'fft_coefficient', 'spectral_centroid',
            'spectral_entropy', 'spectral_roll_off', 'spectral_bandwidth'
        ],
        '复杂度特征': [
            'approximate_entropy', 'sample_entropy', 'permutation_entropy',
            'multiscale_entropy', 'fuzzy_entropy'
        ],
        '变化率特征': [
            'mean_abs_change', 'mean_change', 'number_peaks',
            'number_crossings', 'number_cwt_peaks'
        ],
        '自相关特征': [
            'autocorrelation', 'partial_autocorrelation', 'lag',
            'cross_correlation', 'autocorrelation_lag'
        ],
        '分形特征': [
            'detrended_fluctuation_analysis', 'hurst_exponent',
            'fractal_dimension', 'higuchi_fractal_dimension'
        ]
    }


def create_tsfresh_feature_config() -> Dict[str, Dict]:
    """
    创建tsfresh特征配置
    
    Returns:
    Dict: tsfresh特征配置字典
    """
    return {
        'mean': {'f_agg': 'mean'},
        'std': {'f_agg': 'std'},
        'max': {'f_agg': 'max'},
        'min': {'f_agg': 'min'},
        'skewness': {'f_agg': 'skew'},
        'kurtosis': {'f_agg': 'kurt'},
        'abs_energy': {},
        'absolute_sum_of_changes': {},
        'approximate_entropy': {'m': 2, 'r': 0.2},
        'sample_entropy': {'m': 2, 'r': 0.2},
        'permutation_entropy': {'tau': 1, 'dimension': 3},
        'autocorrelation': {'lag': 1},
        'partial_autocorrelation': {'lag': 1},
        'hurst_exponent': {},
        'detrended_fluctuation_analysis': {}
    }


def main():
    """主函数，演示tsfresh因子计算"""
    print("=" * 60)
    print("tsfresh因子计算模板演示")
    print("=" * 60)
    
    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.02),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    print(f"示例数据形状: {data.shape}")
    print(f"数据列: {list(data.columns)}")
    
    # 计算tsfresh因子
    features = ['mean', 'std', 'max', 'min', 'mean_abs_change']
    result = calculate_tsfresh_factors(data, features)
    
    print(f"\n计算后的数据形状: {result.shape}")
    print(f"新增的tsfresh特征列: {[col for col in result.columns if col.startswith('tsfresh_')]}")
    
    # 显示特征分类
    categories = get_tsfresh_feature_categories()
    print(f"\ntsfresh特征分类:")
    for category, features in categories.items():
        print(f"  {category}: {len(features)}个特征")
    
    # 显示特征配置
    config = create_tsfresh_feature_config()
    print(f"\ntsfresh特征配置示例:")
    for feature, params in list(config.items())[:5]:
        print(f"  {feature}: {params}")
    
    print("\n" + "=" * 60)
    print("tsfresh因子计算模板演示完成！")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
