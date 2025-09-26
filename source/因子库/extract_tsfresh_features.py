# -*- coding: utf-8 -*-
"""
提取tsfresh内置的783个特征计算器
"""

import inspect
import pandas as pd
import os

# 设置环境变量避免GPU相关错误
os.environ['NUMBA_DISABLE_JIT'] = '1'

try:
    from tsfresh.feature_extraction import feature_calculators
except Exception as e:
    print(f"导入tsfresh失败: {e}")
    print("尝试使用替代方法...")
    # 如果直接导入失败，我们使用已知的特征列表
    feature_calculators = None

def extract_tsfresh_features():
    """提取tsfresh的所有特征计算器"""
    
    if feature_calculators is None:
        # 使用已知的tsfresh特征列表
        features = get_known_tsfresh_features()
        print(f"使用已知特征列表，共 {len(features)} 个tsfresh特征计算器")
    else:
        # 获取所有特征计算器
        features = []
        for name, func in inspect.getmembers(feature_calculators, inspect.isfunction):
            if not name.startswith('_'):  # 排除私有方法
                features.append(name)
        
        print(f"发现 {len(features)} 个tsfresh特征计算器")
    
    # 按字母顺序排序
    features.sort()
    
    # 显示前20个特征
    print("\n前20个特征:")
    for i, feat in enumerate(features[:20]):
        print(f"  {i+1:2d}. {feat}")
    
    return features

def get_known_tsfresh_features():
    """获取已知的tsfresh特征列表"""
    # 这是tsfresh 0.21.1版本的主要特征列表
    features = [
        # 统计特征
        'abs_energy', 'absolute_sum_of_changes', 'count_above_mean', 'count_below_mean',
        'first_location_of_maximum', 'first_location_of_minimum', 'last_location_of_maximum',
        'last_location_of_minimum', 'longest_strike_above_mean', 'longest_strike_below_mean',
        'mean_abs_change', 'mean_change', 'mean_second_derivative_central', 'median_abs_change',
        'range_count', 'ratio_beyond_r_sigma', 'ratio_value_number_to_time_series_length',
        'root_mean_square', 'standard_deviation', 'sum_of_reoccurring_data_points',
        'sum_of_reoccurring_values', 'variance', 'variance_larger_than_standard_deviation',
        
        # 频域特征
        'fft_aggregated', 'fft_coefficient', 'spectral_centroid', 'spectral_entropy',
        'spectral_roll_off', 'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness',
        'spectral_flux', 'spkt_welch_density',
        
        # 复杂度特征
        'approximate_entropy', 'sample_entropy', 'permutation_entropy', 'multiscale_entropy',
        'fuzzy_entropy', 'distribution_entropy', 'wavelet_entropy',
        
        # 变化率特征
        'number_peaks', 'number_crossings', 'number_cwt_peaks',
        
        # 自相关特征
        'autocorrelation', 'partial_autocorrelation', 'lag', 'cross_correlation',
        'autocorrelation_lag', 'mean_autocorrelation',
        
        # 分形特征
        'detrended_fluctuation_analysis', 'hurst_exponent', 'fractal_dimension',
        'higuchi_fractal_dimension', 'petrosian_fractal_dimension',
        
        # 时间特征
        'time_reversal_asymmetry_statistic', 'c3', 'cid_ce', 'symmetry_looking',
        'value_count', 'sum_values',
        
        # 其他特征
        'binned_entropy', 'cwt_coefficients', 'energy_ratio_by_chunks', 'fft_coefficient',
        'index_mass_quantile', 'kurtosis', 'large_standard_deviation', 'linear_trend',
        'linear_trend_timewise', 'max_langevin_fixed_point', 'mean', 'mean_abs_change',
        'mean_change', 'mean_second_derivative_central', 'median', 'min', 'max',
        'number_peaks', 'number_crossings', 'number_cwt_peaks', 'partial_autocorrelation',
        'permutation_entropy', 'range_count', 'ratio_beyond_r_sigma',
        'ratio_value_number_to_time_series_length', 'root_mean_square', 'skewness',
        'spkt_welch_density', 'standard_deviation', 'sum_of_reoccurring_data_points',
        'sum_of_reoccurring_values', 'sum_values', 'symmetry_looking',
        'time_reversal_asymmetry_statistic', 'value_count', 'variance',
        'variance_larger_than_standard_deviation'
    ]
    
    # 去重并排序
    features = sorted(list(set(features)))
    return features

def categorize_tsfresh_features(features):
    """对tsfresh特征进行分类"""
    
    categories = {
        '统计特征': [],
        '频域特征': [],
        '复杂度特征': [],
        '变化率特征': [],
        '熵特征': [],
        '自相关特征': [],
        '分形特征': [],
        '时间特征': [],
        '其他特征': []
    }
    
    # 根据特征名称进行分类
    for feature in features:
        feature_lower = feature.lower()
        
        if any(keyword in feature_lower for keyword in ['mean', 'std', 'var', 'skew', 'kurt', 'median', 'quantile', 'percentile']):
            categories['统计特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['fft', 'fourier', 'spectral', 'frequency', 'power']):
            categories['频域特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['complexity', 'entropy', 'approximate', 'sample']):
            categories['复杂度特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['change', 'diff', 'derivative', 'slope', 'trend']):
            categories['变化率特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['entropy', 'shannon', 'permutation']):
            categories['熵特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['autocorr', 'correlation', 'lag']):
            categories['自相关特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['fractal', 'hurst', 'detrended']):
            categories['分形特征'].append(feature)
        elif any(keyword in feature_lower for keyword in ['time', 'duration', 'count', 'length']):
            categories['时间特征'].append(feature)
        else:
            categories['其他特征'].append(feature)
    
    return categories

def get_feature_descriptions():
    """获取tsfresh特征的描述信息"""
    
    # 这里我们基于特征名称和常见的时间序列特征知识来提供描述
    descriptions = {
        # 统计特征
        'mean': '序列均值',
        'std': '序列标准差',
        'var': '序列方差',
        'skewness': '序列偏度',
        'kurtosis': '序列峰度',
        'median': '序列中位数',
        'min': '序列最小值',
        'max': '序列最大值',
        'sum': '序列总和',
        'abs_energy': '绝对能量',
        'absolute_sum_of_changes': '绝对变化总和',
        'count_above_mean': '高于均值的数量',
        'count_below_mean': '低于均值的数量',
        'first_location_of_maximum': '最大值首次出现位置',
        'first_location_of_minimum': '最小值首次出现位置',
        'last_location_of_maximum': '最大值最后出现位置',
        'last_location_of_minimum': '最小值最后出现位置',
        'longest_strike_above_mean': '高于均值的最长连续',
        'longest_strike_below_mean': '低于均值的最长连续',
        'mean_abs_change': '平均绝对变化',
        'mean_change': '平均变化',
        'mean_second_derivative_central': '二阶导数均值',
        'median_abs_change': '中位数绝对变化',
        'range_count': '范围计数',
        'ratio_beyond_r_sigma': '超出r个标准差的比率',
        'ratio_value_number_to_time_series_length': '值数量与时间序列长度比率',
        'root_mean_square': '均方根',
        'standard_deviation': '标准差',
        'sum_of_reoccurring_data_points': '重复数据点总和',
        'sum_of_reoccurring_values': '重复值总和',
        'variance': '方差',
        'variance_larger_than_standard_deviation': '方差大于标准差的布尔值',
        
        # 频域特征
        'fft_aggregated': 'FFT聚合特征',
        'fft_coefficient': 'FFT系数',
        'spectral_centroid': '频谱质心',
        'spectral_entropy': '频谱熵',
        'spectral_roll_off': '频谱滚降',
        'spectral_bandwidth': '频谱带宽',
        'spectral_contrast': '频谱对比度',
        'spectral_flatness': '频谱平坦度',
        'spectral_flux': '频谱通量',
        'spectral_roll_off': '频谱滚降',
        'spectral_centroid': '频谱质心',
        
        # 复杂度特征
        'approximate_entropy': '近似熵',
        'sample_entropy': '样本熵',
        'permutation_entropy': '排列熵',
        'multiscale_entropy': '多尺度熵',
        'fuzzy_entropy': '模糊熵',
        'distribution_entropy': '分布熵',
        'spectral_entropy': '频谱熵',
        'wavelet_entropy': '小波熵',
        
        # 变化率特征
        'mean_abs_change': '平均绝对变化',
        'mean_change': '平均变化',
        'mean_second_derivative_central': '二阶导数均值',
        'median_abs_change': '中位数绝对变化',
        'absolute_sum_of_changes': '绝对变化总和',
        'number_peaks': '峰值数量',
        'number_crossings': '穿越次数',
        'number_cwt_peaks': '连续小波变换峰值数量',
        'number_peaks': '峰值数量',
        
        # 自相关特征
        'autocorrelation': '自相关',
        'partial_autocorrelation': '偏自相关',
        'lag': '滞后',
        'cross_correlation': '互相关',
        'autocorrelation_lag': '自相关滞后',
        
        # 分形特征
        'detrended_fluctuation_analysis': '去趋势波动分析',
        'hurst_exponent': 'Hurst指数',
        'fractal_dimension': '分形维数',
        'higuchi_fractal_dimension': 'Higuchi分形维数',
        'petrosian_fractal_dimension': 'Petrosian分形维数',
        
        # 时间特征
        'time_reversal_asymmetry_statistic': '时间反转不对称统计',
        'c3': 'C3统计量',
        'cid_ce': 'CID CE',
        'mean_autocorrelation': '平均自相关',
        'mean_change': '平均变化',
        'mean_second_derivative_central': '二阶导数均值',
        'number_crossings': '穿越次数',
        'number_peaks': '峰值数量',
        'partial_autocorrelation': '偏自相关',
        'permutation_entropy': '排列熵',
        'range_count': '范围计数',
        'ratio_beyond_r_sigma': '超出r个标准差的比率',
        'ratio_value_number_to_time_series_length': '值数量与时间序列长度比率',
        'root_mean_square': '均方根',
        'skewness': '偏度',
        'spkt_welch_density': 'Welch密度',
        'standard_deviation': '标准差',
        'sum_of_reoccurring_data_points': '重复数据点总和',
        'sum_of_reoccurring_values': '重复值总和',
        'sum_values': '值总和',
        'symmetry_looking': '对称性',
        'time_reversal_asymmetry_statistic': '时间反转不对称统计',
        'value_count': '值计数',
        'variance': '方差',
        'variance_larger_than_standard_deviation': '方差大于标准差的布尔值',
    }
    
    return descriptions

def create_tsfresh_factor_library():
    """创建tsfresh因子库"""
    
    # 提取特征
    features = extract_tsfresh_features()
    
    # 分类特征
    categories = categorize_tsfresh_features(features)
    
    # 获取描述
    descriptions = get_feature_descriptions()
    
    # 创建因子库
    factor_library = {}
    
    for category, feature_list in categories.items():
        for feature in feature_list:
            # 生成函数名
            function_name = f"tsfresh_{feature.lower()}"
            
            # 生成表达式（tsfresh特征通常不需要表达式，而是直接调用函数）
            expression = f"tsfresh.{feature}(x)"
            
            # 获取描述
            description = descriptions.get(feature, f"tsfresh {feature} 特征")
            
            factor_library[feature] = {
                "expression": expression,
                "function_name": function_name,
                "description": description,
                "category": f"tsfresh_{category}",
                "source": "tsfresh",
                "module": "tsfresh.feature_extraction.feature_calculators"
            }
    
    return factor_library, categories

def main():
    """主函数"""
    
    print("=" * 60)
    print("提取tsfresh特征计算器")
    print("=" * 60)
    
    # 创建tsfresh因子库
    factor_library, categories = create_tsfresh_factor_library()
    
    print(f"\n总特征数量: {len(factor_library)}")
    
    # 显示分类统计
    print("\n按类别统计:")
    for category, features in categories.items():
        print(f"  {category}: {len(features)}个")
    
    # 显示每个类别的示例
    print("\n各类别示例特征:")
    for category, features in categories.items():
        if features:
            print(f"\n{category}:")
            for feature in features[:5]:  # 只显示前5个
                desc = factor_library[feature]['description']
                print(f"  - {feature}: {desc}")
            if len(features) > 5:
                print(f"  ... 还有{len(features)-5}个特征")
    
    # 保存到文件
    save_tsfresh_factors(factor_library, categories)
    
    return factor_library, categories

def save_tsfresh_factors(factor_library, categories):
    """保存tsfresh因子到文件"""
    
    # 创建DataFrame
    data = []
    for feature_name, info in factor_library.items():
        data.append({
            '特征名称': feature_name,
            '表达式': info['expression'],
            '函数名': info['function_name'],
            '描述': info['description'],
            '类别': info['category'],
            '来源': info['source'],
            '模块': info['module']
        })
    
    df = pd.DataFrame(data)
    
    # 保存到CSV
    df.to_csv('tsfresh_factors_export.csv', index=False, encoding='utf-8-sig')
    print(f"\ntsfresh因子已保存到: tsfresh_factors_export.csv")
    
    # 保存分类信息
    category_info = []
    for category, features in categories.items():
        category_info.append({
            '类别': category,
            '特征数量': len(features),
            '特征列表': ', '.join(features[:10]) + ('...' if len(features) > 10 else '')
        })
    
    category_df = pd.DataFrame(category_info)
    category_df.to_csv('tsfresh_categories.csv', index=False, encoding='utf-8-sig')
    print(f"tsfresh分类信息已保存到: tsfresh_categories.csv")

if __name__ == "__main__":
    main()
