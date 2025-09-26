#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsfresh时间序列特征工程与IC评分分析案例

本案例演示：
1. 生成模拟时间序列数据
2. 使用tsfresh进行自动特征工程
3. 计算各特征的IC评分（信息系数）
4. 特征重要性分析和可视化
"""

import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_time_series_data(n_samples=100, n_timesteps=50):
    """
    生成模拟时间序列数据
    
    参数:
    n_samples: 样本数量
    n_timesteps: 时间步长
    
    返回:
    time_series_df: 时间序列数据框
    target_values: 目标变量
    """
    print("正在生成模拟时间序列数据...")
    
    # 生成时间序列数据
    time_series_data = []
    target_values = []
    
    for i in range(n_samples):
        # 基础趋势
        trend = np.linspace(0, 10, n_timesteps)
        
        # 添加随机噪声
        noise = np.random.normal(0, 1, n_timesteps)
        
        # 添加周期性成分
        periodic = 2 * np.sin(2 * np.pi * np.arange(n_timesteps) / 10)
        
        # 添加跳跃
        jump = np.zeros(n_timesteps)
        if np.random.random() > 0.7:
            jump_point = np.random.randint(10, n_timesteps-10)
            jump[jump_point:] = np.random.normal(2, 0.5)
        
        # 组合所有成分
        series = trend + noise + periodic + jump
        
        # 创建时间戳
        timestamps = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        # 添加到数据列表
        for t, value in enumerate(series):
            time_series_data.append({
                'id': i,
                'time': timestamps[t],
                'value': value
            })
        
        # 生成目标变量（基于时间序列特征）
        # 目标变量 = 趋势强度 + 波动性 + 跳跃强度
        trend_strength = np.corrcoef(np.arange(n_timesteps), series)[0, 1]
        volatility = np.std(series)
        jump_intensity = np.sum(np.abs(np.diff(series)))
        
        target = 0.3 * trend_strength + 0.4 * volatility + 0.3 * jump_intensity / n_timesteps
        target_values.append(target)
    
    time_series_df = pd.DataFrame(time_series_data)
    target_df = pd.DataFrame({
        'id': range(n_samples),
        'target': target_values
    })
    
    print(f"生成了 {n_samples} 个样本，每个样本 {n_timesteps} 个时间点")
    return time_series_df, target_df

def extract_features_with_tsfresh(time_series_df, target_df, feature_level='minimal'):
    """
    使用tsfresh提取时间序列特征
    
    参数:
    time_series_df: 时间序列数据
    target_df: 目标变量数据
    feature_level: 特征提取级别
        - 'minimal': 最小特征集（快速，约10个特征）
        - 'efficient': 高效特征集（平衡，约50-100个特征）
        - 'comprehensive': 全面特征集（完整，约1000+个特征）
        - 'custom': 自定义特征集
    
    返回:
    features_df: 特征数据框
    """
    print(f"正在使用tsfresh提取特征（级别：{feature_level}）...")
    
    try:
        import os
        os.environ["NUMBA_DISABLE_CUDA"] = "1"   # 禁用 CUDA
        os.environ["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "0"
        # 将导入移到函数内部
        from tsfresh import extract_features, select_features
        from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
        
        # 根据级别选择特征提取参数
        if feature_level == 'minimal':
            fc_parameters = MinimalFCParameters()
            print("使用最小特征集（约10个特征）")
        elif feature_level == 'efficient':
            fc_parameters = EfficientFCParameters()
            print("使用高效特征集（约50-100个特征）")
        elif feature_level == 'comprehensive':
            fc_parameters = ComprehensiveFCParameters()
            print("使用全面特征集（约1000+个特征）")
        elif feature_level == 'custom':
            # 自定义特征集 - 选择最常用的特征类型
            fc_parameters = {
                'abs_energy': None,
                'absolute_maximum': None,
                'absolute_sum_of_changes': None,
                'autocorrelation': [{'lag': 1}, {'lag': 2}, {'lag': 5}],
                'binned_entropy': [{'max_bins': 10}],
                'change_quantiles': [{'ql': 0.1, 'qh': 0.9, 'isabs': False}, {'ql': 0.2, 'qh': 0.8, 'isabs': False}],
                'cid_ce': [{'normalize': True}],
                'count_above_mean': None,
                'count_below_mean': None,
                'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 5}],
                'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 1}],
                'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}],
                'fft_coefficient': [{'coeff': 0, 'attr': 'real'}, {'coeff': 1, 'attr': 'real'}, {'coeff': 2, 'attr': 'real'}],
                'first_location_of_maximum': None,
                'first_location_of_minimum': None,
                'fourier_entropy': [{'bins': 10}],
                'has_duplicate': None,
                'has_duplicate_max': None,
                'has_duplicate_min': None,
                'index_mass_quantile': [{'q': 0.1}, {'q': 0.5}, {'q': 0.9}],
                'kurtosis': None,
                'last_location_of_maximum': None,
                'last_location_of_minimum': None,
                'linear_trend': [{'attr': 'slope'}, {'attr': 'intercept'}, {'attr': 'rvalue'}, {'attr': 'pvalue'}, {'attr': 'stderr'}],
                'longest_strike_above_mean': None,
                'longest_strike_below_mean': None,
                'max_langevin_fixed_point': [{'m': 3, 'r': 30}],
                'maximum': None,
                'mean': None,
                'mean_abs_change': None,
                'mean_change': None,
                'mean_second_derivative_central': None,
                'median': None,
                'minimum': None,
                'number_crossing_m': [{'m': 0}],
                'number_cwt_peaks': [{'n': 5}],
                'number_peaks': [{'n': 5}],
                'partial_autocorrelation': [{'lag': 1}, {'lag': 2}, {'lag': 5}],
                'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
                'percentage_of_reoccurring_values_to_all_values': None,
                'permutation_entropy': [{'tau': 1, 'dim': 3}],
                'quantile': [{'q': 0.1}, {'q': 0.25}, {'q': 0.5}, {'q': 0.75}, {'q': 0.9}],
                'range_count': [{'min': 0, 'max': 1}],
                'ratio_beyond_r_sigma': [{'r': 0.5}, {'r': 1}, {'r': 1.5}, {'r': 2}],
                'ratio_value_number_to_time_series_length': None,
                'root_mean_square': None,
                'sample_entropy': None,
                'set_property': [{'property': 'min'}, {'property': 'max'}],
                'skewness': None,
                'spkt_welch_density': [{'coeff': 2}],
                'standard_deviation': None,
                'sum_of_reoccurring_data_points': None,
                'sum_of_reoccurring_values': None,
                'sum_values': None,
                'symmetry_looking': None,
                'time_reversal_asymmetry_statistic': [{'lag': 1}],
                'value_count': [{'value': 0}],
                'variance': None,
                'variance_larger_than_standard_deviation': None
            }
            print("使用自定义特征集（约80个特征）")
        else:
            print(f"未知的特征级别 '{feature_level}'，使用最小特征集")
            fc_parameters = MinimalFCParameters()
        
        # 提取特征
        print("开始特征提取，这可能需要一些时间...")
        features_df = extract_features(time_series_df, 
                                     column_id='id', 
                                     column_sort='time', 
                                     column_value='value',
                                     default_fc_parameters=fc_parameters,
                                     disable_progressbar=False,  # 显示进度条
                                     n_jobs=0)  # 使用所有CPU核心
        
        print(f"提取了 {features_df.shape[1]} 个特征")
        
        # 处理缺失值
        features_df = features_df.fillna(0)
        
        # 合并目标变量
        features_with_target = features_df.merge(target_df, left_index=True, right_on='id')
        
        return features_with_target
        
    except ImportError:
        print("tsfresh未安装，使用模拟特征...")
        traceback.print_exc()
        return generate_mock_features(target_df)
    except Exception as e:
        print(f"tsfresh特征提取失败: {e}")
        print("使用模拟特征...")
        return generate_mock_features(target_df)

def generate_mock_features(target_df):
    """
    生成模拟特征（当tsfresh不可用时使用）
    """
    n_samples = len(target_df)
    
    # 生成一些模拟特征
    mock_features = pd.DataFrame({
        'id': target_df['id'],
        'trend_strength': np.random.normal(0.5, 0.2, n_samples),
        'volatility': np.random.normal(1.0, 0.3, n_samples),
        'mean_value': np.random.normal(5.0, 1.0, n_samples),
        'std_value': np.random.normal(1.5, 0.5, n_samples),
        'max_value': np.random.normal(8.0, 1.5, n_samples),
        'min_value': np.random.normal(2.0, 1.0, n_samples),
        'range_value': np.random.normal(6.0, 2.0, n_samples),
        'skewness': np.random.normal(0.0, 0.5, n_samples),
        'kurtosis': np.random.normal(3.0, 1.0, n_samples),
        'autocorr_lag1': np.random.normal(0.8, 0.1, n_samples),
        'autocorr_lag2': np.random.normal(0.6, 0.2, n_samples),
        'linear_trend': np.random.normal(0.1, 0.05, n_samples),
        'seasonal_strength': np.random.normal(0.3, 0.1, n_samples),
        'jump_count': np.random.poisson(2, n_samples),
        'target': target_df['target']
    })
    
    print(f"生成了 {mock_features.shape[1]-2} 个模拟特征")
    return mock_features

def calculate_ic_scores(features_df):
    """
    计算各特征的IC评分（信息系数）
    
    参数:
    features_df: 包含特征和目标变量的数据框
    
    返回:
    ic_scores: IC评分数据框
    """
    print("正在计算IC评分...")
    
    # 获取特征列（排除id和target）
    feature_columns = [col for col in features_df.columns if col not in ['id', 'target']]
    target = features_df['target']
    
    ic_scores = []
    
    for feature in feature_columns:
        feature_values = features_df[feature]
        
        # 计算Pearson相关系数
        pearson_corr, pearson_p = stats.pearsonr(feature_values, target)
        
        # 计算Spearman相关系数
        spearman_corr, spearman_p = stats.spearmanr(feature_values, target)
        
        # 计算Kendall相关系数
        kendall_corr, kendall_p = stats.kendalltau(feature_values, target)
        
        ic_scores.append({
            'feature': feature,
            'pearson_ic': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_ic': spearman_corr,
            'spearman_p': spearman_p,
            'kendall_ic': kendall_corr,
            'kendall_p': kendall_p,
            'mean_ic': np.mean([abs(pearson_corr), abs(spearman_corr), abs(kendall_corr)])
        })
    
    ic_df = pd.DataFrame(ic_scores)
    
    # 按平均IC评分排序
    ic_df = ic_df.sort_values('mean_ic', ascending=False)
    
    print(f"计算了 {len(ic_df)} 个特征的IC评分")
    return ic_df

def visualize_ic_scores(ic_df, top_n=20):
    """
    可视化IC评分结果
    
    参数:
    ic_df: IC评分数据框
    top_n: 显示前N个特征
    """
    print("正在生成可视化图表...")
    
    # 过滤有效数据
    valid_ic_df = ic_df.dropna(subset=['mean_ic']).copy()
    if len(valid_ic_df) == 0:
        print("警告：没有有效的IC评分数据用于可视化")
        return
    
    # 按平均IC评分排序
    valid_ic_df = valid_ic_df.sort_values('mean_ic', ascending=False)
    top_features = valid_ic_df.head(top_n)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('时间序列特征IC评分分析结果', fontsize=16, fontweight='bold')
    
    # 1. 前N个特征的平均IC评分柱状图
    axes[0, 0].barh(range(len(top_features)), top_features['mean_ic'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0, 0].set_xlabel('平均IC评分')
    axes[0, 0].set_title(f'前{top_n}个特征的平均IC评分')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 三种IC评分的对比散点图
    axes[0, 1].scatter(valid_ic_df['pearson_ic'], valid_ic_df['spearman_ic'], 
                        alpha=0.6, s=30)
    axes[0, 1].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
    axes[0, 1].set_xlabel('Pearson IC')
    axes[0, 1].set_ylabel('Spearman IC')
    axes[0, 1].set_title('Pearson vs Spearman IC评分对比')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. IC评分分布直方图
    axes[1, 0].hist(valid_ic_df['mean_ic'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(valid_ic_df['mean_ic'].mean(), color='red', linestyle='--', 
                        label=f'平均值: {valid_ic_df["mean_ic"].mean():.3f}')
    axes[1, 0].set_xlabel('平均IC评分')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('IC评分分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 前N个特征的三种IC评分热力图
    top_features_for_heatmap = top_features[['pearson_ic', 'spearman_ic', 'kendall_ic']].T
    # 确保列名数量与特征数量匹配
    top_features_for_heatmap.columns = top_features['feature'].tolist()
    
    sns.heatmap(top_features_for_heatmap.T, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                ax=axes[1, 1],
                cbar_kws={'label': 'IC评分'})
    axes[1, 1].set_title('前N个特征的三种IC评分热力图')
    axes[1, 1].set_xlabel('特征')
    axes[1, 1].set_ylabel('IC类型')
    
    plt.tight_layout()
    
    # 确保可视化输出目录存在 - 修复路径问题
    output_dir = Path(__file__).parent.parent / "可视化文件"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ic_analysis_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图表已保存到: {output_path}")

def generate_feature_importance_report(ic_df, output_file='feature_importance_report.txt'):
    """
    生成特征重要性报告
    
    参数:
    ic_df: IC评分数据框
    output_file: 输出文件名
    """
    print("正在生成特征重要性报告...")
    
    # 确保报告输出目录存在 - 修复路径问题
    if output_file.startswith('../'):
        # 如果是相对路径，转换为绝对路径
        output_dir = Path(__file__).parent.parent / output_file[3:]
        output_file = str(output_dir)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("时间序列特征IC评分分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"特征总数: {len(ic_df)}\n")
        f.write(f"平均IC评分: {ic_df['mean_ic'].mean():.4f}\n")
        f.write(f"IC评分标准差: {ic_df['mean_ic'].std():.4f}\n\n")
        
        f.write("特征重要性排名 (按平均IC评分):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'排名':<4} {'特征名':<20} {'平均IC':<8} {'Pearson':<10} {'Spearman':<10} {'Kendall':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, (_, row) in enumerate(ic_df.iterrows(), 1):
            f.write(f"{i:<4} {row['feature']:<20} {row['mean_ic']:<8.4f} {row['pearson_ic']:<10.4f} {row['spearman_ic']:<10.4f} {row['kendall_ic']:<10.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("高价值特征 (平均IC > 0.1):\n")
        f.write("-" * 60 + "\n")
        
        high_value_features = ic_df[ic_df['mean_ic'] > 0.1]
        for _, row in high_value_features.iterrows():
            f.write(f"• {row['feature']}: {row['mean_ic']:.4f}\n")
        
        f.write(f"\n高价值特征数量: {len(high_value_features)} / {len(ic_df)}\n")
        f.write(f"高价值特征比例: {len(high_value_features)/len(ic_df)*100:.1f}%\n")
    
    print(f"报告已保存到: {output_file}")

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="tsfresh时间序列特征工程与IC评分分析")
    parser.add_argument("--feature-level", type=str, default="minimal", 
                        choices=["minimal", "efficient", "comprehensive", "custom"],
                        help="特征提取级别：minimal(快速), efficient(平衡), comprehensive(全面), custom(自定义)")
    parser.add_argument("--samples", type=int, default=200, help="样本数量")
    parser.add_argument("--timesteps", type=int, default=100, help="时间步长")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("tsfresh时间序列特征工程与IC评分分析案例")
    print("=" * 60)
    print(f"特征提取级别: {args.feature_level}")
    print(f"样本数量: {args.samples}")
    print(f"时间步长: {args.timesteps}")
    print("=" * 60)
    
    # 1. 生成数据
    time_series_df, target_df = generate_time_series_data(n_samples=args.samples, n_timesteps=args.timesteps)
    
    # 2. 特征提取
    features_df = extract_features_with_tsfresh(time_series_df, target_df, feature_level=args.feature_level)
    
    # 3. 计算IC评分
    ic_df = calculate_ic_scores(features_df)
    
    # 4. 可视化结果
    visualize_ic_scores(ic_df, top_n=min(20, len(ic_df)))
    
    # 5. 生成报告
    generate_feature_importance_report(ic_df)
    
    # 6. 显示结果摘要
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)
    print(f"总特征数: {len(ic_df)}")
    print(f"平均IC评分: {ic_df['mean_ic'].mean():.4f}")
    print(f"最高IC评分: {ic_df['mean_ic'].max():.4f}")
    print(f"高价值特征数 (IC > 0.1): {len(ic_df[ic_df['mean_ic'] > 0.1])}")
    
    print("\n前5个最重要的特征:")
    for i, (_, row) in enumerate(ic_df.head(5).iterrows(), 1):
        print(f"{i}. {row['feature']}: {row['mean_ic']:.4f}")
    
    print("\n分析完成！请查看生成的图表和报告文件。")
    print(f"特征提取级别: {args.feature_level}")
    print("如需提取更多特征，请使用: --feature-level comprehensive")

if __name__ == "__main__":
    main() 