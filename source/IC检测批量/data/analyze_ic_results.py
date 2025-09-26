#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC结果分析脚本
分析IC值最大和最小的各20个因子
"""

import json
import pandas as pd

def analyze_ic_results():
    """分析IC结果"""
    # 读取IC结果 - 自动找到前缀为ic_results_batch的JSON文件
    import glob
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本目录: {script_dir}")
    
    # 查找脚本目录下所有ic_results_batch开头的JSON文件
    json_pattern = os.path.join(script_dir, 'ic_results_batch_*.json')
    json_files = glob.glob(json_pattern)
    
    print(f"查找模式: {json_pattern}")
    print(f"找到文件: {json_files}")
    
    if not json_files:
        print("错误：未找到ic_results_batch开头的JSON文件")
        return
    
    # 使用最新的文件（按修改时间排序）
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"使用文件: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        ic_data = json.load(f)
    
    # 转换为DataFrame便于分析
    results = []
    for factor_name, data in ic_data.items():
        results.append({
            'factor_name': factor_name,
            'pearson_ic': data['pearson_ic'],
            'spearman_ic': data['spearman_ic'],
            'sample_size': data['sample_size']
        })
    
    df = pd.DataFrame(results)
    
    # 按Pearson IC排序
    df_sorted = df.sort_values('pearson_ic', ascending=False)
    
    print("=" * 80)
    print("IC结果分析报告")
    print("=" * 80)
    print(f"总因子数量: {len(df)}")
    print(f"样本数量: {df['sample_size'].iloc[0]:,}")
    print()
    
    # 最大IC值TOP20
    print("最大IC值TOP20 (按Pearson IC排序)")
    print("-" * 80)
    top20 = df_sorted.head(20)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"{i:2d}. {row['factor_name']:<40} | Pearson: {row['pearson_ic']:8.6f} | Spearman: {row['spearman_ic']:8.6f}")
    
    print()
    
    # 最小IC值BOTTOM20
    print("最小IC值BOTTOM20 (按Pearson IC排序)")
    print("-" * 80)
    bottom20 = df_sorted.tail(20)
    for i, (_, row) in enumerate(bottom20.iterrows(), 1):
        print(f"{i:2d}. {row['factor_name']:<40} | Pearson: {row['pearson_ic']:8.6f} | Spearman: {row['spearman_ic']:8.6f}")
    
    print()
    
    # 统计信息
    print("统计信息")
    print("-" * 80)
    print(f"Pearson IC 最大值: {df['pearson_ic'].max():.6f}")
    print(f"Pearson IC 最小值: {df['pearson_ic'].min():.6f}")
    print(f"Pearson IC 平均值: {df['pearson_ic'].mean():.6f}")
    print(f"Pearson IC 中位数: {df['pearson_ic'].median():.6f}")
    print(f"Pearson IC 标准差: {df['pearson_ic'].std():.6f}")
    print()
    
    print(f"Spearman IC 最大值: {df['spearman_ic'].max():.6f}")
    print(f"Spearman IC 最小值: {df['spearman_ic'].min():.6f}")
    print(f"Spearman IC 平均值: {df['spearman_ic'].mean():.6f}")
    print(f"Spearman IC 中位数: {df['spearman_ic'].median():.6f}")
    print(f"Spearman IC 标准差: {df['spearman_ic'].std():.6f}")
    print()
    
    # IC值分布
    print("IC值分布")
    print("-" * 80)
    positive_pearson = (df['pearson_ic'] > 0).sum()
    negative_pearson = (df['pearson_ic'] < 0).sum()
    zero_pearson = (df['pearson_ic'] == 0).sum()
    
    print(f"正IC值因子: {positive_pearson} 个 ({positive_pearson/len(df)*100:.1f}%)")
    print(f"负IC值因子: {negative_pearson} 个 ({negative_pearson/len(df)*100:.1f}%)")
    print(f"零IC值因子: {zero_pearson} 个 ({zero_pearson/len(df)*100:.1f}%)")
    print()
    
    # 按绝对值排序的TOP20
    print("按绝对值排序的TOP20 (最强预测能力)")
    print("-" * 80)
    df['abs_pearson_ic'] = df['pearson_ic'].abs()
    df_abs_sorted = df.sort_values('abs_pearson_ic', ascending=False)
    top20_abs = df_abs_sorted.head(20)
    for i, (_, row) in enumerate(top20_abs.iterrows(), 1):
        print(f"{i:2d}. {row['factor_name']:<40} | Pearson: {row['pearson_ic']:8.6f} | Spearman: {row['spearman_ic']:8.6f} | 绝对值: {row['abs_pearson_ic']:8.6f}")

if __name__ == "__main__":
    analyze_ic_results()
