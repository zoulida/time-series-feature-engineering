#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列数据与目标变量关系示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_example_data():
    """创建示例数据"""
    np.random.seed(42)
    
    # 生成3个样本的时间序列数据
    time_series_data = []
    target_values = []
    
    for sample_id in range(3):
        # 为每个样本生成10个时间点的数据
        n_timesteps = 10
        timestamps = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
        
        # 生成时间序列（添加趋势和噪声）
        trend = np.linspace(0, 5, n_timesteps)
        noise = np.random.normal(0, 0.5, n_timesteps)
        values = trend + noise
        
        # 添加到时间序列数据
        for t in range(n_timesteps):
            time_series_data.append({
                'id': sample_id,
                'time': timestamps[t],
                'value': values[t]
            })
        
        # 生成目标变量（基于时间序列特征）
        trend_strength = np.corrcoef(np.arange(n_timesteps), values)[0, 1]
        volatility = np.std(values)
        target = 0.6 * trend_strength + 0.4 * volatility
        target_values.append(target)
    
    time_series_df = pd.DataFrame(time_series_data)
    target_df = pd.DataFrame({
        'id': range(3),
        'target': target_values
    })
    
    return time_series_df, target_df

def visualize_relationship(time_series_df, target_df):
    """可视化数据关系"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('时间序列数据与目标变量关系示例', fontsize=16, fontweight='bold')
    
    # 1. 时间序列数据展示
    for sample_id in range(3):
        sample_data = time_series_df[time_series_df['id'] == sample_id]
        axes[0, 0].plot(sample_data['time'], sample_data['value'], 
                       marker='o', label=f'样本 {sample_id}')
    
    axes[0, 0].set_title('时间序列数据 (time_series_df)')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('值')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 目标变量展示
    axes[0, 1].bar(target_df['id'], target_df['target'], 
                   color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_title('目标变量 (target_df)')
    axes[0, 1].set_xlabel('样本ID')
    axes[0, 1].set_ylabel('目标值')
    axes[0, 1].set_xticks(target_df['id'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 数据表格展示
    axes[1, 0].axis('off')
    axes[1, 0].text(0.1, 0.9, 'time_series_df 结构:', fontsize=12, fontweight='bold')
    axes[1, 0].text(0.1, 0.8, time_series_df.head(10).to_string(), fontsize=8, 
                    family='monospace', verticalalignment='top')
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.9, 'target_df 结构:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.8, target_df.to_string(index=False), fontsize=10, 
                    family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('../可视化文件/data_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()

def explain_relationship():
    """解释数据关系"""
    print("=" * 60)
    print("时间序列数据与目标变量关系解释")
    print("=" * 60)
    
    print("\n📊 数据结构关系:")
    print("1. time_series_df (时间序列数据):")
    print("   - 包含多个样本的完整时间序列")
    print("   - 每个样本有多个时间点的观测值")
    print("   - 格式: [id, time, value]")
    print("   - 用于特征提取")
    
    print("\n2. target_df (目标变量数据):")
    print("   - 每个样本对应一个目标值")
    print("   - 格式: [id, target]")
    print("   - 用于计算IC评分")
    
    print("\n🔗 关系说明:")
    print("- 两个数据框通过 'id' 列关联")
    print("- time_series_df 中的每个样本ID对应 target_df 中的一个目标值")
    print("- 特征提取: 从 time_series_df 提取特征")
    print("- IC评分: 计算特征与 target_df 中目标值的相关性")
    
    print("\n📈 实际应用场景:")
    print("- 股票预测: time_series_df=股价历史, target_df=未来收益率")
    print("- 设备故障预测: time_series_df=传感器数据, target_df=故障标签")
    print("- 销售预测: time_series_df=历史销售数据, target_df=下月销售额")

if __name__ == "__main__":
    # 创建示例数据
    time_series_df, target_df = create_example_data()
    
    # 显示数据
    print("time_series_df (前10行):")
    print(time_series_df.head(10))
    print("\ntarget_df:")
    print(target_df)
    
    # 可视化关系
    visualize_relationship(time_series_df, target_df)
    
    # 解释关系
    explain_relationship() 