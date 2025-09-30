# -*- coding: utf-8 -*-
"""
测试因子计算功能
验证Alpha158和tsfresh因子的计算是否正确
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'source', 'IC检测批量'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'source', '因子库'))

from step3_generate_training_data_batch import TrainingDataBatchGenerator
from config_batch import get_config

def create_test_data():
    """创建测试数据"""
    np.random.seed(42)  # 固定随机种子
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 确保high >= max(open, close), low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_alpha158_factors():
    """测试Alpha158因子计算"""
    print("=" * 60)
    print("测试Alpha158因子计算")
    print("=" * 60)
    
    # 创建测试数据
    data = create_test_data()
    print(f"测试数据形状: {data.shape}")
    print(f"数据列: {list(data.columns)}")
    
    # 创建训练数据生成器
    config = get_config()
    generator = TrainingDataBatchGenerator(config)
    
    # 测试几个Alpha158因子
    test_factors = [
        {
            'name': 'ALPHA_KMID',
            'info': {
                'expression': '($close-$open)/$open',
                'function_name': 'kbar_mid_ratio',
                'description': 'K线实体相对开盘价的比例'
            }
        },
        {
            'name': 'ALPHA_KLEN', 
            'info': {
                'expression': '($high-$low)/$open',
                'function_name': 'kbar_length_ratio',
                'description': 'K线长度相对开盘价的比例'
            }
        },
        {
            'name': 'ALPHA_MA5',
            'info': {
                'expression': 'Mean($close, 5)/$close',
                'function_name': 'moving_average_5d',
                'description': '5天移动平均相对当前价格'
            }
        }
    ]
    
    for factor in test_factors:
        print(f"\n测试因子: {factor['name']}")
        print(f"表达式: {factor['info']['expression']}")
        
        try:
            # 计算因子
            result = generator._calculate_alpha158_factor(data, factor['name'], factor['info'])
            
            print(f"结果形状: {result.shape}")
            print(f"非NaN值数量: {np.sum(~np.isnan(result))}")
            print(f"结果范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
            print(f"结果均值: {np.nanmean(result):.4f}")
            print(f"结果标准差: {np.nanstd(result):.4f}")
            
            # 显示前10个值
            print(f"前10个值: {result[:10]}")
            
        except Exception as e:
            print(f"计算失败: {e}")

def test_tsfresh_factors():
    """测试tsfresh因子计算"""
    print("\n" + "=" * 60)
    print("测试tsfresh因子计算")
    print("=" * 60)
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建训练数据生成器
    config = get_config()
    generator = TrainingDataBatchGenerator(config)
    
    # 测试几个tsfresh因子
    test_factors = [
        {
            'name': 'TSFRESH_abs_energy',
            'info': {
                'function_name': 'tsfresh_abs_energy',
                'description': '绝对能量'
            }
        },
        {
            'name': 'TSFRESH_standard_deviation',
            'info': {
                'function_name': 'tsfresh_standard_deviation', 
                'description': '标准差'
            }
        },
        {
            'name': 'TSFRESH_number_peaks',
            'info': {
                'function_name': 'tsfresh_number_peaks',
                'description': '峰值数量'
            }
        }
    ]
    
    for factor in test_factors:
        print(f"\n测试因子: {factor['name']}")
        print(f"函数名: {factor['info']['function_name']}")
        
        try:
            # 计算因子
            result = generator._calculate_tsfresh_factor(data, factor['name'], factor['info'])
            
            print(f"结果形状: {result.shape}")
            print(f"非NaN值数量: {np.sum(~np.isnan(result))}")
            print(f"结果范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
            print(f"结果均值: {np.nanmean(result):.4f}")
            print(f"结果标准差: {np.nanstd(result):.4f}")
            
            # 显示前10个值
            print(f"前10个值: {result[:10]}")
            
        except Exception as e:
            print(f"计算失败: {e}")

def test_expression_evaluation():
    """测试表达式解析功能"""
    print("\n" + "=" * 60)
    print("测试表达式解析功能")
    print("=" * 60)
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建训练数据生成器
    config = get_config()
    generator = TrainingDataBatchGenerator(config)
    
    # 测试表达式
    test_expressions = [
        '($close-$open)/$open',
        '($high-$low)/$open', 
        'Mean($close, 5)/$close',
        'Std($close, 10)/$close',
        'Greater($open, $close)',
        'Less($open, $close)'
    ]
    
    for expr in test_expressions:
        print(f"\n测试表达式: {expr}")
        
        try:
            result = generator._evaluate_expression(expr, data)
            
            print(f"结果形状: {result.shape}")
            print(f"非NaN值数量: {np.sum(~np.isnan(result))}")
            print(f"结果范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
            print(f"结果均值: {np.nanmean(result):.4f}")
            
        except Exception as e:
            print(f"解析失败: {e}")

def main():
    """主函数"""
    print("开始测试因子计算功能...")
    
    # 测试表达式解析
    test_expression_evaluation()
    
    # 测试Alpha158因子
    test_alpha158_factors()
    
    # 测试tsfresh因子
    test_tsfresh_factors()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()