# -*- coding: utf-8 -*-
"""
测试表达式解析功能
验证新增的Alpha158函数解析是否正确
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'source', 'IC检测批量'))

from step3_generate_training_data_batch import TrainingDataBatchGenerator
from config_batch import get_config

def test_expression_parsing():
    """测试表达式解析功能"""
    
    # 创建测试数据
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    print("测试数据:")
    print(data)
    print()
    
    # 创建训练数据生成器
    config = get_config()
    generator = TrainingDataBatchGenerator(config)
    
    # 测试表达式
    test_expressions = [
        # 基本函数
        "($close-$open)/$open",
        "($high-$low)/$open",
        "Mean($close, 5)/$close",
        "Std($close, 5)/$close",
        
        # 新增函数
        "Quantile($close, 5, 0.8)/$close",
        "IdxMax($high, 5)/5",
        "IdxMin($low, 5)/5",
        "Rank($close, 5)",
        "Corr($close, Log($volume+1), 5)",
        "Rsquare($close, 5)",
        "Resi($close, 5)/$close",
        "Slope($close, 5)/$close",
        
        # 复杂表达式
        "Mean($close>Ref($close, 1), 5)",
        "Mean($close<Ref($close, 1), 5)",
        "($vwap-$close)/$close",
        
        # 组合表达式
        "Mean($close>Ref($close, 1), 5)-Mean($close<Ref($close, 1), 5)",
        "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
        "(IdxMax($high, 5)-IdxMin($low, 5))/5"
    ]
    
    print("=" * 80)
    print("测试表达式解析功能")
    print("=" * 80)
    
    for i, expr in enumerate(test_expressions, 1):
        print(f"\n{i:2d}. 测试表达式: {expr}")
        
        try:
            result = generator._evaluate_expression(expr, data)
            
            print(f"    结果形状: {result.shape}")
            print(f"    非NaN值数量: {np.sum(~np.isnan(result))}")
            print(f"    结果范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
            print(f"    结果均值: {np.nanmean(result):.4f}")
            print(f"    前5个值: {result[:5]}")
            
        except Exception as e:
            print(f"    ❌ 解析失败: {e}")

def test_specific_functions():
    """测试特定函数"""
    
    # 创建测试数据
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    config = get_config()
    generator = TrainingDataBatchGenerator(config)
    
    print("\n" + "=" * 80)
    print("测试特定函数")
    print("=" * 80)
    
    # 测试Quantile函数
    print("\n1. 测试Quantile函数:")
    expr = "Quantile($close, 5, 0.8)"
    try:
        result = generator._evaluate_expression(expr, data)
        print(f"   表达式: {expr}")
        print(f"   结果: {result}")
        print(f"   手动验证: {data['close'].rolling(5).quantile(0.8).values}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 测试IdxMax函数
    print("\n2. 测试IdxMax函数:")
    expr = "IdxMax($high, 5)"
    try:
        result = generator._evaluate_expression(expr, data)
        print(f"   表达式: {expr}")
        print(f"   结果: {result}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 测试Corr函数
    print("\n3. 测试Corr函数:")
    expr = "Corr($close, Log($volume+1), 5)"
    try:
        result = generator._evaluate_expression(expr, data)
        print(f"   表达式: {expr}")
        print(f"   结果: {result}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")

if __name__ == "__main__":
    test_expression_parsing()
    test_specific_functions()
