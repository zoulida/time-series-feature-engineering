# -*- coding: utf-8 -*-
"""
测试Alpha158因子计算函数
验证新的直接计算函数方式是否正常工作
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'source', '因子库'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'source', 'IC检测批量'))

from alpha158_factors import Alpha158Factors
from step3_generate_training_data_batch import TrainingDataBatchGenerator
from config_batch import get_config

def test_alpha158_functions():
    """测试Alpha158因子计算函数"""
    
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
    
    # 创建Alpha158因子库实例
    alpha158_lib = Alpha158Factors()
    
    # 测试K线形态因子
    print("=" * 60)
    print("测试K线形态因子")
    print("=" * 60)
    
    kbar_factors = alpha158_lib.get_factors_by_category('K线形态')
    for factor_name, factor_info in kbar_factors.items():
        print(f"\n测试因子: {factor_name}")
        print(f"函数名: {factor_info['function_name']}")
        print(f"参数: {factor_info['parameters']}")
        
        try:
            # 获取计算函数
            calc_func = getattr(alpha158_lib, factor_info['function_name'])
            
            # 准备参数
            args = []
            for param in factor_info['parameters']:
                if param in data.columns:
                    args.append(data[param])
                else:
                    print(f"  缺少参数: {param}")
                    continue
            
            # 计算因子
            result = calc_func(*args)
            print(f"  结果形状: {result.shape}")
            print(f"  非NaN值数量: {result.notna().sum()}")
            print(f"  结果范围: [{result.min():.4f}, {result.max():.4f}]")
            print(f"  前5个值: {result.head().values}")
            
        except Exception as e:
            print(f"  计算失败: {e}")
    
    # 测试价格因子
    print("\n" + "=" * 60)
    print("测试价格因子")
    print("=" * 60)
    
    price_factors = alpha158_lib.get_factors_by_category('价格因子')
    for factor_name, factor_info in price_factors.items():
        print(f"\n测试因子: {factor_name}")
        print(f"函数名: {factor_info['function_name']}")
        print(f"参数: {factor_info['parameters']}")
        
        try:
            # 获取计算函数
            calc_func = getattr(alpha158_lib, factor_info['function_name'])
            
            # 准备参数
            args = []
            for param in factor_info['parameters']:
                if param in data.columns:
                    args.append(data[param])
                else:
                    print(f"  缺少参数: {param}")
                    continue
            
            # 计算因子
            result = calc_func(*args)
            print(f"  结果形状: {result.shape}")
            print(f"  非NaN值数量: {result.notna().sum()}")
            print(f"  结果范围: [{result.min():.4f}, {result.max():.4f}]")
            print(f"  前5个值: {result.head().values}")
            
        except Exception as e:
            print(f"  计算失败: {e}")
    
    # 测试滚动统计因子
    print("\n" + "=" * 60)
    print("测试滚动统计因子")
    print("=" * 60)
    
    rolling_factors = alpha158_lib.get_factors_by_category('滚动统计')
    # 只测试前5个因子
    for i, (factor_name, factor_info) in enumerate(rolling_factors.items()):
        if i >= 5:
            break
            
        print(f"\n测试因子: {factor_name}")
        print(f"函数名: {factor_info['function_name']}")
        print(f"参数: {factor_info['parameters']}")
        
        try:
            # 获取计算函数
            calc_func = getattr(alpha158_lib, factor_info['function_name'])
            
            # 准备参数
            args = []
            for param in factor_info['parameters']:
                if param in data.columns:
                    args.append(data[param])
                else:
                    print(f"  缺少参数: {param}")
                    continue
            
            # 计算因子
            result = calc_func(*args)
            print(f"  结果形状: {result.shape}")
            print(f"  非NaN值数量: {result.notna().sum()}")
            print(f"  结果范围: [{result.min():.4f}, {result.max():.4f}]")
            print(f"  前5个值: {result.head().values}")
            
        except Exception as e:
            print(f"  计算失败: {e}")

def test_training_data_generator():
    """测试训练数据生成器中的Alpha158因子计算"""
    
    print("\n" + "=" * 60)
    print("测试训练数据生成器")
    print("=" * 60)
    
    # 创建训练数据生成器
    config = get_config()
    generator = TrainingDataBatchGenerator(config)
    
    # 创建测试数据
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    # 获取Alpha158因子信息
    alpha158_lib = Alpha158Factors()
    kbar_factors = alpha158_lib.get_factors_by_category('K线形态')
    
    # 测试几个因子
    test_factors = list(kbar_factors.items())[:3]
    
    for factor_name, factor_info in test_factors:
        print(f"\n测试因子: {factor_name}")
        try:
            result = generator._calculate_alpha158_factor(data, factor_name, factor_info)
            print(f"  结果形状: {result.shape}")
            print(f"  非NaN值数量: {np.sum(~np.isnan(result))}")
            print(f"  结果范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
            print(f"  前5个值: {result[:5]}")
        except Exception as e:
            print(f"  计算失败: {e}")

if __name__ == "__main__":
    print("测试Alpha158因子计算函数")
    print("=" * 60)
    
    # 测试直接函数调用
    test_alpha158_functions()
    
    # 测试训练数据生成器
    test_training_data_generator()
    
    print("\n测试完成！")
