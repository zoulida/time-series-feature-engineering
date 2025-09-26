#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试GPLearn功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def test_gplearn():
    """测试gplearn功能"""
    print("开始详细测试GPLearn...")
    
    try:
        from gplearn.genetic import SymbolicTransformer
        print("✅ SymbolicTransformer导入成功")
        
        # 创建测试数据
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        print(f"测试数据形状: X={X.shape}, y={y.shape}")
        
        # 创建转换器
        transformer = SymbolicTransformer(
            population_size=10,
            generations=2,
            n_components=3,
            function_set=['add', 'sub', 'mul', 'div'],
            metric='mse',
            random_state=42,
            n_jobs=1
        )
        print("✅ SymbolicTransformer创建成功")
        
        # 尝试拟合
        print("尝试拟合...")
        try:
            X_transformed = transformer.fit_transform(X, y)
            print(f"✅ fit_transform成功，输出形状: {X_transformed.shape}")
            
            # 检查属性
            print("检查transformer属性:")
            attrs = [attr for attr in dir(transformer) if not attr.startswith('_')]
            print(f"可用属性: {attrs}")
            
            if hasattr(transformer, 'best_programs_'):
                print(f"best_programs_存在，长度: {len(transformer.best_programs_)}")
                for i, program in enumerate(transformer.best_programs_):
                    print(f"程序{i}: {program}")
            else:
                print("❌ best_programs_不存在")
                
        except Exception as e:
            print(f"❌ fit_transform失败: {e}")
            print(f"错误类型: {type(e)}")
            
            # 尝试分别调用fit和transform
            print("尝试分别调用fit和transform...")
            try:
                transformer.fit(X, y)
                print("✅ fit成功")
                X_transformed = transformer.transform(X)
                print(f"✅ transform成功，输出形状: {X_transformed.shape}")
            except Exception as e2:
                print(f"❌ fit+transform也失败: {e2}")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_gplearn()
