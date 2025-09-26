#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLearn兼容性补丁
修复numpy兼容性问题
"""

import numpy as np
import sys

def apply_numpy_patch():
    """应用numpy兼容性补丁"""
    # 修复np.int问题
    if not hasattr(np, 'int'):
        np.int = int
    
    # 修复其他可能的兼容性问题
    if not hasattr(np, 'float'):
        np.float = float
    
    if not hasattr(np, 'bool'):
        np.bool = bool

def patch_gplearn():
    """补丁gplearn"""
    try:
        # 应用numpy补丁
        apply_numpy_patch()
        
        # 导入gplearn
        from gplearn.genetic import SymbolicTransformer
        
        # 创建测试数据
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        # 测试SymbolicTransformer
        transformer = SymbolicTransformer(
            population_size=100,
            generations=2,
            n_components=3,
            function_set=['add', 'sub', 'mul', 'div'],
            metric='spearman',
            random_state=42,
            n_jobs=1,
            hall_of_fame=50
        )
        
        X_transformed = transformer.fit_transform(X, y)
        print(f"✅ GPLearn补丁成功！输出形状: {X_transformed.shape}")
        
        # 检查特征表达式
        if hasattr(transformer, 'best_programs_'):
            print(f"✅ 找到 {len(transformer.best_programs_)} 个特征程序")
            for i, program in enumerate(transformer.best_programs_):
                print(f"   程序{i}: {program}")
        
        return transformer
        
    except Exception as e:
        print(f"❌ GPLearn补丁失败: {e}")
        return None

if __name__ == "__main__":
    print("应用GPLearn兼容性补丁...")
    transformer = patch_gplearn()
    if transformer:
        print("🎉 补丁应用成功！")
    else:
        print("❌ 补丁应用失败")
