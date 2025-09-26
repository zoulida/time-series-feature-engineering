#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整修复GPLearn兼容性问题
"""

import subprocess
import sys
import os

def check_current_versions():
    """检查当前版本"""
    print("="*60)
    print("检查当前环境版本...")
    print("="*60)
    
    try:
        import numpy as np
        print(f"NumPy版本: {np.__version__}")
    except ImportError:
        print("NumPy未安装")
    
    try:
        import sklearn
        print(f"Scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn未安装")
    
    try:
        import gplearn
        print(f"GPLearn版本: {gplearn.__version__}")
    except ImportError:
        print("GPLearn未安装")
    
    print("="*60)

def fix_gplearn_compatibility():
    """修复gplearn兼容性"""
    print("\n开始修复GPLearn兼容性问题...")
    
    try:
        # 步骤1: 卸载当前版本
        print("步骤1: 卸载当前版本...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "gplearn", "-y"], 
                      check=True, capture_output=True)
        print("✅ GPLearn已卸载")
        
        # 步骤2: 降级numpy到兼容版本
        print("\n步骤2: 降级NumPy到兼容版本...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3"], 
                      check=True, capture_output=True)
        print("✅ NumPy已降级到1.24.3")
        
        # 步骤3: 安装兼容的gplearn版本
        print("\n步骤3: 安装兼容的GPLearn版本...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gplearn==0.4.1"], 
                      check=True, capture_output=True)
        print("✅ GPLearn 0.4.1已安装")
        
        # 步骤4: 测试安装
        print("\n步骤4: 测试安装...")
        import gplearn
        from gplearn.genetic import SymbolicTransformer
        print(f"✅ GPLearn版本: {gplearn.__version__}")
        
        # 步骤5: 测试基本功能
        print("\n步骤5: 测试基本功能...")
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        transformer = SymbolicTransformer(
            population_size=10,
            generations=2,
            n_components=2,
            function_set=['add', 'sub', 'mul', 'div'],
            metric='spearman',
            random_state=42,
            n_jobs=1
        )
        
        X_transformed = transformer.fit_transform(X, y)
        print(f"✅ 基本功能测试成功，输出形状: {X_transformed.shape}")
        
        # 步骤6: 检查特征表达式
        if hasattr(transformer, 'best_programs_'):
            print(f"✅ 找到 {len(transformer.best_programs_)} 个特征程序")
            for i, program in enumerate(transformer.best_programs_):
                print(f"   程序{i}: {program}")
        else:
            print("⚠️ 无法获取特征程序")
        
        print("\n🎉 GPLearn修复完成！现在可以正常运行特征提取程序。")
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def create_alternative_solution():
    """创建替代方案"""
    print("\n" + "="*60)
    print("如果GPLearn修复失败，使用替代方案")
    print("="*60)
    
    print("替代方案1: 使用增强特征工程")
    print("   - 运行: python source/gplearnFactor/enhanced_feature_engineering.py")
    print("   - 手动创建21个复杂特征组合")
    print("   - 包含价格、趋势、RSI、动量等特征")
    
    print("\n替代方案2: 使用featuretools")
    print("   - 安装: pip install featuretools")
    print("   - 自动特征工程库")
    
    print("\n替代方案3: 使用tsfresh")
    print("   - 安装: pip install tsfresh")
    print("   - 专门用于时间序列特征提取")

def main():
    """主函数"""
    print("GPLearn兼容性修复工具")
    print("="*60)
    
    # 检查当前版本
    check_current_versions()
    
    # 尝试修复
    success = fix_gplearn_compatibility()
    
    if not success:
        create_alternative_solution()
    
    print("\n修复完成！")

if __name__ == "__main__":
    main()
