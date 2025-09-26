#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试gplearn是否正常工作的脚本
"""

def test_gplearn():
    """测试gplearn功能"""
    print("=" * 50)
    print("测试gplearn功能")
    print("=" * 50)
    
    try:
        # 测试导入
        print("1. 测试导入gplearn...")
        import gplearn
        print(f"   ✓ gplearn版本: {gplearn.__version__}")
        
        # 测试SymbolicTransformer
        print("2. 测试SymbolicTransformer...")
        from gplearn.genetic import SymbolicTransformer
        print("   ✓ SymbolicTransformer导入成功")
        
        # 测试SymbolicRegressor
        print("3. 测试SymbolicRegressor...")
        from gplearn.genetic import SymbolicRegressor
        print("   ✓ SymbolicRegressor导入成功")
        
        # 测试SymbolicClassifier
        print("4. 测试SymbolicClassifier...")
        from gplearn.genetic import SymbolicClassifier
        print("   ✓ SymbolicClassifier导入成功")
        
        # 测试SymbolicSelector
        print("5. 测试SymbolicSelector...")
        from gplearn.genetic import SymbolicSelector
        print("   ✓ SymbolicSelector导入成功")
        
        print("\n所有测试通过！gplearn可以正常使用。")
        return True
        
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        print("\ngplearn未正确安装，请运行 install_gplearn.py")
        return False
    except Exception as e:
        print(f"   ✗ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    try:
        print("\n" + "=" * 50)
        print("测试基本功能")
        print("=" * 50)
        
        import numpy as np
        from gplearn.genetic import SymbolicTransformer
        
        # 创建测试数据
        print("1. 创建测试数据...")
        X = np.random.randn(100, 5)
        y = X[:, 0] + X[:, 1] * 2 + np.random.randn(100) * 0.1
        
        # 测试SymbolicTransformer
        print("2. 测试SymbolicTransformer...")
        transformer = SymbolicTransformer(
            population_size=100,
            generations=5,
            n_components=3,
            random_state=42
        )
        
        X_transformed = transformer.fit_transform(X, y)
        print(f"   ✓ 特征转换成功，输出形状: {X_transformed.shape}")
        
        # 测试SymbolicRegressor
        print("3. 测试SymbolicRegressor...")
        from gplearn.genetic import SymbolicRegressor
        
        regressor = SymbolicRegressor(
            population_size=100,
            generations=5,
            random_state=42
        )
        
        regressor.fit(X, y)
        score = regressor.score(X, y)
        print(f"   ✓ 回归器训练成功，R²分数: {score:.4f}")
        
        print("\n基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"   ✗ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试gplearn...")
    
    # 测试导入
    if not test_gplearn():
        return
    
    # 测试基本功能
    if not test_basic_functionality():
        return
    
    print("\n" + "=" * 50)
    print("所有测试通过！gplearn可以正常使用。")
    print("=" * 50)
    print("\n现在可以运行特征提取程序了：")
    print("python run_gplearn_analysis.py")

if __name__ == "__main__":
    main()
