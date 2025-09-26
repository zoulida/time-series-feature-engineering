#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复GPLearn版本兼容性问题
"""

import subprocess
import sys

def fix_gplearn():
    """修复gplearn版本问题"""
    print("正在修复GPLearn版本兼容性问题...")
    
    try:
        # 卸载当前版本
        print("1. 卸载当前gplearn版本...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "gplearn", "-y"], 
                      check=True, capture_output=True)
        
        # 安装兼容版本
        print("2. 安装兼容版本的gplearn...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gplearn==0.4.2"], 
                      check=True, capture_output=True)
        
        # 测试安装
        print("3. 测试gplearn安装...")
        import gplearn
        from gplearn.genetic import SymbolicTransformer
        print(f"✅ GPLearn版本: {gplearn.__version__}")
        
        # 测试基本功能
        print("4. 测试基本功能...")
        transformer = SymbolicTransformer(
            population_size=10,
            generations=2,
            n_components=2,
            random_state=42
        )
        print("✅ SymbolicTransformer创建成功")
        
        print("\n🎉 GPLearn修复完成！现在可以正常运行特征提取程序。")
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        print("\n建议手动安装:")
        print("pip uninstall gplearn -y")
        print("pip install gplearn==0.4.2")

if __name__ == "__main__":
    fix_gplearn()
