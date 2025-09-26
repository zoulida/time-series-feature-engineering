#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装gplearn和相关依赖的脚本
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ 安装 {package} 失败")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("安装gplearn和相关依赖")
    print("=" * 60)
    
    # 需要安装的包列表
    packages = [
        "gplearn",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0"
    ]
    
    print("开始安装包...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n安装完成！成功安装 {success_count}/{len(packages)} 个包")
    
    if success_count == len(packages):
        print("所有依赖已成功安装，可以运行gplearn特征提取程序了！")
    else:
        print("部分包安装失败，请检查错误信息并手动安装")
    
    # 验证gplearn安装
    try:
        import gplearn
        print(f"✓ gplearn版本: {gplearn.__version__}")
    except ImportError:
        print("✗ gplearn导入失败，请检查安装")

if __name__ == "__main__":
    main()
