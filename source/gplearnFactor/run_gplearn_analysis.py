#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行GPLearn特征提取分析的简化脚本
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    """主函数"""
    print("=" * 60)
    print("GPLearn时序特征提取和IC测试")
    print("=" * 60)
    
    try:
        # 导入特征提取器
        from gplearn_feature_extraction import GPLearnFeatureExtractor
        
        # 创建特征提取器
        extractor = GPLearnFeatureExtractor()
        
        # 设置文件路径
        target_file = os.path.join(current_dir, '..', '数据文件', 'tsfresh_target_panel.csv')
        long_file = os.path.join(current_dir, '..', '数据文件', 'tsfresh_long.csv')
        
        # 检查文件是否存在
        if not os.path.exists(target_file):
            print(f"错误：目标文件不存在: {target_file}")
            return
        
        if not os.path.exists(long_file):
            print(f"错误：时序数据文件不存在: {long_file}")
            return
        
        print("文件检查通过，开始处理...")
        
        # 加载数据
        extractor.load_data(target_file=target_file, long_file=long_file)
        
        # 准备特征
        extractor.prepare_features(window_sizes=[5, 10, 20])
        
        # 创建gplearn特征
        extractor.create_gplearn_features(
            n_features=20,  # 减少特征数量以加快处理
            population_size=300,  # 减少种群大小
            generations=10   # 减少代数
        )
        
        # 计算IC值
        extractor.calculate_ic()
        
        # 分析IC结果
        extractor.analyze_ic()
        
        # 可视化结果
        extractor.visualize_results()
        
        # 保存结果
        extractor.save_results()
        
        print("\n程序执行完成！")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请先运行 install_gplearn.py 安装gplearn")
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
