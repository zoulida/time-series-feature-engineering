# -*- coding: utf-8 -*-
"""
测试配置集成效果
验证config_batch.py的配置是否正确应用到工作流中
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from config_batch import get_config
from step2_select_factors_batch import FactorBatchSelector

def test_config_integration():
    """测试配置集成效果"""
    print("=" * 60)
    print("测试配置集成效果")
    print("=" * 60)
    
    # 获取配置
    batch_config = get_config()
    
    print(f"当前配置:")
    print(f"  股票数量: {batch_config.PRODUCTION_STOCKS}")
    print(f"  因子数量: {batch_config.PRODUCTION_FACTORS}")
    print(f"  因子库模式: {batch_config.FACTOR_LIBRARY_MODE}")
    print(f"  批次大小: {batch_config.BATCH_SIZE}")
    print(f"  最大内存: {batch_config.MAX_MEMORY_GB}GB")
    print()
    
    # 测试因子源获取
    factor_sources = batch_config.get_factor_sources()
    print(f"因子源列表: {factor_sources}")
    print()
    
    # 测试因子选择器
    print("测试因子选择器...")
    try:
        # 创建因子选择器
        factor_selector = FactorBatchSelector(None, "data", batch_config)
        
        # 获取因子库统计信息
        stats = factor_selector.factor_lib.get_statistics()
        print(f"因子库统计信息:")
        print(f"  总因子数量: {stats['总因子数量']}")
        print(f"  可用因子源: {', '.join(stats['可用因子源'])}")
        
        print("\n按来源统计:")
        for source, count in stats["按来源统计"].items():
            print(f"  {source}: {count}个")
        
        print("\n按类别统计 (前10个):")
        sorted_categories = sorted(stats["按类别统计"].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            print(f"  {category}: {count}个")
        
        # 测试因子选择
        print(f"\n测试选择 {batch_config.PRODUCTION_FACTORS} 个因子...")
        selected_factors = factor_selector.select_factors_batch(batch_config.PRODUCTION_FACTORS)
        print(f"成功选择了 {len(selected_factors)} 个因子")
        
        # 显示选中的因子来源分布
        if selected_factors:
            source_count = {}
            for factor_name in selected_factors:
                factor_info = factor_selector.factor_lib.get_factor_info(factor_name)
                if factor_info:
                    source = factor_info.get('source', '未知')
                    source_count[source] = source_count.get(source, 0) + 1
            
            print("\n选中因子的来源分布:")
            for source, count in source_count.items():
                print(f"  {source}: {count}个")
        
        print("\n✅ 配置集成测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 配置集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_different_modes():
    """测试不同因子库模式"""
    print("\n" + "=" * 60)
    print("测试不同因子库模式")
    print("=" * 60)
    
    modes = ['all', 'alpha158', 'tsfresh', 'custom', 'mixed']
    
    for mode in modes:
        print(f"\n测试模式: {mode}")
        try:
            # 创建配置
            batch_config = get_config()
            batch_config.FACTOR_LIBRARY_MODE = mode
            
            # 如果是mixed模式，设置开关
            if mode == 'mixed':
                batch_config.ENABLE_ALPHA158 = True
                batch_config.ENABLE_TSFRESH = False
                batch_config.ENABLE_CUSTOM = True
            
            # 获取因子源
            factor_sources = batch_config.get_factor_sources()
            print(f"  因子源: {factor_sources}")
            
            # 创建因子选择器
            factor_selector = FactorBatchSelector(None, "data", batch_config)
            stats = factor_selector.factor_lib.get_statistics()
            print(f"  总因子数量: {stats['总因子数量']}")
            print(f"  可用因子源: {', '.join(stats['可用因子源'])}")
            
        except Exception as e:
            print(f"  ❌ 模式 {mode} 测试失败: {str(e)}")

if __name__ == "__main__":
    # 测试基本配置集成
    success = test_config_integration()
    
    if success:
        # 测试不同模式
        test_different_modes()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
