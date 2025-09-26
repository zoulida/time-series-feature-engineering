# -*- coding: utf-8 -*-
"""
配置使用示例
演示如何使用简化后的config_batch.py与unified_factor_library.py配合使用
"""

from config_batch import get_config, CONFIG_DESCRIPTION
import sys
import os

# 添加因子库路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))

def main():
    """主函数，演示配置的使用"""
    print("=" * 60)
    print("配置使用示例")
    print("=" * 60)
    
    # 获取配置
    config = get_config()
    
    # 显示当前配置
    print(f"股票数量: {config.PRODUCTION_STOCKS}")
    print(f"因子数量: {config.PRODUCTION_FACTORS}")
    print(f"因子库模式: {config.FACTOR_LIBRARY_MODE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"数据目录: {config.DATA_DIR}")
    
    # 获取因子源列表
    factor_sources = config.get_factor_sources()
    print(f"因子源列表: {factor_sources}")
    
    # 演示不同模式的使用
    print("\n" + "=" * 40)
    print("不同因子库模式演示")
    print("=" * 40)
    
    # 模式1: 使用所有因子库
    config.FACTOR_LIBRARY_MODE = 'all'
    sources = config.get_factor_sources()
    print(f"模式 'all': {sources}")
    
    # 模式2: 只使用Alpha158
    config.FACTOR_LIBRARY_MODE = 'alpha158'
    sources = config.get_factor_sources()
    print(f"模式 'alpha158': {sources}")
    
    # 模式3: 只使用tsfresh
    config.FACTOR_LIBRARY_MODE = 'tsfresh'
    sources = config.get_factor_sources()
    print(f"模式 'tsfresh': {sources}")
    
    # 模式4: 只使用自定义因子
    config.FACTOR_LIBRARY_MODE = 'custom'
    sources = config.get_factor_sources()
    print(f"模式 'custom': {sources}")
    
    # 模式5: 混合模式
    config.FACTOR_LIBRARY_MODE = 'mixed'
    config.ENABLE_ALPHA158 = True
    config.ENABLE_TSFRESH = False
    config.ENABLE_CUSTOM = True
    sources = config.get_factor_sources()
    print(f"模式 'mixed' (Alpha158+Custom): {sources}")
    
    # 演示与unified_factor_library的配合使用
    print("\n" + "=" * 40)
    print("与unified_factor_library配合使用")
    print("=" * 40)
    
    try:
        from unified_factor_library import UnifiedFactorLibrary
        
        # 使用配置创建因子库
        config.FACTOR_LIBRARY_MODE = 'all'
        factor_sources = config.get_factor_sources()
        
        print(f"正在创建因子库，使用因子源: {factor_sources}")
        factor_lib = UnifiedFactorLibrary(sources=factor_sources)
        
        # 获取统计信息
        stats = factor_lib.get_statistics()
        print(f"总因子数量: {stats['总因子数量']}")
        print(f"可用因子源: {', '.join(stats['可用因子源'])}")
        
        print("\n按来源统计:")
        for source, count in stats["按来源统计"].items():
            print(f"  {source}: {count}个")
            
    except ImportError as e:
        print(f"无法导入unified_factor_library: {e}")
        print("请确保unified_factor_library.py文件存在且路径正确")
    except Exception as e:
        print(f"创建因子库时出错: {e}")
    
    print("\n" + "=" * 60)
    print("配置使用示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
