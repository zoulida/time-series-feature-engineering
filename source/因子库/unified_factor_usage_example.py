# -*- coding: utf-8 -*-
"""
统一因子库使用示例
演示如何使用UnifiedFactorLibrary类来管理和使用各种因子
"""

import pandas as pd
import numpy as np
from unified_factor_library import UnifiedFactorLibrary, create_unified_factor_library

def demo_basic_usage():
    """演示基本使用方法"""
    print("=" * 60)
    print("统一因子库基本使用演示")
    print("=" * 60)
    
    # 创建统一因子库
    factor_lib = create_unified_factor_library()
    
    # 获取统计信息
    stats = factor_lib.get_statistics()
    print(f"总因子数量: {stats['总因子数量']}")
    print(f"可用因子源: {', '.join(stats['可用因子源'])}")
    
    # 按来源查看因子
    print("\n按来源统计:")
    for source, count in stats["按来源统计"].items():
        print(f"  {source}: {count}个")
    
    # 查看前10个因子
    print("\n前10个因子:")
    sample_factors = list(factor_lib.factors.keys())[:10]
    for factor in sample_factors:
        info = factor_lib.factors[factor]
        print(f"  {factor}: {info['description']} ({info['source']})")
    
    return factor_lib


def demo_factor_filtering():
    """演示因子过滤功能"""
    print("\n" + "=" * 60)
    print("因子过滤功能演示")
    print("=" * 60)
    
    factor_lib = create_unified_factor_library()
    
    # 1. 按来源过滤
    print("\n1. 按来源过滤:")
    alpha_factors = factor_lib.get_factors_by_source('Alpha158')
    print(f"Alpha158因子数量: {len(alpha_factors)}")
    
    tsfresh_factors = factor_lib.get_factors_by_source('tsfresh')
    print(f"tsfresh因子数量: {len(tsfresh_factors)}")
    
    # 2. 按类别过滤
    print("\n2. 按类别过滤:")
    kbar_factors = factor_lib.get_factors_by_category('K线形态')
    print(f"K线形态因子数量: {len(kbar_factors)}")
    
    statistical_factors = factor_lib.get_factors_by_category('统计')
    print(f"统计特征因子数量: {len(statistical_factors)}")
    
    # 3. 关键词搜索
    print("\n3. 关键词搜索:")
    mean_factors = factor_lib.search_factors("mean")
    print(f"包含'mean'的因子数量: {len(mean_factors)}")
    
    volume_factors = factor_lib.search_factors("volume")
    print(f"包含'volume'的因子数量: {len(volume_factors)}")
    
    # 4. 复合过滤
    print("\n4. 复合过滤:")
    filtered = factor_lib.filter_factors(
        sources=['Alpha158'],
        categories=['K线形态'],
        keywords=['mid']
    )
    print(f"Alpha158中K线形态且包含'mid'的因子数量: {len(filtered)}")
    
    for factor_name, info in list(filtered.items())[:5]:
        print(f"  {factor_name}: {info['description']}")


def demo_custom_factors():
    """演示自定义因子添加"""
    print("\n" + "=" * 60)
    print("自定义因子添加演示")
    print("=" * 60)
    
    factor_lib = create_unified_factor_library()
    
    # 定义自定义因子
    custom_factors = {
        "MY_RSI": {
            "expression": "RSI($close, 14)",
            "function_name": "my_rsi_14",
            "description": "自定义RSI指标，14天周期",
            "category": "技术指标",
            "formula": "RSI(收盘价, 14)"
        },
        "MY_BOLLINGER": {
            "expression": "($close - Mean($close, 20)) / (2 * Std($close, 20))",
            "function_name": "my_bollinger_bands",
            "description": "自定义布林带位置指标",
            "category": "技术指标",
            "formula": "(收盘价 - 20日均价) / (2 * 20日标准差)"
        },
        "MY_VOLATILITY": {
            "expression": "Std($close, 20) / Mean($close, 20)",
            "function_name": "my_volatility_ratio",
            "description": "自定义波动率比率",
            "category": "风险指标",
            "formula": "20日标准差 / 20日均价"
        }
    }
    
    # 添加自定义因子
    factor_lib.add_custom_factors(custom_factors)
    
    # 查看添加后的统计信息
    stats = factor_lib.get_statistics()
    print(f"添加自定义因子后的总数量: {stats['总因子数量']}")
    
    # 查看自定义因子
    custom_factors_result = factor_lib.get_factors_by_source('custom')
    print(f"\n自定义因子:")
    for factor_name, info in custom_factors_result.items():
        print(f"  {factor_name}: {info['description']}")


def demo_factor_export():
    """演示因子导出功能"""
    print("\n" + "=" * 60)
    print("因子导出功能演示")
    print("=" * 60)
    
    factor_lib = create_unified_factor_library()
    
    # 导出所有因子
    df = factor_lib.export_to_csv("unified_factors_demo.csv")
    print(f"已导出 {len(df)} 个因子到 unified_factors_demo.csv")
    
    # 显示导出数据的结构
    print(f"\n导出数据列: {list(df.columns)}")
    print(f"数据形状: {df.shape}")
    
    # 显示前5行
    print("\n前5行数据:")
    print(df.head().to_string(index=False))


def demo_factor_calculation():
    """演示因子计算（模拟）"""
    print("\n" + "=" * 60)
    print("因子计算演示（模拟）")
    print("=" * 60)
    
    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.01) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.01) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
        'volume': np.random.randint(1000, 10000, 100),
        'vwap': 100 + np.cumsum(np.random.randn(100) * 0.01)
    })
    
    print(f"示例数据形状: {data.shape}")
    print(f"数据列: {list(data.columns)}")
    
    # 模拟计算一些因子
    result = data.copy()
    
    # 模拟Alpha158因子计算
    result['ALPHA_KMID'] = (result['close'] - result['open']) / result['open']
    result['ALPHA_KLEN'] = (result['high'] - result['low']) / result['open']
    result['ALPHA_MA5'] = result['close'].rolling(5).mean() / result['close']
    result['ALPHA_STD20'] = result['close'].rolling(20).std() / result['close']
    
    # 模拟tsfresh因子计算
    result['TSFRESH_mean'] = result['close'].rolling(5).mean()
    result['TSFRESH_std'] = result['close'].rolling(5).std()
    result['TSFRESH_max'] = result['close'].rolling(5).max()
    result['TSFRESH_min'] = result['close'].rolling(5).min()
    
    print(f"\n计算后的数据形状: {result.shape}")
    print(f"新增的因子列: {[col for col in result.columns if col.startswith(('ALPHA_', 'TSFRESH_'))]}")
    
    # 显示因子值统计
    factor_cols = [col for col in result.columns if col.startswith(('ALPHA_', 'TSFRESH_'))]
    print(f"\n因子值统计:")
    for col in factor_cols:
        print(f"  {col}: 均值={result[col].mean():.4f}, 标准差={result[col].std():.4f}")


def demo_advanced_usage():
    """演示高级用法"""
    print("\n" + "=" * 60)
    print("高级用法演示")
    print("=" * 60)
    
    factor_lib = create_unified_factor_library()
    
    # 1. 获取因子摘要
    summary_df = factor_lib.get_factor_summary()
    print(f"因子摘要数据形状: {summary_df.shape}")
    
    # 2. 按类别统计
    category_stats = summary_df['类别'].value_counts()
    print(f"\n按类别统计 (前10个):")
    for category, count in category_stats.head(10).items():
        print(f"  {category}: {count}个")
    
    # 3. 按来源统计
    source_stats = summary_df['来源'].value_counts()
    print(f"\n按来源统计:")
    for source, count in source_stats.items():
        print(f"  {source}: {count}个")
    
    # 4. 查找特定类型的因子
    print(f"\n查找技术指标相关因子:")
    tech_factors = factor_lib.search_factors("技术")
    print(f"包含'技术'的因子数量: {len(tech_factors)}")
    
    # 5. 获取特定因子的详细信息
    if factor_lib.factors:
        first_factor = list(factor_lib.factors.keys())[0]
        factor_info = factor_lib.get_factor_info(first_factor)
        print(f"\n第一个因子的详细信息:")
        print(f"  因子名: {first_factor}")
        print(f"  描述: {factor_info.get('description', 'N/A')}")
        print(f"  表达式: {factor_info.get('expression', 'N/A')}")
        print(f"  类别: {factor_info.get('category', 'N/A')}")
        print(f"  来源: {factor_info.get('source', 'N/A')}")


def main():
    """主函数"""
    print("统一因子库使用示例")
    print("本示例演示了UnifiedFactorLibrary的各种功能")
    
    try:
        # 基本使用
        factor_lib = demo_basic_usage()
        
        # 因子过滤
        demo_factor_filtering()
        
        # 自定义因子
        demo_custom_factors()
        
        # 因子导出
        demo_factor_export()
        
        # 因子计算
        demo_factor_calculation()
        
        # 高级用法
        demo_advanced_usage()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
