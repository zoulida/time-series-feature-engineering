# -*- coding: utf-8 -*-
"""
MyFactors集成示例
展示如何将MyFactors集成到统一因子库中

作者: 用户自定义
创建时间: 2024年
"""

import pandas as pd
import numpy as np
from unified_factor_library import UnifiedFactorLibrary
from myfactors import MyFactors


def integrate_myfactors_with_unified_library():
    """将MyFactors集成到统一因子库中"""
    print("=" * 60)
    print("MyFactors集成到统一因子库")
    print("=" * 60)
    
    # 1. 创建统一因子库
    print("1. 创建统一因子库...")
    try:
        unified_lib = UnifiedFactorLibrary(sources=['alpha158', 'tsfresh'])
        print(f"   成功加载统一因子库，包含 {len(unified_lib.get_all_factors())} 个因子")
    except Exception as e:
        print(f"   统一因子库加载失败: {e}")
        return
    
    # 2. 创建MyFactors
    print("\n2. 创建MyFactors...")
    my_factors = MyFactors()
    print(f"   成功创建MyFactors，包含 {len(my_factors.get_all_factors())} 个因子")
    
    # 3. 将MyFactors添加到统一因子库
    print("\n3. 将MyFactors添加到统一因子库...")
    my_factors_dict = my_factors.get_all_factors()
    unified_lib.add_custom_factors(my_factors_dict)
    
    # 4. 查看集成后的统计信息
    print("\n4. 集成后的统计信息:")
    stats = unified_lib.get_statistics()
    print(f"   总因子数量: {stats['总因子数量']}")
    print(f"   按来源统计: {stats['按来源统计']}")
    
    # 5. 查看MyFactors因子
    print("\n5. MyFactors因子:")
    my_factors_in_unified = unified_lib.get_factors_by_source('custom')
    for name, info in my_factors_in_unified.items():
        print(f"   {name}: {info['description']}")
    
    return unified_lib


def test_integrated_factors():
    """测试集成后的因子库"""
    print("\n" + "=" * 60)
    print("测试集成后的因子库")
    print("=" * 60)
    
    # 创建集成后的因子库
    unified_lib = integrate_myfactors_with_unified_library()
    if unified_lib is None:
        return
    
    # 创建示例数据
    print("\n创建示例数据...")
    np.random.seed(42)
    n_days = 50
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # 生成模拟数据
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    data.set_index('date', inplace=True)
    
    print(f"   数据形状: {data.shape}")
    
    # 测试MyFactors中的因子
    print("\n测试MyFactors因子...")
    try:
        # 获取MyFactors因子信息
        my_factor_name = 'CUSTOM_ZHANGTING_VOLUME_PRICE_FACTOR'
        factor_info = unified_lib.get_factor_info(my_factor_name)
        
        if factor_info:
            print(f"   因子名称: {my_factor_name}")
            print(f"   描述: {factor_info['description']}")
            print(f"   表达式: {factor_info['expression']}")
            
            # 注意：这里只是演示，实际计算需要实现对应的计算函数
            print("   注意: 实际计算需要实现对应的计算函数")
        else:
            print(f"   未找到因子: {my_factor_name}")
            
    except Exception as e:
        print(f"   测试MyFactors因子时出错: {e}")
    
    # 测试其他因子
    print("\n测试其他因子...")
    try:
        # 获取Alpha158因子示例
        alpha_factors = unified_lib.get_factors_by_source('Alpha158')
        if alpha_factors:
            sample_alpha = list(alpha_factors.keys())[0]
            print(f"   Alpha158因子示例: {sample_alpha}")
        
        # 获取tsfresh因子示例
        tsfresh_factors = unified_lib.get_factors_by_source('tsfresh')
        if tsfresh_factors:
            sample_tsfresh = list(tsfresh_factors.keys())[0]
            print(f"   tsfresh因子示例: {sample_tsfresh}")
            
    except Exception as e:
        print(f"   测试其他因子时出错: {e}")


def demonstrate_factor_search():
    """演示因子搜索功能"""
    print("\n" + "=" * 60)
    print("演示因子搜索功能")
    print("=" * 60)
    
    # 创建集成后的因子库
    unified_lib = integrate_myfactors_with_unified_library()
    if unified_lib is None:
        return
    
    # 搜索包含"zhangting"的因子
    print("\n1. 搜索包含'zhangting'的因子:")
    zhangting_factors = unified_lib.search_factors("zhangting")
    for name, info in zhangting_factors.items():
        print(f"   {name}: {info['description']}")
    
    # 搜索包含"volume"的因子
    print("\n2. 搜索包含'volume'的因子:")
    volume_factors = unified_lib.search_factors("volume")
    print(f"   找到 {len(volume_factors)} 个包含'volume'的因子")
    for name in list(volume_factors.keys())[:5]:  # 显示前5个
        print(f"   {name}")
    
    # 按类别搜索
    print("\n3. 按类别搜索'复合因子':")
    compound_factors = unified_lib.get_factors_by_category("复合因子")
    for name, info in compound_factors.items():
        print(f"   {name}: {info['description']}")
    
    # 复合搜索
    print("\n4. 复合搜索(来源=custom, 关键词=zhangting):")
    filtered = unified_lib.filter_factors(
        sources=['custom'],
        keywords=['zhangting']
    )
    for name, info in filtered.items():
        print(f"   {name}: {info['description']}")


def export_integrated_factors():
    """导出集成后的因子库"""
    print("\n" + "=" * 60)
    print("导出集成后的因子库")
    print("=" * 60)
    
    # 创建集成后的因子库
    unified_lib = integrate_myfactors_with_unified_library()
    if unified_lib is None:
        return
    
    # 导出所有因子
    print("\n导出所有因子...")
    df = unified_lib.export_to_csv("integrated_factors_export.csv")
    print(f"   已导出 {len(df)} 个因子到 integrated_factors_export.csv")
    
    # 只导出MyFactors
    print("\n只导出MyFactors...")
    my_factors_df = df[df['来源'] == 'custom']
    my_factors_df.to_csv("my_factors_only_export.csv", index=False, encoding='utf-8-sig')
    print(f"   已导出 {len(my_factors_df)} 个MyFactors到 my_factors_only_export.csv")
    
    # 显示导出结果摘要
    print("\n导出结果摘要:")
    print(f"   总因子数: {len(df)}")
    print(f"   Alpha158因子: {len(df[df['来源'] == 'Alpha158'])}")
    print(f"   tsfresh因子: {len(df[df['来源'] == 'tsfresh'])}")
    print(f"   MyFactors: {len(df[df['来源'] == 'custom'])}")


def main():
    """主函数"""
    print("MyFactors集成示例")
    print("=" * 60)
    
    try:
        # 集成MyFactors到统一因子库
        integrate_myfactors_with_unified_library()
        
        # 测试集成后的因子库
        test_integrated_factors()
        
        # 演示因子搜索功能
        demonstrate_factor_search()
        
        # 导出集成后的因子库
        export_integrated_factors()
        
        print("\n" + "=" * 60)
        print("MyFactors集成示例完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"集成示例过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
