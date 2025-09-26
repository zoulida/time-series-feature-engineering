# -*- coding: utf-8 -*-
"""
综合因子库使用示例
演示如何使用整合了Alpha158和tsfresh的综合因子库
"""

import pandas as pd
import numpy as np
from comprehensive_factor_library import (
    get_comprehensive_factors,
    get_alpha158_factors,
    get_tsfresh_factors,
    get_factors_by_source,
    get_factors_by_category
)

def demonstrate_comprehensive_factors():
    """演示综合因子库的使用"""
    
    print("=" * 60)
    print("综合因子库使用演示")
    print("=" * 60)
    
    # 1. 获取所有因子
    all_factors = get_comprehensive_factors()
    print(f"\n1. 总因子数量: {len(all_factors)}")
    
    # 2. 按来源获取因子
    alpha_factors = get_alpha158_factors()
    tsfresh_factors = get_tsfresh_factors()
    
    print(f"\n2. 按来源统计:")
    print(f"   Alpha158因子: {len(alpha_factors)}个")
    print(f"   tsfresh因子: {len(tsfresh_factors)}个")
    
    # 3. 显示Alpha158因子示例
    print(f"\n3. Alpha158因子示例 (前10个):")
    alpha_sample = list(alpha_factors.keys())[:10]
    for factor in alpha_sample:
        info = alpha_factors[factor]
        print(f"   {factor}: {info['description']}")
    
    # 4. 显示tsfresh因子示例
    print(f"\n4. tsfresh因子示例 (前10个):")
    tsfresh_sample = list(tsfresh_factors.keys())[:10]
    for factor in tsfresh_sample:
        info = tsfresh_factors[factor]
        print(f"   {factor}: {info['description']}")
    
    # 5. 按类别查询因子
    print(f"\n5. 按类别查询因子:")
    
    # 查询统计特征
    statistical_factors = get_factors_by_category('统计')
    print(f"   统计特征: {len(statistical_factors)}个")
    for factor in list(statistical_factors.keys())[:5]:
        print(f"     - {factor}")
    
    # 查询频域特征
    frequency_factors = get_factors_by_category('频域')
    print(f"   频域特征: {len(frequency_factors)}个")
    for factor in list(frequency_factors.keys())[:5]:
        print(f"     - {factor}")
    
    # 查询K线形态特征
    kbar_factors = get_factors_by_category('K线')
    print(f"   K线形态特征: {len(kbar_factors)}个")
    for factor in list(kbar_factors.keys())[:5]:
        print(f"     - {factor}")

def create_factor_calculation_example():
    """创建因子计算示例"""
    
    print(f"\n6. 因子计算示例:")
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100),
        'vwap': 100 + np.cumsum(np.random.randn(100) * 0.5)
    })
    
    print("   示例数据:")
    print(f"   数据形状: {data.shape}")
    print(f"   列名: {list(data.columns)}")
    print(f"   前5行数据:")
    print(data.head())
    
    # Alpha158因子计算示例
    print(f"\n   Alpha158因子计算示例:")
    alpha_factors = get_alpha158_factors()
    
    # 选择几个简单的Alpha158因子进行计算
    sample_alpha_factors = ['ALPHA_KMID', 'ALPHA_KLEN', 'ALPHA_OPEN0', 'ALPHA_HIGH0']
    
    for factor_name in sample_alpha_factors:
        if factor_name in alpha_factors:
            info = alpha_factors[factor_name]
            expression = info['expression']
            print(f"     {factor_name}: {info['description']}")
            print(f"       表达式: {expression}")
            
            # 简单的表达式计算（这里只是演示，实际需要更复杂的解析）
            if 'KMID' in factor_name:
                # ($close-$open)/$open
                data[factor_name] = (data['close'] - data['open']) / data['open']
            elif 'KLEN' in factor_name:
                # ($high-$low)/$open
                data[factor_name] = (data['high'] - data['low']) / data['open']
            elif 'OPEN0' in factor_name:
                # $open/$close
                data[factor_name] = data['open'] / data['close']
            elif 'HIGH0' in factor_name:
                # $high/$close
                data[factor_name] = data['high'] / data['close']
    
    print(f"   计算后的数据形状: {data.shape}")
    print(f"   新增因子列: {[col for col in data.columns if col.startswith('ALPHA_')]}")
    
    # tsfresh因子计算示例
    print(f"\n   tsfresh因子计算示例:")
    tsfresh_factors = get_tsfresh_factors()
    
    # 选择几个简单的tsfresh因子
    sample_tsfresh_factors = ['TSFRESH_mean', 'TSFRESH_std', 'TSFRESH_skewness', 'TSFRESH_kurtosis']
    
    for factor_name in sample_tsfresh_factors:
        if factor_name in tsfresh_factors:
            info = tsfresh_factors[factor_name]
            print(f"     {factor_name}: {info['description']}")
            print(f"       函数名: {info['function_name']}")
            
            # 简单的统计计算
            if 'mean' in factor_name:
                data[factor_name] = data['close'].rolling(window=20).mean()
            elif 'std' in factor_name:
                data[factor_name] = data['close'].rolling(window=20).std()
            elif 'skewness' in factor_name:
                data[factor_name] = data['close'].rolling(window=20).skew()
            elif 'kurtosis' in factor_name:
                data[factor_name] = data['close'].rolling(window=20).kurt()
    
    print(f"   最终数据形状: {data.shape}")
    print(f"   所有因子列: {[col for col in data.columns if col.startswith(('ALPHA_', 'TSFRESH_'))]}")
    
    return data

def create_factor_analysis_report():
    """创建因子分析报告"""
    
    print(f"\n7. 因子分析报告:")
    
    # 获取所有因子
    all_factors = get_comprehensive_factors()
    
    # 创建分析报告
    report = {
        "总因子数量": len(all_factors),
        "Alpha158因子数量": len(get_alpha158_factors()),
        "tsfresh因子数量": len(get_tsfresh_factors()),
        "因子来源分布": {},
        "因子类别分布": {}
    }
    
    # 统计来源分布
    for factor_name, info in all_factors.items():
        source = info.get('source', '未知')
        if source not in report["因子来源分布"]:
            report["因子来源分布"][source] = 0
        report["因子来源分布"][source] += 1
    
    # 统计类别分布
    for factor_name, info in all_factors.items():
        category = info.get('category', '未知')
        if category not in report["因子类别分布"]:
            report["因子类别分布"][category] = 0
        report["因子类别分布"][category] += 1
    
    print(f"   总因子数量: {report['总因子数量']}")
    print(f"   Alpha158因子: {report['Alpha158因子数量']}")
    print(f"   tsfresh因子: {report['tsfresh因子数量']}")
    
    print(f"\n   因子来源分布:")
    for source, count in report["因子来源分布"].items():
        print(f"     {source}: {count}个")
    
    print(f"\n   因子类别分布 (前10个):")
    sorted_categories = sorted(report["因子类别分布"].items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:10]:
        print(f"     {category}: {count}个")
    
    return report

def export_factor_usage_examples():
    """导出因子使用示例"""
    
    print(f"\n8. 导出因子使用示例:")
    
    # 获取所有因子
    all_factors = get_comprehensive_factors()
    
    # 创建使用示例数据
    examples = []
    
    # Alpha158因子示例
    alpha_factors = get_alpha158_factors()
    for factor_name in list(alpha_factors.keys())[:20]:  # 前20个
        info = alpha_factors[factor_name]
        examples.append({
            '因子名称': factor_name,
            '原始名称': factor_name.replace('ALPHA_', ''),
            '表达式': info.get('expression', ''),
            '函数名': info.get('function_name', ''),
            '描述': info.get('description', ''),
            '类别': info.get('category', ''),
            '来源': 'Alpha158',
            '使用示例': f"data['{factor_name}'] = {info.get('expression', '')}"
        })
    
    # tsfresh因子示例
    tsfresh_factors = get_tsfresh_factors()
    for factor_name in list(tsfresh_factors.keys())[:20]:  # 前20个
        info = tsfresh_factors[factor_name]
        examples.append({
            '因子名称': factor_name,
            '原始名称': factor_name.replace('TSFRESH_', ''),
            '表达式': info.get('expression', ''),
            '函数名': info.get('function_name', ''),
            '描述': info.get('description', ''),
            '类别': info.get('category', ''),
            '来源': 'tsfresh',
            '使用示例': f"tsfresh.{factor_name.replace('TSFRESH_', '')}(data['close'])"
        })
    
    # 保存示例
    df = pd.DataFrame(examples)
    df.to_csv('factor_usage_examples.csv', index=False, encoding='utf-8-sig')
    print(f"   因子使用示例已保存到: factor_usage_examples.csv")
    
    return df

def main():
    """主函数"""
    
    # 演示综合因子库
    demonstrate_comprehensive_factors()
    
    # 创建因子计算示例
    data = create_factor_calculation_example()
    
    # 创建因子分析报告
    report = create_factor_analysis_report()
    
    # 导出因子使用示例
    examples_df = export_factor_usage_examples()
    
    print(f"\n" + "=" * 60)
    print("综合因子库演示完成！")
    print("=" * 60)
    
    print(f"\n生成的文件:")
    print(f"  - comprehensive_factors_export.csv: 综合因子库")
    print(f"  - comprehensive_factor_usage_guide.md: 使用指南")
    print(f"  - factor_usage_examples.csv: 使用示例")
    print(f"  - tsfresh_factors_export.csv: tsfresh因子")
    print(f"  - tsfresh_categories.csv: tsfresh分类")

if __name__ == "__main__":
    main()
