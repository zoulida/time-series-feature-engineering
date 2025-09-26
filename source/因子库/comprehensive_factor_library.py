# -*- coding: utf-8 -*-
"""
综合因子库
整合Alpha158因子和tsfresh特征的综合因子库
"""

import pandas as pd
from factor_library import ALL_FACTORS as ALPHA158_FACTORS
from extract_tsfresh_features import create_tsfresh_factor_library

def create_comprehensive_factor_library():
    """创建综合因子库"""
    
    # 获取Alpha158因子
    alpha158_factors = ALPHA158_FACTORS.copy()
    
    # 获取tsfresh因子
    tsfresh_factors, _ = create_tsfresh_factor_library()
    
    # 合并因子库
    comprehensive_factors = {}
    
    # 添加Alpha158因子
    for name, info in alpha158_factors.items():
        comprehensive_factors[f"ALPHA_{name}"] = {
            **info,
            "source": "Alpha158",
            "category": f"Alpha158_{info.get('category', '未知')}"
        }
    
    # 添加tsfresh因子
    for name, info in tsfresh_factors.items():
        comprehensive_factors[f"TSFRESH_{name}"] = {
            **info,
            "source": "tsfresh",
            "category": info.get('category', 'tsfresh_其他特征')
        }
    
    return comprehensive_factors, alpha158_factors, tsfresh_factors

def get_factor_statistics(comprehensive_factors):
    """获取因子库统计信息"""
    
    stats = {
        "总因子数量": len(comprehensive_factors),
        "Alpha158因子": 0,
        "tsfresh因子": 0,
        "按来源统计": {},
        "按类别统计": {}
    }
    
    for name, info in comprehensive_factors.items():
        source = info.get('source', '未知')
        category = info.get('category', '未知')
        
        # 统计来源
        if source not in stats["按来源统计"]:
            stats["按来源统计"][source] = 0
        stats["按来源统计"][source] += 1
        
        if source == "Alpha158":
            stats["Alpha158因子"] += 1
        elif source == "tsfresh":
            stats["tsfresh因子"] += 1
        
        # 统计类别
        if category not in stats["按类别统计"]:
            stats["按类别统计"][category] = 0
        stats["按类别统计"][category] += 1
    
    return stats

def export_comprehensive_factors(comprehensive_factors):
    """导出综合因子库"""
    
    # 创建DataFrame
    data = []
    for factor_name, info in comprehensive_factors.items():
        data.append({
            '因子名称': factor_name,
            '原始名称': factor_name.replace('ALPHA_', '').replace('TSFRESH_', ''),
            '表达式': info.get('expression', ''),
            '函数名': info.get('function_name', ''),
            '描述': info.get('description', ''),
            '类别': info.get('category', ''),
            '来源': info.get('source', ''),
            '模块': info.get('module', '')
        })
    
    df = pd.DataFrame(data)
    
    # 保存到CSV
    df.to_csv('comprehensive_factors_export.csv', index=False, encoding='utf-8-sig')
    print(f"综合因子库已保存到: comprehensive_factors_export.csv")
    
    return df

def create_factor_usage_guide():
    """创建因子使用指南"""
    
    guide = """
# 综合因子库使用指南

## 概述
本因子库整合了Alpha158因子和tsfresh特征，总共包含{}个因子。

## 因子分类

### Alpha158因子 (158个)
- 来源: qlib.contrib.data.handler.Alpha158
- 特点: 专门为量化投资设计的因子
- 分类: K线形态、价格、滚动统计、成交量等

### tsfresh因子 (74个)
- 来源: tsfresh.feature_extraction.feature_calculators
- 特点: 通用时间序列特征提取
- 分类: 统计、频域、复杂度、变化率、熵、自相关、分形等

## 使用方法

### 1. 基本使用
```python
from comprehensive_factor_library import get_comprehensive_factors

# 获取所有因子
factors = get_comprehensive_factors()

# 获取特定来源的因子
alpha_factors = {k: v for k, v in factors.items() if v['source'] == 'Alpha158'}
tsfresh_factors = {k: v for k, v in factors.items() if v['source'] == 'tsfresh'}
```

### 2. 因子查询
```python
# 按类别查询
kbar_factors = {k: v for k, v in factors.items() if 'K线形态' in v['category']}
statistical_factors = {k: v for k, v in factors.items() if '统计' in v['category']}

# 按来源查询
alpha_factors = {k: v for k, v in factors.items() if k.startswith('ALPHA_')}
tsfresh_factors = {k: v for k, v in factors.items() if k.startswith('TSFRESH_')}
```

### 3. 因子计算
```python
# Alpha158因子计算
from qlib.contrib.data.handler import Alpha158
alpha_handler = Alpha158()

# tsfresh因子计算
from tsfresh.feature_extraction import extract_features
tsfresh_features = extract_features(data, default_fc_parameters=tsfresh_settings)
```

## 因子命名规则

- Alpha158因子: ALPHA_<原始名称>
- tsfresh因子: TSFRESH_<原始名称>

## 注意事项

1. Alpha158因子需要OHLCV数据
2. tsfresh因子需要时间序列数据
3. 建议根据具体需求选择合适的因子子集
4. 因子计算前请确保数据质量

## 扩展因子库

可以通过修改comprehensive_factor_library.py来添加自定义因子：

```python
# 添加自定义因子
CUSTOM_FACTORS = {
    "CUSTOM_MY_FACTOR": {
        "expression": "自定义表达式",
        "function_name": "my_custom_factor",
        "description": "我的自定义因子",
        "category": "自定义类别",
        "source": "custom"
    }
}

# 合并到综合因子库
comprehensive_factors.update(CUSTOM_FACTORS)
```
"""
    
    return guide

def main():
    """主函数"""
    
    print("=" * 60)
    print("创建综合因子库")
    print("=" * 60)
    
    # 创建综合因子库
    comprehensive_factors, alpha158_factors, tsfresh_factors = create_comprehensive_factor_library()
    
    # 获取统计信息
    stats = get_factor_statistics(comprehensive_factors)
    
    # 显示统计信息
    print(f"\n总因子数量: {stats['总因子数量']}")
    print(f"Alpha158因子: {stats['Alpha158因子']}")
    print(f"tsfresh因子: {stats['tsfresh因子']}")
    
    print("\n按来源统计:")
    for source, count in stats["按来源统计"].items():
        print(f"  {source}: {count}个")
    
    print("\n按类别统计 (前10个):")
    sorted_categories = sorted(stats["按类别统计"].items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:10]:
        print(f"  {category}: {count}个")
    
    # 导出因子库
    df = export_comprehensive_factors(comprehensive_factors)
    
    # 创建使用指南
    guide = create_factor_usage_guide()
    with open('comprehensive_factor_usage_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    print(f"使用指南已保存到: comprehensive_factor_usage_guide.md")
    
    # 显示示例因子
    print("\n示例因子:")
    sample_factors = list(comprehensive_factors.keys())[:10]
    for factor in sample_factors:
        info = comprehensive_factors[factor]
        print(f"  {factor}: {info['description']} ({info['source']})")
    
    print("\n" + "=" * 60)
    print("综合因子库创建完成！")
    print("=" * 60)
    
    return comprehensive_factors, stats

# 提供便捷的访问函数
def get_comprehensive_factors():
    """获取综合因子库"""
    comprehensive_factors, _, _ = create_comprehensive_factor_library()
    return comprehensive_factors

def get_alpha158_factors():
    """获取Alpha158因子"""
    return {k: v for k, v in get_comprehensive_factors().items() if v['source'] == 'Alpha158'}

def get_tsfresh_factors():
    """获取tsfresh因子"""
    return {k: v for k, v in get_comprehensive_factors().items() if v['source'] == 'tsfresh'}

def get_factors_by_source(source):
    """按来源获取因子"""
    return {k: v for k, v in get_comprehensive_factors().items() if v['source'] == source}

def get_factors_by_category(category):
    """按类别获取因子"""
    return {k: v for k, v in get_comprehensive_factors().items() if category in v['category']}

if __name__ == "__main__":
    main()
