
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
