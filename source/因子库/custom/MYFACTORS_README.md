# 自定义因子库 - MyFactors

## 概述

MyFactors是一个用户自定义的量化因子库，专门用于实现个性化的量化投资因子。该库提供了灵活的框架来定义、计算和管理自定义因子。

## 文件结构

```
因子库/
├── myfactors.py                    # 自定义因子库主文件
├── myfactors_usage_example.py      # 使用示例
├── MYFACTORS_README.md             # 说明文档
└── zhangtingCalculation.py         # 涨停判断模块（外部依赖）
```

## 主要特性

- ✅ **模块化设计**: 易于添加新的自定义因子
- ✅ **涨停判断集成**: 内置涨停判断功能
- ✅ **复合因子支持**: 支持多条件组合的复合因子
- ✅ **灵活参数配置**: 支持自定义参数设置
- ✅ **数据导出功能**: 支持因子结果导出
- ✅ **详细统计信息**: 提供因子计算统计

## 已实现的因子

### 1. ZHANGTING_VOLUME_PRICE_FACTOR (涨停+成交量+价格复合因子)

**因子描述**: 7日内有涨停，成交量20日均线当日大于7天前1.5倍，价格不高于close20日均线10%的复合因子

**因子逻辑**:
- **条件1**: 7个交易日内有涨停
- **条件2**: 成交量20日均线当日大于7天前1.5倍  
- **条件3**: 价格不高于close20日均线10%

**因子值**: 三个条件等权组合，范围[0, 3]
- 0: 不满足任何条件
- 1: 满足1个条件
- 2: 满足2个条件  
- 3: 满足所有条件

**参数配置**:
```python
{
    "涨停判断周期": 7,
    "成交量均线周期": 20,
    "成交量比较周期": 7,
    "成交量倍数": 1.5,
    "价格均线周期": 20,
    "价格上限比例": 0.1
}
```

## 快速开始

### 1. 基本使用

```python
from myfactors import MyFactors, create_my_factors

# 创建因子库
my_factors = create_my_factors()

# 查看可用因子
print(my_factors.list_factors())

# 计算因子
factor_values = my_factors.calculate_factor(
    'ZHANGTING_VOLUME_PRICE_FACTOR',
    close=data['close'],
    high=data['high'], 
    volume=data['volume'],
    stock_code='000001.SZ'
)
```

### 2. 使用示例数据

```python
# 运行完整示例
python myfactors_usage_example.py
```

### 3. 添加自定义因子

```python
# 添加新的自定义因子
my_factors.add_custom_factor(
    name="MY_RSI_FACTOR",
    expression="MY_RSI_FACTOR($close, 14)",
    function_name="my_rsi_factor",
    description="自定义RSI因子",
    category="技术指标",
    parameters={"周期": 14}
)
```

## 详细使用说明

### 因子计算

```python
# 计算涨停+成交量+价格复合因子
factor_values = my_factors.calculate_factor(
    'ZHANGTING_VOLUME_PRICE_FACTOR',
    close=close_series,      # 收盘价序列
    high=high_series,        # 最高价序列
    volume=volume_series,    # 成交量序列
    stock_code='000001.SZ'   # 股票代码
)
```

### 因子信息查询

```python
# 获取因子信息
info = my_factors.get_factor_info('ZHANGTING_VOLUME_PRICE_FACTOR')
print(info['description'])
print(info['parameters'])

# 获取因子描述
desc = my_factors.get_factor_description('ZHANGTING_VOLUME_PRICE_FACTOR')

# 获取因子函数名
func_name = my_factors.get_factor_function_name('ZHANGTING_VOLUME_PRICE_FACTOR')
```

### 统计信息

```python
# 获取因子库统计
stats = my_factors.get_statistics()
print(f"总因子数量: {stats['总因子数量']}")
print(f"因子列表: {stats['因子列表']}")
print(f"按类别统计: {stats['按类别统计']}")
```

### 数据导出

```python
# 导出因子库信息
df = my_factors.export_factors("my_factors.csv")

# 导出因子计算结果
result_df = data.copy()
result_df['factor_values'] = factor_values
result_df.to_csv("factor_results.csv")
```

## 因子实现细节

### 涨停判断逻辑

```python
def _check_zhangting_in_period(self, close, high, stock_code, period=7):
    """
    检查指定周期内是否有涨停
    
    使用zhangtingCalculation模块的limitUp函数计算涨停价
    比较最高价是否达到涨停价
    """
```

### 成交量条件判断

```python
def _check_volume_condition(self, volume, ma_period=20, compare_period=7, multiplier=1.5):
    """
    检查成交量条件
    
    计算成交量20日均线
    比较当日均线与compare_period天前的倍数关系
    """
```

### 价格条件判断

```python
def _check_price_condition(self, close, ma_period=20, max_ratio=0.1):
    """
    检查价格条件
    
    计算收盘价20日均线
    检查当前价格是否不高于均线的(1+max_ratio)倍
    """
```

## 依赖要求

### 必需依赖
- pandas >= 1.0.0
- numpy >= 1.18.0

### 外部模块
- zhangtingCalculation.py (涨停判断模块)
  - 路径: `d:\pythonProject\JQKA\indicator\zhangtingCalculation.py`
  - 功能: 提供涨停价计算和股票代码振幅判断

## 数据格式要求

### 输入数据格式
```python
# DataFrame格式，包含以下列：
data = pd.DataFrame({
    'close': close_prices,    # 收盘价
    'high': high_prices,      # 最高价  
    'volume': volumes,        # 成交量
    # 其他列...
})
```

### 时间序列要求
- 数据需要按时间顺序排列
- 建议至少包含20个交易日的数据
- 支持日频数据

## 扩展因子库

### 添加新因子的步骤

1. **定义因子函数**:
```python
def my_new_factor(self, param1, param2, **kwargs):
    """新因子计算函数"""
    # 实现因子逻辑
    return result_series
```

2. **注册因子**:
```python
self.factors["MY_NEW_FACTOR"] = {
    "expression": "MY_NEW_FACTOR($close, $high)",
    "function_name": "my_new_factor", 
    "description": "新因子描述",
    "category": "因子类别",
    "parameters": {"参数1": "值1"}
}
```

3. **测试因子**:
```python
# 使用示例数据测试
factor_values = my_factors.calculate_factor('MY_NEW_FACTOR', ...)
```

## 注意事项

1. **涨停判断模块**: 确保zhangtingCalculation.py模块路径正确
2. **数据质量**: 输入数据需要清洗，避免缺失值和异常值
3. **计算效率**: 对于大量数据，建议分批处理
4. **参数调优**: 根据实际市场情况调整因子参数

## 示例输出

### 因子值统计示例
```
因子值范围: 0.00 到 3.00
因子值统计:
  均值: 0.45
  标准差: 0.78
  非零值数量: 23

因子值分布:
  值=0: 77次 (77.0%)
  值=1: 15次 (15.0%) 
  值=2: 6次 (6.0%)
  值=3: 2次 (2.0%)
```

### 最近因子值示例
```
最近10天的因子值:
  2024-01-15: 0.00
  2024-01-16: 1.00
  2024-01-17: 0.00
  2024-01-18: 2.00
  2024-01-19: 0.00
  ...
```

## 更新日志

- **v1.0** (2024): 初始版本，实现涨停+成交量+价格复合因子

## 联系方式

如有问题或建议，请联系开发者。

---

*最后更新: 2024年*
