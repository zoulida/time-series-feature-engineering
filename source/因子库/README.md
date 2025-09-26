# 因子库

本目录包含多种因子库的实现，用于量化投资中的特征工程。现已重构为统一的因子库架构。

## 文件结构

### 核心文件
- `unified_factor_library.py` - **统一因子库类**（推荐使用）
- `unified_factor_usage_example.py` - 统一因子库使用示例

### 各因子源实现
- `alpha158_factors.py` - Alpha158因子库实现
- `extract_tsfresh_features.py` - tsfresh特征提取
- `tsfresh_factor_library.py` - tsfresh因子库（重命名）
- `tsfresh_factor_usage_example.py` - tsfresh使用示例（重命名）
- `tsfresh_factor_calculation_template.py` - tsfresh计算模板（重命名）
- `tsfresh_factor_usage_examples.csv` - tsfresh使用示例数据（重命名）

### 综合因子库
- `comprehensive_factor_library.py` - 综合因子库（旧版本）
- `comprehensive_factor_example.py` - 综合因子库示例

## 推荐使用方法

### 统一因子库（推荐）

```python
from unified_factor_library import UnifiedFactorLibrary, create_unified_factor_library

# 创建统一因子库（加载所有可用因子源）
factor_lib = create_unified_factor_library()

# 或者指定特定的因子源
factor_lib = UnifiedFactorLibrary(sources=['alpha158', 'tsfresh'])

# 获取所有因子
all_factors = factor_lib.get_all_factors()

# 按来源获取因子
alpha_factors = factor_lib.get_factors_by_source('Alpha158')
tsfresh_factors = factor_lib.get_factors_by_source('tsfresh')

# 按类别获取因子
kbar_factors = factor_lib.get_factors_by_category('K线形态')
statistical_factors = factor_lib.get_factors_by_category('统计')

# 搜索因子
mean_factors = factor_lib.search_factors("mean")

# 复合过滤
filtered = factor_lib.filter_factors(
    sources=['Alpha158'],
    categories=['K线形态'],
    keywords=['mid']
)

# 添加自定义因子
custom_factors = {
    "MY_RSI": {
        "expression": "RSI($close, 14)",
        "function_name": "my_rsi_14",
        "description": "自定义RSI指标",
        "category": "技术指标"
    }
}
factor_lib.add_custom_factors(custom_factors)

# 导出因子库
df = factor_lib.export_to_csv("my_factors.csv")

# 获取统计信息
stats = factor_lib.get_statistics()
print(f"总因子数量: {stats['总因子数量']}")
```

### 传统使用方法

#### Alpha158因子

```python
from alpha158_factors import Alpha158Factors

# 创建因子库实例
factor_lib = Alpha158Factors()

# 获取所有因子
factors = factor_lib.get_all_factors()

# 按类别获取因子
kbar_factors = factor_lib.get_factors_by_category('K线形态')
```

#### tsfresh特征

```python
from extract_tsfresh_features import create_tsfresh_factor_library

# 创建tsfresh因子库
tsfresh_factors, categories = create_tsfresh_factor_library()
```

## 因子分类

### Alpha158因子 (158个)
- **K线形态因子**: KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
- **价格因子**: OPEN0, HIGH0, LOW0, VWAP0
- **滚动统计因子**: ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD
- **成交量因子**: VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD

### tsfresh因子 (74个)
- **统计特征**: mean, std, var, skewness, kurtosis, median, min, max, sum, abs_energy
- **频域特征**: fft_aggregated, fft_coefficient, spectral_centroid, spectral_entropy
- **复杂度特征**: approximate_entropy, sample_entropy, permutation_entropy
- **变化率特征**: mean_abs_change, mean_change, number_peaks, number_crossings
- **熵特征**: approximate_entropy, sample_entropy, permutation_entropy, multiscale_entropy
- **自相关特征**: autocorrelation, partial_autocorrelation, lag, cross_correlation
- **分形特征**: detrended_fluctuation_analysis, hurst_exponent, fractal_dimension
- **时间特征**: time_reversal_asymmetry_statistic, c3, cid_ce, symmetry_looking

## 统一因子库特性

### 支持的因子源
1. **Alpha158** - 量化投资专用因子
2. **tsfresh** - 通用时间序列特征
3. **custom** - 用户自定义因子

### 主要功能
- ✅ 统一接口管理多种因子源
- ✅ 灵活的因子过滤和搜索
- ✅ 支持自定义因子添加
- ✅ 因子统计和导出功能
- ✅ 因子信息查询和描述
- ✅ 可扩展的架构设计

### 因子命名规则
- Alpha158因子: `ALPHA_<原始名称>`
- tsfresh因子: `TSFRESH_<原始名称>`
- 自定义因子: `CUSTOM_<原始名称>`

## 使用示例

运行完整的使用示例：

```bash
python unified_factor_usage_example.py
```

## 因子表达式说明

### Alpha158因子表达式语法
- `$close`: 收盘价
- `$open`: 开盘价
- `$high`: 最高价
- `$low`: 最低价
- `$volume`: 成交量
- `$vwap`: 成交量加权平均价
- `Ref($close, n)`: n天前的收盘价
- `Mean($close, n)`: n天收盘价均值
- `Std($close, n)`: n天收盘价标准差
- `Max($high, n)`: n天最高价的最大值
- `Min($low, n)`: n天最低价的最小值
- `Rank($close, n)`: 当前收盘价在n天中的排名
- `Corr(x, y, n)`: x和y的n天相关系数
- `Greater(a, b)`: a和b中的较大值
- `Less(a, b)`: a和b中的较小值

### tsfresh因子表达式语法
- `tsfresh.{feature_name}(x)`: 调用tsfresh特征计算函数
- 例如: `tsfresh.mean(x)`, `tsfresh.std(x)`, `tsfresh.approximate_entropy(x)`

## 注意事项

1. **依赖要求**: 确保安装了必要的依赖包（pandas, numpy, tsfresh等）
2. **数据格式**: 
   - Alpha158因子需要OHLCV数据
   - tsfresh因子需要时间序列数据
3. **性能考虑**: 建议根据具体需求选择合适的因子子集
4. **扩展性**: 可以通过继承`FactorSource`类来添加新的因子源

## 扩展因子库

### 添加自定义因子源

```python
from unified_factor_library import FactorSource

class MyCustomFactorSource(FactorSource):
    def __init__(self):
        self.factors = {...}  # 你的因子定义
    
    def get_factors(self):
        return self.factors
    
    def get_source_name(self):
        return "my_custom"
    
    def get_factor_count(self):
        return len(self.factors)

# 在统一因子库中使用
factor_lib = UnifiedFactorLibrary()
factor_lib.sources['my_custom'] = MyCustomFactorSource()
```

### 添加单个自定义因子

```python
custom_factors = {
    "MY_RSI": {
        "expression": "RSI($close, 14)",
        "function_name": "my_rsi_14",
        "description": "自定义RSI指标，14天周期",
        "category": "技术指标",
        "formula": "RSI(收盘价, 14)"
    }
}

factor_lib.add_custom_factors(custom_factors)
```

## 更新日志

- **v2.0**: 重构为统一因子库架构，支持多种因子源
- **v1.0**: 基础因子库实现

## 文件结构

```
因子库/
├── README.md                                    # 说明文档
├── unified_factor_library.py                    # 统一因子库类（推荐）
├── unified_factor_usage_example.py              # 统一因子库使用示例
├── alpha158_factors.py                          # Alpha158因子库
├── extract_tsfresh_features.py                  # tsfresh特征提取
├── tsfresh_factor_library.py                    # tsfresh因子库
├── tsfresh_factor_usage_example.py              # tsfresh使用示例
├── tsfresh_factor_calculation_template.py       # tsfresh计算模板
├── tsfresh_factor_usage_examples.csv            # tsfresh使用示例数据
├── comprehensive_factor_library.py              # 综合因子库（旧版本）
└── comprehensive_factor_example.py              # 综合因子库示例
```

## 版本信息

- 基于qlib Alpha158因子集和tsfresh特征库
- 提取时间: 2024年
- 总因子数量: 232个（Alpha158: 158个 + tsfresh: 74个）
- 支持窗口期: 5, 10, 20, 30, 60天（Alpha158）