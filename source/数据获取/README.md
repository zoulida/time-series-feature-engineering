# 股票数据获取模块

这是一个简洁的股票数据获取模块，支持自定义开始和结束日期参数，可以从指定目录获取股票数据。

## 功能特性

- 🚀 **灵活的数据获取**: 支持自定义日期范围和股票选择
- 📊 **多股票处理**: 支持单只股票、多只股票或全部股票数据获取
- 📈 **面板数据支持**: 自动创建面板数据格式
- 💾 **数据导出**: 支持CSV格式导出
- 📋 **数据摘要**: 自动生成数据摘要信息

## 文件结构

```
数据获取/
├── stock_data_fetcher.py    # 主数据获取模块
├── config.py               # 配置文件
├── utils.py                # 工具函数
├── run_data_fetcher.py     # 运行脚本
└── README.md               # 说明文档
```

## 快速开始

### 1. 基本使用

```python
from stock_data_fetcher import StockDataFetcher

# 创建数据获取器
fetcher = StockDataFetcher()

# 获取可用股票列表
stocks = fetcher.get_available_stocks()
print(f"可用股票数量: {len(stocks)}")

# 加载指定日期范围的数据
start_date = "20250101"
end_date = "20250331"
stock_data = fetcher.load_multiple_stocks_data(
    stocks[:5], start_date, end_date
)

# 显示数据摘要
summary = fetcher.get_data_summary(stock_data)
print(summary)
```

### 2. 命令行使用

```bash
# 处理所有股票，最近90天数据
python run_data_fetcher.py --all-stocks

# 指定日期范围和股票
python run_data_fetcher.py --start-date 20250101 --end-date 20250331 --stocks 000001.SZ 000002.SZ

# 从文件读取股票列表
python run_data_fetcher.py --stock-file stock_list.txt --start-date 20250101 --end-date 20250331
```

## 核心功能

### StockDataFetcher 类

#### 主要方法：

- `get_available_stocks()`: 获取可用股票代码列表
- `load_single_stock_data(stock_code, start_date, end_date)`: 加载单只股票数据
- `load_multiple_stocks_data(stock_codes, start_date, end_date)`: 加载多只股票数据
- `load_all_stocks_data(start_date, end_date)`: 加载所有股票数据
- `create_panel_data(stock_data, target_column)`: 创建面板数据
- `get_data_summary(stock_data)`: 获取数据摘要
- `save_data(data, output_path, format)`: 保存数据

### 使用示例

```python
# 1. 加载单只股票
df = fetcher.load_single_stock_data('000001.SZ', '20250101', '20250331')

# 2. 加载多只股票
stocks = ['000001.SZ', '000002.SZ', '000858.SZ']
stock_data = fetcher.load_multiple_stocks_data(stocks, '20250101', '20250331')

# 3. 创建面板数据
panel_data = fetcher.create_panel_data(stock_data, 'close')

# 4. 保存数据
fetcher.save_data(df, 'output/000001_SZ.csv', 'csv')
```

## 数据格式

### 输入数据格式

CSV文件应包含以下列：
- `date`: 日期 (YYYYMMDD格式)
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `amount`: 成交额
- `preClose`: 前收盘价
- `suspendFlag`: 停牌标志

### 输出数据格式

- 原始数据：包含所有原始列 + `stock_code` 列
- 面板数据：以日期为索引，股票代码为列名的宽格式数据

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data-dir` | 数据目录路径 | `--data-dir /path/to/data` |
| `--start-date` | 开始日期 | `--start-date 20250101` |
| `--end-date` | 结束日期 | `--end-date 20250331` |
| `--stocks` | 指定股票代码 | `--stocks 000001.SZ 000002.SZ` |
| `--all-stocks` | 处理所有股票 | `--all-stocks` |
| `--stock-file` | 从文件读取股票列表 | `--stock-file stock_list.txt` |
| `--output-dir` | 输出目录 | `--output-dir output` |

## 配置说明

在 `config.py` 中可以修改：

```python
# 数据路径配置
DEFAULT_DATA_DIR = r"f:\stockdata\getDayKlineData\20241101-20250922-front"
OUTPUT_DIR = "output"

# 日期格式
DATE_FORMAT = '%Y%m%d'
```

## 注意事项

1. **数据目录**: 确保数据目录路径正确，包含CSV格式的股票数据文件
2. **日期格式**: 所有日期参数使用YYYYMMDD格式
3. **股票代码**: 使用标准格式，如000001.SZ、000300.SH等
4. **内存使用**: 处理大量股票时注意内存使用情况

## 依赖库

```txt
pandas>=1.3.0
numpy>=1.21.0
pathlib
logging
datetime
```

## 许可证

MIT License