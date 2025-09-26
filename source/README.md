# 时序特征工程 - 源代码目录

本目录包含时序特征工程项目的所有源代码、数据和结果文件。

## 📁 文件说明

### 🔧 程序文件

| 文件名 | 功能描述 |
|--------|----------|
| `prepare_tsfresh_input.py` | 数据预处理脚本，将多股票CSV整理为tsfresh长表格式 |
| `run_factor_ic.py` | 因子提取与IC计算主脚本，构建滚动窗口并计算IC评分 |
| `tsfresh_ic_analysis.py` | IC分析核心程序，包含特征重要性计算和可视化 |
| `enhanced_feature_extractor.py` | 增强版特征提取器，不依赖GPU/CUDA |
| `data_relationship_example.py` | 数据关系示例，展示time_series_df与target_df的关系 |

### 📊 数据文件

| 文件名 | 描述 |
|--------|------|
| `000001.SZ.csv` | 示例股票数据（平安银行） |
| `tsfresh_long.csv` | 处理后的长表数据，格式：[id,time,value] |
| `tsfresh_target_panel.csv` | 目标变量面板数据，格式：[id,time,target] |
| `window_features.csv` | 滚动窗口特征矩阵 |

### 📈 结果文件

| 文件名 | 描述 |
|--------|------|
| `ic_scores.csv` | IC评分结果，包含各特征的相关系数 |
| `feature_importance_report.txt` | 特征重要性详细报告 |
| `window_feature_ic_report.txt` | 窗口特征IC分析报告 |

### 🎨 可视化文件

| 文件名 | 描述 |
|--------|------|
| `ic_analysis_results.png` | IC分析结果可视化图表 |
| `data_relationship.png` | 数据关系示例图表 |

### ⚙️ 配置文件

| 文件名 | 描述 |
|--------|------|
| `requirements.txt` | 完整依赖包列表 |
| `requirements_simple.txt` | 简化依赖包列表（推荐） |

## 🚀 使用流程

### 步骤1：环境准备
```bash
# 激活虚拟环境（在项目根目录）
.\venv\Scripts\Activate.ps1

# 进入程序文件目录
cd source/程序文件

# 安装依赖
pip install -r ../配置文件/requirements_simple.txt
```

### 步骤2：数据预处理
```bash
# 处理A股数据，生成长表和目标变量
python prepare_tsfresh_input.py \
  --data-dir F:\stockdata\getDayKlineData\20241101-20250818-front \
  --single \
  --use-date \
  --out-long ../数据文件/tsfresh_long.csv \
  --out-target ../数据文件/tsfresh_target_panel.csv
```

**参数说明：**
- `--data-dir`: 原始CSV数据目录
- `--single`: 使用单变量模式（收盘价）
- `--use-date`: 使用date列作为时间轴
- `--out-long`: 长表输出文件
- `--out-target`: 目标变量输出文件

### 步骤3：因子提取与IC计算
```bash
# 构建滚动窗口，提取因子并计算IC
python run_factor_ic.py \
  --long-csv ../数据文件/tsfresh_long.csv \
  --target-csv ../数据文件/tsfresh_target_panel.csv \
  --window 60 \
  --out-features ../数据文件/window_features.csv \
  --out-ic ../结果文件/ic_scores.csv
```

**参数说明：**
- `--long-csv`: 长表数据文件
- `--target-csv`: 目标变量文件
- `--window`: 滚动窗口长度（交易日）
- `--out-features`: 特征矩阵输出文件
- `--out-ic`: IC评分输出文件

## 📊 数据格式说明

### 长表格式 (tsfresh_long.csv)
```csv
id,time,value
000001.SZ,2024-11-01,11.068
000001.SZ,2024-11-04,11.098
...
```

### 目标变量格式 (tsfresh_target_panel.csv)
```csv
id,time,target
000001.SZ,2024-11-01,11.548
000001.SZ,2024-11-04,11.678
...
```
其中target为未来20日最高收盘价（排除当天）

### 特征矩阵格式 (window_features.csv)
```csv
window_id,stock_id,anchor_time,feature1,feature2,...,target
000001.SZ|20241201,000001.SZ,2024-12-01,0.123,0.456,...,11.548
...
```

## 🔍 核心功能

### 1. 数据预处理
- 支持多股票CSV数据批量处理
- 自动过滤停牌记录
- 生成tsfresh兼容的长表格式
- 计算未来20日最高价作为预测目标

### 2. 特征提取
- **tsfresh模式**: 使用tsfresh库提取时间序列特征
- **增强模式**: 使用自研特征提取器（无GPU依赖）
- 支持45+种统计特征（趋势、波动、周期性等）

### 3. 滚动窗口
- 构建时间滚动窗口进行特征提取
- 窗口ID格式：`{stock_id}|{YYYYMMDD}`
- 自动对齐目标变量

### 4. IC评分
- 计算Pearson、Spearman、Kendall三种相关系数
- 生成特征重要性排名
- 可视化IC分布和特征重要性

## 🛠️ 故障排除

### 常见问题

1. **tsfresh安装失败**
   ```bash
   # 使用增强版特征提取器
   # 脚本会自动回退到enhanced_feature_extractor.py
   ```

2. **内存不足**
   ```bash
   # 减少窗口长度
   --window 30
   
   # 或分批处理数据
   ```

3. **数据格式错误**
   ```bash
   # 检查CSV文件格式
   # 确保包含必要的列：date/time, open, high, low, close, volume, amount
   ```

## 📝 注意事项

1. 确保虚拟环境已激活
2. 大数据集处理需要足够内存
3. 首次运行可能需要较长时间
4. 建议先在小数据集上测试

## 🔗 相关文档

- [tsfresh官方文档](https://tsfresh.readthedocs.io/)
- [pandas时间序列处理](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [scipy统计分析](https://docs.scipy.org/doc/scipy/reference/stats.html) 