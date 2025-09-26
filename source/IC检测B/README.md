# IC检测工作流

这是一个完整的IC（信息系数）检测工作流，用于分析股票因子的预测能力。

## 工作流概述

该工作流包含4个主要步骤：

1. **步骤1：获取股票数据** (`step1_get_stock_data.py`)
   - 使用 `stock_data_fetcher.py` 获取最大时段的5只股票数据
   - 自动分析所有股票的时间范围，选择共同的最大时间段
   - 保存股票数据到 `data/` 目录

2. **步骤2：选择因子** (`step2_select_factors.py`)
   - 使用 `alpha158_factors.py` 随机选择8个因子
   - 确保因子类别多样性（K线形态、价格因子、滚动统计等）
   - 保存选中的因子信息

3. **步骤3：生成训练数据** (`step3_generate_training_data.py`)
   - 基于原始股票数据计算因子值
   - 计算不同周期的收益率（1天、5天、10天、20天）
   - 生成包含原始数据、因子数据和收益率的完整数据集

4. **步骤4：IC检测分析** (`step4_ic_analysis.py`)
   - 使用改进的IC分析方法
   - 计算皮尔逊和斯皮尔曼相关系数
   - 进行显著性检验和统计分析
   - 生成IC报告和因子排名

## 文件结构

```
IC检测B/
├── step1_get_stock_data.py      # 步骤1：获取股票数据
├── step2_select_factors.py      # 步骤2：选择因子
├── step3_generate_training_data.py  # 步骤3：生成训练数据
├── step4_ic_analysis.py         # 步骤4：IC检测分析
├── run_ic_workflow.py           # 主工作流程序
├── run_workflow.bat             # Windows批处理启动脚本
├── run_workflow.ps1             # PowerShell启动脚本
├── README.md                    # 说明文档
└── data/                        # 数据输出目录
    ├── selected_stock_codes.txt     # 选中的股票代码
    ├── selected_factors.json        # 选中的因子信息
    ├── training_data_full.csv       # 完整训练数据
    ├── ic_analysis_data.csv         # IC分析数据
    ├── ic_results.json              # IC分析结果
    ├── ic_report.csv                # IC分析报告
    ├── ic_summary.csv               # IC分析摘要
    └── workflow_summary.json        # 工作流摘要
```

## 使用方法

### 方法1：运行主工作流（推荐）

```bash
# 直接运行主工作流
python run_ic_workflow.py
```

### 方法2：使用启动脚本

**Windows批处理：**
```cmd
run_workflow.bat
```

**PowerShell：**
```powershell
.\run_workflow.ps1
```

### 方法3：分步执行

如果需要单独执行某个步骤：

```bash
# 步骤1：获取股票数据
python step1_get_stock_data.py

# 步骤2：选择因子
python step2_select_factors.py

# 步骤3：生成训练数据
python step3_generate_training_data.py

# 步骤4：IC检测分析
python step4_ic_analysis.py
```

## 依赖项

### 必需依赖
- `pandas` - 数据处理
- `numpy` - 数值计算
- `json` - JSON数据处理
- `datetime` - 日期时间处理
- `scipy` - 科学计算（用于统计检验）

## 输出结果

### 数据文件
- `selected_stock_codes.txt` - 选中的5只股票代码
- `selected_factors.json` - 选中的8个因子详细信息
- `training_data_full.csv` - 完整的训练数据集
- `ic_analysis_data.csv` - 用于IC分析的数据

### 分析结果
- `ic_results.json` - 详细的IC分析结果
- `ic_report.csv` - IC分析报告表格
- `ic_summary.csv` - IC分析摘要和排名
- `workflow_summary.json` - 工作流执行摘要

### 日志文件
- `ic_workflow.log` - 工作流执行日志

## 因子类型

工作流会从Alpha158因子库中随机选择8个因子，包括：

1. **K线形态因子** - 如KMID（K线实体比率）、KLEN（K线长度比率）等
2. **价格因子** - 如OPEN、HIGH、LOW相对收盘价的比率
3. **滚动统计因子** - 如ROC（变化率）、MA（移动平均）、STD（标准差）等
4. **成交量因子** - 如VMA（成交量移动平均）、VSTD（成交量标准差）等

## IC分析说明

IC（Information Coefficient）是衡量因子预测能力的重要指标：

- **IC > 0.05** - 因子预测能力较强
- **0.02 < IC ≤ 0.05** - 因子预测能力中等
- **0 < IC ≤ 0.02** - 因子预测能力较弱
- **IC ≤ 0** - 因子无预测能力或预测方向错误

## 注意事项

1. 确保 `stock_data_fetcher.py` 中的数据目录路径正确
2. 如果 `alphalens` 不可用，程序会自动使用简化的IC计算方法
3. 工作流会自动创建 `data/` 目录保存所有输出文件
4. 每个步骤都会生成详细的日志信息

## 故障排除

### 常见问题

1. **数据目录不存在**
   - 检查 `stock_data_fetcher.py` 中的 `data_dir` 路径
   - 确保股票数据文件存在

2. **IC分析问题**
   - 程序使用改进的IC分析方法，不依赖外部库
   - 包含皮尔逊和斯皮尔曼相关系数计算

3. **内存不足**
   - 减少股票数量或时间范围
   - 修改 `step1_get_stock_data.py` 中的股票选择逻辑

4. **数据格式错误**
   - 检查股票数据文件格式
   - 确保包含必要的列：date, open, high, low, close, volume

## 扩展功能

可以根据需要扩展以下功能：

1. 增加更多因子类型
2. 支持更多股票
3. 添加更多技术指标
4. 实现更复杂的IC分析
5. 添加可视化功能

## 联系信息

如有问题或建议，请查看日志文件或联系开发团队。
