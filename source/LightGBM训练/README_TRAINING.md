# LightGBM训练管道

本目录包含用于训练LightGBM回归模型预测`return_15d`的完整管道。

## 功能概述

- **目标变量**: `return_15d` (15日收益率)
- **特征变量**: 
  - `CUSTOM_PRICE_SCORE_FACTOR` (价格评分因子)
  - `CUSTOM_ZHANGTING_SCORE_FACTOR` (涨停评分因子)
  - `CUSTOM_VOLUME_SCORE_FACTOR` (成交量评分因子)
- **数据划分**: 训练集80%，验证集10%，测试集10%（按时间顺序）
- **IC分析**: 仅对测试集进行IC分析

## 文件说明

### 核心程序
- `data_preprocessing.py` - 数据预处理程序
- `lightgbm_training.py` - LightGBM模型训练程序
- `run_training_pipeline.py` - 主训练管道程序

### 运行脚本
- `run_training.bat` - Windows批处理脚本
- `run_training.ps1` - PowerShell脚本

### 输出目录
- `processed_data/` - 处理后的训练/验证/测试数据
- `model_results/` - 模型训练结果和预测
- `*.log` - 日志文件
- `training_pipeline_report.*` - 训练报告

## 使用方法

### 方法1: 使用批处理脚本（推荐）
```cmd
run_training.bat
```

### 方法2: 使用PowerShell脚本
```powershell
.\run_training.ps1
```

### 方法3: 直接运行Python程序
```bash
python run_training_pipeline.py
```

### 方法4: 分步执行
```bash
# 步骤1: 数据预处理
python data_preprocessing.py

# 步骤2: 模型训练
python lightgbm_training.py
```

## 数据要求

程序会自动查找`../IC检测批量/data/`目录下最新的`training_data_batch_*.csv`文件。

数据文件必须包含以下列：
- `date` - 日期
- `return_15d` - 目标变量（15日收益率）
- `factor_CUSTOM_PRICE_SCORE_FACTOR` - 价格评分因子
- `factor_CUSTOM_ZHANGTING_SCORE_FACTOR` - 涨停评分因子
- `factor_CUSTOM_VOLUME_SCORE_FACTOR` - 成交量评分因子

## 输出结果

### 数据文件
- `processed_data/train_data.csv` - 训练集数据
- `processed_data/val_data.csv` - 验证集数据
- `processed_data/test_data.csv` - 测试集数据
- `processed_data/data_stats.json` - 数据统计信息

### 模型结果
- `model_results/predictions.csv` - 所有数据集的预测结果
- `model_results/model_metrics.json` - 模型性能指标
- `model_results/ic_results.json` - IC分析结果（仅测试集）
- `model_results/feature_importance.csv` - 特征重要性
- `model_results/lightgbm_model.txt` - 训练好的模型文件

### 报告文件
- `training_pipeline_report.json` - 完整训练报告（JSON格式）
- `training_pipeline_report.txt` - 训练报告（文本格式）

## 模型性能指标

- **RMSE** - 均方根误差
- **MAE** - 平均绝对误差
- **R²** - 决定系数

## IC分析

IC（信息系数）分析仅对测试集进行，包括：
- **皮尔逊相关系数** - 线性相关性
- **斯皮尔曼相关系数** - 单调相关性
- **p值** - 统计显著性

## 依赖项

程序会自动安装以下依赖：
- `pandas` - 数据处理
- `numpy` - 数值计算
- `scikit-learn` - 机器学习工具
- `scipy` - 科学计算
- `lightgbm` - LightGBM模型

## 注意事项

1. 确保数据文件路径正确
2. 程序会自动处理缺失值和异常值
3. 数据按时间顺序划分，确保时间序列的连续性
4. IC分析仅使用测试集数据，避免过拟合
5. 所有日志会保存到对应的`.log`文件中

## 故障排除

### 常见问题

1. **数据文件未找到**
   - 检查`../IC检测批量/data/`目录是否存在
   - 确保有`training_data_batch_*.csv`文件

2. **LightGBM安装失败**
   - 手动安装: `pip install lightgbm`
   - 检查Python版本兼容性

3. **内存不足**
   - 减少数据量或使用更小的模型参数
   - 增加系统内存

4. **特征列缺失**
   - 检查数据文件是否包含所需的因子列
   - 确认列名拼写正确

## 扩展功能

可以根据需要修改：
- 模型参数（在`lightgbm_training.py`中）
- 数据划分比例（在`data_preprocessing.py`中）
- 特征选择（在`data_preprocessing.py`中）
- 评估指标（在`lightgbm_training.py`中）

## 联系信息

如有问题，请查看日志文件或联系开发团队。
