# IC检测工作流批量版

## 概述

这是IC检测工作流的批量版本，专门用于处理大规模股票和因子分析。支持分批处理、进度监控、错误恢复等高级功能。

## 功能特点

- 🚀 **大规模处理**: 支持3500只股票、242个因子的批量分析
- 📊 **分批处理**: 自动分批处理，避免内存溢出
- 📈 **进度监控**: 实时显示处理进度和内存使用情况
- 🔄 **错误恢复**: 支持从中断点继续运行
- ⚡ **并行计算**: 支持多进程并行处理
- 💾 **内存优化**: 智能内存管理和垃圾回收
- 📋 **详细报告**: 生成完整的IC分析报告

## 文件结构

```
IC检测批量/
├── run_batch.py                    # 主启动脚本
├── run_ic_workflow_batch.py        # 批量工作流核心
├── config_batch.py                 # 配置文件
├── README.md                       # 说明文档
└── data/                          # 数据目录（自动创建）
    ├── selected_stock_codes.txt    # 选中的股票代码
    ├── selected_factors.csv        # 选中的因子详情
    ├── training_data_batch_*.csv   # 训练数据
    └── ic_results_batch_*.json     # IC分析结果
```

## 快速开始

### 1. 测试模式（推荐先运行）

```bash
# 使用默认测试配置（100只股票，50个因子）
python run_batch.py --mode test

# 自定义测试参数
python run_batch.py --mode test --stocks 200 --factors 100
```

### 2. 生产模式

```bash
# 使用默认生产配置（3500只股票，242个因子）
python run_batch.py --mode production

# 自定义生产参数
python run_batch.py --mode production --stocks 2000 --factors 150
```

### 3. 自定义模式

```bash
# 使用自定义配置
python run_batch.py --mode custom --stocks 500 --factors 80 --batch-size 100
```

## 配置说明

### 运行模式

| 模式 | 股票数量 | 因子数量 | 适用场景 |
|------|----------|----------|----------|
| test | 100 | 50 | 快速测试、开发调试 |
| production | 3500 | 242 | 完整分析、最终结果 |
| custom | 可自定义 | 可自定义 | 灵活配置 |

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stocks` | 股票数量 | 测试:100, 生产:3500 |
| `--factors` | 因子数量 | 测试:50, 生产:242 |
| `--batch-size` | 批次大小 | 500 |
| `--memory-limit` | 内存限制(GB) | 4 |
| `--workers` | 并行进程数 | 4 |
| `--no-recovery` | 禁用错误恢复 | False |
| `--no-parallel` | 禁用并行处理 | False |

## 使用示例

### 示例1: 快速测试

```bash
# 50只股票，20个因子，小批次处理
python run_batch.py --mode test --stocks 50 --factors 20 --batch-size 25
```

### 示例2: 中等规模分析

```bash
# 500只股票，100个因子
python run_batch.py --mode custom --stocks 500 --factors 100 --batch-size 100
```

### 示例3: 大规模生产分析

```bash
# 3500只股票，242个因子，高内存配置
python run_batch.py --mode production --memory-limit 8 --workers 6
```

### 示例4: 查看配置信息

```bash
# 查看当前配置
python run_batch.py --config-info

# 查看特定模式配置
python run_batch.py --mode production --config-info
```

## 性能估算

### 测试模式 (100只股票，50个因子)
- 运行时间: 约5-10分钟
- 内存使用: 约500MB
- 输出文件: 约50MB

### 生产模式 (3500只股票，242个因子)
- 运行时间: 约2-3小时
- 内存使用: 约2-4GB
- 输出文件: 约500MB-1GB

## 监控和恢复

### 进度监控
程序会实时显示：
- 当前处理进度
- 内存使用情况
- 预估剩余时间
- 处理速度统计

### 错误恢复
- 自动保存检查点
- 支持从中断点继续
- 详细的错误日志
- 内存不足自动处理

### 日志文件
- `ic_workflow_batch.log`: 详细运行日志
- `workflow_checkpoint.pkl`: 检查点文件（自动清理）

## 输出结果

### 数据文件
- `selected_stock_codes.txt`: 选中的股票代码列表
- `selected_factors.csv`: 选中的因子详细信息
- `training_data_batch_*.csv`: 完整的训练数据
- `ic_results_batch_*.json`: IC分析结果

### 报告文件
- `ic_report_batch_*.csv`: IC分析报告
- 包含每个因子的IC值、相关性等统计信息

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小: `--batch-size 100`
   - 减少并行进程: `--workers 2`
   - 增加内存限制: `--memory-limit 8`

2. **运行时间过长**
   - 使用测试模式先验证
   - 减少股票或因子数量
   - 启用并行处理

3. **程序中断**
   - 检查检查点文件是否存在
   - 查看日志文件了解中断原因
   - 重新运行会自动恢复

### 性能优化建议

1. **内存优化**
   - 使用SSD硬盘提高I/O速度
   - 确保有足够的内存空间
   - 定期清理临时文件

2. **并行优化**
   - 根据CPU核心数调整workers
   - 避免过度并行导致资源竞争
   - 监控系统负载

3. **批次优化**
   - 根据内存大小调整批次大小
   - 平衡内存使用和处理效率
   - 监控内存使用情况

## 注意事项

1. **首次运行建议使用测试模式**
2. **生产模式需要较长时间，建议在稳定环境下运行**
3. **定期检查磁盘空间，确保有足够空间保存结果**
4. **如遇到问题，查看日志文件获取详细信息**

## 技术支持

如有问题或建议，请查看：
- 日志文件: `ic_workflow_batch.log`
- 配置文件: `config_batch.py`
- 错误信息: 控制台输出
