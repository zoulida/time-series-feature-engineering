# GPLearn时序特征提取和IC测试

## 概述

本项目使用gplearn（Genetic Programming for Symbolic Regression）对时序数据进行特征提取，并计算特征与目标变量的信息系数（IC值）。

## 功能特点

- **时序特征提取**：使用滚动窗口计算统计量和技术指标
- **符号回归特征**：通过遗传算法自动发现特征组合
- **IC值计算**：计算特征与目标变量的信息系数
- **结果可视化**：生成特征重要性排名和IC值分析图表
- **结果保存**：自动保存所有分析结果

## 文件说明

- `gplearn_feature_extraction.py` - 完整的特征提取程序
- `run_gplearn_analysis.py` - 简化的运行脚本
- `install_gplearn.py` - 安装gplearn和相关依赖
- `test_gplearn.py` - 测试gplearn是否正常工作
- `run_analysis.bat` - Windows批处理文件
- `run_analysis.ps1` - PowerShell脚本

## 快速开始

### 方法1：使用批处理文件（推荐Windows用户）
```bash
# 双击运行
run_analysis.bat
```

### 方法2：使用PowerShell脚本
```bash
# 在PowerShell中运行
.\run_analysis.ps1
```

### 方法3：手动运行
```bash
# 1. 安装依赖
python install_gplearn.py

# 2. 测试gplearn
python test_gplearn.py

# 3. 运行特征提取
python run_gplearn_analysis.py
```

## 程序流程

1. **数据加载** - 加载目标数据和时序数据
2. **特征准备** - 计算滚动统计量和技术指标
3. **GPLearn特征提取** - 使用符号回归创建新特征
4. **IC值计算** - 计算特征与目标变量的信息系数
5. **结果分析** - 特征排名和统计分析
6. **可视化** - 生成分析图表
7. **保存结果** - 保存所有分析结果

## 输出文件

- `source/结果文件/gplearn_features.csv` - 提取的特征矩阵
- `source/结果文件/gplearn_ic_results.csv` - 详细的IC计算结果
- `source/结果文件/gplearn_ic_summary.csv` - 特征IC值摘要
- `source/可视化文件/gplearn_ic_analysis.png` - IC分析图表

## 注意事项

1. 确保Python环境已正确安装
2. gplearn计算密集，建议在性能较好的机器上运行
3. 根据数据规模和计算资源调整参数
4. 结果会自动保存到相应目录

## 故障排除

如果遇到问题，请查看：
1. Python版本是否兼容（建议3.7+）
2. 依赖包是否正确安装
3. 数据文件路径是否正确
4. 计算资源是否充足

---

**版本**：1.0  
**更新日期**：2024年  
**兼容性**：Python 3.7+
