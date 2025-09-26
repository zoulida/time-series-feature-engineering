# 时序特征工程与IC评分分析项目

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-zoulida%2Ftime--series--feature--engineering-brightgreen.svg)](https://github.com/zoulida/time-series-feature-engineering)

一个完整的时序特征工程与IC评分分析项目，专为量化投资设计，包含数据预处理、特征提取、因子库、IC分析等完整解决方案。

## 🌟 项目特点

- **完整的特征工程流程**: 从原始数据到特征提取的完整pipeline
- **多种因子库支持**: Alpha158、tsfresh、自定义因子等
- **批量处理能力**: 支持大规模股票数据的批量处理
- **IC评分分析**: 提供Pearson、Spearman、Kendall等多种相关系数分析
- **模块化设计**: 清晰的代码结构，易于维护和扩展
- **内存优化**: 针对大数据集的内存优化处理

## 📁 项目结构

```
时序特征工程/
├── source/                           # 源代码目录
│   ├── IC检测批量/                   # IC检测批量处理模块
│   │   ├── shared/                   # 共享组件
│   │   ├── step1_get_stock_data_batch.py
│   │   ├── step2_select_factors_batch.py
│   │   ├── step3_generate_training_data_batch.py
│   │   ├── step4_ic_analysis_batch.py
│   │   └── run_batch_refactored.py   # 主运行脚本
│   ├── 因子库/                       # 因子库模块
│   │   ├── unified_factor_library.py # 统一因子库
│   │   ├── alpha158_factors.py       # Alpha158因子
│   │   ├── tsfresh_factor_library.py # tsfresh因子
│   │   └── custom/                   # 自定义因子
│   ├── 数据获取/                     # 数据获取模块
│   ├── 程序文件/                     # 核心程序文件
│   └── 配置文件/                     # 配置文件
├── output/                           # 输出结果目录
│   ├── features/                     # 特征文件
│   ├── processed_data/               # 处理后数据
│   ├── raw_data/                     # 原始数据
│   └── reports/                      # 分析报告
├── venv/                             # Python虚拟环境
└── README.md                         # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/zoulida/time-series-feature-engineering.git
cd time-series-feature-engineering

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r source/配置文件/requirements_simple.txt
```

### 2. 运行IC检测工作流

```bash
# 进入IC检测目录
cd source/IC检测批量

# 运行重构版工作流
python run_batch_refactored.py
```

### 3. 使用因子库

```python
from source.因子库.unified_factor_library import create_unified_factor_library

# 创建统一因子库
factor_lib = create_unified_factor_library()

# 获取所有因子
all_factors = factor_lib.get_all_factors()

# 按类别获取因子
kbar_factors = factor_lib.get_factors_by_category('K线形态')
```

## 📊 核心功能

### 1. 数据预处理
- 支持多股票CSV数据批量处理
- 自动过滤停牌记录
- 生成tsfresh兼容的长表格式
- 计算未来收益率作为预测目标

### 2. 特征提取
- **Alpha158因子**: 158个量化投资专用因子
- **tsfresh特征**: 74个通用时间序列特征
- **自定义因子**: 支持用户自定义因子添加
- **统一接口**: 统一的因子库管理接口

### 3. IC评分分析
- Pearson相关系数
- Spearman相关系数
- Kendall相关系数
- 特征重要性排名
- 可视化分析报告

### 4. 批量处理
- 内存优化的大规模数据处理
- 实时进度监控
- 错误恢复机制
- 模块化设计

## 🔧 技术栈

- **Python 3.12+**
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **matplotlib/seaborn**: 数据可视化
- **scipy**: 统计分析
- **tsfresh**: 时间序列特征工程
- **scikit-learn**: 机器学习工具

## 📈 使用示例

### 基本使用

```python
# 导入必要模块
from source.IC检测批量.run_batch_refactored import run_ic_workflow

# 运行IC工作流
results = run_ic_workflow(
    stock_count=100,      # 股票数量
    factor_count=50,      # 因子数量
    window_size=20,       # 窗口大小
    test_mode=True        # 测试模式
)
```

### 因子库使用

```python
from source.因子库.unified_factor_library import create_unified_factor_library

# 创建因子库
factor_lib = create_unified_factor_library()

# 获取统计信息
stats = factor_lib.get_statistics()
print(f"总因子数量: {stats['总因子数量']}")

# 搜索因子
mean_factors = factor_lib.search_factors("mean")

# 按来源过滤
alpha_factors = factor_lib.get_factors_by_source('Alpha158')
```

## 📊 输出结果

运行完成后，您将获得：

- **特征矩阵**: CSV格式的特征数据
- **IC评分**: 各因子的相关系数排名
- **分析报告**: 详细的IC分析报告
- **可视化图表**: 特征重要性和IC分布图

## 🛠️ 配置说明

### 测试模式
- 股票数量: 100只
- 因子数量: 50个
- 适合快速验证功能

### 生产模式
- 股票数量: 5000只
- 因子数量: 500个
- 适合大规模数据分析

## 📝 注意事项

1. 确保Python版本为3.12+
2. 大数据集处理需要足够内存
3. 建议先在小数据集上测试
4. 确保数据源路径正确

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

### 贡献指南
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [https://github.com/zoulida/time-series-feature-engineering](https://github.com/zoulida/time-series-feature-engineering)
- 问题反馈: [Issues](https://github.com/zoulida/time-series-feature-engineering/issues)

## 🙏 致谢

- 感谢qlib项目提供的Alpha158因子库
- 感谢tsfresh项目提供的时间序列特征工程工具
- 感谢所有贡献者和用户的支持

---

⭐ 如果这个项目对您有帮助，请给它一个星标！ 