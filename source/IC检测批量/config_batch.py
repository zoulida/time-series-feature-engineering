# -*- coding: utf-8 -*-
"""
批量IC检测配置文件
"""

class BatchConfig:
    """批量检测配置类"""
    
    def __init__(self):
        # ==================== 生产模式配置 ====================
        self.PRODUCTION_STOCKS = 5000  # 生产模式股票数量（最大）
        self.PRODUCTION_FACTORS = 500  # 生产模式因子数量（Alpha158全部因子）
        
        # ==================== 因子库配置 ================================================================================================================================================================================================================================================
        # 因子库选择模式：'all', 'alpha158', 'tsfresh', 'custom', 'mixed'
        self.FACTOR_LIBRARY_MODE = 'all'  # 因子库模式
        
        # 各因子库开关（当模式为'mixed'时使用）
        self.ENABLE_ALPHA158 = True   # 是否启用Alpha158因子库
        self.ENABLE_TSFRESH = True    # 是否启用tsfresh因子库
        self.ENABLE_CUSTOM = True     # 是否启用自定义因子库
        
        # 因子库优先级配置（按比例分配，当模式为'mixed'时使用）
        self.ALPHA158_RATIO = 0.6  # Alpha158因子占比 (60%)
        self.TSFRESH_RATIO = 0.3   # tsfresh因子占比 (30%)
        self.CUSTOM_RATIO = 0.1    # 自定义因子占比 (10%)
        
        # ==================== 分批处理配置 ====================
        self.BATCH_SIZE = 500  # 每批处理的股票数量
        self.MAX_MEMORY_GB = 4  # 最大内存使用限制(GB)
        self.CHECKPOINT_INTERVAL = 100  # 检查点间隔（每处理多少只股票保存一次）
        
        # ==================== 数据配置 ====================
        self.DATA_DIR = "data"  # 数据保存目录
        self.LOG_LEVEL = "INFO"  # 日志级别
        self.ENABLE_PROGRESS_BAR = True  # 是否显示进度条
        
        # ==================== 输出配置 ====================
        self.SAVE_INTERMEDIATE_RESULTS = True  # 是否保存中间结果
        self.GENERATE_DETAILED_REPORT = True  # 是否生成详细报告
        self.EXPORT_TO_EXCEL = False  # 是否导出到Excel





    def get_factor_sources(self):
        """
        根据因子库模式获取因子源列表
        
        Returns:
            List[str]: 因子源列表，用于unified_factor_library
        """
        if self.FACTOR_LIBRARY_MODE == 'all':
            return ['alpha158', 'tsfresh', 'custom']
        elif self.FACTOR_LIBRARY_MODE == 'alpha158':
            return ['alpha158']
        elif self.FACTOR_LIBRARY_MODE == 'tsfresh':
            return ['tsfresh']
        elif self.FACTOR_LIBRARY_MODE == 'custom':
            return ['custom']
        elif self.FACTOR_LIBRARY_MODE == 'mixed':
            sources = []
            if self.ENABLE_ALPHA158:
                sources.append('alpha158')
            if self.ENABLE_TSFRESH:
                sources.append('tsfresh')
            if self.ENABLE_CUSTOM:
                sources.append('custom')
            return sources
        else:
            return ['alpha158', 'tsfresh', 'custom']  # 默认返回所有


def get_config():
    """
    获取配置
    
    Returns:
        BatchConfig: 配置对象
    """
    return BatchConfig()


# 配置说明
CONFIG_DESCRIPTION = """
配置说明：

=== 生产模式配置 ===
- PRODUCTION_STOCKS: 生产模式股票数量（默认: 5000）
- PRODUCTION_FACTORS: 生产模式因子数量（默认: 158）

=== 因子库配置 ===
- FACTOR_LIBRARY_MODE: 因子库选择模式
  * 'all': 使用所有因子库（Alpha158 + tsfresh + custom）
  * 'alpha158': 只使用Alpha158因子库
  * 'tsfresh': 只使用tsfresh因子库
  * 'custom': 只使用自定义因子库
  * 'mixed': 混合模式，通过开关控制

- 各因子库开关（当模式为'mixed'时使用）:
  * ENABLE_ALPHA158: 是否启用Alpha158因子库 (True/False)
  * ENABLE_TSFRESH: 是否启用tsfresh因子库 (True/False)
  * ENABLE_CUSTOM: 是否启用自定义因子库 (True/False)

- 因子库比例配置（当模式为'mixed'时使用）:
  * ALPHA158_RATIO: Alpha158因子占比 (0.0-1.0)
  * TSFRESH_RATIO: tsfresh因子占比 (0.0-1.0)
  * CUSTOM_RATIO: 自定义因子占比 (0.0-1.0)

=== 分批处理配置 ===
- BATCH_SIZE: 每批处理的股票数量，影响内存使用
- MAX_MEMORY_GB: 最大内存限制，超过会触发垃圾回收
- CHECKPOINT_INTERVAL: 检查点间隔，用于错误恢复

=== 数据配置 ===
- DATA_DIR: 数据保存目录
- LOG_LEVEL: 日志级别
- ENABLE_PROGRESS_BAR: 是否显示进度条

=== 输出配置 ===
- SAVE_INTERMEDIATE_RESULTS: 是否保存中间结果
- GENERATE_DETAILED_REPORT: 是否生成详细报告
- EXPORT_TO_EXCEL: 是否导出到Excel

=== 使用方法 ===
1. 直接修改 config_batch.py 中的配置参数
2. 在代码中调用 get_config() 获取配置
3. 使用 config.get_factor_sources() 获取因子源列表，配合unified_factor_library使用

=== 与unified_factor_library配合使用 ===
```python
from config_batch import get_config
from unified_factor_library import UnifiedFactorLibrary

# 获取配置
config = get_config()

# 根据配置创建因子库
factor_lib = UnifiedFactorLibrary(sources=config.get_factor_sources())
```
"""
