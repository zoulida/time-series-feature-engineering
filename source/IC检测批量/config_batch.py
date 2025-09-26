# -*- coding: utf-8 -*-
"""
批量IC检测配置文件
"""

class BatchConfig:
    """批量检测配置类"""
    
    def __init__(self):
        # ==================== 运行模式配置 ====================
        self.TEST_MODE = True  # 是否启用测试模式
        self.PRODUCTION_MODE = False  # 是否启用生产模式
        
        # ==================== 测试模式配置 ====================
        self.TEST_STOCKS = 100  # 测试模式股票数量
        self.TEST_FACTORS = 50  # 测试模式因子数量
        
        # ==================== 生产模式配置 ====================
        self.PRODUCTION_STOCKS = 5000  # 生产模式股票数量（最大）
        self.PRODUCTION_FACTORS = 158  # 生产模式因子数量（Alpha158全部因子）
        
        # ==================== 因子库配置 ====================
        self.ENABLE_ALPHA158 = True  # 是否启用Alpha158因子库
        self.ENABLE_TSFRESH = True  # 是否启用tsfresh因子库
        self.ENABLE_CUSTOM = True  # 是否启用自定义因子库
        
        # 因子库优先级配置（按比例分配）
        self.ALPHA158_RATIO = 0.6  # Alpha158因子占比 (60%)
        self.TSFRESH_RATIO = 0.3   # tsfresh因子占比 (30%)
        self.CUSTOM_RATIO = 0.1    # 自定义因子占比 (10%)
        
        # 因子类别优先级配置
        self.PRIORITY_CATEGORIES = {
            'Alpha158_K线形态': 8,        # K线形态因子8个
            'Alpha158_成交量因子': 6,       # 成交量因子6个
            'Alpha158_滚动统计': 10,        # 滚动统计因子10个
            'Alpha158_价格因子': 2,         # 价格因子2个
            'tsfresh_统计特征': 8,          # tsfresh统计特征8个
            'tsfresh_频域特征': 4,          # tsfresh频域特征4个
            '复合因子': 3,                 # 自定义复合因子3个
        }
        
        # ==================== 分批处理配置 ====================
        self.BATCH_SIZE = 500  # 每批处理的股票数量
        self.MAX_MEMORY_GB = 4  # 最大内存使用限制(GB)
        self.CHECKPOINT_INTERVAL = 100  # 检查点间隔（每处理多少只股票保存一次）
        
        # ==================== 并行处理配置 ====================
        self.ENABLE_PARALLEL = True  # 是否启用并行处理
        self.MAX_WORKERS = 4  # 最大并行进程数
        
        # ==================== 错误恢复配置 ====================
        self.ENABLE_RECOVERY = True  # 是否启用错误恢复
        self.AUTO_SAVE_INTERVAL = 300  # 自动保存间隔（秒）
        
        # ==================== 数据配置 ====================
        self.DATA_DIR = "data"  # 数据保存目录
        self.LOG_LEVEL = "INFO"  # 日志级别
        self.ENABLE_PROGRESS_BAR = True  # 是否显示进度条
        
        # ==================== 性能优化配置 ====================
        self.USE_FLOAT32 = True  # 是否使用float32节省内存
        self.ENABLE_MEMORY_MONITORING = True  # 是否启用内存监控
        self.GC_INTERVAL = 50  # 垃圾回收间隔（每处理多少只股票）
        
        # ==================== 输出配置 ====================
        self.SAVE_INTERMEDIATE_RESULTS = True  # 是否保存中间结果
        self.GENERATE_DETAILED_REPORT = True  # 是否生成详细报告
        self.EXPORT_TO_EXCEL = False  # 是否导出到Excel





class DefaultConfig(BatchConfig):
    """默认配置 - 可通过修改配置文件调整参数"""
    
    def __init__(self):
        super().__init__()
        # 默认使用生产模式配置，可通过修改配置文件调整
        self.TEST_MODE = False
        self.PRODUCTION_MODE = True
        self.CUSTOM_MODE = False
        self.PRODUCTION_STOCKS = 5000  # 股票数量
        self.PRODUCTION_FACTORS = 158  # 因子数量
        self.BATCH_SIZE = 500  # 批次大小
        self.MAX_MEMORY_GB = 8  # 内存限制
        self.CHECKPOINT_INTERVAL = 100  # 检查点间隔
        self.MAX_WORKERS = 4  # 并行进程数
        
        # 因子库配置 - 默认启用所有因子库
        self.ENABLE_ALPHA158 = True
        self.ENABLE_TSFRESH = True
        self.ENABLE_CUSTOM = True
        
        # 因子库比例配置 - 默认比例
        self.ALPHA158_RATIO = 0.6  # 60% Alpha158因子
        self.TSFRESH_RATIO = 0.3   # 30% tsfresh因子
        self.CUSTOM_RATIO = 0.1    # 10% 自定义因子


class TestConfig(BatchConfig):
    """测试配置 - 少量因子快速测试"""
    
    def __init__(self):
        super().__init__()
        self.TEST_MODE = True
        self.PRODUCTION_MODE = False
        self.TEST_STOCKS = 50  # 测试50只股票
        self.TEST_FACTORS = 20  # 测试20个因子
        
        # 测试模式因子库配置
        self.ENABLE_ALPHA158 = True
        self.ENABLE_TSFRESH = True
        self.ENABLE_CUSTOM = True
        
        # 测试模式比例配置
        self.ALPHA158_RATIO = 0.5  # 50% Alpha158因子
        self.TSFRESH_RATIO = 0.3   # 30% tsfresh因子
        self.CUSTOM_RATIO = 0.2    # 20% 自定义因子
        
        # 简化的类别配置
        self.PRIORITY_CATEGORIES = {
            'Alpha158_K线形态': 3,        # K线形态因子3个
            'Alpha158_成交量因子': 2,       # 成交量因子2个
            'Alpha158_滚动统计': 5,        # 滚动统计因子5个
            'tsfresh_统计特征': 4,          # tsfresh统计特征4个
            '复合因子': 2,                 # 自定义复合因子2个
        }


class Alpha158OnlyConfig(BatchConfig):
    """只使用Alpha158因子的配置"""
    
    def __init__(self):
        super().__init__()
        self.TEST_MODE = False
        self.PRODUCTION_MODE = True
        self.PRODUCTION_STOCKS = 1000
        self.PRODUCTION_FACTORS = 50
        
        # 只启用Alpha158因子库
        self.ENABLE_ALPHA158 = True
        self.ENABLE_TSFRESH = False
        self.ENABLE_CUSTOM = False
        
        # 100% Alpha158因子
        self.ALPHA158_RATIO = 1.0
        self.TSFRESH_RATIO = 0.0
        self.CUSTOM_RATIO = 0.0


class CustomOnlyConfig(BatchConfig):
    """只使用自定义因子的配置"""
    
    def __init__(self):
        super().__init__()
        self.TEST_MODE = True
        self.PRODUCTION_MODE = False
        self.TEST_STOCKS = 100
        self.TEST_FACTORS = 5  # 只测试自定义因子
        
        # 只启用自定义因子库
        self.ENABLE_ALPHA158 = False
        self.ENABLE_TSFRESH = False
        self.ENABLE_CUSTOM = True
        
        # 100% 自定义因子
        self.ALPHA158_RATIO = 0.0
        self.TSFRESH_RATIO = 0.0
        self.CUSTOM_RATIO = 1.0


def get_config(config_type="default"):
    """
    获取配置
    
    Args:
        config_type: 配置类型，可选值:
            - "default": 默认配置（生产模式，所有因子库）
            - "test": 测试配置（少量因子快速测试）
            - "alpha158": 只使用Alpha158因子
            - "custom": 只使用自定义因子
            - "mixed": 混合配置（可自定义比例）
    
    Returns:
        BatchConfig: 配置对象
    """
    if config_type == "test":
        return TestConfig()
    elif config_type == "alpha158":
        return Alpha158OnlyConfig()
    elif config_type == "custom":
        return CustomOnlyConfig()
    elif config_type == "mixed":
        return DefaultConfig()  # 可以进一步自定义
    else:
        return DefaultConfig()


# 配置说明
CONFIG_DESCRIPTION = """
配置说明：

=== 配置类型 ===
1. "default": 默认配置（生产模式，所有因子库）
   - 股票数量: 5000
   - 因子数量: 158
   - 因子库: Alpha158(60%) + tsfresh(30%) + 自定义(10%)

2. "test": 测试配置（少量因子快速测试）
   - 股票数量: 50
   - 因子数量: 20
   - 因子库: Alpha158(50%) + tsfresh(30%) + 自定义(20%)

3. "alpha158": 只使用Alpha158因子
   - 股票数量: 1000
   - 因子数量: 50
   - 因子库: 100% Alpha158

4. "custom": 只使用自定义因子
   - 股票数量: 100
   - 因子数量: 5
   - 因子库: 100% 自定义因子

=== 因子库配置参数 ===
- ENABLE_ALPHA158: 是否启用Alpha158因子库 (True/False)
- ENABLE_TSFRESH: 是否启用tsfresh因子库 (True/False)
- ENABLE_CUSTOM: 是否启用自定义因子库 (True/False)
- ALPHA158_RATIO: Alpha158因子占比 (0.0-1.0)
- TSFRESH_RATIO: tsfresh因子占比 (0.0-1.0)
- CUSTOM_RATIO: 自定义因子占比 (0.0-1.0)

=== 因子类别优先级配置 ===
PRIORITY_CATEGORIES: 指定各类别因子的数量
- Alpha158_K线形态: K线形态因子数量
- Alpha158_成交量因子: 成交量因子数量
- Alpha158_滚动统计: 滚动统计因子数量
- Alpha158_价格因子: 价格因子数量
- tsfresh_统计特征: tsfresh统计特征数量
- tsfresh_频域特征: tsfresh频域特征数量
- 复合因子: 自定义复合因子数量

=== 使用方法 ===
1. 直接修改 config_batch.py 中的配置类
2. 在代码中调用 get_config("配置类型")
3. 运行程序时选择配置类型

=== 性能优化 ===
- BATCH_SIZE: 每批处理的股票数量，影响内存使用
- MAX_MEMORY_GB: 最大内存限制，超过会触发垃圾回收
- CHECKPOINT_INTERVAL: 检查点间隔，用于错误恢复
"""
