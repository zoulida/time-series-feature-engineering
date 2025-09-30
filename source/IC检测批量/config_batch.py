# -*- coding: utf-8 -*-
"""
批量IC检测配置文件
"""

class BatchConfig:
    """批量检测配置类 - 统一管理所有配置"""
    
    def __init__(self):
        # ==================== 生产模式配置 ====================
        self.PRODUCTION_STOCKS = 20  # 生产模式股票数量（最大）
        self.PRODUCTION_FACTORS = 500  # 生产模式因子数量（Alpha158全部因子）
        
        # ==================== 因子库配置 ================================================================================================================================================================================================================================================
        # 因子库选择模式：'all', 'alpha158', 'tsfresh', 'custom', 'mixed'
        self.FACTOR_LIBRARY_MODE = 'tsfresh'  # 因子库模式
        
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
        
        # ==================== 并行配置 ====================
        self.max_workers = 4  # 最大并行进程数（避免系统过载，建议2-8个）
        self.enable_parallel = True  # 是否启用并行处理（True=多进程，False=单进程）
        
        # ==================== 恢复配置 ====================
        self.enable_recovery = True  # 是否启用错误恢复（支持从中断点继续运行）
        
        # ==================== 数据配置 ====================
        self.DATA_DIR = "data"  # 数据保存目录
        self.LOG_LEVEL = "INFO"  # 日志级别
        self.ENABLE_PROGRESS_BAR = True  # 是否显示进度条
        
        # ==================== 收益率计算配置 ====================
        # 收益率计算方法：'max_future_15d' 或 'next_day'
        self.RETURN_CALCULATION_METHOD = 'max_future_15d'  # 默认使用未来15天最大收益率
        self.FUTURE_DAYS = 15  # 未来天数（当方法为max_future_15d时使用）
        
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
    
    def get_workflow_config(self):
        """获取工作流配置"""
        return {
            'stocks': self.PRODUCTION_STOCKS,
            'factors': self.PRODUCTION_FACTORS,
            'mode': '生产模式'
        }
    
    def get_batch_size(self):
        """获取批次大小"""
        return self.BATCH_SIZE
    
    def get_max_memory_gb(self):
        """获取最大内存限制"""
        return self.MAX_MEMORY_GB


def get_config():
    """
    获取配置
    
    Returns:
        BatchConfig: 配置对象
    """
    return BatchConfig()


def main():
    """主函数 - 配置完成后直接执行工作流"""
    try:
        print("=" * 60)
        print("IC检测工作流批量版 - 配置执行")
        print("=" * 60)
        print("正在导入工作流模块...")
        
        # 导入工作流模块
        from run_batch_refactored import main as run_workflow
        
        print("配置加载完成，开始执行工作流...")
        print()
        
        # 执行工作流
        success = run_workflow()
        
        return success
        
    except ImportError as e:
        print(f"❌ 导入工作流模块失败: {str(e)}")
        print("请确保 run_batch_refactored.py 文件存在且可访问")
        return False
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        return False


if __name__ == "__main__":
    """直接执行配置文件"""
    import sys
    
    print("=" * 60)
    print("IC检测工作流批量版 - 配置执行")
    print("=" * 60)
    print("当前配置:")
    
    # 获取并显示当前配置
    config = get_config()
    print(f"  模式: 生产模式")
    print(f"  股票数量: {config.PRODUCTION_STOCKS}")
    print(f"  因子数量: {config.PRODUCTION_FACTORS}")
    print(f"  因子库模式: {config.FACTOR_LIBRARY_MODE}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  最大内存: {config.MAX_MEMORY_GB}GB")
    print(f"  收益率计算方法: {config.RETURN_CALCULATION_METHOD}")
    if config.RETURN_CALCULATION_METHOD == 'max_future_15d':
        print(f"  未来天数: {config.FUTURE_DAYS}")
    print()
    
    # 直接执行工作流
    try:
        print("开始执行工作流...")
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户取消执行")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        sys.exit(1)


# 配置说明
CONFIG_DESCRIPTION = r"""
配置说明：

=== 生产模式配置 ===
- PRODUCTION_STOCKS: 生产模式股票数量（默认: 20）
- PRODUCTION_FACTORS: 生产模式因子数量（默认: 500）

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

=== 收益率计算配置 ===
- RETURN_CALCULATION_METHOD: 收益率计算方法
  * 'max_future_15d': 计算未来15天中收益率的最大值（默认）
  * 'next_day': 后一个交易日的收益率
- FUTURE_DAYS: 未来天数（当方法为max_future_15d时使用，默认15天）

=== 输出配置 ===
- SAVE_INTERMEDIATE_RESULTS: 是否保存中间结果
- GENERATE_DETAILED_REPORT: 是否生成详细报告
- EXPORT_TO_EXCEL: 是否导出到Excel

=== 使用方法 ===
1. 直接修改 config_batch.py 中的配置参数
2. 在代码中调用 get_config() 获取配置
3. 使用 config.get_factor_sources() 获取因子源列表，配合unified_factor_library使用
4. 直接执行配置文件：python config_batch.py

=== 与unified_factor_library配合使用 ===
```python
from config_batch import get_config
from unified_factor_library import UnifiedFactorLibrary

# 获取配置
config = get_config()

# 根据配置创建因子库
factor_lib = UnifiedFactorLibrary(sources=config.get_factor_sources())
```

=== 收益率计算方法配置示例 ===
```python
from config_batch import get_config

# 获取配置
config = get_config()

# 方法1：使用未来15天最大收益率（默认）
config.RETURN_CALCULATION_METHOD = 'max_future_15d'
config.FUTURE_DAYS = 15

# 方法2：使用下一个交易日收益率
config.RETURN_CALCULATION_METHOD = 'next_day'

# 方法3：使用未来10天最大收益率
config.RETURN_CALCULATION_METHOD = 'max_future_15d'
config.FUTURE_DAYS = 10
```

=== 生产模式配置示例 ===
```python
from config_batch import get_config

# 获取配置
config = get_config()

# 调整股票数量
config.PRODUCTION_STOCKS = 100  # 处理100只股票

# 调整因子数量
config.PRODUCTION_FACTORS = 200  # 使用200个因子

# 调整批次大小
config.BATCH_SIZE = 50  # 每批处理50只股票
```

=== 直接执行配置文件 ===
```bash
# 方法1：直接执行配置文件
python config_batch.py

# 方法2：在命令行中执行
cd "D:\pythonProject\时序特征工程\source\IC检测批量"
python config_batch.py
```

执行后会显示当前配置并直接开始执行工作流：
```
============================================================
IC检测工作流批量版 - 配置执行
============================================================
当前配置:
  模式: 生产模式
  股票数量: 20
  因子数量: 500
  因子库模式: custom
  批次大小: 500
  最大内存: 4GB
  收益率计算方法: max_future_15d
  未来天数: 15

开始执行工作流...
```
"""
