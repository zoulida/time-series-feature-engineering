# -*- coding: utf-8 -*-
"""
IC检测工作流批量版 - 重构版
主工作流协调器，调用各个步骤模块完成IC检测流程

主要功能：
1. 协调各个步骤模块的执行
2. 管理配置和错误处理
3. 提供统一的执行入口
4. 支持进度监控和性能统计
"""

import sys
import os
import logging
import time
from datetime import datetime

# 禁用Numba GPU加速以避免CUDA版本警告
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CUDA_DISABLE'] = '1'
os.environ['NUMBA_DISABLE_CUDA'] = '1'

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

# 导入各个步骤模块
from step1_get_stock_data_batch import StockDataBatchFetcher
from step2_select_factors_batch import FactorBatchSelector
from step3_generate_training_data_batch import TrainingDataBatchGenerator
from step4_ic_analysis_batch import ICBatchAnalyzer

# 导入共享组件
from shared import MemoryMonitor, ProgressTracker, check_dependencies, cleanup_old_files

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ic_workflow_batch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchICWorkflow:
    """批量IC检测工作流 - 主协调器
    
    主要功能：
    1. 协调各个步骤模块的执行
    2. 管理配置和错误处理
    3. 提供统一的执行入口
    4. 支持进度监控和性能统计
    """
    
    def __init__(self, batch_config):
        """
        初始化批量IC检测工作流
        
        Args:
            batch_config: 批量配置对象（来自config_batch.py）
        """
        self.config = batch_config  # 直接使用batch_config作为主配置
        self.memory_monitor = MemoryMonitor(self.config.get_max_memory_gb())
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 清空之前生成的数据文件（避免磁盘空间浪费）
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """清理旧文件"""
        file_patterns = [
            'training_data_batch_*.csv',  # 训练数据文件
            'ic_results_batch_*.json',   # IC分析结果JSON
            'ic_report_batch_*.csv',     # IC分析报告CSV
            'selected_stock_codes.txt',  # 选中的股票代码
            'selected_factors.csv',      # 选中的因子CSV
            'selected_factors.json',     # 选中的因子JSON
            'workflow_checkpoint.pkl'    # 检查点文件
        ]
        cleanup_old_files(self.data_dir, file_patterns)
    
    def get_workflow_config(self):
        """获取工作流配置"""
        return self.config.get_workflow_config()
    
    def run_batch_workflow(self):
        """运行批量工作流"""
        try:
            logger.info("=" * 60)
            logger.info("IC检测工作流批量版开始执行")
            logger.info("=" * 60)
            
            # 获取配置
            workflow_config = self.get_workflow_config()
            logger.info(f"运行模式: {workflow_config['mode']}")
            logger.info(f"股票数量: {workflow_config['stocks']}")
            logger.info(f"因子数量: {workflow_config['factors']}")
            
            # 检查依赖
            if not check_dependencies(self.memory_monitor):
                return False
            
            # 开始新的工作流
            return self._start_new_workflow(workflow_config)
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return False
    
    def _start_new_workflow(self, workflow_config):
        """开始新的工作流 - 执行完整的IC检测流程
        
        工作流包含4个主要步骤：
        1. 获取股票数据 - 从数据源加载指定数量的股票数据
        2. 选择因子 - 从统一因子库中选择指定数量的因子
        3. 生成训练数据 - 计算因子值和收益率，分批处理避免内存溢出
        4. IC分析 - 计算每个因子的信息系数，评估因子有效性
        
        Args:
            workflow_config (dict): 工作流配置参数
            
        Returns:
            bool: 工作流是否执行成功
        """
        # 创建进度跟踪器
        total_steps = 4  # 4个主要步骤
        progress_tracker = ProgressTracker(total_steps)
        
        try:
            total_start_time = time.time()
            start_datetime = datetime.now()
            logger.info(f"工作流开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 步骤1: 获取股票数据
            step1_start = time.time()
            progress_tracker.update("获取股票数据", f"目标: {workflow_config['stocks']}只股票")
            
            stock_fetcher = StockDataBatchFetcher(self.config, self.memory_monitor)
            stock_data = stock_fetcher.get_stock_data_batch(workflow_config['stocks'])
            
            step1_time = time.time() - step1_start
            logger.info(f"步骤1完成，耗时: {step1_time:.2f}秒")
            
            # 检查内存使用情况
            memory_info = self.memory_monitor.check_memory()
            if not memory_info['is_safe']:
                self.memory_monitor.force_gc()
            
            # 步骤2: 选择因子
            step2_start = time.time()
            progress_tracker.update("选择因子", f"目标: {workflow_config['factors']}个因子")
            
            factor_selector = FactorBatchSelector(self.config, self.data_dir)
            selected_factors = factor_selector.select_factors_batch(workflow_config['factors'])
            
            step2_time = time.time() - step2_start
            logger.info(f"步骤2完成，耗时: {step2_time:.2f}秒")
            
            # 步骤3: 生成训练数据（分批处理）
            step3_start = time.time()
            progress_tracker.update("生成训练数据", "分批处理中...")
            
            training_generator = TrainingDataBatchGenerator(self.config, self.memory_monitor, self.config)
            training_data = training_generator.generate_training_data_batch(stock_data, selected_factors, progress_tracker)
            
            step3_time = time.time() - step3_start
            logger.info(f"步骤3完成，耗时: {step3_time:.2f}秒")
            
            # 步骤4: IC分析
            step4_start = time.time()
            progress_tracker.update("IC分析", f"分析{len(selected_factors)}个因子")
            
            ic_analyzer = ICBatchAnalyzer(self.config, self.data_dir)
            ic_results = ic_analyzer.perform_ic_analysis_batch(training_data, selected_factors)
            
            step4_time = time.time() - step4_start
            logger.info(f"步骤4完成，耗时: {step4_time:.2f}秒")
            
            total_time = time.time() - total_start_time
            
            # 输出详细的性能统计信息
            self._display_performance_summary(step1_time, step2_time, step3_time, step4_time, total_time, start_datetime)
            
            # 自动运行IC结果分析
            self._run_ic_analysis()
            
            return True
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return False
    
    def _display_performance_summary(self, step1_time, step2_time, step3_time, step4_time, total_time, start_datetime):
        """显示性能统计摘要"""
        # 计算完成时间
        end_datetime = datetime.now()
        
        logger.info("工作流执行完成")
        logger.info("=" * 60)
        logger.info(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"完成时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        logger.info("各阶段耗时统计:")
        logger.info(f"  步骤1 - 获取股票数据: {step1_time:.2f}秒 ({step1_time/total_time*100:.1f}%)")
        logger.info(f"  步骤2 - 选择因子: {step2_time:.2f}秒 ({step2_time/total_time*100:.1f}%)")
        logger.info(f"  步骤3 - 生成训练数据: {step3_time:.2f}秒 ({step3_time/total_time*100:.1f}%)")
        logger.info(f"  步骤4 - IC分析: {step4_time:.2f}秒 ({step4_time/total_time*100:.1f}%)")
        logger.info("=" * 60)
    
    def _run_ic_analysis(self):
        """运行IC结果分析"""
        try:
            logger.info("=" * 60)
            logger.info("开始IC结果分析")
            logger.info("=" * 60)
            
            # 构建analyze_ic_results.py的路径
            analyze_script_path = os.path.join(self.data_dir, 'analyze_ic_results.py')
            
            if not os.path.exists(analyze_script_path):
                logger.warning(f"IC分析脚本不存在: {analyze_script_path}")
                return
            
            # 运行IC分析脚本
            import subprocess
            result = subprocess.run(
                [sys.executable, analyze_script_path],
                cwd=self.data_dir,
                capture_output=True,
                text=True,
                encoding='gbk',  # 使用gbk编码处理中文输出
                errors='ignore'  # 忽略编码错误
            )
            
            if result.returncode == 0:
                logger.info("IC结果分析完成")
                print("\n" + "=" * 80)
                print("IC结果分析报告")
                print("=" * 80)
                print(result.stdout)
            else:
                logger.error(f"IC分析脚本执行失败: {result.stderr}")
                
        except Exception as e:
            logger.error(f"运行IC分析失败: {str(e)}")


def main():
    """主函数 - 程序入口点
    
    执行完整的IC检测工作流：
    1. 创建配置对象（包含所有运行参数）
    2. 初始化批量工作流实例
    3. 运行工作流（包含4个主要步骤）
    4. 输出执行结果
    
    配置说明：
    - 生产模式：20只股票，500个因子（完整分析）
    """
    # 导入配置
    from config_batch import get_config
    
    # 创建配置对象
    config = get_config()
    
    # 创建批量工作流实例
    workflow = BatchICWorkflow(config)
    
    # 运行工作流
    success = workflow.run_batch_workflow()
    
    # 输出执行结果
    if success:
        print("🎉 批量工作流执行成功！")
        print("📁 结果文件保存在: D:\\pythonProject\\时序特征工程\\source\\IC检测批量\\data")
    else:
        print("❌ 批量工作流执行失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
