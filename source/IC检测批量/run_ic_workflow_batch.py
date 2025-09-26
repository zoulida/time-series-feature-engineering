# -*- coding: utf-8 -*-
"""
IC检测工作流批量版
支持分批处理、进度监控、错误恢复、性能优化

主要功能：
1. 批量获取股票数据（支持5000只股票）
2. 智能选择因子（支持统一因子库：Alpha158、tsfresh等）
3. 分批生成训练数据（避免内存溢出）
4. 高效IC分析（Pearson和Spearman相关系数）
5. 自动清理旧文件（避免磁盘空间浪费）
6. 实时进度监控和性能统计
"""

import sys
import os
import time
import logging
import psutil  # 系统资源监控
import gc  # 垃圾回收
from datetime import datetime
from tqdm import tqdm  # 进度条显示
import json
import pickle  # 检查点保存
import pandas as pd
import numpy as np

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '数据获取'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))

from stock_data_fetcher import StockDataFetcher  # 股票数据获取器
from unified_factor_library import UnifiedFactorLibrary  # 统一因子库

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


class WorkflowConfig:
    """工作流配置类 - 管理所有运行参数和性能设置"""
    def __init__(self):
        # 测试配置 - 用于快速验证功能
        self.test_mode = True  # 是否启用测试模式（True=测试模式，False=生产模式）
        self.test_stocks = 100  # 测试模式股票数量（快速验证用，建议100-500只）
        self.test_factors = 50  # 测试模式因子数量（快速验证用，建议20-100个）
        
        # 生产配置 - 用于大规模数据分析
        self.production_stocks = 5000  # 生产模式股票数量（最大支持5000只股票）
        self.production_factors = 500  # 生产模式因子数量（统一因子库全部因子）
        
        # 分批配置 - 优化内存使用和性能
        self.batch_size = 500  # 每批处理的股票数量（平衡内存使用和处理效率）
        self.max_memory_gb = 4  # 最大内存使用限制(GB)（超过会触发垃圾回收）
        
        # 并行配置 - 提升计算效率
        self.max_workers = 4  # 最大并行进程数（避免系统过载，建议2-8个）
        self.enable_parallel = True  # 是否启用并行处理（True=多进程，False=单进程）
        
        # 恢复配置 - 支持断点续传
        self.checkpoint_interval = 100  # 检查点间隔（每处理多少只股票保存一次）
        self.enable_recovery = True  # 是否启用错误恢复（支持从中断点继续运行）


class MemoryMonitor:
    """内存监控类 - 实时监控系统内存使用情况，防止内存溢出"""
    def __init__(self, max_memory_gb=4):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024  # 转换为字节
        self.initial_memory = psutil.virtual_memory().used  # 记录初始内存使用量
        
    def check_memory(self):
        """检查当前内存使用情况
        
        Returns:
            dict: 包含内存使用信息的字典
                - used_gb: 程序使用的内存(GB)
                - total_gb: 系统总内存使用(GB)
                - percentage: 内存使用百分比
                - is_safe: 是否安全（未超过限制）
        """
        current_memory = psutil.virtual_memory().used
        used_gb = (current_memory - self.initial_memory) / (1024**3)  # 计算程序使用的内存
        total_gb = current_memory / (1024**3)  # 系统总内存使用
        
        return {
            'used_gb': round(used_gb, 2),
            'total_gb': round(total_gb, 2),
            'percentage': round((current_memory / psutil.virtual_memory().total) * 100, 1),
            'is_safe': current_memory < self.max_memory_bytes
        }
    
    def force_gc(self):
        """强制垃圾回收 - 释放未使用的内存"""
        gc.collect()  # 执行Python垃圾回收
        logger.info("执行垃圾回收")


class ProgressTracker:
    """进度跟踪类 - 实时显示工作流执行进度、耗时统计和剩余时间估算"""
    def __init__(self, total_steps):
        self.total_steps = total_steps  # 总步骤数
        self.current_step = 0  # 当前步骤
        self.start_time = time.time()  # 开始时间
        self.checkpoints = []  # 检查点列表（用于错误恢复）
        
    def update(self, step_name, additional_info=""):
        """更新进度信息
        
        Args:
            step_name (str): 当前步骤名称
            additional_info (str): 附加信息（可选）
        """
        self.current_step += 1
        elapsed = time.time() - self.start_time  # 已用时间
        remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)  # 估算剩余时间
        
        progress_info = {
            'step': self.current_step,
            'total': self.total_steps,
            'step_name': step_name,
            'elapsed': round(elapsed, 2),
            'remaining': round(remaining, 2),
            'percentage': round((self.current_step / self.total_steps) * 100, 1),
            'additional_info': additional_info,
            'timestamp': datetime.now().isoformat()
        }
        
        self.checkpoints.append(progress_info)  # 保存检查点
        
        logger.info(f"进度: {self.current_step}/{self.total_steps} ({progress_info['percentage']}%) - {step_name}")
        if additional_info:
            logger.info(f"  详细信息: {additional_info}")
            
        return progress_info
    
    def save_checkpoint(self, data, checkpoint_file):
        """保存检查点"""
        checkpoint_data = {
            'progress': {
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'checkpoints': self.checkpoints
            },
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"检查点已保存: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_file):
        """加载检查点"""
        if not os.path.exists(checkpoint_file):
            return None, None
            
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.current_step = checkpoint_data['progress']['current_step']
        self.checkpoints = checkpoint_data['progress']['checkpoints']
        
        logger.info(f"从检查点恢复: 步骤 {self.current_step}/{self.total_steps}")
        return checkpoint_data['data'], checkpoint_data['progress']


class BatchICWorkflow:
    """批量IC检测工作流 - 核心工作流管理类
    
    主要功能：
    1. 批量获取股票数据（支持5000只股票）
    2. 智能选择因子（支持158个Alpha158因子）
    3. 分批生成训练数据（避免内存溢出）
    4. 高效IC分析（Pearson和Spearman相关系数）
    5. 自动清理旧文件（避免磁盘空间浪费）
    6. 实时进度监控和性能统计
    7. 错误恢复和断点续传
    """
    
    def __init__(self, config=None):
        self.config = config or WorkflowConfig()  # 工作流配置
        self.memory_monitor = MemoryMonitor(self.config.MAX_MEMORY_GB)  # 内存监控器
        self.checkpoint_file = 'workflow_checkpoint.pkl'  # 检查点文件
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')  # 数据目录
        os.makedirs(self.data_dir, exist_ok=True)  # 确保数据目录存在
        
        # 清空之前生成的数据文件（避免磁盘空间浪费）
        self.cleanup_old_files()
    
    def cleanup_old_files(self):
        """清理之前生成的数据文件 - 避免磁盘空间浪费和文件冲突
        
        清理的文件类型：
        - 训练数据文件 (training_data_batch_*.csv)
        - IC分析结果 (ic_results_batch_*.json, ic_report_batch_*.csv)
        - 中间文件 (selected_stock_codes.txt, selected_factors.*)
        - 检查点文件 (workflow_checkpoint.pkl)
        """
        try:
            # 要清理的文件模式
            file_patterns = [
                'training_data_batch_*.csv',  # 训练数据文件
                'ic_results_batch_*.json',   # IC分析结果JSON
                'ic_report_batch_*.csv',     # IC分析报告CSV
                'selected_stock_codes.txt',  # 选中的股票代码
                'selected_factors.csv',      # 选中的因子CSV
                'selected_factors.json',     # 选中的因子JSON
                'workflow_checkpoint.pkl'    # 检查点文件
            ]
            
            cleaned_files = []
            for pattern in file_patterns:
                import glob
                files = glob.glob(os.path.join(self.data_dir, pattern))
                for file_path in files:
                    try:
                        os.remove(file_path)
                        cleaned_files.append(os.path.basename(file_path))
                    except Exception as e:
                        logger.warning(f"无法删除文件 {file_path}: {e}")
            
            if cleaned_files:
                logger.info(f"已清理 {len(cleaned_files)} 个旧文件: {', '.join(cleaned_files[:5])}{'...' if len(cleaned_files) > 5 else ''}")
            else:
                logger.info("没有找到需要清理的旧文件")
                
        except Exception as e:
            logger.warning(f"清理旧文件时出错: {e}")
        
    def get_workflow_config(self):
        """获取工作流配置"""
        if self.config.TEST_MODE:
            return {
                'stocks': self.config.TEST_STOCKS,
                'factors': self.config.TEST_FACTORS,
                'mode': '测试模式'
            }
        else:
            return {
                'stocks': self.config.PRODUCTION_STOCKS,
                'factors': self.config.PRODUCTION_FACTORS,
                'mode': '生产模式'
            }
    
    def check_dependencies(self):
        """检查依赖项"""
        logger.info("检查依赖项...")
        
        # 检查内存
        memory_info = self.memory_monitor.check_memory()
        logger.info(f"内存使用: {memory_info['used_gb']}GB / {memory_info['total_gb']}GB ({memory_info['percentage']}%)")
        
        if not memory_info['is_safe']:
            logger.warning("内存使用接近限制，建议减少批次大小")
        
        # 检查磁盘空间
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        logger.info(f"可用磁盘空间: {free_gb:.1f}GB")
        
        if free_gb < 2:
            logger.warning("磁盘空间不足，可能影响文件保存")
        
        logger.info("依赖项检查完成")
        return True
    
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
            if not self.check_dependencies():
                return False
            
            # 检查是否有检查点可恢复
            if self.config.ENABLE_RECOVERY:
                checkpoint_data, progress = self.load_checkpoint()
                if checkpoint_data:
                    logger.info("从检查点恢复工作流")
                    return self.resume_from_checkpoint(checkpoint_data, progress)
            
            # 开始新的工作流
            return self.start_new_workflow(workflow_config)
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return False
    
    def load_checkpoint(self):
        """加载检查点"""
        if not os.path.exists(self.checkpoint_file):
            return None, None
            
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            return checkpoint_data, checkpoint_data.get('progress')
        except Exception as e:
            logger.warning(f"加载检查点失败: {str(e)}")
            return None, None
    
    def resume_from_checkpoint(self, checkpoint_data, progress):
        """从检查点恢复"""
        # 这里可以实现从检查点恢复的逻辑
        logger.info("检查点恢复功能待实现")
        return False
    
    def start_new_workflow(self, workflow_config):
        """开始新的工作流 - 执行完整的IC检测流程
        
        工作流包含4个主要步骤：
        1. 获取股票数据 - 从数据源加载指定数量的股票数据
        2. 选择因子 - 从Alpha158因子库中选择指定数量的因子
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
            
            # 步骤1: 获取股票数据
            step1_start = time.time()
            progress_tracker.update("获取股票数据", f"目标: {workflow_config['stocks']}只股票")
            stock_data = self.get_stock_data_batch(workflow_config['stocks'])
            step1_time = time.time() - step1_start
            logger.info(f"步骤1完成，耗时: {step1_time:.2f}秒")
            
            # 检查内存使用情况
            memory_info = self.memory_monitor.check_memory()
            if not memory_info['is_safe']:
                self.memory_monitor.force_gc()  # 内存不足时强制垃圾回收
            
            # 步骤2: 选择因子
            step2_start = time.time()
            progress_tracker.update("选择因子", f"目标: {workflow_config['factors']}个因子")
            selected_factors = self.select_factors_batch(workflow_config['factors'])
            step2_time = time.time() - step2_start
            logger.info(f"步骤2完成，耗时: {step2_time:.2f}秒")
            
            # 步骤3: 生成训练数据（分批处理）
            step3_start = time.time()
            progress_tracker.update("生成训练数据", "分批处理中...")
            training_data = self.generate_training_data_batch(stock_data, selected_factors, progress_tracker)
            step3_time = time.time() - step3_start
            logger.info(f"步骤3完成，耗时: {step3_time:.2f}秒")
            
            # 步骤4: IC分析
            step4_start = time.time()
            progress_tracker.update("IC分析", f"分析{len(selected_factors)}个因子")
            ic_results = self.perform_ic_analysis_batch(training_data, selected_factors)
            step4_time = time.time() - step4_start
            logger.info(f"步骤4完成，耗时: {step4_time:.2f}秒")
            
            # 保存最终结果
            self.save_final_results(ic_results, progress_tracker)
            
            total_time = time.time() - total_start_time
            
            # 输出详细的性能统计信息
            logger.info("=" * 60)
            logger.info("工作流执行完成")
            logger.info("=" * 60)
            logger.info("各阶段耗时统计:")
            logger.info(f"  步骤1 - 获取股票数据: {step1_time:.2f}秒 ({step1_time/total_time*100:.1f}%)")
            logger.info(f"  步骤2 - 选择因子: {step2_time:.2f}秒 ({step2_time/total_time*100:.1f}%)")
            logger.info(f"  步骤3 - 生成训练数据: {step3_time:.2f}秒 ({step3_time/total_time*100:.1f}%)")
            logger.info(f"  步骤4 - IC分析: {step4_time:.2f}秒 ({step4_time/total_time*100:.1f}%)")
            logger.info(f"  总耗时: {total_time:.2f}秒")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            # 保存检查点（支持错误恢复）
            if self.config.ENABLE_RECOVERY:
                progress_tracker.save_checkpoint({'error': str(e)}, self.checkpoint_file)
            return False
    
    def get_stock_data_batch(self, num_stocks):
        """批量获取股票数据"""
        try:
            # 创建数据获取器
            fetcher = StockDataFetcher()
            
            # 获取所有可用股票
            all_stocks = fetcher.get_available_stocks()
            logger.info(f"发现 {len(all_stocks)} 只股票")
            
            if len(all_stocks) < num_stocks:
                logger.warning(f"可用股票数量不足，需要{num_stocks}只，实际只有{len(all_stocks)}只")
                num_stocks = len(all_stocks)
            
            # 选择指定数量的股票
            selected_stocks = all_stocks[:num_stocks]
            logger.info(f"选择的股票: {len(selected_stocks)}只")
            
            # 获取股票数据
            stock_data = {}
            max_start_date = None
            min_end_date = None
            
            logger.info("正在分析股票数据时间范围...")
            for i, stock_code in enumerate(tqdm(selected_stocks, desc="加载股票数据")):
                try:
                    df = fetcher.load_single_stock_data(stock_code)
                    stock_data[stock_code] = df
                    
                    # 更新最大时间范围
                    stock_start = df['date'].min()
                    stock_end = df['date'].max()
                    
                    if max_start_date is None or stock_start > max_start_date:
                        max_start_date = stock_start
                    if min_end_date is None or stock_end < min_end_date:
                        min_end_date = stock_end
                        
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 数据失败: {str(e)}")
                    continue
            
            if not stock_data:
                raise ValueError("未能获取任何股票数据")
            
            # 只过滤掉0条记录的股票，其他都保留
            logger.info("过滤0条记录的股票...")
            valid_stock_data = {}
            empty_stocks = []
            
            for stock_code, df in stock_data.items():
                if len(df) > 0:  # 只要有数据就保留
                    valid_stock_data[stock_code] = df
                else:
                    empty_stocks.append(stock_code)
            
            logger.info(f"有效数据股票: {len(valid_stock_data)} 只")
            logger.info(f"空数据股票: {len(empty_stocks)} 只")
            if empty_stocks:
                logger.info("空数据股票:")
                for stock_code in empty_stocks[:5]:
                    logger.info(f"  {stock_code}")
            
            # 直接使用所有有效数据，不进行时间范围过滤
            if valid_stock_data:
                filtered_stock_data = valid_stock_data
                logger.info(f"保留所有有效数据，共 {len(filtered_stock_data)} 只股票")
                
                # 统计数据量分布
                data_counts = [len(df) for df in filtered_stock_data.values()]
                logger.info(f"数据量统计: 最少 {min(data_counts)} 条，最多 {max(data_counts)} 条，平均 {sum(data_counts)/len(data_counts):.1f} 条")
            else:
                raise ValueError("没有找到有效数据的股票")
            
            # 保存股票代码列表
            stock_codes_path = os.path.join(self.data_dir, 'selected_stock_codes.txt')
            with open(stock_codes_path, 'w', encoding='utf-8') as f:
                for stock_code in filtered_stock_data.keys():
                    f.write(f"{stock_code}\n")
            logger.info(f"股票代码列表已保存到: {stock_codes_path}")
            
            return filtered_stock_data
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {str(e)}")
            raise
    
    def select_factors_batch(self, num_factors):
        """批量选择因子"""
        try:
            # 创建统一因子库实例
            factor_lib = UnifiedFactorLibrary()
            
            # 获取所有因子
            all_factors = factor_lib.list_factors()
            logger.info(f"统一因子库总共有 {len(all_factors)} 个因子")
            
            if len(all_factors) < num_factors:
                logger.warning(f"可用因子数量不足，需要{num_factors}个，实际只有{len(all_factors)}个")
                num_factors = len(all_factors)
            
            # 按类别分组因子
            factor_categories = {}
            for factor_name in all_factors:
                factor_info = factor_lib.get_factor_info(factor_name)
                category = factor_info.get('category', '未知')
                if category not in factor_categories:
                    factor_categories[category] = []
                factor_categories[category].append(factor_name)
            
            logger.info("因子类别分布:")
            for category, factors in factor_categories.items():
                logger.info(f"  {category}: {len(factors)} 个因子")
            
            # 从每个类别中随机选择因子，确保多样性
            selected_factors = []
            factors_per_category = max(1, num_factors // len(factor_categories))
            remaining_factors = num_factors
            
            for category, factors in factor_categories.items():
                if remaining_factors <= 0:
                    break
                
                # 从当前类别选择因子
                select_count = min(factors_per_category, remaining_factors, len(factors))
                selected_from_category = np.random.choice(factors, select_count, replace=False).tolist()
                selected_factors.extend(selected_from_category)
                remaining_factors -= select_count
                
                logger.info(f"从 {category} 类别选择了 {select_count} 个因子")
            
            # 如果还有剩余因子，随机补充
            if remaining_factors > 0:
                remaining_all = [f for f in all_factors if f not in selected_factors]
                if remaining_all:
                    additional_count = min(remaining_factors, len(remaining_all))
                    additional_factors = np.random.choice(remaining_all, additional_count, replace=False).tolist()
                    selected_factors.extend(additional_factors)
                    logger.info(f"随机补充了 {additional_count} 个因子")
            
            logger.info(f"最终选择了 {len(selected_factors)} 个因子")
            
            # 保存选中的因子信息
            self.save_selected_factors(selected_factors, factor_lib)
            
            return selected_factors
            
        except Exception as e:
            logger.error(f"选择因子失败: {str(e)}")
            raise
    
    def save_selected_factors(self, selected_factors, factor_lib):
        """保存选中的因子信息"""
        try:
            # 创建因子详情
            factor_details = []
            for factor_name in selected_factors:
                factor_info = factor_lib.get_factor_info(factor_name)
                if factor_info:
                    factor_details.append({
                        'factor_name': factor_name,
                        'function_name': factor_info.get('function_name', ''),
                        'category': factor_info.get('category', ''),
                        'description': factor_info.get('description', ''),
                        'expression': factor_info.get('expression', ''),
                        'formula': factor_info.get('formula', '')
                    })
            
            # 保存为CSV
            factor_df = pd.DataFrame(factor_details)
            csv_path = os.path.join(self.data_dir, 'selected_factors.csv')
            factor_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"选中因子详情已保存到: {csv_path}")
            
            # 保存为JSON
            json_path = os.path.join(self.data_dir, 'selected_factors.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'selected_factors': selected_factors,
                    'factor_details': factor_details,
                    'selection_time': datetime.now().isoformat(),
                    'total_count': len(selected_factors)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"选中因子信息已保存到: {json_path}")
            
        except Exception as e:
            logger.error(f"保存因子信息失败: {str(e)}")
    
    def generate_training_data_batch(self, stock_data, selected_factors, progress_tracker):
        """批量生成训练数据（分批处理）"""
        try:
            logger.info("开始分批处理训练数据生成...")
            
            # 分批处理股票
            stock_codes = list(stock_data.keys())
            batches = [stock_codes[i:i+self.config.BATCH_SIZE] 
                      for i in range(0, len(stock_codes), self.config.BATCH_SIZE)]
            
            logger.info(f"分为 {len(batches)} 个批次处理，每批最多 {self.config.BATCH_SIZE} 只股票")
            
            all_training_data = []
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"处理批次 {batch_idx + 1}/{len(batches)}: {len(batch)}只股票")
                
                # 处理当前批次
                batch_data = self.process_stock_batch(batch, stock_data, selected_factors)
                all_training_data.append(batch_data)
                
                # 检查内存
                memory_info = self.memory_monitor.check_memory()
                logger.info(f"内存使用: {memory_info['used_gb']}GB ({memory_info['percentage']}%)")
                
                if not memory_info['is_safe']:
                    self.memory_monitor.force_gc()
                
                # 保存中间检查点
                if self.config.ENABLE_RECOVERY and (batch_idx + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    progress_tracker.save_checkpoint({
                        'processed_batches': batch_idx + 1,
                        'total_batches': len(batches),
                        'training_data': all_training_data
                    }, self.checkpoint_file)
            
            # 合并所有批次数据
            logger.info("合并所有批次数据...")
            combined_data = pd.concat(all_training_data, ignore_index=True)
            
            # 保存训练数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            training_data_path = os.path.join(self.data_dir, f'training_data_batch_{timestamp}.csv')
            combined_data.to_csv(training_data_path, index=False, encoding='utf-8-sig')
            logger.info(f"训练数据已保存到: {training_data_path}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"生成训练数据失败: {str(e)}")
            raise
    
    def process_stock_batch(self, stock_batch, stock_data, selected_factors):
        """处理单个股票批次"""
        try:
            # 创建统一因子库实例
            factor_lib = UnifiedFactorLibrary()
            
            all_training_data = []
            
            for stock_code in stock_batch:
                if stock_code not in stock_data:
                    continue
                
                df = stock_data[stock_code].copy()
                logger.info(f"处理股票 {stock_code}: {len(df)} 条记录")
                
                # 计算因子 - 使用批量添加避免碎片化
                factor_data = {}
                for factor_name in selected_factors:
                    factor_col = f'factor_{factor_name}'
                    try:
                        factor_values = self.calculate_factor(df, factor_name, factor_lib)
                        factor_data[factor_col] = factor_values
                    except Exception as e:
                        logger.warning(f"计算因子 {factor_name} 失败: {str(e)}")
                        factor_data[factor_col] = np.full(len(df), np.nan)
                
                # 计算收益率
                returns = self.calculate_returns(df)
                factor_data.update(returns)
                
                # 批量添加所有列
                if factor_data:
                    factor_df = pd.DataFrame(factor_data, index=df.index)
                    df = pd.concat([df, factor_df], axis=1)
                
                # 添加技术指标
                df = self.add_technical_indicators(df)
                
                # 添加股票代码
                df['stock_code'] = stock_code
                
                all_training_data.append(df)
            
            # 合并当前批次数据
            if all_training_data:
                batch_df = pd.concat(all_training_data, ignore_index=True)
                # 移除包含NaN的行
                initial_rows = len(batch_df)
                batch_df = batch_df.dropna()
                removed_rows = initial_rows - len(batch_df)
                if removed_rows > 0:
                    logger.info(f"批次移除了 {removed_rows} 行包含NaN的数据")
                
                return batch_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"处理股票批次失败: {str(e)}")
            raise
    
    def calculate_factor(self, data, factor_name, factor_lib):
        """计算单个因子"""
        try:
            factor_info = factor_lib.get_factor_info(factor_name)
            if not factor_info:
                return np.nan
            
            # 根据因子来源选择不同的计算方式
            source = factor_info.get('source', '')
            
            if source == 'Alpha158':
                # Alpha158因子使用原始计算逻辑
                return self._calculate_alpha158_factor(data, factor_name, factor_info)
            elif source == 'tsfresh':
                # tsfresh因子使用时间序列特征提取
                return self._calculate_tsfresh_factor(data, factor_name, factor_info)
            else:
                # 其他因子使用简化计算
                return self._calculate_generic_factor(data, factor_name, factor_info)
            
        except Exception as e:
            logger.warning(f"计算因子 {factor_name} 失败: {str(e)}")
            return np.nan
    
    def _calculate_alpha158_factor(self, data, factor_name, factor_info):
        """计算Alpha158因子"""
        # 这里应该实现Alpha158因子的具体计算逻辑
        # 目前使用简化版本
        return np.random.randn(len(data))
    
    def _calculate_tsfresh_factor(self, data, factor_name, factor_info):
        """计算tsfresh因子"""
        # 这里应该实现tsfresh因子的具体计算逻辑
        # 目前使用简化版本
        return np.random.randn(len(data))
    
    def _calculate_generic_factor(self, data, factor_name, factor_info):
        """计算通用因子"""
        # 根据因子表达式或函数名计算
        expression = factor_info.get('expression', '')
        if expression:
            # 这里应该解析表达式并计算
            # 目前使用简化版本
            return np.random.randn(len(data))
        else:
            # 使用随机值作为占位符
            return np.random.randn(len(data))
    
    def calculate_returns(self, data):
        """计算收益率"""
        returns = {}
        
        # 计算未来15天中收益率的最大值
        if len(data) > 15:
            # 计算未来1-15天的所有收益率
            future_returns = []
            for i in range(1, 16):  # 1天到15天
                future_return = data['close'].shift(-i) / data['close'] - 1
                future_returns.append(future_return)
            
            # 将未来收益率组合成DataFrame
            future_returns_df = pd.DataFrame(future_returns).T
            # 计算每行（每个时间点）的最大收益率
            max_future_return = future_returns_df.max(axis=1)
            returns['return_15d'] = max_future_return
            
            # 保留过去15天收益率作为参考
            past_return = data['close'] / data['close'].shift(15) - 1
            returns['past_return_15d'] = past_return
        
        return returns
    
    def add_technical_indicators(self, data):
        """添加技术指标 - 使用批量添加避免碎片化"""
        # 准备所有技术指标数据
        indicators = {}
        
        # 价格变化
        indicators['price_change'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        indicators['volume_change'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-12)
        indicators['high_low_ratio'] = data['high'] / data['low']
        indicators['close_open_ratio'] = data['close'] / data['open']
        
        # 移动平均
        if len(data) > 20:
            indicators['sma_5'] = data['close'].rolling(5).mean()
            indicators['sma_20'] = data['close'].rolling(20).mean()
            # 布林带
            bb_mean = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            indicators['bb_upper'] = bb_mean + 2 * bb_std
            indicators['bb_lower'] = bb_mean - 2 * bb_std
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / data['close']
        
        # 批量添加所有技术指标
        if indicators:
            indicators_df = pd.DataFrame(indicators, index=data.index)
            data = pd.concat([data, indicators_df], axis=1)
        
        return data
    
    def perform_ic_analysis_batch(self, training_data, selected_factors):
        """批量IC分析 - 计算所有因子的信息系数(Information Coefficient)
        
        IC分析是量化投资中的核心指标，用于衡量因子与未来收益率的相关性：
        - Pearson IC: 线性相关系数，衡量线性关系强度
        - Spearman IC: 秩相关系数，衡量单调关系强度
        
        Args:
            training_data (DataFrame): 包含因子值和收益率的训练数据
            selected_factors (list): 选中的因子列表
            
        Returns:
            dict: IC分析结果，包含每个因子的IC值和统计信息
        """
        try:
            logger.info("开始IC分析...")
            
            # 获取因子列和收益率列
            factor_cols = [col for col in training_data.columns if col.startswith('factor_')]
            return_cols = [col for col in training_data.columns if col.startswith('return_15d')]
            
            logger.info(f"找到 {len(factor_cols)} 个因子列")
            logger.info(f"找到 {len(return_cols)} 个收益率列")
            
            if not factor_cols or not return_cols:
                raise ValueError("未找到因子列或收益率列")
            
            # 进行IC分析
            ic_results = {}
            
            for factor_col in tqdm(factor_cols, desc="IC分析"):
                factor_name = factor_col.replace('factor_', '')
                
                for return_col in return_cols:
                    return_period = return_col.replace('return_', '').replace('d', '')
                    
                    # 计算IC - 只使用有效数据（非NaN值）
                    valid_data = training_data[[factor_col, return_col]].dropna()
                    if len(valid_data) > 10:  # 至少需要10个有效样本
                        # 皮尔逊相关系数 - 衡量线性关系
                        ic_value = valid_data[factor_col].corr(valid_data[return_col])
                        # 斯皮尔曼相关系数 - 衡量单调关系
                        spearman_ic = valid_data[factor_col].corr(valid_data[return_col], method='spearman')
                        
                        ic_results[f"{factor_name}_{return_period}d"] = {
                            'pearson_ic': ic_value,
                            'spearman_ic': spearman_ic,
                            'sample_size': len(valid_data)
                        }
                        
                        logger.info(f"  {factor_name} - {return_period}天收益率IC: {ic_value:.4f} (Spearman: {spearman_ic:.4f})")
            
            return ic_results
            
        except Exception as e:
            logger.error(f"IC分析失败: {str(e)}")
            raise
    
    def save_final_results(self, ic_results, progress_tracker):
        """保存最终结果"""
        try:
            # 保存IC结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(self.data_dir, f'ic_results_batch_{timestamp}.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(ic_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"IC结果已保存到: {results_file}")
            
            # 生成IC报告
            self.generate_ic_report(ic_results, timestamp)
            
            # 清理检查点文件
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logger.info("检查点文件已清理")
                
        except Exception as e:
            logger.error(f"保存最终结果失败: {str(e)}")
    
    def generate_ic_report(self, ic_results, timestamp):
        """生成IC报告"""
        try:
            # 创建IC报告DataFrame
            report_data = []
            for factor_return, metrics in ic_results.items():
                factor_name, return_period = factor_return.rsplit('_', 1)
                report_data.append({
                    'factor_name': factor_name,
                    'return_period': return_period,
                    'pearson_ic': metrics['pearson_ic'],
                    'spearman_ic': metrics['spearman_ic'],
                    'sample_size': metrics['sample_size']
                })
            
            report_df = pd.DataFrame(report_data)
            
            # 保存报告
            report_file = os.path.join(self.data_dir, f'ic_report_batch_{timestamp}.csv')
            report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
            logger.info(f"IC报告已保存到: {report_file}")
            
            # 显示摘要
            print("\n=== IC分析结果摘要 ===")
            print(f"总因子数: {len(report_df['factor_name'].unique())}")
            print(f"平均Pearson IC: {report_df['pearson_ic'].mean():.4f}")
            print(f"平均Spearman IC: {report_df['spearman_ic'].mean():.4f}")
            
            # 显示前10个最佳因子（包含描述信息）
            best_factors = report_df.nlargest(10, 'pearson_ic')
            print("\n=== 前10个最佳因子 ===")
            
            # 获取因子描述信息
            try:
                from unified_factor_library import UnifiedFactorLibrary
                factor_lib = UnifiedFactorLibrary()
                
                for i, (_, row) in enumerate(best_factors.iterrows(), 1):
                    factor_name = row['factor_name']
                    ic_value = row['pearson_ic']
                    
                    # 获取因子描述
                    factor_info = factor_lib.get_factor_info(factor_name)
                    description = factor_info.get('description', '暂无描述') if factor_info else '暂无描述'
                    category = factor_info.get('category', '未知类别') if factor_info else '未知类别'
                    source = factor_info.get('source', '未知来源') if factor_info else '未知来源'
                    
                    print(f"{i:2d}. {factor_name}: {ic_value:.4f}")
                    print(f"    来源: {source}")
                    print(f"    类别: {category}")
                    print(f"    描述: {description}")
                    print()
                    
            except Exception as e:
                # 如果获取描述失败，只显示基本信息
                logger.warning(f"获取因子描述失败: {e}")
                for i, (_, row) in enumerate(best_factors.iterrows(), 1):
                    print(f"{i:2d}. {row['factor_name']}: {row['pearson_ic']:.4f}")
                
        except Exception as e:
            logger.error(f"生成IC报告失败: {str(e)}")


def main():
    """主函数 - 程序入口点
    
    执行完整的IC检测工作流：
    1. 创建配置对象（包含所有运行参数）
    2. 初始化批量工作流实例
    3. 运行工作流（包含4个主要步骤）
    4. 输出执行结果
    
    配置说明：
    - 测试模式：100只股票，50个因子（快速验证）
    - 生产模式：5000只股票，158个因子（完整分析）
    """
    # 创建配置对象
    config = WorkflowConfig()
    
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
