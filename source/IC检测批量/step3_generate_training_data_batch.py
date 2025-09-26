# -*- coding: utf-8 -*-
"""
步骤3: 批量生成训练数据
负责分批生成训练数据，包含因子计算、收益率计算和技术指标添加
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from unified_factor_library import UnifiedFactorLibrary
from shared import MemoryMonitor, log_performance_stats

logger = logging.getLogger(__name__)


class TrainingDataBatchGenerator:
    """批量训练数据生成器"""
    
    def __init__(self, config, memory_monitor=None):
        """
        初始化批量训练数据生成器
        
        Args:
            config: 配置对象
            memory_monitor: 内存监控器实例
        """
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.factor_lib = UnifiedFactorLibrary()
        
    def generate_training_data_batch(self, stock_data, selected_factors, progress_tracker=None):
        """批量生成训练数据（分批处理）
        
        Args:
            stock_data (dict): 股票数据字典
            selected_factors (list): 选中的因子列表
            progress_tracker: 进度跟踪器实例
            
        Returns:
            pd.DataFrame: 合并后的训练数据
        """
        try:
            start_time = os.times().elapsed
            
            logger.info("开始分批处理训练数据生成...")
            
            # 分批处理股票
            stock_codes = list(stock_data.keys())
            batches = [stock_codes[i:i+self.config.batch_size] 
                      for i in range(0, len(stock_codes), self.config.batch_size)]
            
            logger.info(f"分为 {len(batches)} 个批次处理，每批最多 {self.config.batch_size} 只股票")
            
            all_training_data = []
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"处理批次 {batch_idx + 1}/{len(batches)}: {len(batch)}只股票")
                
                # 处理当前批次
                batch_data = self._process_stock_batch(batch, stock_data, selected_factors)
                all_training_data.append(batch_data)
                
                # 检查内存
                memory_info = self.memory_monitor.check_memory()
                logger.info(f"内存使用: {memory_info['used_gb']}GB ({memory_info['percentage']}%)")
                
                if not memory_info['is_safe']:
                    self.memory_monitor.force_gc()
                
                # 更新进度
                if progress_tracker:
                    progress_tracker.update(f"处理批次 {batch_idx + 1}/{len(batches)}", 
                                          f"已处理 {len(batch)} 只股票")
            
            # 合并所有批次数据
            logger.info("合并所有批次数据...")
            combined_data = pd.concat(all_training_data, ignore_index=True)
            
            # 保存训练数据
            from shared import get_timestamp
            timestamp = get_timestamp()
            training_data_path = os.path.join(self.data_dir, f'training_data_batch_{timestamp}.csv')
            combined_data.to_csv(training_data_path, index=False, encoding='utf-8-sig')
            logger.info(f"训练数据已保存到: {training_data_path}")
            
            end_time = os.times().elapsed
            log_performance_stats("生成训练数据", start_time, end_time, 
                                f"成功生成{len(combined_data)}条训练数据")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"生成训练数据失败: {str(e)}")
            raise
    
    def _process_stock_batch(self, stock_batch, stock_data, selected_factors):
        """处理单个股票批次
        
        Args:
            stock_batch (list): 当前批次的股票代码列表
            stock_data (dict): 股票数据字典
            selected_factors (list): 选中的因子列表
            
        Returns:
            pd.DataFrame: 当前批次的训练数据
        """
        try:
            all_training_data = []
            
            for stock_code in stock_batch:
                if stock_code not in stock_data:
                    continue
                
                df = stock_data[stock_code].copy()
                logger.info(f"处理股票 {stock_code}: {len(df)} 条记录")
                
                # 计算因子 - 使用批量添加避免碎片化
                factor_data = self._calculate_factors_for_stock(df, selected_factors)
                
                # 计算收益率
                returns = self._calculate_returns(df)
                factor_data.update(returns)
                
                # 批量添加所有列
                if factor_data:
                    factor_df = pd.DataFrame(factor_data, index=df.index)
                    df = pd.concat([df, factor_df], axis=1)
                
                # 添加技术指标
                df = self._add_technical_indicators(df)
                
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
                
                # 重新排列列顺序，将return_15d放到最后一列（IC检测目标）
                if 'return_15d' in batch_df.columns:
                    # 获取所有列名
                    all_columns = list(batch_df.columns)
                    # 移除return_15d
                    other_columns = [col for col in all_columns if col != 'return_15d']
                    # 重新排列：其他列 + return_15d
                    new_column_order = other_columns + ['return_15d']
                    batch_df = batch_df[new_column_order]
                    logger.info("已将return_15d列移动到最后一列（IC检测目标）")
                
                return batch_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"处理股票批次失败: {str(e)}")
            raise
    
    def _calculate_factors_for_stock(self, data, selected_factors):
        """为单只股票计算因子
        
        Args:
            data (pd.DataFrame): 股票数据
            selected_factors (list): 选中的因子列表
            
        Returns:
            dict: 因子数据字典
        """
        factor_data = {}
        for factor_name in selected_factors:
            factor_col = f'factor_{factor_name}'
            try:
                factor_values = self._calculate_factor(data, factor_name)
                factor_data[factor_col] = factor_values
            except Exception as e:
                logger.warning(f"计算因子 {factor_name} 失败: {str(e)}")
                factor_data[factor_col] = np.full(len(data), np.nan)
        
        return factor_data
    
    def _calculate_factor(self, data, factor_name):
        """计算单个因子
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            
        Returns:
            np.ndarray: 因子值数组
        """
        try:
            factor_info = self.factor_lib.get_factor_info(factor_name)
            if not factor_info:
                return np.full(len(data), np.nan)
            
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
            return np.full(len(data), np.nan)
    
    def _calculate_alpha158_factor(self, data, factor_name, factor_info):
        """计算Alpha158因子
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            factor_info (dict): 因子信息
            
        Returns:
            np.ndarray: 因子值数组
        """
        # 这里应该实现Alpha158因子的具体计算逻辑
        # 目前使用简化版本
        return np.random.randn(len(data))
    
    def _calculate_tsfresh_factor(self, data, factor_name, factor_info):
        """计算tsfresh因子
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            factor_info (dict): 因子信息
            
        Returns:
            np.ndarray: 因子值数组
        """
        # 这里应该实现tsfresh因子的具体计算逻辑
        # 目前使用简化版本
        return np.random.randn(len(data))
    
    def _calculate_generic_factor(self, data, factor_name, factor_info):
        """计算通用因子
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            factor_info (dict): 因子信息
            
        Returns:
            np.ndarray: 因子值数组
        """
        # 根据因子表达式或函数名计算
        expression = factor_info.get('expression', '')
        if expression:
            # 这里应该解析表达式并计算
            # 目前使用简化版本
            return np.random.randn(len(data))
        else:
            # 使用随机值作为占位符
            return np.random.randn(len(data))
    
    def _calculate_returns(self, data):
        """计算收益率
        
        Args:
            data (pd.DataFrame): 股票数据
            
        Returns:
            dict: 收益率数据字典
        """
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
    
    def _add_technical_indicators(self, data):
        """添加技术指标 - 使用批量添加避免碎片化
        
        Args:
            data (pd.DataFrame): 股票数据
            
        Returns:
            pd.DataFrame: 添加技术指标后的数据
        """
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
    
    def get_training_data_summary(self, training_data):
        """获取训练数据摘要信息
        
        Args:
            training_data (pd.DataFrame): 训练数据
            
        Returns:
            dict: 训练数据摘要信息
        """
        if training_data.empty:
            return {}
        
        # 统计因子列和收益率列
        factor_cols = [col for col in training_data.columns if col.startswith('factor_')]
        return_cols = [col for col in training_data.columns if col.startswith('return_')]
        
        return {
            'total_records': len(training_data),
            'total_columns': len(training_data.columns),
            'factor_columns': len(factor_cols),
            'return_columns': len(return_cols),
            'stock_codes': training_data['stock_code'].nunique() if 'stock_code' in training_data.columns else 0,
            'date_range': {
                'start': training_data['date'].min() if 'date' in training_data.columns else None,
                'end': training_data['date'].max() if 'date' in training_data.columns else None
            } if 'date' in training_data.columns else {}
        }
