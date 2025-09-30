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
    
    def __init__(self, config, memory_monitor=None, batch_config=None):
        """
        初始化批量训练数据生成器
        
        Args:
            config: 工作流配置对象
            memory_monitor: 内存监控器实例
            batch_config: 批量配置对象（包含收益率计算配置）
        """
        self.config = config
        self.batch_config = batch_config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 根据配置的因子库模式创建因子库
        if hasattr(config, 'get_factor_sources'):
            factor_sources = config.get_factor_sources()
            self.factor_lib = UnifiedFactorLibrary(sources=factor_sources)
            logger.info(f"使用因子库模式: {config.FACTOR_LIBRARY_MODE}")
            logger.info(f"加载的因子源: {factor_sources}")
        else:
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
            batch_size = self.config.get_batch_size()
            batches = [stock_codes[i:i+batch_size] 
                      for i in range(0, len(stock_codes), batch_size)]
            
            logger.info(f"分为 {len(batches)} 个批次处理，每批最多 {batch_size} 只股票")
            
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
            
            # 添加进度条
            from tqdm import tqdm
            import time
            
            start_time = time.time()
            logger.info(f"开始处理 {len(stock_batch)} 只股票...")
            
            # 使用tqdm显示进度条
            for i, stock_code in enumerate(tqdm(stock_batch, desc="处理股票", unit="只")):
                if stock_code not in stock_data:
                    continue
                
                stock_start_time = time.time()
                df = stock_data[stock_code].copy()
                
                # 添加VWAP计算（为ALPHA_VWAP0因子提供数据）
                df = self._add_vwap_calculation(df)
                
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
                
                # 计算单只股票处理耗时
                stock_time = time.time() - stock_start_time
                logger.debug(f"处理股票 {stock_code}: {len(df)} 条记录，耗时 {stock_time:.2f}秒")
            
            total_time = time.time() - start_time
            logger.info(f"批次处理完成，总耗时: {total_time:.2f}秒，平均每只股票: {total_time/len(stock_batch):.2f}秒")
            
            # 合并当前批次数据
            if all_training_data:
                batch_df = pd.concat(all_training_data, ignore_index=True)
                # 只移除关键列包含NaN的行（保留因子列中的NaN，因为IC分析会处理）
                initial_rows = len(batch_df)
                key_columns = ['date', 'close', 'return_15d']  # 只检查关键列
                available_key_columns = [col for col in key_columns if col in batch_df.columns]
                if available_key_columns:
                    batch_df = batch_df.dropna(subset=available_key_columns)
                removed_rows = initial_rows - len(batch_df)
                if removed_rows > 0:
                    logger.info(f"批次移除了 {removed_rows} 行关键列包含NaN的数据")
                
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
        """计算Alpha158因子 - 使用直接计算函数
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            factor_info (dict): 因子信息
            
        Returns:
            np.ndarray: 因子值数组
        """
        try:
            function_name = factor_info.get('function_name', '')
            if not function_name:
                logger.warning(f"Alpha158因子 {factor_name} 没有函数名")
                return np.full(len(data), np.nan)
            
            # 获取Alpha158因子库实例
            if not hasattr(self, '_alpha158_lib'):
                import sys
                import os
                # 添加因子库路径
                factor_lib_path = os.path.join(os.path.dirname(__file__), '..', '因子库')
                if factor_lib_path not in sys.path:
                    sys.path.append(factor_lib_path)
                from alpha158_factors import Alpha158Factors
                self._alpha158_lib = Alpha158Factors()
            
            # 获取计算函数
            if not hasattr(self._alpha158_lib, function_name):
                logger.warning(f"Alpha158因子库中没有函数: {function_name}")
                return np.full(len(data), np.nan)
            
            calc_func = getattr(self._alpha158_lib, function_name)
            
            # 获取函数参数
            parameters = factor_info.get('parameters', [])
            
            # 准备函数参数
            args = []
            for param in parameters:
                if param in data.columns:
                    args.append(data[param])
                else:
                    logger.warning(f"Alpha158因子 {factor_name} 缺少参数: {param}")
                    return np.full(len(data), np.nan)
            
            # 调用计算函数
            result = calc_func(*args)
            return result.values if hasattr(result, 'values') else result
                
        except Exception as e:
            logger.error(f"计算Alpha158因子 {factor_name} 失败: {str(e)}")
            return np.full(len(data), np.nan)
    
    def _calculate_tsfresh_factor(self, data, factor_name, factor_info):
        """计算tsfresh因子 - 使用直接pandas方法
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            factor_info (dict): 因子信息
            
        Returns:
            np.ndarray: 因子值数组
        """
        try:
            # 获取因子函数名
            function_name = factor_info.get('function_name', '')
            if not function_name:
                logger.warning(f"tsfresh因子 {factor_name} 没有函数名")
                return np.full(len(data), np.nan)
            
            # 使用close价格序列作为主要输入
            price_series = data['close']
            
            # 简化的tsfresh因子计算 - 直接使用pandas方法
            # 根据因子名称计算对应的特征，使用滚动窗口计算整个时间序列
            if 'mean' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).mean().values
            elif 'std' in function_name.lower() or 'standard_deviation' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=2).std().values
            elif 'var' in function_name.lower() or 'variance' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=2).var().values
            elif 'max' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).max().values
            elif 'min' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).min().values
            elif 'median' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).median().values
            elif 'skew' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=3).skew().values
            elif 'kurt' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=3).kurt().values
            elif 'sum' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).sum().values
            elif 'count' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).count().values
            elif 'abs_energy' in function_name.lower():
                factor_values = (price_series ** 2).rolling(5, min_periods=1).sum().values
            elif 'absolute_sum_of_changes' in function_name.lower():
                factor_values = price_series.diff().abs().rolling(5, min_periods=1).sum().values
            elif 'count_above_mean' in function_name.lower():
                mean_val = price_series.rolling(5, min_periods=1).mean()
                factor_values = (price_series > mean_val).rolling(5, min_periods=1).sum().values
            elif 'count_below_mean' in function_name.lower():
                mean_val = price_series.rolling(5, min_periods=1).mean()
                factor_values = (price_series < mean_val).rolling(5, min_periods=1).sum().values
            elif 'mean_abs_change' in function_name.lower():
                factor_values = price_series.diff().abs().rolling(5, min_periods=1).mean().values
            elif 'mean_change' in function_name.lower():
                factor_values = price_series.diff().rolling(5, min_periods=1).mean().values
            elif 'number_peaks' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: self._count_peaks(x)).values
            elif 'autocorrelation' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=3).apply(lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan).values
            elif 'linear_trend' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan).values
            elif 'first_location_of_maximum' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: x.idxmax() if len(x) > 0 else np.nan).values
            elif 'first_location_of_minimum' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: x.idxmin() if len(x) > 0 else np.nan).values
            elif 'last_location_of_maximum' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: x[x == x.max()].index[-1] if len(x) > 0 else np.nan).values
            elif 'last_location_of_minimum' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: x[x == x.min()].index[-1] if len(x) > 0 else np.nan).values
            elif 'longest_strike_above_mean' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: self._longest_strike_above_mean(x)).values
            elif 'longest_strike_below_mean' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: self._longest_strike_below_mean(x)).values
            elif 'number_crossing_m' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: self._count_crossings(x)).values
            elif 'ratio_beyond_r_sigma' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: self._ratio_beyond_r_sigma(x)).values
            elif 'variance_larger_than_standard_deviation' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: 1 if x.var() > x.std() else 0).values
            elif 'fft_coefficient' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: np.real(np.fft.fft(x.values)[0]) if len(x) > 0 else np.nan).values
            elif 'fft_aggregated' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: np.mean(np.abs(np.fft.fft(x.values))) if len(x) > 0 else np.nan).values
            elif 'binned_entropy' in function_name.lower():
                factor_values = price_series.rolling(5, min_periods=1).apply(lambda x: self._binned_entropy(x)).values
            else:
                # 默认使用均值
                factor_values = price_series.rolling(5, min_periods=1).mean().values
            
            return np.array(factor_values)
                
        except Exception as e:
            logger.error(f"计算tsfresh因子 {factor_name} 失败: {str(e)}")
            return np.full(len(data), np.nan)
    
    def _count_peaks(self, series):
        """计算峰值数量"""
        if len(series) < 3:
            return 0
        peaks = 0
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                peaks += 1
        return peaks
    
    def _count_crossings(self, series):
        """计算交叉数量（与均值交叉）"""
        if len(series) < 2:
            return 0
        mean_val = series.mean()
        crossings = 0
        for i in range(1, len(series)):
            if (series.iloc[i-1] < mean_val and series.iloc[i] > mean_val) or \
               (series.iloc[i-1] > mean_val and series.iloc[i] < mean_val):
                crossings += 1
        return crossings
    
    def _longest_strike_above_mean(self, series):
        """计算均值上方最长连续"""
        if len(series) < 2:
            return 0
        above_mean = series > series.mean()
        if not above_mean.any():
            return 0
        strikes = (above_mean != above_mean.shift()).cumsum()
        longest_strike = strikes[above_mean].value_counts().max() if above_mean.any() else 0
        return longest_strike
    
    def _longest_strike_below_mean(self, series):
        """计算均值下方最长连续"""
        if len(series) < 2:
            return 0
        below_mean = series < series.mean()
        if not below_mean.any():
            return 0
        strikes = (below_mean != below_mean.shift()).cumsum()
        longest_strike = strikes[below_mean].value_counts().max() if below_mean.any() else 0
        return longest_strike
    
    def _ratio_beyond_r_sigma(self, series):
        """计算R-sigma外比例"""
        if len(series) < 2:
            return 0
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return 0
        beyond_sigma = abs(series - mean_val) > std_val
        return beyond_sigma.sum() / len(series)
    
    def _binned_entropy(self, series):
        """计算分箱熵"""
        if len(series) < 2:
            return 0
        try:
            hist, _ = np.histogram(series.values, bins=10)
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return entropy
        except:
            return 0
    
    def _evaluate_expression(self, expression, data):
        """解析并计算因子表达式
        
        Args:
            expression (str): 因子表达式
            data (pd.DataFrame): 股票数据
            
        Returns:
            np.ndarray: 计算结果
        """
        try:
            # 替换表达式中的变量
            expr = expression
            
            # 替换$close, $high, $low, $open, $volume等变量
            if '$close' in expr and 'close' in data.columns:
                expr = expr.replace('$close', 'data["close"]')
            if '$high' in expr and 'high' in data.columns:
                expr = expr.replace('$high', 'data["high"]')
            if '$low' in expr and 'low' in data.columns:
                expr = expr.replace('$low', 'data["low"]')
            if '$open' in expr and 'open' in data.columns:
                expr = expr.replace('$open', 'data["open"]')
            if '$volume' in expr and 'volume' in data.columns:
                expr = expr.replace('$volume', 'data["volume"]')
            
            # 替换函数名 - 使用正则表达式更精确地替换
            import re
            
            # 替换Ref函数
            expr = re.sub(r'Ref\(([^,]+),\s*(\d+)\)', r'\1.shift(\2)', expr)
            
            # 替换Mean函数
            expr = re.sub(r'Mean\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).mean()', expr)
            
            # 替换Std函数
            expr = re.sub(r'Std\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).std()', expr)
            
            # 先处理IdxMax和IdxMin函数（避免被Max和Min函数误匹配）
            # IdxMax函数: IdxMax(data["high"], 20) -> data["high"].rolling(20).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)
            expr = re.sub(r'IdxMax\(data\["([^"]+)"\],\s*(\d+)\)', r'data["\1"].rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
            
            # IdxMin函数: IdxMin(data["low"], 20) -> data["low"].rolling(20).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)
            expr = re.sub(r'IdxMin\(data\["([^"]+)"\],\s*(\d+)\)', r'data["\1"].rolling(\2).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)', expr)
            
            # 替换Max函数
            expr = re.sub(r'Max\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).max()', expr)
            
            # 替换Min函数
            expr = re.sub(r'Min\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).min()', expr)
            
            # 替换Sum函数
            expr = re.sub(r'Sum\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).sum()', expr)
            
            # 替换其他函数
            expr = expr.replace('Greater(', 'np.maximum(')
            expr = expr.replace('Less(', 'np.minimum(')
            expr = expr.replace('Abs(', 'np.abs(')
            
            # 添加更多Alpha158函数支持
            
            # Quantile函数: Quantile($close, 20, 0.8) -> data["close"].rolling(20).quantile(0.8)
            expr = re.sub(r'Quantile\(([^,]+),\s*(\d+),\s*([0-9.]+)\)', r'\1.rolling(\2).quantile(\3)', expr)
            
            # Rank函数: Rank($close, 20) -> data["close"].rolling(20).rank(pct=True)
            expr = re.sub(r'Rank\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).rank(pct=True)', expr)
            
            # Corr函数: Corr($close, $volume, 20) -> data["close"].rolling(20).corr(data["volume"])
            expr = re.sub(r'Corr\(([^,]+),\s*([^,]+),\s*(\d+)\)', r'\1.rolling(\3).corr(\2)', expr)
            
            # Rsquare函数: Rsquare($close, 20) -> data["close"].rolling(20).apply(lambda x: x.corr(pd.Series(range(len(x))))**2 if len(x) > 1 else np.nan)
            expr = re.sub(r'Rsquare\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).apply(lambda x: x.corr(pd.Series(range(len(x))))**2 if len(x) > 1 else np.nan)', expr)
            
            # Resi函数: Resi($close, 20) -> data["close"].rolling(20).apply(lambda x: x.iloc[-1] - x.mean() if len(x) > 0 else np.nan)
            expr = re.sub(r'Resi\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).apply(lambda x: x.iloc[-1] - x.mean() if len(x) > 0 else np.nan)', expr)
            
            # Slope函数: Slope($close, 20) -> data["close"].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan)
            expr = re.sub(r'Slope\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan)', expr)
            
            # Log函数: Log($volume+1) -> np.log(data["volume"]+1)
            expr = re.sub(r'Log\(([^)]+)\)', r'np.log(\1)', expr)
            
            # 处理复杂的条件表达式
            # Mean($close>Ref($close, 1), 10) -> (data["close"] > data["close"].shift(1)).rolling(10).mean()
            expr = re.sub(r'Mean\(([^>]+)>([^,]+),\s*(\d+)\)', r'(\1 > \2).rolling(\3).mean()', expr)
            expr = re.sub(r'Mean\(([^<]+)<([^,]+),\s*(\d+)\)', r'(\1 < \2).rolling(\3).mean()', expr)
            
            # 处理更复杂的条件表达式，需要先处理括号内的内容
            # 例如: Mean($close>Ref($close, 1), 5)-Mean($close<Ref($close, 1), 5)
            # 先处理第一个Mean
            expr = re.sub(r'Mean\(([^)]*\$[^)]*>[^)]*),\s*(\d+)\)', 
                         lambda m: f'({m.group(1)}).rolling({m.group(2)}).mean()', expr)
            # 再处理第二个Mean
            expr = re.sub(r'Mean\(([^)]*\$[^)]*<[^)]*),\s*(\d+)\)', 
                         lambda m: f'({m.group(1)}).rolling({m.group(2)}).mean()', expr)
            
            # 处理vwap变量（如果存在）
            if '$vwap' in expr:
                # 简单的vwap计算：(high + low + close) / 3
                expr = expr.replace('$vwap', '((data["high"] + data["low"] + data["close"]) / 3)')
            
            
            # 调试信息
            logger.debug(f"原始表达式: {expression}")
            logger.debug(f"解析后表达式: {expr}")
            
            # 安全地执行表达式
            result = eval(expr)
            return result.values if hasattr(result, 'values') else result
            
        except Exception as e:
            logger.warning(f"表达式解析失败: {expression}, 错误: {e}")
            return np.full(len(data), np.nan)
    
    def _calculate_generic_factor(self, data, factor_name, factor_info):
        """计算通用因子
        
        Args:
            data (pd.DataFrame): 股票数据
            factor_name (str): 因子名称
            factor_info (dict): 因子信息
            
        Returns:
            np.ndarray: 因子值数组
        """
        try:
            # 获取因子函数名
            function_name = factor_info.get('function_name', '')
            if function_name:
                # 尝试从myfactors模块导入并调用函数
                try:
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))
                    from custom.myfactors import MyFactors
                    my_factors = MyFactors()
                    
                    # 获取计算函数
                    if hasattr(my_factors, function_name):
                        factor_func = getattr(my_factors, function_name)
                        
                        # 准备函数参数
                        kwargs = {}
                        if 'close' in data.columns:
                            kwargs['close'] = data['close']
                        if 'high' in data.columns:
                            kwargs['high'] = data['high']
                        if 'volume' in data.columns:
                            kwargs['volume'] = data['volume']
                        if 'stock_code' in data.columns:
                            kwargs['stock_code'] = data['stock_code'].iloc[0] if len(data) > 0 else '000001.SZ'
                        
                        # 调用因子计算函数
                        result = factor_func(**kwargs)
                        return result.values if hasattr(result, 'values') else result
                    else:
                        logger.warning(f"因子函数 {function_name} 不存在于MyFactors类中")
                        return np.full(len(data), np.nan)
                        
                except ImportError as e:
                    logger.warning(f"无法导入MyFactors模块: {e}")
                    return np.full(len(data), np.nan)
                except Exception as e:
                    logger.warning(f"计算因子 {factor_name} 时出错: {e}")
                    return np.full(len(data), np.nan)
            else:
                # 如果没有函数名，尝试解析表达式
                expression = factor_info.get('expression', '')
                if expression == 'close' and 'close' in data.columns:
                    return data['close'].values
                else:
                    logger.warning(f"无法解析因子表达式: {expression}")
                    return np.full(len(data), np.nan)
                    
        except Exception as e:
            logger.warning(f"计算因子 {factor_name} 失败: {str(e)}")
            return np.full(len(data), np.nan)
    
    def _calculate_returns(self, data):
        """计算收益率
        
        Args:
            data (pd.DataFrame): 股票数据
            
        Returns:
            dict: 收益率数据字典
        """
        returns = {}
        
        # 根据配置选择收益率计算方法
        if self.batch_config:
            method = self.batch_config.RETURN_CALCULATION_METHOD
            future_days = self.batch_config.FUTURE_DAYS
        else:
            # 如果没有batch_config，使用默认配置
            method = 'max_future_15d'
            future_days = 15
        
        if method == 'max_future_15d':
            # 方法1：计算未来15天中收益率的最大值
            if len(data) > future_days:
                # 计算未来1到future_days天的所有收益率
                future_returns = []
                for i in range(1, future_days + 1):  # 1天到future_days天
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
                
        elif method == 'next_day':
            # 方法2：后一个交易日的收益率
            if len(data) > 1:
                # 计算下一个交易日的收益率
                next_day_return = data['close'].shift(-1) / data['close'] - 1
                returns['return_15d'] = next_day_return  # 保持列名一致，便于后续处理
                
                # 保留过去1天收益率作为参考
                past_return = data['close'] / data['close'].shift(1) - 1
                returns['past_return_15d'] = past_return
                
        else:
            # 默认使用max_future_15d方法
            logger.warning(f"未知的收益率计算方法: {method}，使用默认方法")
            if len(data) > 15:
                future_returns = []
                for i in range(1, 16):
                    future_return = data['close'].shift(-i) / data['close'] - 1
                    future_returns.append(future_return)
                
                future_returns_df = pd.DataFrame(future_returns).T
                max_future_return = future_returns_df.max(axis=1)
                returns['return_15d'] = max_future_return
                
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
    
    def _add_vwap_calculation(self, data):
        """添加VWAP（成交量加权平均价格）计算
        
        Args:
            data (pd.DataFrame): 股票数据
            
        Returns:
            pd.DataFrame: 添加了vwap列的数据
        """
        try:
            # 检查必需的列是否存在
            required_cols = ['high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"缺少VWAP计算所需的列: {missing_cols}")
                return data
            
            # 计算VWAP = (high + low + close) / 3 * volume 的加权平均
            # 使用典型价格 (high + low + close) / 3
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            
            # 计算VWAP：典型价格按成交量加权
            data['vwap'] = (typical_price * data['volume']).rolling(window=len(data), min_periods=1).sum() / data['volume'].rolling(window=len(data), min_periods=1).sum()
            
            logger.debug(f"成功计算VWAP，数据长度: {len(data)}")
            return data
            
        except Exception as e:
            logger.warning(f"计算VWAP失败: {str(e)}")
            return data
