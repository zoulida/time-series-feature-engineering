# -*- coding: utf-8 -*-
"""
步骤1: 批量获取股票数据
负责从数据源加载指定数量的股票数据，支持批量处理和内存优化
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '数据获取'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from stock_data_fetcher import StockDataFetcher
from shared import MemoryMonitor, log_performance_stats

logger = logging.getLogger(__name__)


class StockDataBatchFetcher:
    """批量股票数据获取器"""
    
    def __init__(self, config, memory_monitor=None):
        """
        初始化批量股票数据获取器
        
        Args:
            config: 配置对象
            memory_monitor: 内存监控器实例
        """
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_stock_data_batch(self, num_stocks):
        """批量获取股票数据
        
        Args:
            num_stocks (int): 需要获取的股票数量
            
        Returns:
            dict: 股票数据字典，键为股票代码，值为DataFrame
        """
        try:
            start_time = os.times().elapsed
            
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
            stock_data = self._load_stock_data(selected_stocks, fetcher)
            
            # 过滤有效数据
            filtered_stock_data = self._filter_valid_data(stock_data)
            
            # 保存股票代码列表
            self._save_stock_codes(filtered_stock_data)
            
            end_time = os.times().elapsed
            log_performance_stats("获取股票数据", start_time, end_time, 
                                f"成功获取{len(filtered_stock_data)}只股票数据")
            
            return filtered_stock_data
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {str(e)}")
            raise
    
    def _load_stock_data(self, selected_stocks, fetcher):
        """加载股票数据
        
        Args:
            selected_stocks (list): 选中的股票代码列表
            fetcher: 数据获取器实例
            
        Returns:
            dict: 股票数据字典
        """
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
        
        logger.info(f"时间范围: {max_start_date} 到 {min_end_date}")
        return stock_data
    
    def _filter_valid_data(self, stock_data):
        """过滤有效数据
        
        Args:
            stock_data (dict): 原始股票数据
            
        Returns:
            dict: 过滤后的股票数据
        """
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
            logger.info(f"保留所有有效数据，共 {len(valid_stock_data)} 只股票")
            
            # 统计数据量分布
            data_counts = [len(df) for df in valid_stock_data.values()]
            logger.info(f"数据量统计: 最少 {min(data_counts)} 条，最多 {max(data_counts)} 条，平均 {sum(data_counts)/len(data_counts):.1f} 条")
        else:
            raise ValueError("没有找到有效数据的股票")
        
        return valid_stock_data
    
    def _save_stock_codes(self, stock_data):
        """保存股票代码列表
        
        Args:
            stock_data (dict): 股票数据字典
        """
        stock_codes_path = os.path.join(self.data_dir, 'selected_stock_codes.txt')
        with open(stock_codes_path, 'w', encoding='utf-8') as f:
            for stock_code in stock_data.keys():
                f.write(f"{stock_code}\n")
        logger.info(f"股票代码列表已保存到: {stock_codes_path}")
    
    def get_data_summary(self, stock_data):
        """获取数据摘要信息
        
        Args:
            stock_data (dict): 股票数据字典
            
        Returns:
            dict: 数据摘要信息
        """
        if not stock_data:
            return {}
        
        data_counts = [len(df) for df in stock_data.values()]
        total_records = sum(data_counts)
        
        return {
            'total_stocks': len(stock_data),
            'total_records': total_records,
            'min_records': min(data_counts),
            'max_records': max(data_counts),
            'avg_records': round(sum(data_counts) / len(data_counts), 1),
            'date_range': self._get_date_range(stock_data)
        }
    
    def _get_date_range(self, stock_data):
        """获取数据日期范围
        
        Args:
            stock_data (dict): 股票数据字典
            
        Returns:
            dict: 日期范围信息
        """
        all_dates = []
        for df in stock_data.values():
            all_dates.extend(df['date'].tolist())
        
        if not all_dates:
            return {}
        
        return {
            'start_date': min(all_dates),
            'end_date': max(all_dates),
            'total_days': len(set(all_dates))
        }
