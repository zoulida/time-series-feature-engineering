# -*- coding: utf-8 -*-
"""
步骤1：获取股票数据
使用stock_data_fetcher.py获取最大时段的5只股票数据
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '数据获取'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))

from stock_data_fetcher import StockDataFetcher
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_stock_data():
    """
    获取最大时段的5只股票数据（不保存文件）
    
    Returns:
        dict: 包含5只股票数据的字典
    """
    try:
        # 创建数据获取器
        fetcher = StockDataFetcher()
        
        # 获取所有可用股票
        all_stocks = fetcher.get_available_stocks()
        logger.info(f"发现 {len(all_stocks)} 只股票")
        
        if len(all_stocks) < 5:
            raise ValueError(f"可用股票数量不足，需要至少5只，实际只有{len(all_stocks)}只")
        
        # 选择前5只股票
        selected_stocks = all_stocks[:5]
        logger.info(f"选择的股票: {selected_stocks}")
        
        # 获取所有股票的数据摘要，找出最大时间范围
        logger.info("正在分析股票数据时间范围...")
        stock_data = {}
        max_start_date = None
        min_end_date = None
        
        for stock_code in selected_stocks:
            try:
                # 不指定日期范围，获取所有数据
                df = fetcher.load_single_stock_data(stock_code)
                stock_data[stock_code] = df
                
                # 更新最大时间范围
                stock_start = df['date'].min()
                stock_end = df['date'].max()
                
                if max_start_date is None or stock_start > max_start_date:
                    max_start_date = stock_start
                if min_end_date is None or stock_end < min_end_date:
                    min_end_date = stock_end
                    
                logger.info(f"股票 {stock_code}: {stock_start.strftime('%Y-%m-%d')} 到 {stock_end.strftime('%Y-%m-%d')}")
                
            except Exception as e:
                logger.error(f"加载股票 {stock_code} 失败: {str(e)}")
                continue
        
        if not stock_data:
            raise ValueError("没有成功加载任何股票数据")
        
        # 过滤到共同的时间范围
        logger.info(f"共同时间范围: {max_start_date.strftime('%Y-%m-%d')} 到 {min_end_date.strftime('%Y-%m-%d')}")
        
        filtered_stock_data = {}
        for stock_code, df in stock_data.items():
            # 过滤到共同时间范围
            filtered_df = df[(df['date'] >= max_start_date) & (df['date'] <= min_end_date)].copy()
            if len(filtered_df) > 0:
                filtered_stock_data[stock_code] = filtered_df
                logger.info(f"股票 {stock_code} 过滤后数据量: {len(filtered_df)} 条")
        
        # 保存数据
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        # 直接返回数据，不保存文件
        logger.info(f"步骤1完成：成功获取 {len(filtered_stock_data)} 只股票数据")
        return filtered_stock_data
        
    except Exception as e:
        logger.error(f"步骤1失败: {str(e)}")
        raise


def main():
    """主函数"""
    logger.info("开始执行步骤1：获取股票数据")
    
    try:
        stock_data = get_stock_data()
        
        # 保存数据摘要（仅用于记录）
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建数据摘要
        summary_data = []
        for stock_code, df in stock_data.items():
            summary = {
                'stock_code': stock_code,
                'start_date': df['date'].min().strftime('%Y-%m-%d'),
                'end_date': df['date'].max().strftime('%Y-%m-%d'),
                'total_days': len(df),
                'avg_close': df['close'].mean(),
                'avg_volume': df['volume'].mean(),
                'price_range': f"{df['close'].min():.2f} - {df['close'].max():.2f}"
            }
            summary_data.append(summary)
        
        # 保存摘要
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'stock_data_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据摘要已保存到: {summary_path}")
        
        # 保存股票代码列表
        stock_codes_path = os.path.join(output_dir, 'selected_stock_codes.txt')
        with open(stock_codes_path, 'w', encoding='utf-8') as f:
            for stock_code in stock_data.keys():
                f.write(f"{stock_code}\n")
        logger.info(f"股票代码列表已保存到: {stock_codes_path}")
        
        logger.info("步骤1执行成功！")
        
        # 显示数据概览
        print("\n=== 股票数据概览 ===")
        for stock_code, df in stock_data.items():
            print(f"股票 {stock_code}: {len(df)} 条记录, "
                  f"时间范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        logger.error(f"步骤1执行失败: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("步骤1执行成功！")
    else:
        print("步骤1执行失败！")
        sys.exit(1)
