"""
股票数据获取模块
支持自定义开始和结束日期参数，从指定目录获取股票数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self, data_dir: str = r"f:\stockdata\getDayKlineData\20241101-20250922-front"):
        """
        初始化股票数据获取器
        
        Args:
            data_dir: 股票数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.supported_columns = [
            'date', 'time', 'open', 'high', 'low', 'close', 
            'volume', 'amount', 'settelementPrice', 'openInterest', 
            'preClose', 'suspendFlag'
        ]
        
        # 验证数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        logger.info(f"股票数据获取器初始化完成，数据目录: {self.data_dir}")
    
    def get_available_stocks(self) -> List[str]:
        """
        获取可用的股票代码列表
        
        Returns:
            股票代码列表
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        stock_codes = [f.stem for f in csv_files]
        logger.info(f"发现 {len(stock_codes)} 只股票")
        return sorted(stock_codes)
    
    def load_single_stock_data(self, stock_code: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载单只股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (格式: YYYYMMDD)
            end_date: 结束日期 (格式: YYYYMMDD)
            
        Returns:
            股票数据DataFrame
        """
        file_path = self.data_dir / f"{stock_code}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"股票数据文件不存在: {file_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 验证列名
            if not all(col in df.columns for col in self.supported_columns):
                logger.warning(f"股票 {stock_code} 的列名可能不标准")
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)
            
            # 日期过滤
            if start_date:
                start_dt = pd.to_datetime(start_date, format='%Y%m%d')
                df = df[df['date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date, format='%Y%m%d')
                df = df[df['date'] <= end_dt]
            
            # 添加股票代码列
            df['stock_code'] = stock_code
            
            logger.info(f"成功加载股票 {stock_code} 数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载股票 {stock_code} 数据失败: {str(e)}")
            raise
    
    def load_multiple_stocks_data(self, stock_codes: List[str], 
                                start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载多只股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (格式: YYYYMMDD)
            end_date: 结束日期 (格式: YYYYMMDD)
            
        Returns:
            股票数据字典 {股票代码: DataFrame}
        """
        stock_data = {}
        failed_stocks = []
        
        for stock_code in stock_codes:
            try:
                df = self.load_single_stock_data(stock_code, start_date, end_date)
                stock_data[stock_code] = df
            except Exception as e:
                logger.error(f"加载股票 {stock_code} 失败: {str(e)}")
                failed_stocks.append(stock_code)
        
        if failed_stocks:
            logger.warning(f"以下股票加载失败: {failed_stocks}")
        
        logger.info(f"成功加载 {len(stock_data)} 只股票数据")
        return stock_data
    
    def load_all_stocks_data(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有股票数据
        
        Args:
            start_date: 开始日期 (格式: YYYYMMDD)
            end_date: 结束日期 (格式: YYYYMMDD)
            
        Returns:
            所有股票数据字典
        """
        stock_codes = self.get_available_stocks()
        return self.load_multiple_stocks_data(stock_codes, start_date, end_date)
    
    def create_panel_data(self, stock_data: Dict[str, pd.DataFrame], 
                         target_column: str = 'close') -> pd.DataFrame:
        """
        创建面板数据格式
        
        Args:
            stock_data: 股票数据字典
            target_column: 目标列名
            
        Returns:
            面板数据DataFrame
        """
        panel_data = []
        
        for stock_code, df in stock_data.items():
            if target_column in df.columns:
                temp_df = df[['date', target_column]].copy()
                temp_df['stock_code'] = stock_code
                temp_df = temp_df.rename(columns={target_column: 'value'})
                panel_data.append(temp_df)
        
        if not panel_data:
            raise ValueError(f"未找到目标列 {target_column}")
        
        result_df = pd.concat(panel_data, ignore_index=True)
        
        # 转换为宽格式
        panel_df = result_df.pivot(index='date', columns='stock_code', values='value')
        panel_df = panel_df.fillna(method='ffill')  # 前向填充缺失值
        
        logger.info(f"创建面板数据完成，形状: {panel_df.shape}")
        return panel_df
    
    def get_data_summary(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        获取数据摘要信息
        
        Args:
            stock_data: 股票数据字典
            
        Returns:
            数据摘要DataFrame
        """
        summary_data = []
        
        for stock_code, df in stock_data.items():
            summary = {
                'stock_code': stock_code,
                'start_date': df['date'].min().strftime('%Y%m%d'),
                'end_date': df['date'].max().strftime('%Y%m%d'),
                'total_days': len(df),
                'missing_days': 0,  # 可以进一步计算
                'avg_volume': df['volume'].mean() if 'volume' in df.columns else 0,
                'avg_amount': df['amount'].mean() if 'amount' in df.columns else 0,
                'price_range': f"{df['close'].min():.2f} - {df['close'].max():.2f}" if 'close' in df.columns else "N/A"
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_data(self, data: pd.DataFrame, output_path: str, 
                  format: str = 'csv') -> None:
        """
        保存数据到文件
        
        Args:
            data: 要保存的数据
            output_path: 输出路径
            format: 文件格式 ('csv', 'excel', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            data.to_csv(output_path, index=True, encoding='utf-8-sig')
        elif format.lower() == 'excel':
            data.to_excel(output_path, index=True)
        elif format.lower() == 'parquet':
            data.to_parquet(output_path, index=True)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        logger.info(f"数据已保存到: {output_path}")


def main():
    """主函数示例"""
    # 创建数据获取器
    fetcher = StockDataFetcher()
    
    # 获取可用股票列表
    stocks = fetcher.get_available_stocks()
    print(f"可用股票数量: {len(stocks)}")
    print(f"前10只股票: {stocks[:10]}")
    
    # 加载指定日期范围的数据
    start_date = "20250101"
    end_date = "20250331"
    
    # 加载前5只股票的数据作为示例
    sample_stocks = stocks[:5]
    stock_data = fetcher.load_multiple_stocks_data(
        sample_stocks, start_date, end_date
    )
    
    # 显示数据摘要
    summary = fetcher.get_data_summary(stock_data)
    print("\n数据摘要:")
    print(summary)
    
    # 创建面板数据
    panel_data = fetcher.create_panel_data(stock_data, 'close')
    print(f"\n面板数据形状: {panel_data.shape}")
    print(panel_data.head())


if __name__ == "__main__":
    main()