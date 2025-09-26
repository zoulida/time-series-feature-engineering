# -*- coding: utf-8 -*-
"""
步骤3：生成训练数据
生成包含原始数据、因子数据和收益率的数据集
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '数据获取'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))

from stock_data_fetcher import StockDataFetcher
from alpha158_factors import Alpha158Factors
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_previous_data():
    """直接获取前两个步骤的数据，避免文件冗余"""
    try:
        # 直接调用步骤1获取股票数据
        from step1_get_stock_data import get_stock_data
        stock_data = get_stock_data()
        
        # 直接调用步骤2获取选中的因子
        from step2_select_factors import select_factors
        selected_factors = select_factors()
        
        logger.info(f"获取股票数据: {len(stock_data)} 只股票")
        logger.info(f"获取选中因子: {len(selected_factors)} 个")
        
        return stock_data, selected_factors
        
    except Exception as e:
        logger.error(f"获取前置数据失败: {str(e)}")
        raise


def calculate_factors(data, factor_name, factor_lib):
    """
    计算单个因子的值
    
    Args:
        data: 股票数据DataFrame
        factor_name: 因子名称
        factor_lib: 因子库实例
        
    Returns:
        pd.Series: 因子值序列
    """
    try:
        factor_info = factor_lib.get_factor_info(factor_name)
        if not factor_info:
            logger.warning(f"未找到因子 {factor_name} 的信息")
            return pd.Series(index=data.index, dtype=float)
        
        # 获取因子表达式
        expression = factor_info.get('expression', '')
        if not expression:
            logger.warning(f"因子 {factor_name} 没有表达式")
            return pd.Series(index=data.index, dtype=float)
        
        # 这里简化处理，实际应该实现表达式解析
        # 为了演示，我们使用一些基本的因子计算
        if 'KMID' in factor_name:
            # K线实体比率
            return (data['close'] - data['open']) / data['open']
        elif 'KLEN' in factor_name:
            # K线长度比率
            return (data['high'] - data['low']) / data['open']
        elif 'RO' in factor_name and any(d in factor_name for d in ['5', '10', '20', '30', '60']):
            # 变化率因子
            window = int(''.join([d for d in factor_name if d.isdigit()]))
            if len(data) > window:
                return data['close'].shift(window) / data['close']
        elif 'MA' in factor_name and any(d in factor_name for d in ['5', '10', '20', '30', '60']):
            # 移动平均因子
            window = int(''.join([d for d in factor_name if d.isdigit()]))
            if len(data) > window:
                return data['close'].rolling(window).mean() / data['close']
        elif 'STD' in factor_name and any(d in factor_name for d in ['5', '10', '20', '30', '60']):
            # 标准差因子
            window = int(''.join([d for d in factor_name if d.isdigit()]))
            if len(data) > window:
                return data['close'].rolling(window).std() / data['close']
        elif 'MAX' in factor_name and any(d in factor_name for d in ['5', '10', '20', '30', '60']):
            # 最大值因子
            window = int(''.join([d for d in factor_name if d.isdigit()]))
            if len(data) > window:
                return data['high'].rolling(window).max() / data['close']
        elif 'MIN' in factor_name and any(d in factor_name for d in ['5', '10', '20', '30', '60']):
            # 最小值因子
            window = int(''.join([d for d in factor_name if d.isdigit()]))
            if len(data) > window:
                return data['low'].rolling(window).min() / data['close']
        elif 'VMA' in factor_name and any(d in factor_name for d in ['5', '10', '20', '30', '60']):
            # 成交量移动平均因子
            window = int(''.join([d for d in factor_name if d.isdigit()]))
            if len(data) > window:
                return data['volume'].rolling(window).mean() / (data['volume'] + 1e-12)
        else:
            # 默认返回随机值（用于演示）
            logger.warning(f"因子 {factor_name} 使用默认计算方式")
            return pd.Series(np.random.randn(len(data)), index=data.index)
        
        return pd.Series(index=data.index, dtype=float)
        
    except Exception as e:
        logger.warning(f"计算因子 {factor_name} 失败: {str(e)}")
        return pd.Series(index=data.index, dtype=float)


def calculate_returns(data, periods=[15]):
    """
    计算不同周期的收益率
    
    Args:
        data: 股票数据DataFrame
        periods: 收益率周期列表
        
    Returns:
        dict: 各周期收益率字典
    """
    returns = {}
    
    for period in periods:
        if len(data) > period:
            # 计算未来收益率
            future_return = data['close'].shift(-period) / data['close'] - 1
            returns[f'return_{period}d'] = future_return
            
            # 计算历史收益率
            past_return = data['close'] / data['close'].shift(period) - 1
            returns[f'past_return_{period}d'] = past_return
    
    return returns


def generate_training_data(stock_data, selected_factors, factor_lib):
    """
    生成训练数据
    
    Args:
        stock_data: 股票数据字典
        selected_factors: 选中的因子列表
        factor_lib: 因子库实例
        
    Returns:
        pd.DataFrame: 训练数据
    """
    try:
        all_training_data = []
        
        for stock_code, data in stock_data.items():
            logger.info(f"处理股票 {stock_code} 的训练数据...")
            
            # 创建基础数据框
            training_df = data[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            training_df['stock_code'] = stock_code
            
            # 计算因子
            logger.info(f"计算股票 {stock_code} 的 {len(selected_factors)} 个因子...")
            for factor_name in selected_factors:
                factor_values = calculate_factors(data, factor_name, factor_lib)
                training_df[f'factor_{factor_name}'] = factor_values
            
            # 计算收益率
            logger.info(f"计算股票 {stock_code} 的收益率...")
            returns = calculate_returns(data)
            for return_name, return_values in returns.items():
                training_df[return_name] = return_values
            
            # 计算其他衍生指标
            training_df['price_change'] = (training_df['close'] - training_df['close'].shift(1)) / training_df['close'].shift(1)
            training_df['volume_change'] = (training_df['volume'] - training_df['volume'].shift(1)) / (training_df['volume'].shift(1) + 1e-12)
            training_df['high_low_ratio'] = training_df['high'] / training_df['low']
            training_df['close_open_ratio'] = training_df['close'] / training_df['open']
            
            # 添加技术指标
            if len(training_df) > 20:
                # 简单移动平均
                training_df['sma_5'] = training_df['close'].rolling(5).mean()
                training_df['sma_20'] = training_df['close'].rolling(20).mean()
                # 布林带
                training_df['bb_upper'] = training_df['close'].rolling(20).mean() + 2 * training_df['close'].rolling(20).std()
                training_df['bb_lower'] = training_df['close'].rolling(20).mean() - 2 * training_df['close'].rolling(20).std()
                training_df['bb_width'] = (training_df['bb_upper'] - training_df['bb_lower']) / training_df['close']
            
            all_training_data.append(training_df)
        
        # 合并所有股票数据
        combined_data = pd.concat(all_training_data, ignore_index=True)
        
        # 按日期和股票代码排序
        combined_data = combined_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
        
        # 移除包含NaN的行（前几行可能有NaN值）
        initial_rows = len(combined_data)
        combined_data = combined_data.dropna()
        removed_rows = initial_rows - len(combined_data)
        logger.info(f"移除了 {removed_rows} 行包含NaN的数据")
        
        logger.info(f"训练数据生成完成，总计 {len(combined_data)} 行，{len(combined_data.columns)} 列")
        return combined_data
        
    except Exception as e:
        logger.error(f"生成训练数据失败: {str(e)}")
        raise


def save_training_data(training_data, data_directory):
    """
    保存训练数据
    
    Args:
        training_data: 训练数据DataFrame
        data_directory: 数据保存目录
    """
    try:
        # 保存完整数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_data_path = os.path.join(data_directory, f'training_data_full_{timestamp}.csv')
        training_data.to_csv(full_data_path, index=False, encoding='utf-8-sig')
        logger.info(f"完整训练数据已保存到: {full_data_path}")
        
        # 保存数据摘要
        summary = {
            'total_rows': len(training_data),
            'total_columns': len(training_data.columns),
            'stock_count': training_data['stock_code'].nunique(),
            'date_range': f"{training_data['date'].min()} 到 {training_data['date'].max()}",
            'columns': list(training_data.columns),
            'data_types': dict(training_data.dtypes)
        }
        
        summary_path = os.path.join(data_directory, 'training_data_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"数据摘要已保存到: {summary_path}")
        
        # 创建用于IC分析的数据格式
        ic_data_path = os.path.join(data_directory, 'ic_analysis_data.csv')
        
        # 选择因子列和收益率列
        factor_cols = [col for col in training_data.columns if col.startswith('factor_')]
        return_cols = [col for col in training_data.columns if col.startswith('return_')]
        
        ic_df = training_data[['date', 'stock_code'] + factor_cols + return_cols].copy()
        ic_df = ic_df.dropna()
        
        ic_df.to_csv(ic_data_path, index=False, encoding='utf-8-sig')
        logger.info(f"IC分析数据已保存到: {ic_data_path}")
        
    except Exception as e:
        logger.error(f"保存训练数据失败: {str(e)}")
        raise


def main():
    """主函数"""
    logger.info("开始执行步骤3：生成训练数据")
    
    try:
        # 加载前置数据
        stock_data, selected_factors = get_previous_data()
        
        if not stock_data:
            logger.error("未找到股票数据")
            return False
        
        if not selected_factors:
            logger.error("未找到选中的因子")
            return False
        
        # 创建因子库实例
        factor_lib = Alpha158Factors()
        
        # 生成训练数据
        training_data = generate_training_data(stock_data, selected_factors, factor_lib)
        
        # 保存数据
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        save_training_data(training_data, data_dir)
        
        # 显示数据概览
        print("\n=== 训练数据概览 ===")
        print(f"总行数: {len(training_data)}")
        print(f"总列数: {len(training_data.columns)}")
        print(f"股票数量: {training_data['stock_code'].nunique()}")
        print(f"时间范围: {training_data['date'].min()} 到 {training_data['date'].max()}")
        
        print("\n=== 列名列表 ===")
        for i, col in enumerate(training_data.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\n=== 前5行数据预览 ===")
        print(training_data.head().to_string())
        
        logger.info("步骤3执行成功！")
        return True
        
    except Exception as e:
        logger.error(f"步骤3执行失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n步骤3执行成功！")
    else:
        print("\n步骤3执行失败！")
        sys.exit(1)
