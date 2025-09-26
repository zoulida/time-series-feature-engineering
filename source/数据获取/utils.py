"""
股票数据获取工具函数
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import re
import logging

logger = logging.getLogger(__name__)


def validate_date_format(date_str: str, format: str = '%Y%m%d') -> bool:
    """
    验证日期格式
    
    Args:
        date_str: 日期字符串
        format: 日期格式
        
    Returns:
        是否有效
    """
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False


def validate_stock_code(stock_code: str) -> bool:
    """
    验证股票代码格式
    
    Args:
        stock_code: 股票代码
        
    Returns:
        是否有效
    """
    # 基本格式检查：6位数字 + .SH/.SZ/.BJ
    pattern = r'^\d{6}\.(SH|SZ|BJ)$'
    return bool(re.match(pattern, stock_code))


def get_trading_days(start_date: str, end_date: str, 
                    exclude_weekends: bool = True) -> List[str]:
    """
    获取交易日列表
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        exclude_weekends: 是否排除周末
        
    Returns:
        交易日列表
    """
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    trading_days = []
    current = start
    
    while current <= end:
        if not exclude_weekends or current.weekday() < 5:  # 0-4为周一到周五
            trading_days.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    return trading_days


def main():
    """主函数示例"""
    # 测试工具函数
    print("测试日期格式验证:")
    print(f"20250101: {validate_date_format('20250101')}")
    print(f"2025-01-01: {validate_date_format('2025-01-01', '%Y-%m-%d')}")
    
    print("\n测试股票代码验证:")
    print(f"000001.SZ: {validate_stock_code('000001.SZ')}")
    print(f"000001: {validate_stock_code('000001')}")
    
    print("\n测试交易日获取:")
    trading_days = get_trading_days('20250101', '20250110')
    print(f"2025年1月前10个交易日: {trading_days}")


if __name__ == "__main__":
    main()