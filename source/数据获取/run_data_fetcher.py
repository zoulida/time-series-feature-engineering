"""
股票数据获取运行脚本
提供命令行接口
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from stock_data_fetcher import StockDataFetcher
from config import get_config, validate_config, create_output_dirs
from utils import validate_date_format, validate_stock_code

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_data_fetcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票数据获取工具')
    
    # 基本参数
    parser.add_argument('--data-dir', type=str, 
                       default=r"f:\stockdata\getDayKlineData\20241101-20250922-front",
                       help='股票数据目录路径')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录路径')
    
    # 日期参数
    parser.add_argument('--start-date', type=str, 
                       help='开始日期 (格式: YYYYMMDD)')
    parser.add_argument('--end-date', type=str,
                       help='结束日期 (格式: YYYYMMDD)')
    
    # 股票选择
    parser.add_argument('--stocks', type=str, nargs='+',
                       help='指定股票代码列表')
    parser.add_argument('--stock-file', type=str,
                       help='从文件读取股票代码列表')
    parser.add_argument('--all-stocks', action='store_true',
                       help='处理所有股票')
    
    # 其他选项
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def load_stock_list(stock_file: str) -> list:
    """从文件加载股票代码列表"""
    try:
        with open(stock_file, 'r', encoding='utf-8') as f:
            stocks = [line.strip() for line in f if line.strip()]
        logger.info(f"从文件加载了 {len(stocks)} 只股票")
        return stocks
    except Exception as e:
        logger.error(f"加载股票文件失败: {e}")
        return []


def get_default_dates() -> tuple:
    """获取默认日期范围"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 默认最近90天
    
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证配置
    config = get_config()
    if not validate_config(config):
        logger.error("配置验证失败")
        return 1
    
    # 创建输出目录
    create_output_dirs(config)
    
    # 获取日期范围
    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        start_date, end_date = get_default_dates()
        logger.info(f"使用默认日期范围: {start_date} 到 {end_date}")
    
    # 验证日期格式
    if not validate_date_format(start_date) or not validate_date_format(end_date):
        logger.error("日期格式无效，请使用 YYYYMMDD 格式")
        return 1
    
    # 创建数据获取器
    try:
        fetcher = StockDataFetcher(args.data_dir)
    except Exception as e:
        logger.error(f"创建数据获取器失败: {e}")
        return 1
    
    # 获取股票列表
    if args.all_stocks:
        stocks = fetcher.get_available_stocks()
    elif args.stocks:
        stocks = args.stocks
        # 验证股票代码格式
        invalid_stocks = [s for s in stocks if not validate_stock_code(s)]
        if invalid_stocks:
            logger.warning(f"以下股票代码格式可能无效: {invalid_stocks}")
    elif args.stock_file:
        stocks = load_stock_list(args.stock_file)
        if not stocks:
            logger.error("未能加载任何股票代码")
            return 1
    else:
        # 默认处理前10只股票
        stocks = fetcher.get_available_stocks()[:10]
        logger.info(f"未指定股票，默认处理前10只: {stocks}")
    
    logger.info(f"开始处理 {len(stocks)} 只股票，日期范围: {start_date} 到 {end_date}")
    
    # 加载股票数据
    try:
        stock_data = fetcher.load_multiple_stocks_data(stocks, start_date, end_date)
        if not stock_data:
            logger.error("未能加载任何股票数据")
            return 1
    except Exception as e:
        logger.error(f"加载股票数据失败: {e}")
        return 1
    
    # 数据摘要
    summary = fetcher.get_data_summary(stock_data)
    print("\n数据摘要:")
    print(summary.to_string(index=False))
    
    # 保存原始数据
    raw_data_dir = Path(args.output_dir) / 'raw_data'
    raw_data_dir.mkdir(exist_ok=True)
    
    for stock_code, df in stock_data.items():
        output_file = raw_data_dir / f"{stock_code}_{start_date}_{end_date}.csv"
        fetcher.save_data(df, str(output_file), 'csv')
    
    logger.info(f"原始数据已保存到: {raw_data_dir}")
    
    # 创建面板数据
    try:
        panel_data = fetcher.create_panel_data(stock_data, 'close')
        
        # 保存面板数据
        panel_file = Path(args.output_dir) / f"panel_data_{start_date}_{end_date}.csv"
        fetcher.save_data(panel_data, str(panel_file), 'csv')
        
        logger.info(f"面板数据已保存到: {panel_file}")
    except Exception as e:
        logger.error(f"创建面板数据失败: {e}")
    
    logger.info("数据处理完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())