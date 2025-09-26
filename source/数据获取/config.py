"""
股票数据获取配置文件
"""

import os
from pathlib import Path

# 数据路径配置
DEFAULT_DATA_DIR = r"f:\stockdata\getDayKlineData\20241101-20250922-front"
OUTPUT_DIR = "output"

# 数据列配置
STOCK_DATA_COLUMNS = [
    'date', 'time', 'open', 'high', 'low', 'close', 
    'volume', 'amount', 'settelementPrice', 'openInterest', 
    'preClose', 'suspendFlag'
]

# 默认日期格式
DATE_FORMAT = '%Y%m%d'

# 股票代码配置
STOCK_CODE_PATTERNS = {
    'SH': r'^\d{6}\.SH$',  # 上海证券交易所
    'SZ': r'^\d{6}\.SZ$',  # 深圳证券交易所
    'BJ': r'^\d{6}\.BJ$'   # 北京证券交易所
}

def get_config():
    """获取完整配置"""
    return {
        'data_dir': DEFAULT_DATA_DIR,
        'output_dir': OUTPUT_DIR,
        'columns': STOCK_DATA_COLUMNS,
        'date_format': DATE_FORMAT,
        'stock_patterns': STOCK_CODE_PATTERNS
    }

def validate_config(config: dict) -> bool:
    """验证配置有效性"""
    required_keys = ['data_dir', 'output_dir', 'columns']
    
    for key in required_keys:
        if key not in config:
            print(f"配置缺少必需项: {key}")
            return False
    
    # 检查数据目录是否存在
    if not os.path.exists(config['data_dir']):
        print(f"数据目录不存在: {config['data_dir']}")
        return False
    
    return True

def create_output_dirs(config: dict) -> None:
    """创建输出目录"""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print(f"输出目录已创建: {output_dir}")

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print("配置验证:", validate_config(config))
    create_output_dirs(config)