# -*- coding: utf-8 -*-
"""
批量IC检测工作流运行脚本 - 重构版
使用分步骤模块结构，代码更清晰、更易维护
"""

import sys
import os
import logging
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from run_ic_workflow_batch_refactored import BatchICWorkflow
from config_batch import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ic_workflow_batch_refactored.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    try:
        start_time = datetime.now()
        print("=" * 60)
        print("IC检测工作流批量版 - 重构版")
        print("=" * 60)
        print(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 创建配置对象 - 使用config_batch.py的配置
        batch_config = get_config()
        
        # 显示配置信息
        print("当前配置:")
        print(f"  模式: 生产模式")
        print(f"  股票数量: {batch_config.PRODUCTION_STOCKS}")
        print(f"  因子数量: {batch_config.PRODUCTION_FACTORS}")
        print(f"  因子库模式: {batch_config.FACTOR_LIBRARY_MODE}")
        print(f"  批次大小: {batch_config.BATCH_SIZE}")
        print(f"  最大内存: {batch_config.MAX_MEMORY_GB}GB")
        print(f"  收益率计算方法: {batch_config.RETURN_CALCULATION_METHOD}")
        if batch_config.RETURN_CALCULATION_METHOD == 'max_future_15d':
            print(f"  未来天数: {batch_config.FUTURE_DAYS}")
        print()
        
        # 创建批量工作流实例
        workflow = BatchICWorkflow(batch_config)
        
        # 运行工作流
        print("开始执行工作流...")
        success = workflow.run_batch_workflow()
        
        # 输出执行结果
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        print("\n" + "=" * 60)
        if success:
            print("🎉 批量工作流执行成功！")
            print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏰ 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ 总耗时: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟)")
            print("📁 结果文件保存在: D:\\pythonProject\\时序特征工程\\source\\IC检测批量\\data")
            print("📊 可以运行 data/analyze_ic_results.py 查看详细分析结果")
        else:
            print("❌ 批量工作流执行失败！")
            print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏰ 失败时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ 总耗时: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟)")
            print("📋 请查看日志文件了解详细错误信息")
        print("=" * 60)
        
        return success
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        print(f"❌ 程序执行失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
