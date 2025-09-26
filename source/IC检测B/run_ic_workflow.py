# -*- coding: utf-8 -*-
"""
IC检测工作流主程序
按顺序执行4个步骤：获取数据 -> 选择因子 -> 生成训练数据 -> IC检测
"""

import sys
import os
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ic_workflow.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_step(step_name, step_module, step_function):
    """
    运行单个步骤
    
    Args:
        step_name: 步骤名称
        step_module: 步骤模块
        step_function: 步骤函数
        
    Returns:
        bool: 是否成功
    """
    try:
        logger.info(f"开始执行 {step_name}")
        start_time = time.time()
        
        # 导入并执行步骤
        if step_module == "step1_get_stock_data":
            from step1_get_stock_data import main as step1_main
            success = step1_main()
        elif step_module == "step2_select_factors":
            from step2_select_factors import main as step2_main
            success = step2_main() is not None
        elif step_module == "step3_generate_training_data":
            from step3_generate_training_data import main as step3_main
            success = step3_main()
        elif step_module == "step4_ic_analysis":
            from step4_ic_analysis import main as step4_main
            success = step4_main()
        else:
            logger.error(f"未知的步骤模块: {step_module}")
            return False
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            logger.info(f"{step_name} 执行成功，耗时: {duration:.2f}秒")
            return True
        else:
            logger.error(f"{step_name} 执行失败，耗时: {duration:.2f}秒")
            return False
            
    except Exception as e:
        logger.error(f"{step_name} 执行异常: {str(e)}")
        return False


def check_dependencies():
    """检查依赖项"""
    try:
        logger.info("检查依赖项...")
        
        # 检查必要的模块
        required_modules = [
            'pandas', 'numpy', 'json', 'datetime'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            logger.error(f"缺少必要的模块: {missing_modules}")
            return False
        
        # 使用改进的IC分析方法（不依赖alphalens）
        logger.info("使用改进的IC分析方法")
        
        logger.info("依赖项检查完成")
        return True
        
    except Exception as e:
        logger.error(f"依赖项检查失败: {str(e)}")
        return False


def create_workflow_summary():
    """创建工作流摘要"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        summary = {
            'workflow_start_time': datetime.now().isoformat(),
            'workflow_version': '1.0',
            'steps': [
                {
                    'step': 1,
                    'name': '获取股票数据',
                    'module': 'step1_get_stock_data.py',
                    'description': '使用stock_data_fetcher.py获取最大时段的5只股票数据'
                },
                {
                    'step': 2,
                    'name': '选择因子',
                    'module': 'step2_select_factors.py',
                    'description': '使用alpha158_factors.py随机选择8个因子'
                },
                {
                    'step': 3,
                    'name': '生成训练数据',
                    'module': 'step3_generate_training_data.py',
                    'description': '生成包含原始数据、因子数据和收益率的数据集'
                },
                {
                    'step': 4,
                    'name': 'IC检测分析',
                    'module': 'step4_ic_analysis.py',
                    'description': '使用alphalens完成8个因子的IC检测'
                }
            ],
            'output_files': [
                'data/selected_stock_codes.txt',
                'data/selected_factors.json',
                'data/training_data_full.csv',
                'data/ic_analysis_data.csv',
                'data/ic_results.json',
                'data/ic_report.csv',
                'data/ic_summary.csv'
            ]
        }
        
        summary_path = os.path.join(data_dir, 'workflow_summary.json')
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"工作流摘要已保存到: {summary_path}")
        
    except Exception as e:
        logger.warning(f"创建工作流摘要失败: {str(e)}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("IC检测工作流开始执行")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # 检查依赖项
        if not check_dependencies():
            logger.error("依赖项检查失败，工作流终止")
            return False
        
        # 定义工作流步骤
        workflow_steps = [
            {
                'name': '步骤1：获取股票数据',
                'module': 'step1_get_stock_data',
                'function': 'main'
            },
            {
                'name': '步骤2：选择因子',
                'module': 'step2_select_factors',
                'function': 'main'
            },
            {
                'name': '步骤3：生成训练数据',
                'module': 'step3_generate_training_data',
                'function': 'main'
            },
            {
                'name': '步骤4：IC检测分析',
                'module': 'step4_ic_analysis',
                'function': 'main'
            }
        ]
        
        # 执行工作流步骤
        success_count = 0
        total_steps = len(workflow_steps)
        
        for i, step in enumerate(workflow_steps, 1):
            logger.info(f"\n{'='*20} 执行步骤 {i}/{total_steps} {'='*20}")
            
            success = run_step(step['name'], step['module'], step['function'])
            
            if success:
                success_count += 1
                logger.info(f"步骤 {i} 执行成功")
            else:
                logger.error(f"步骤 {i} 执行失败，工作流终止")
                break
        
        # 计算总耗时
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 创建工作流摘要
        create_workflow_summary()
        
        # 输出结果
        logger.info("\n" + "=" * 60)
        logger.info("工作流执行完成")
        logger.info("=" * 60)
        logger.info(f"总步骤数: {total_steps}")
        logger.info(f"成功步骤数: {success_count}")
        logger.info(f"失败步骤数: {total_steps - success_count}")
        logger.info(f"总耗时: {total_duration:.2f}秒")
        
        if success_count == total_steps:
            logger.info("所有步骤执行成功！")
            print("\n🎉 IC检测工作流执行成功！")
            print(f"📊 成功执行了 {success_count}/{total_steps} 个步骤")
            print(f"⏱️  总耗时: {total_duration:.2f}秒")
            print("📁 结果文件保存在 data/ 目录中")
            return True
        else:
            logger.error("工作流执行失败！")
            print(f"\n❌ IC检测工作流执行失败！")
            print(f"📊 成功执行了 {success_count}/{total_steps} 个步骤")
            print(f"⏱️  总耗时: {total_duration:.2f}秒")
            return False
            
    except Exception as e:
        logger.error(f"工作流执行异常: {str(e)}")
        print(f"\n💥 工作流执行异常: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
