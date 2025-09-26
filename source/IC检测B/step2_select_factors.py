# -*- coding: utf-8 -*-
"""
步骤2：随机选择因子
使用alpha158_factors.py随机选择8个因子
"""

import sys
import os
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))

from alpha158_factors import Alpha158Factors
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_factors(num_factors=8, seed=42):
    """
    随机选择指定数量的因子（不保存文件）
    
    Args:
        num_factors: 要选择的因子数量
        seed: 随机种子，确保结果可重现
        
    Returns:
        list: 选中的因子名称列表
    """
    try:
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建因子库实例
        factor_lib = Alpha158Factors()
        
        # 获取所有因子
        all_factors = factor_lib.list_factors()
        logger.info(f"Alpha158因子库总共有 {len(all_factors)} 个因子")
        
        # 按类别分组因子
        factor_categories = {}
        for factor_name in all_factors:
            factor_info = factor_lib.get_factor_info(factor_name)
            category = factor_info.get('category', '未知')
            if category not in factor_categories:
                factor_categories[category] = []
            factor_categories[category].append(factor_name)
        
        logger.info("因子类别分布:")
        for category, factors in factor_categories.items():
            logger.info(f"  {category}: {len(factors)} 个因子")
        
        # 从每个类别中随机选择因子，确保多样性
        selected_factors = []
        factors_per_category = max(1, num_factors // len(factor_categories))
        remaining_factors = num_factors
        
        for category, factors in factor_categories.items():
            if remaining_factors <= 0:
                break
                
            # 计算这个类别要选择的因子数量
            category_count = min(factors_per_category, remaining_factors, len(factors))
            
            # 随机选择
            selected_from_category = random.sample(factors, category_count)
            selected_factors.extend(selected_from_category)
            remaining_factors -= category_count
            
            logger.info(f"从 {category} 类别选择了 {len(selected_from_category)} 个因子")
        
        # 如果还需要更多因子，从剩余因子中随机选择
        if remaining_factors > 0:
            remaining_all = [f for f in all_factors if f not in selected_factors]
            if remaining_all:
                additional_factors = random.sample(remaining_all, min(remaining_factors, len(remaining_all)))
                selected_factors.extend(additional_factors)
                logger.info(f"额外选择了 {len(additional_factors)} 个因子")
        
        # 确保数量正确
        if len(selected_factors) > num_factors:
            selected_factors = selected_factors[:num_factors]
        elif len(selected_factors) < num_factors:
            # 如果还不够，从所有因子中随机补充
            all_remaining = [f for f in all_factors if f not in selected_factors]
            needed = num_factors - len(selected_factors)
            additional = random.sample(all_remaining, min(needed, len(all_remaining)))
            selected_factors.extend(additional)
        
        logger.info(f"最终选择了 {len(selected_factors)} 个因子")
        return selected_factors
        
    except Exception as e:
        logger.error(f"选择因子失败: {str(e)}")
        raise


def save_selected_factors(selected_factors, factor_lib):
    """
    保存选中的因子信息
    
    Args:
        selected_factors: 选中的因子名称列表
        factor_lib: 因子库实例
    """
    try:
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建因子详细信息
        factor_details = []
        for factor_name in selected_factors:
            factor_info = factor_lib.get_factor_info(factor_name)
            if factor_info:
                factor_details.append({
                    'factor_name': factor_name,
                    'function_name': factor_info.get('function_name', ''),
                    'category': factor_info.get('category', ''),
                    'description': factor_info.get('description', ''),
                    'expression': factor_info.get('expression', ''),
                    'formula': factor_info.get('formula', '')
                })
        
        # 直接返回选中的因子，不保存文件
        logger.info(f"成功选择 {len(selected_factors)} 个因子")
        
    except Exception as e:
        logger.error(f"保存因子信息失败: {str(e)}")
        raise


def display_selected_factors(selected_factors, factor_lib):
    """
    显示选中的因子信息
    
    Args:
        selected_factors: 选中的因子名称列表
        factor_lib: 因子库实例
    """
    print("\n=== 选中的8个因子 ===")
    print(f"{'序号':<4} {'因子名称':<12} {'类别':<12} {'描述':<50}")
    print("-" * 80)
    
    for i, factor_name in enumerate(selected_factors, 1):
        factor_info = factor_lib.get_factor_info(factor_name)
        if factor_info:
            category = factor_info.get('category', '未知')
            description = factor_info.get('description', '无描述')
            # 截断过长的描述
            if len(description) > 45:
                description = description[:42] + "..."
            print(f"{i:<4} {factor_name:<12} {category:<12} {description:<50}")
    
    print("\n=== 因子类别统计 ===")
    category_count = {}
    for factor_name in selected_factors:
        factor_info = factor_lib.get_factor_info(factor_name)
        if factor_info:
            category = factor_info.get('category', '未知')
            category_count[category] = category_count.get(category, 0) + 1
    
    for category, count in category_count.items():
        print(f"{category}: {count} 个因子")


def main():
    """主函数"""
    logger.info("开始执行步骤2：随机选择因子")
    
    try:
        # 创建因子库实例
        factor_lib = Alpha158Factors()
        
        # 随机选择8个因子
        selected_factors = select_factors(num_factors=8, seed=42)
        
        # 保存选中的因子信息（仅用于记录）
        save_selected_factors(selected_factors, factor_lib)
        
        # 显示选中的因子
        display_selected_factors(selected_factors, factor_lib)
        
        logger.info("步骤2执行成功！")
        return selected_factors
        
    except Exception as e:
        logger.error(f"步骤2执行失败: {str(e)}")
        return None


if __name__ == "__main__":
    selected_factors = main()
    if selected_factors:
        print(f"\n步骤2执行成功！共选择了 {len(selected_factors)} 个因子")
    else:
        print("步骤2执行失败！")
        sys.exit(1)
