# -*- coding: utf-8 -*-
"""
步骤2: 批量选择因子
负责从统一因子库中选择指定数量的因子，支持分类选择和多样性保证
"""

import sys
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from unified_factor_library import UnifiedFactorLibrary
from shared import log_performance_stats

logger = logging.getLogger(__name__)


class FactorBatchSelector:
    """批量因子选择器"""
    
    def __init__(self, config, data_dir):
        """
        初始化批量因子选择器
        
        Args:
            config: 配置对象（BatchConfig）
            data_dir (str): 数据目录路径
        """
        self.config = config
        self.data_dir = data_dir
        
        # 根据config的因子库模式创建因子库
        if hasattr(config, 'get_factor_sources'):
            factor_sources = config.get_factor_sources()
            self.factor_lib = UnifiedFactorLibrary(sources=factor_sources)
            logger.info(f"使用因子库模式: {config.FACTOR_LIBRARY_MODE}")
            logger.info(f"加载的因子源: {factor_sources}")
        else:
            self.factor_lib = UnifiedFactorLibrary()
            logger.info("使用默认因子库配置")
        
    def select_factors_batch(self, num_factors):
        """批量选择因子
        
        Args:
            num_factors (int): 需要选择的因子数量
            
        Returns:
            list: 选中的因子名称列表
        """
        try:
            start_time = os.times().elapsed
            
            # 获取所有因子
            all_factors = self.factor_lib.list_factors()
            logger.info(f"统一因子库总共有 {len(all_factors)} 个因子")
            
            if len(all_factors) < num_factors:
                logger.warning(f"可用因子数量不足，需要{num_factors}个，实际只有{len(all_factors)}个")
                num_factors = len(all_factors)
            
            # 按类别分组因子
            factor_categories = self._group_factors_by_category(all_factors)
            
            # 从每个类别中选择因子，确保多样性
            selected_factors = self._select_factors_from_categories(factor_categories, num_factors)
            
            # 保存选中的因子信息
            self._save_selected_factors(selected_factors)
            
            end_time = os.times().elapsed
            log_performance_stats("选择因子", start_time, end_time, 
                                f"成功选择{len(selected_factors)}个因子")
            
            return selected_factors
            
        except Exception as e:
            logger.error(f"选择因子失败: {str(e)}")
            raise
    
    def _group_factors_by_category(self, all_factors):
        """按类别分组因子
        
        Args:
            all_factors (list): 所有因子名称列表
            
        Returns:
            dict: 按类别分组的因子字典
        """
        factor_categories = {}
        for factor_name in all_factors:
            factor_info = self.factor_lib.get_factor_info(factor_name)
            category = factor_info.get('category', '未知') if factor_info else '未知'
            if category not in factor_categories:
                factor_categories[category] = []
            factor_categories[category].append(factor_name)
        
        logger.info("因子类别分布:")
        for category, factors in factor_categories.items():
            logger.info(f"  {category}: {len(factors)} 个因子")
        
        return factor_categories
    
    def _select_factors_from_categories(self, factor_categories, num_factors):
        """从各个类别中选择因子
        
        Args:
            factor_categories (dict): 按类别分组的因子字典
            num_factors (int): 需要选择的因子数量
            
        Returns:
            list: 选中的因子名称列表
        """
        selected_factors = []
        factors_per_category = max(1, num_factors // len(factor_categories))
        remaining_factors = num_factors
        
        for category, factors in factor_categories.items():
            if remaining_factors <= 0:
                break
            
            # 从当前类别选择因子
            select_count = min(factors_per_category, remaining_factors, len(factors))
            selected_from_category = np.random.choice(factors, select_count, replace=False).tolist()
            selected_factors.extend(selected_from_category)
            remaining_factors -= select_count
            
            logger.info(f"从 {category} 类别选择了 {select_count} 个因子")
        
        # 如果还有剩余因子，随机补充
        if remaining_factors > 0:
            remaining_all = [f for f in self.factor_lib.list_factors() if f not in selected_factors]
            if remaining_all:
                additional_count = min(remaining_factors, len(remaining_all))
                additional_factors = np.random.choice(remaining_all, additional_count, replace=False).tolist()
                selected_factors.extend(additional_factors)
                logger.info(f"随机补充了 {additional_count} 个因子")
        
        logger.info(f"最终选择了 {len(selected_factors)} 个因子")
        return selected_factors
    
    def _save_selected_factors(self, selected_factors):
        """保存选中的因子信息
        
        Args:
            selected_factors (list): 选中的因子名称列表
        """
        try:
            # 创建因子详情
            factor_details = []
            for factor_name in selected_factors:
                factor_info = self.factor_lib.get_factor_info(factor_name)
                if factor_info:
                    factor_details.append({
                        'factor_name': factor_name,
                        'function_name': factor_info.get('function_name', ''),
                        'category': factor_info.get('category', ''),
                        'description': factor_info.get('description', ''),
                        'expression': factor_info.get('expression', ''),
                        'formula': factor_info.get('formula', ''),
                        'source': factor_info.get('source', '')
                    })
            
            # 保存为CSV
            factor_df = pd.DataFrame(factor_details)
            csv_path = os.path.join(self.data_dir, 'selected_factors.csv')
            factor_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"选中因子详情已保存到: {csv_path}")
            
            # 保存为JSON
            json_path = os.path.join(self.data_dir, 'selected_factors.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'selected_factors': selected_factors,
                    'factor_details': factor_details,
                    'selection_time': datetime.now().isoformat(),
                    'total_count': len(selected_factors)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"选中因子信息已保存到: {json_path}")
            
        except Exception as e:
            logger.error(f"保存因子信息失败: {str(e)}")
    
    def get_factor_summary(self, selected_factors):
        """获取因子摘要信息
        
        Args:
            selected_factors (list): 选中的因子名称列表
            
        Returns:
            dict: 因子摘要信息
        """
        if not selected_factors:
            return {}
        
        # 统计因子来源分布
        source_count = {}
        category_count = {}
        
        for factor_name in selected_factors:
            factor_info = self.factor_lib.get_factor_info(factor_name)
            if factor_info:
                source = factor_info.get('source', '未知')
                category = factor_info.get('category', '未知')
                
                source_count[source] = source_count.get(source, 0) + 1
                category_count[category] = category_count.get(category, 0) + 1
        
        return {
            'total_factors': len(selected_factors),
            'source_distribution': source_count,
            'category_distribution': category_count
        }
    
    def validate_factors(self, selected_factors):
        """验证选中的因子
        
        Args:
            selected_factors (list): 选中的因子名称列表
            
        Returns:
            dict: 验证结果
        """
        valid_factors = []
        invalid_factors = []
        
        for factor_name in selected_factors:
            factor_info = self.factor_lib.get_factor_info(factor_name)
            if factor_info and factor_info.get('function_name'):
                valid_factors.append(factor_name)
            else:
                invalid_factors.append(factor_name)
        
        return {
            'valid_count': len(valid_factors),
            'invalid_count': len(invalid_factors),
            'valid_factors': valid_factors,
            'invalid_factors': invalid_factors
        }
