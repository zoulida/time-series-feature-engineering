# -*- coding: utf-8 -*-
"""
步骤4: 批量IC分析
负责计算所有因子的信息系数(Information Coefficient)，生成IC报告
"""

import sys
import os
import logging
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '因子库'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from unified_factor_library import UnifiedFactorLibrary
from shared import log_performance_stats, get_timestamp

logger = logging.getLogger(__name__)


class ICBatchAnalyzer:
    """批量IC分析器"""
    
    def __init__(self, config, data_dir):
        """
        初始化批量IC分析器
        
        Args:
            config: 配置对象
            data_dir (str): 数据目录路径
        """
        self.config = config
        self.data_dir = data_dir
        self.factor_lib = UnifiedFactorLibrary()
        
    def perform_ic_analysis_batch(self, training_data, selected_factors):
        """批量IC分析 - 计算所有因子的信息系数(Information Coefficient)
        
        IC分析是量化投资中的核心指标，用于衡量因子与未来收益率的相关性：
        - Pearson IC: 线性相关系数，衡量线性关系强度
        - Spearman IC: 秩相关系数，衡量单调关系强度
        
        Args:
            training_data (pd.DataFrame): 包含因子值和收益率的训练数据
            selected_factors (list): 选中的因子列表
            
        Returns:
            dict: IC分析结果，包含每个因子的IC值和统计信息
        """
        try:
            start_time = os.times().elapsed
            
            logger.info("开始IC分析...")
            
            # 获取因子列和收益率列
            factor_cols = [col for col in training_data.columns if col.startswith('factor_')]
            return_cols = [col for col in training_data.columns if col.startswith('return_15d')]
            
            logger.info(f"找到 {len(factor_cols)} 个因子列")
            logger.info(f"找到 {len(return_cols)} 个收益率列")
            
            if not factor_cols or not return_cols:
                raise ValueError("未找到因子列或收益率列")
            
            # 进行IC分析
            ic_results = self._calculate_ic_values(training_data, factor_cols, return_cols)
            
            # 保存IC结果
            self._save_ic_results(ic_results)
            
            # 生成IC报告
            self._generate_ic_report(ic_results)
            
            end_time = os.times().elapsed
            log_performance_stats("IC分析", start_time, end_time, 
                                f"成功分析{len(ic_results)}个因子-收益率组合")
            
            return ic_results
            
        except Exception as e:
            logger.error(f"IC分析失败: {str(e)}")
            raise
    
    def _calculate_ic_values(self, training_data, factor_cols, return_cols):
        """计算IC值
        
        Args:
            training_data (pd.DataFrame): 训练数据
            factor_cols (list): 因子列名列表
            return_cols (list): 收益率列名列表
            
        Returns:
            dict: IC分析结果
        """
        ic_results = {}
        
        for factor_col in tqdm(factor_cols, desc="IC分析"):
            factor_name = factor_col.replace('factor_', '')
            
            for return_col in return_cols:
                return_period = return_col.replace('return_', '').replace('d', '')
                
                # 计算IC - 只使用有效数据（非NaN值）
                valid_data = training_data[[factor_col, return_col]].dropna()
                if len(valid_data) > 10:  # 至少需要10个有效样本
                    # 皮尔逊相关系数 - 衡量线性关系
                    ic_value = valid_data[factor_col].corr(valid_data[return_col])
                    # 斯皮尔曼相关系数 - 衡量单调关系
                    spearman_ic = valid_data[factor_col].corr(valid_data[return_col], method='spearman')
                    
                    ic_results[f"{factor_name}_{return_period}d"] = {
                        'pearson_ic': ic_value,
                        'spearman_ic': spearman_ic,
                        'sample_size': len(valid_data)
                    }
                    
                    logger.info(f"  {factor_name} - {return_period}天收益率IC: {ic_value:.4f} (Spearman: {spearman_ic:.4f})")
                else:
                    logger.warning(f"因子 {factor_name} 和收益率 {return_col} 有效数据不足，跳过IC计算。")
        
        return ic_results
    
    def _save_ic_results(self, ic_results):
        """保存IC结果
        
        Args:
            ic_results (dict): IC分析结果
        """
        try:
            timestamp = get_timestamp()
            results_file = os.path.join(self.data_dir, f'ic_results_batch_{timestamp}.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(ic_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"IC结果已保存到: {results_file}")
            
        except Exception as e:
            logger.error(f"保存IC结果失败: {str(e)}")
    
    def _generate_ic_report(self, ic_results):
        """生成IC报告
        
        Args:
            ic_results (dict): IC分析结果
        """
        try:
            # 创建IC报告DataFrame
            report_data = []
            for factor_return, metrics in ic_results.items():
                factor_name, return_period = factor_return.rsplit('_', 1)
                report_data.append({
                    'factor_name': factor_name,
                    'return_period': return_period,
                    'pearson_ic': metrics['pearson_ic'],
                    'spearman_ic': metrics['spearman_ic'],
                    'sample_size': metrics['sample_size']
                })
            
            report_df = pd.DataFrame(report_data)
            
            # 保存报告
            timestamp = get_timestamp()
            report_file = os.path.join(self.data_dir, f'ic_report_batch_{timestamp}.csv')
            report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
            logger.info(f"IC报告已保存到: {report_file}")
            
            # 显示摘要
            self._display_ic_summary(report_df)
            
        except Exception as e:
            logger.error(f"生成IC报告失败: {str(e)}")
    
    def _display_ic_summary(self, report_df):
        """显示IC分析摘要
        
        Args:
            report_df (pd.DataFrame): IC报告数据框
        """
        try:
            print("\n=== IC分析结果摘要 ===")
            print(f"总因子数: {len(report_df['factor_name'].unique())}")
            print(f"平均Pearson IC: {report_df['pearson_ic'].mean():.4f}")
            print(f"平均Spearman IC: {report_df['spearman_ic'].mean():.4f}")
            
            # 显示前10个最佳因子（包含描述信息）
            best_factors = report_df.nlargest(10, 'pearson_ic')
            print("\n=== 前10个最佳因子 ===")
            
            # 获取因子描述信息
            try:
                for i, (_, row) in enumerate(best_factors.iterrows(), 1):
                    factor_name = row['factor_name']
                    ic_value = row['pearson_ic']
                    
                    # 获取因子描述
                    factor_info = self.factor_lib.get_factor_info(factor_name)
                    description = factor_info.get('description', '暂无描述') if factor_info else '暂无描述'
                    category = factor_info.get('category', '未知类别') if factor_info else '未知类别'
                    source = factor_info.get('source', '未知来源') if factor_info else '未知来源'
                    
                    print(f"{i:2d}. {factor_name}: {ic_value:.4f}")
                    print(f"    来源: {source}")
                    print(f"    类别: {category}")
                    print(f"    描述: {description}")
                    print()
                    
            except Exception as e:
                # 如果获取描述失败，只显示基本信息
                logger.warning(f"获取因子描述失败: {e}")
                for i, (_, row) in enumerate(best_factors.iterrows(), 1):
                    print(f"{i:2d}. {row['factor_name']}: {row['pearson_ic']:.4f}")
            
        except Exception as e:
            logger.error(f"显示IC摘要失败: {str(e)}")
    
    def get_ic_statistics(self, ic_results):
        """获取IC统计信息
        
        Args:
            ic_results (dict): IC分析结果
            
        Returns:
            dict: IC统计信息
        """
        if not ic_results:
            return {}
        
        # 提取所有IC值
        pearson_ics = [metrics['pearson_ic'] for metrics in ic_results.values()]
        spearman_ics = [metrics['spearman_ic'] for metrics in ic_results.values()]
        sample_sizes = [metrics['sample_size'] for metrics in ic_results.values()]
        
        return {
            'total_factors': len(ic_results),
            'pearson_ic': {
                'mean': np.mean(pearson_ics),
                'std': np.std(pearson_ics),
                'min': np.min(pearson_ics),
                'max': np.max(pearson_ics),
                'median': np.median(pearson_ics)
            },
            'spearman_ic': {
                'mean': np.mean(spearman_ics),
                'std': np.std(spearman_ics),
                'min': np.min(spearman_ics),
                'max': np.max(spearman_ics),
                'median': np.median(spearman_ics)
            },
            'sample_size': {
                'mean': np.mean(sample_sizes),
                'min': np.min(sample_sizes),
                'max': np.max(sample_sizes)
            },
            'positive_ic_count': sum(1 for ic in pearson_ics if ic > 0),
            'negative_ic_count': sum(1 for ic in pearson_ics if ic < 0),
            'zero_ic_count': sum(1 for ic in pearson_ics if ic == 0)
        }
    
    def get_top_factors(self, ic_results, top_n=10, metric='pearson_ic'):
        """获取最佳因子
        
        Args:
            ic_results (dict): IC分析结果
            top_n (int): 返回前N个因子
            metric (str): 排序指标 ('pearson_ic' 或 'spearman_ic')
            
        Returns:
            list: 最佳因子列表
        """
        if not ic_results:
            return []
        
        # 按指定指标排序
        sorted_factors = sorted(
            ic_results.items(),
            key=lambda x: abs(x[1][metric]),  # 按绝对值排序
            reverse=True
        )
        
        return sorted_factors[:top_n]
    
    def validate_ic_results(self, ic_results):
        """验证IC结果
        
        Args:
            ic_results (dict): IC分析结果
            
        Returns:
            dict: 验证结果
        """
        if not ic_results:
            return {'valid': False, 'error': 'IC结果为空'}
        
        valid_count = 0
        invalid_count = 0
        errors = []
        
        for factor_name, metrics in ic_results.items():
            try:
                # 检查必要的键
                required_keys = ['pearson_ic', 'spearman_ic', 'sample_size']
                if not all(key in metrics for key in required_keys):
                    invalid_count += 1
                    errors.append(f"{factor_name}: 缺少必要的键")
                    continue
                
                # 检查数值有效性
                if not isinstance(metrics['pearson_ic'], (int, float)):
                    invalid_count += 1
                    errors.append(f"{factor_name}: Pearson IC不是数值")
                    continue
                
                if not isinstance(metrics['spearman_ic'], (int, float)):
                    invalid_count += 1
                    errors.append(f"{factor_name}: Spearman IC不是数值")
                    continue
                
                if not isinstance(metrics['sample_size'], int):
                    invalid_count += 1
                    errors.append(f"{factor_name}: 样本大小不是整数")
                    continue
                
                valid_count += 1
                
            except Exception as e:
                invalid_count += 1
                errors.append(f"{factor_name}: 验证错误 - {str(e)}")
        
        return {
            'valid': invalid_count == 0,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'errors': errors[:10]  # 只返回前10个错误
        }
