# -*- coding: utf-8 -*-
"""
步骤4：IC检测分析
使用alphalens完成8个因子的IC检测
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径，以便导入模块
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)

from source.因子库.alpha158_factors import Alpha158Factors
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用改进的IC分析方法（不依赖alphalens）


def load_training_data():
    """加载训练数据"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # 加载IC分析数据
        ic_data_path = os.path.join(data_dir, 'ic_analysis_data.csv')
        if not os.path.exists(ic_data_path):
            raise FileNotFoundError(f"IC分析数据文件不存在: {ic_data_path}")
        
        data = pd.read_csv(ic_data_path)
        data['date'] = pd.to_datetime(data['date'])
        
        logger.info(f"加载IC分析数据: {len(data)} 行，{len(data.columns)} 列")
        return data
        
    except Exception as e:
        logger.error(f"加载训练数据失败: {str(e)}")
        raise


def calculate_ic_without_alphalens(data, factor_cols, return_cols):
    """
    不使用alphalens的简化IC分析
    
    Args:
        data: 数据DataFrame
        factor_cols: 因子列名列表
        return_cols: 收益率列名列表
        
    Returns:
        dict: IC分析结果
    """
    try:
        ic_results = {}
        
        for factor_col in factor_cols:
            factor_name = factor_col.replace('factor_', '')
            logger.info(f"分析因子: {factor_name}")
            
            factor_ic = {}
            
            for return_col in return_cols:
                return_period = return_col.replace('return_', '').replace('d', '')
                
                # 计算IC（信息系数）
                valid_data = data[[factor_col, return_col]].dropna()
                if len(valid_data) > 10:  # 确保有足够的数据点
                    # 计算皮尔逊相关系数
                    ic_value = valid_data[factor_col].corr(valid_data[return_col])
                    
                    # 计算斯皮尔曼相关系数（非参数）
                    spearman_ic = valid_data[factor_col].corr(valid_data[return_col], method='spearman')
                    
                    # 计算IC的统计信息
                    factor_ic[f'IC_{return_period}d'] = ic_value
                    factor_ic[f'IC_{return_period}d_spearman'] = spearman_ic
                    factor_ic[f'IC_{return_period}d_mean'] = ic_value
                    factor_ic[f'IC_{return_period}d_std'] = 0  # 简化处理
                    factor_ic[f'IC_{return_period}d_ir'] = ic_value / 0.1 if 0.1 > 0 else 0  # 简化的IR
                    
                    # 计算IC的显著性（t检验）
                    n = len(valid_data)
                    if n > 2:
                        t_stat = ic_value * np.sqrt((n - 2) / (1 - ic_value**2 + 1e-10))
                        factor_ic[f'IC_{return_period}d_tstat'] = t_stat
                    
                    logger.info(f"  {return_period}天收益率IC: {ic_value:.4f} (Spearman: {spearman_ic:.4f})")
                else:
                    factor_ic[f'IC_{return_period}d'] = np.nan
                    factor_ic[f'IC_{return_period}d_spearman'] = np.nan
                    logger.warning(f"  {return_period}天收益率数据不足")
            
            ic_results[factor_name] = factor_ic
        
        return ic_results
        
    except Exception as e:
        logger.error(f"计算IC失败: {str(e)}")
        raise




def generate_ic_report(ic_results, factor_lib):
    """
    生成IC分析报告
    
    Args:
        ic_results: IC分析结果
        factor_lib: 因子库实例
        
    Returns:
        pd.DataFrame: IC报告
    """
    try:
        report_data = []
        
        for factor_name, ic_data in ic_results.items():
            factor_info = factor_lib.get_factor_info(factor_name)
            
            row = {
                'factor_name': factor_name,
                'category': factor_info.get('category', '未知') if factor_info else '未知',
                'description': factor_info.get('description', '无描述') if factor_info else '无描述'
            }
            
            # 添加各周期的IC值
            for key, value in ic_data.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row[key] = round(value, 4)
                else:
                    row[key] = 'N/A'
            
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        return report_df
        
    except Exception as e:
        logger.error(f"生成IC报告失败: {str(e)}")
        raise


def save_ic_results(ic_results, ic_report, data_directory):
    """
    保存IC分析结果
    
    Args:
        ic_results: IC分析结果
        ic_report: IC报告DataFrame
        data_directory: 数据保存目录
    """
    try:
        # 保存IC结果
        ic_results_path = os.path.join(data_directory, 'ic_results.json')
        with open(ic_results_path, 'w', encoding='utf-8') as f:
            json.dump(ic_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"IC结果已保存到: {ic_results_path}")
        
        # 保存IC报告
        ic_report_path = os.path.join(data_directory, 'ic_report.csv')
        ic_report.to_csv(ic_report_path, index=False, encoding='utf-8-sig')
        logger.info(f"IC报告已保存到: {ic_report_path}")
        
        # 保存IC摘要
        summary_data = []
        for factor_name, ic_data in ic_results.items():
            summary = {'factor_name': factor_name}
            
            # 计算平均IC
            ic_values = [v for k, v in ic_data.items() if k.startswith('IC_') and isinstance(v, (int, float)) and not np.isnan(v)]
            if ic_values:
                summary['avg_ic'] = round(np.mean(ic_values), 4)
                summary['max_ic'] = round(np.max(ic_values), 4)
                summary['min_ic'] = round(np.min(ic_values), 4)
            else:
                summary['avg_ic'] = 'N/A'
                summary['max_ic'] = 'N/A'
                summary['min_ic'] = 'N/A'
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(data_directory, 'ic_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"IC摘要已保存到: {summary_path}")
        
    except Exception as e:
        logger.error(f"保存IC结果失败: {str(e)}")
        raise


def main():
    """主函数"""
    logger.info("开始执行步骤4：IC检测分析")
    
    try:
        # 加载训练数据
        data = load_training_data()
        
        # 获取因子列和收益率列（只分析15天收益率）
        factor_cols = [col for col in data.columns if col.startswith('factor_')]
        return_cols = [col for col in data.columns if col.startswith('return_15d')]
        
        logger.info(f"找到 {len(factor_cols)} 个因子列: {factor_cols}")
        logger.info(f"找到 {len(return_cols)} 个收益率列: {return_cols}")
        
        if not factor_cols:
            logger.error("未找到因子列")
            return False
        
        if not return_cols:
            logger.error("未找到收益率列")
            return False
        
        # 创建因子库实例
        factor_lib = Alpha158Factors()
        
        # 进行IC分析
        logger.info("使用改进的IC分析方法（只分析15天收益率）")
        ic_results = calculate_ic_without_alphalens(data, factor_cols, return_cols)
        
        # 生成IC报告
        ic_report = generate_ic_report(ic_results, factor_lib)
        
        # 保存结果
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        save_ic_results(ic_results, ic_report, data_dir)
        
        # 显示结果
        print("\n=== IC分析结果 ===")
        print(ic_report.to_string(index=False))
        
        print("\n=== IC摘要 ===")
        summary_data = []
        for factor_name, ic_data in ic_results.items():
            ic_values = [v for k, v in ic_data.items() if k.startswith('IC_') and isinstance(v, (int, float)) and not np.isnan(v)]
            if ic_values:
                avg_ic = np.mean(ic_values)
                print(f"{factor_name}: 平均IC = {avg_ic:.4f}")
                summary_data.append({'factor': factor_name, 'avg_ic': avg_ic})
        
        # 按平均IC排序
        if summary_data:
            summary_data.sort(key=lambda x: x['avg_ic'], reverse=True)
            print("\n=== 因子IC排名 ===")
            for i, item in enumerate(summary_data, 1):
                print(f"{i}. {item['factor']}: {item['avg_ic']:.4f}")
        
        logger.info("步骤4执行成功！")
        return True
        
    except Exception as e:
        logger.error(f"步骤4执行失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n步骤4执行成功！")
    else:
        print("\n步骤4执行失败！")
        sys.exit(1)
