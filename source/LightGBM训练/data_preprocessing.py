#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理程序
用于处理训练数据，划分训练集、验证集和测试集
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, data_dir=None):
        """
        初始化数据预处理器
        
        Args:
            data_dir: 数据目录路径，如果为None则自动查找
        """
        if data_dir is None:
            # 自动查找IC检测批量目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 从当前文件位置向上查找IC检测批量目录
            search_dirs = [
                os.path.join(current_dir, "../IC检测批量/data"),
                os.path.join(current_dir, "../../source/IC检测批量/data"),
                os.path.join(current_dir, "../../../source/IC检测批量/data"),
                "source/IC检测批量/data",
                "../source/IC检测批量/data",
                "../../source/IC检测批量/data"
            ]
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    self.data_dir = search_dir
                    break
            else:
                # 如果都找不到，使用默认路径
                self.data_dir = "../IC检测批量/data"
        else:
            self.data_dir = data_dir
        self.training_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def find_latest_training_data(self):
        """找到最新的训练数据文件"""
        logger.info("正在查找最新的训练数据文件...")
        
        # 查找所有training_data_batch文件
        training_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith('training_data_batch_') and file.endswith('.csv'):
                training_files.append(file)
        
        if not training_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到训练数据文件")
        
        # 按文件名排序，获取最新的文件
        training_files.sort(reverse=True)
        latest_file = training_files[0]
        
        logger.info(f"找到最新的训练数据文件: {latest_file}")
        return os.path.join(self.data_dir, latest_file)
    
    def load_training_data(self, file_path):
        """
        加载训练数据
        
        Args:
            file_path: 数据文件路径
        """
        logger.info(f"正在加载训练数据: {file_path}")
        
        try:
            # 读取CSV文件
            self.training_data = pd.read_csv(file_path)
            logger.info(f"成功加载数据，形状: {self.training_data.shape}")
            
            # 显示数据基本信息
            logger.info(f"数据列名: {list(self.training_data.columns)}")
            logger.info(f"数据时间范围: {self.training_data['date'].min()} 到 {self.training_data['date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False
    
    def prepare_features_and_target(self):
        """
        准备特征和目标变量
        """
        logger.info("正在准备特征和目标变量...")
        
        # 目标变量
        target_col = 'return_15d'
        
        # 特征变量（使用现有的因子列名）
        feature_cols = [
            'factor_CUSTOM_PRICE_SCORE_FACTOR',
            'factor_CUSTOM_ZHANGTING_SCORE_FACTOR', 
            'factor_CUSTOM_VOLUME_SCORE_FACTOR'
        ]
        
        # 检查列是否存在
        missing_cols = []
        for col in feature_cols + [target_col]:
            if col not in self.training_data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.error(f"缺少以下列: {missing_cols}")
            logger.info(f"可用的列: {list(self.training_data.columns)}")
            return False
        
        # 选择特征和目标变量
        self.features = self.training_data[feature_cols].copy()
        self.target = self.training_data[target_col].copy()
        
        # 处理缺失值
        logger.info("正在处理缺失值...")
        self.features = self.features.fillna(0)  # 用0填充因子缺失值
        self.target = self.target.fillna(0)      # 用0填充目标变量缺失值
        
        # 移除无穷大值
        self.features = self.features.replace([np.inf, -np.inf], 0)
        self.target = self.target.replace([np.inf, -np.inf], 0)
        
        logger.info(f"特征数据形状: {self.features.shape}")
        logger.info(f"目标变量形状: {self.target.shape}")
        logger.info(f"特征列: {list(self.features.columns)}")
        
        return True
    
    def split_data_by_time(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        按时间顺序划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
        """
        logger.info("正在按时间顺序划分数据集...")
        
        # 确保比例总和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"比例总和不为1，自动调整: {total_ratio}")
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        # 按日期排序
        data_with_date = self.training_data.copy()
        data_with_date['date'] = pd.to_datetime(data_with_date['date'])
        data_with_date = data_with_date.sort_values('date')
        
        # 计算分割点
        total_samples = len(data_with_date)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # 划分数据
        train_indices = data_with_date.index[:train_end]
        val_indices = data_with_date.index[train_end:val_end]
        test_indices = data_with_date.index[val_end:]
        
        # 提取对应的特征和目标
        self.train_data = {
            'X': self.features.loc[train_indices],
            'y': self.target.loc[train_indices],
            'date': data_with_date.loc[train_indices, 'date']
        }
        
        self.val_data = {
            'X': self.features.loc[val_indices],
            'y': self.target.loc[val_indices],
            'date': data_with_date.loc[val_indices, 'date']
        }
        
        self.test_data = {
            'X': self.features.loc[test_indices],
            'y': self.target.loc[test_indices],
            'date': data_with_date.loc[test_indices, 'date']
        }
        
        logger.info(f"训练集大小: {len(self.train_data['X'])}")
        logger.info(f"验证集大小: {len(self.val_data['X'])}")
        logger.info(f"测试集大小: {len(self.test_data['X'])}")
        
        # 显示时间范围
        logger.info(f"训练集时间范围: {self.train_data['date'].min()} 到 {self.train_data['date'].max()}")
        logger.info(f"验证集时间范围: {self.val_data['date'].min()} 到 {self.val_data['date'].max()}")
        logger.info(f"测试集时间范围: {self.test_data['date'].min()} 到 {self.test_data['date'].max()}")
        
        return True
    
    def save_processed_data(self, output_dir="data"):
        """
        保存处理后的数据
        
        Args:
            output_dir: 输出目录
        """
        logger.info("正在保存处理后的数据...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集
        train_df = self.train_data['X'].copy()
        train_df['return_15d'] = self.train_data['y']
        train_df['date'] = self.train_data['date']
        train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        
        # 保存验证集
        val_df = self.val_data['X'].copy()
        val_df['return_15d'] = self.val_data['y']
        val_df['date'] = self.val_data['date']
        val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
        
        # 保存测试集
        test_df = self.test_data['X'].copy()
        test_df['return_15d'] = self.test_data['y']
        test_df['date'] = self.test_data['date']
        test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        # 保存数据统计信息
        stats = {
            'total_samples': len(self.training_data),
            'train_samples': len(self.train_data['X']),
            'val_samples': len(self.val_data['X']),
            'test_samples': len(self.test_data['X']),
            'feature_columns': list(self.features.columns),
            'target_column': 'return_15d',
            'train_date_range': {
                'start': str(self.train_data['date'].min()),
                'end': str(self.train_data['date'].max())
            },
            'val_date_range': {
                'start': str(self.val_data['date'].min()),
                'end': str(self.val_data['date'].max())
            },
            'test_date_range': {
                'start': str(self.test_data['date'].min()),
                'end': str(self.test_data['date'].max())
            },
            'processing_time': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到 {output_dir} 目录")
        logger.info(f"训练集: {len(self.train_data['X'])} 样本")
        logger.info(f"验证集: {len(self.val_data['X'])} 样本")
        logger.info(f"测试集: {len(self.test_data['X'])} 样本")
        
        return True
    
    def get_data_summary(self):
        """获取数据摘要信息"""
        if self.training_data is None:
            return "数据未加载"
        
        summary = {
            '原始数据形状': self.training_data.shape,
            '特征列': list(self.features.columns) if hasattr(self, 'features') else [],
            '目标变量': 'return_15d',
            '训练集大小': len(self.train_data['X']) if self.train_data else 0,
            '验证集大小': len(self.val_data['X']) if self.val_data else 0,
            '测试集大小': len(self.test_data['X']) if self.test_data else 0,
        }
        
        return summary

def main():
    """主函数"""
    logger.info("开始数据预处理...")
    
    try:
        # 创建数据预处理器
        preprocessor = DataPreprocessor()
        
        # 查找最新的训练数据文件
        latest_file = preprocessor.find_latest_training_data()
        
        # 加载数据
        if not preprocessor.load_training_data(latest_file):
            logger.error("数据加载失败")
            return False
        
        # 准备特征和目标变量
        if not preprocessor.prepare_features_and_target():
            logger.error("特征准备失败")
            return False
        
        # 划分数据集
        if not preprocessor.split_data_by_time():
            logger.error("数据划分失败")
            return False
        
        # 保存处理后的数据
        if not preprocessor.save_processed_data():
            logger.error("数据保存失败")
            return False
        
        # 显示数据摘要
        summary = preprocessor.get_data_summary()
        logger.info("数据预处理完成!")
        logger.info(f"数据摘要: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据预处理失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("数据预处理成功完成!")
    else:
        print("数据预处理失败，请查看日志文件。")
