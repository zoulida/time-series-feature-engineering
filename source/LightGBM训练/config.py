#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM训练配置文件
统一管理特征列、模型参数等配置
"""

class TrainingConfig:
    """训练配置类"""
    
    # 目标变量
    TARGET_COLUMN = 'return_15d'
    
    # 特征列配置（使用TOP20中排名4-10的因子）
    FEATURE_COLUMNS = [
        'factor_CUSTOM_PRICE_SCORE_FACTOR',
        'factor_CUSTOM_ZHANGTING_SCORE_FACTOR', 
        'factor_CUSTOM_VOLUME_SCORE_FACTOR',
        # 'factor_TSFRESH_mean_abs_change',              # 排名4
        # 'factor_TSFRESH_change_quantiles',             # 排名5
        # 'factor_TSFRESH_welch',                        # 排名6
        # 'factor_ALPHA_VSTD20',                         # 排名7
        # 'factor_ALPHA_IMIN60',                         # 排名8
        # 'factor_TSFRESH_number_crossing_m',            # 排名9
        # 'factor_ALPHA_CNTN30'                          # 排名10
    ]
    
    # 数据划分比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # LightGBM模型参数
    LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }
    
    # 训练参数
    NUM_BOOST_ROUND = 1000
    EARLY_STOPPING_ROUNDS = 50
    LOG_EVALUATION_PERIOD = 100
    
    # IC计算参数
    IC_WINDOW_SIZE = 252  # 滚动窗口大小（交易日）
    MIN_IC_WINDOW_SIZE = 20  # 最小窗口大小
    
    # 输出目录
    DATA_DIR = "data"
    MODEL_RESULTS_DIR = "model_results"
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_feature_columns(cls):
        """获取特征列列表"""
        return cls.FEATURE_COLUMNS.copy()
    
    @classmethod
    def get_target_column(cls):
        """获取目标列名"""
        return cls.TARGET_COLUMN
    
    @classmethod
    def get_data_split_ratios(cls):
        """获取数据划分比例"""
        return {
            'train': cls.TRAIN_RATIO,
            'val': cls.VAL_RATIO,
            'test': cls.TEST_RATIO
        }
    
    @classmethod
    def get_lightgbm_params(cls):
        """获取LightGBM参数"""
        return cls.LIGHTGBM_PARAMS.copy()
    
    @classmethod
    def get_training_params(cls):
        """获取训练参数"""
        return {
            'num_boost_round': cls.NUM_BOOST_ROUND,
            'early_stopping_rounds': cls.EARLY_STOPPING_ROUNDS,
            'log_evaluation_period': cls.LOG_EVALUATION_PERIOD
        }
    
    @classmethod
    def get_ic_params(cls):
        """获取IC计算参数"""
        return {
            'window_size': cls.IC_WINDOW_SIZE,
            'min_window_size': cls.MIN_IC_WINDOW_SIZE
        }
    
    @classmethod
    def validate_config(cls):
        """验证配置的有效性"""
        # 检查数据划分比例
        total_ratio = cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"数据划分比例总和不为1: {total_ratio}")
        
        # 检查特征列是否为空
        if not cls.FEATURE_COLUMNS:
            raise ValueError("特征列不能为空")
        
        # 检查目标列是否为空
        if not cls.TARGET_COLUMN:
            raise ValueError("目标列不能为空")
        
        return True

# 创建全局配置实例
config = TrainingConfig()

# 验证配置
try:
    config.validate_config()
    print("配置验证通过")
except Exception as e:
    print(f"配置验证失败: {e}")
