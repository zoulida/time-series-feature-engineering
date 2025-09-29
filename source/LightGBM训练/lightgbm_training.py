#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM回归模型训练程序
用于训练模型预测return_15d，并生成IC分析
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 尝试导入LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM未安装，正在尝试安装...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
        print("LightGBM安装成功!")
    except Exception as e:
        print(f"LightGBM安装失败: {e}")
        print("请手动安装: pip install lightgbm")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightgbm_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LightGBMTrainer:
    """LightGBM训练器类"""
    
    def __init__(self, data_dir="data"):
        """
        初始化LightGBM训练器
        
        Args:
            data_dir: 处理后的数据目录
        """
        self.data_dir = data_dir
        self.model = None
        self.feature_importance = None
        self.train_predictions = None
        self.val_predictions = None
        self.test_predictions = None
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM不可用，请先安装")
    
    def load_processed_data(self):
        """加载处理后的数据"""
        logger.info("正在加载处理后的数据...")
        
        try:
            # 加载训练集
            self.train_data = pd.read_csv(os.path.join(self.data_dir, 'train_data.csv'))
            self.train_data['date'] = pd.to_datetime(self.train_data['date'])
            
            # 加载验证集
            self.val_data = pd.read_csv(os.path.join(self.data_dir, 'val_data.csv'))
            self.val_data['date'] = pd.to_datetime(self.val_data['date'])
            
            # 加载测试集
            self.test_data = pd.read_csv(os.path.join(self.data_dir, 'test_data.csv'))
            self.test_data['date'] = pd.to_datetime(self.test_data['date'])
            
            # 准备特征和目标变量
            feature_cols = [
                'factor_CUSTOM_PRICE_SCORE_FACTOR',
                'factor_CUSTOM_ZHANGTING_SCORE_FACTOR', 
                'factor_CUSTOM_VOLUME_SCORE_FACTOR'
            ]
            
            self.X_train = self.train_data[feature_cols]
            self.y_train = self.train_data['return_15d']
            
            self.X_val = self.val_data[feature_cols]
            self.y_val = self.val_data['return_15d']
            
            self.X_test = self.test_data[feature_cols]
            self.y_test = self.test_data['return_15d']
            
            logger.info(f"训练集大小: {self.X_train.shape}")
            logger.info(f"验证集大小: {self.X_val.shape}")
            logger.info(f"测试集大小: {self.X_test.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False
    
    def train_model(self):
        """训练LightGBM模型"""
        logger.info("开始训练LightGBM模型...")
        
        try:
            # 设置LightGBM参数
            params = {
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
            
            # 创建训练数据集
            train_dataset = lgb.Dataset(
                self.X_train, 
                label=self.y_train,
                feature_name=list(self.X_train.columns)
            )
            
            # 创建验证数据集
            val_dataset = lgb.Dataset(
                self.X_val, 
                label=self.y_val,
                reference=train_dataset,
                feature_name=list(self.X_val.columns)
            )
            
            # 训练模型
            self.model = lgb.train(
                params,
                train_dataset,
                valid_sets=[train_dataset, val_dataset],
                valid_names=['train', 'valid'],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # 获取特征重要性
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            logger.info("模型训练完成!")
            logger.info(f"特征重要性:\n{self.feature_importance}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def make_predictions(self):
        """生成预测结果"""
        logger.info("正在生成预测结果...")
        
        try:
            # 训练集预测
            self.train_predictions = self.model.predict(self.X_train, num_iteration=self.model.best_iteration)
            
            # 验证集预测
            self.val_predictions = self.model.predict(self.X_val, num_iteration=self.model.best_iteration)
            
            # 测试集预测
            self.test_predictions = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
            
            logger.info("预测结果生成完成!")
            
            return True
            
        except Exception as e:
            logger.error(f"生成预测失败: {str(e)}")
            return False
    
    def calculate_metrics(self):
        """计算模型评估指标"""
        logger.info("正在计算模型评估指标...")
        
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # 计算训练集指标
            train_mse = mean_squared_error(self.y_train, self.train_predictions)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(self.y_train, self.train_predictions)
            train_r2 = r2_score(self.y_train, self.train_predictions)
            
            # 计算验证集指标
            val_mse = mean_squared_error(self.y_val, self.val_predictions)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(self.y_val, self.val_predictions)
            val_r2 = r2_score(self.y_val, self.val_predictions)
            
            # 计算测试集指标
            test_mse = mean_squared_error(self.y_test, self.test_predictions)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(self.y_test, self.test_predictions)
            test_r2 = r2_score(self.y_test, self.test_predictions)
            
            self.metrics = {
                'train': {
                    'mse': train_mse,
                    'rmse': train_rmse,
                    'mae': train_mae,
                    'r2': train_r2
                },
                'val': {
                    'mse': val_mse,
                    'rmse': val_rmse,
                    'mae': val_mae,
                    'r2': val_r2
                },
                'test': {
                    'mse': test_mse,
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'r2': test_r2
                }
            }
            
            logger.info("模型评估指标:")
            logger.info(f"训练集 - RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
            logger.info(f"验证集 - RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
            logger.info(f"测试集 - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"计算指标失败: {str(e)}")
            return False
    
    def calculate_ic(self):
        """计算IC（信息系数）- 仅分析测试集"""
        logger.info("正在计算IC（仅测试集）...")
        
        try:
            from scipy.stats import pearsonr, spearmanr
            
            # 只计算测试集的IC
            predictions = self.test_predictions
            actual = self.y_test
            
            # 皮尔逊相关系数
            pearson_corr, pearson_p = pearsonr(predictions, actual)
            
            # 斯皮尔曼相关系数
            spearman_corr, spearman_p = spearmanr(predictions, actual)
            
            self.ic_results = {
                'test': {
                    'pearson_ic': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_ic': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'sample_size': len(predictions)
                }
            }
            
            logger.info("IC计算结果（测试集）:")
            logger.info(f"测试集 - 皮尔逊IC: {pearson_corr:.6f} (p={pearson_p:.6f})")
            logger.info(f"测试集 - 斯皮尔曼IC: {spearman_corr:.6f} (p={spearman_p:.6f})")
            logger.info(f"测试集样本数: {len(predictions)}")
            
            return True
            
        except Exception as e:
            logger.error(f"计算IC失败: {str(e)}")
            return False
    
    def save_results(self, output_dir="model_results"):
        """保存训练结果"""
        logger.info("正在保存训练结果...")
        
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存预测结果
            results_data = []
            
            # 训练集结果
            for i, (idx, row) in enumerate(self.train_data.iterrows()):
                results_data.append({
                    'dataset': 'train',
                    'date': row['date'],
                    'actual_return': row['return_15d'],
                    'predicted_return': self.train_predictions[i],
                    'residual': row['return_15d'] - self.train_predictions[i]
                })
            
            # 验证集结果
            for i, (idx, row) in enumerate(self.val_data.iterrows()):
                results_data.append({
                    'dataset': 'val',
                    'date': row['date'],
                    'actual_return': row['return_15d'],
                    'predicted_return': self.val_predictions[i],
                    'residual': row['return_15d'] - self.val_predictions[i]
                })
            
            # 测试集结果
            for i, (idx, row) in enumerate(self.test_data.iterrows()):
                results_data.append({
                    'dataset': 'test',
                    'date': row['date'],
                    'actual_return': row['return_15d'],
                    'predicted_return': self.test_predictions[i],
                    'residual': row['return_15d'] - self.test_predictions[i]
                })
            
            # 保存预测结果CSV
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
            
            # 保存模型指标
            with open(os.path.join(output_dir, 'model_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
            
            # 保存IC结果
            with open(os.path.join(output_dir, 'ic_results.json'), 'w', encoding='utf-8') as f:
                json.dump(self.ic_results, f, ensure_ascii=False, indent=2)
            
            # 保存特征重要性
            self.feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
            
            # 保存模型
            self.model.save_model(os.path.join(output_dir, 'lightgbm_model.txt'))
            
            # 保存训练摘要
            summary = {
                'model_type': 'LightGBM Regression',
                'target_variable': 'return_15d',
                'feature_variables': list(self.X_train.columns),
                'training_samples': len(self.X_train),
                'validation_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'best_iteration': self.model.best_iteration,
                'training_time': datetime.now().isoformat(),
                'metrics': self.metrics,
                'ic_results': self.ic_results
            }
            
            with open(os.path.join(output_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"训练结果已保存到 {output_dir} 目录")
            
            return True
            
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            return False

def main():
    """主函数"""
    logger.info("开始LightGBM模型训练...")
    
    try:
        # 创建训练器
        trainer = LightGBMTrainer()
        
        # 加载数据
        if not trainer.load_processed_data():
            logger.error("数据加载失败")
            return False
        
        # 训练模型
        if not trainer.train_model():
            logger.error("模型训练失败")
            return False
        
        # 生成预测
        if not trainer.make_predictions():
            logger.error("预测生成失败")
            return False
        
        # 计算指标
        if not trainer.calculate_metrics():
            logger.error("指标计算失败")
            return False
        
        # 计算IC
        if not trainer.calculate_ic():
            logger.error("IC计算失败")
            return False
        
        # 保存结果
        if not trainer.save_results():
            logger.error("结果保存失败")
            return False
        
        logger.info("LightGBM模型训练完成!")
        
        # 显示最终结果摘要
        print("\n" + "="*50)
        print("训练结果摘要")
        print("="*50)
        print(f"模型类型: LightGBM回归")
        print(f"目标变量: return_15d")
        print(f"特征变量: {list(trainer.X_train.columns)}")
        print(f"训练样本数: {len(trainer.X_train)}")
        print(f"验证样本数: {len(trainer.X_val)}")
        print(f"测试样本数: {len(trainer.X_test)}")
        print(f"最佳迭代次数: {trainer.model.best_iteration}")
        
        print("\n模型性能:")
        for dataset, metrics in trainer.metrics.items():
            print(f"{dataset.upper()} - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
        
        print("\nIC分析结果（测试集）:")
        test_ic = trainer.ic_results['test']
        print(f"测试集 - 皮尔逊IC: {test_ic['pearson_ic']:.6f}, 斯皮尔曼IC: {test_ic['spearman_ic']:.6f}")
        print(f"测试集样本数: {test_ic['sample_size']}")
        
        print("\n特征重要性:")
        for _, row in trainer.feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.6f}")
        
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"训练过程失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nLightGBM模型训练成功完成!")
    else:
        print("\nLightGBM模型训练失败，请查看日志文件。")
