#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练管道主程序
整合数据预处理和LightGBM模型训练
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_preprocessing import DataPreprocessor
from lightgbm_training import LightGBMTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """训练管道类"""
    
    def __init__(self):
        """初始化训练管道"""
        self.preprocessor = None
        self.trainer = None
        self.start_time = datetime.now()
        
    def run_data_preprocessing(self):
        """运行数据预处理"""
        logger.info("="*60)
        logger.info("步骤1: 数据预处理")
        logger.info("="*60)
        
        try:
            # 创建数据预处理器
            self.preprocessor = DataPreprocessor()
            
            # 查找最新的训练数据文件
            latest_file = self.preprocessor.find_latest_training_data()
            logger.info(f"使用数据文件: {latest_file}")
            
            # 加载数据
            if not self.preprocessor.load_training_data(latest_file):
                logger.error("数据加载失败")
                return False
            
            # 准备特征和目标变量
            if not self.preprocessor.prepare_features_and_target():
                logger.error("特征准备失败")
                return False
            
            # 划分数据集
            if not self.preprocessor.split_data_by_time():
                logger.error("数据划分失败")
                return False
            
            # 保存处理后的数据
            if not self.preprocessor.save_processed_data():
                logger.error("数据保存失败")
                return False
            
            logger.info("数据预处理完成!")
            return True
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return False
    
    def run_model_training(self):
        """运行模型训练"""
        logger.info("="*60)
        logger.info("步骤2: LightGBM模型训练")
        logger.info("="*60)
        
        try:
            # 创建训练器
            self.trainer = LightGBMTrainer()
            
            # 加载处理后的数据
            if not self.trainer.load_processed_data():
                logger.error("数据加载失败")
                return False
            
            # 训练模型
            if not self.trainer.train_model():
                logger.error("模型训练失败")
                return False
            
            # 生成预测
            if not self.trainer.make_predictions():
                logger.error("预测生成失败")
                return False
            
            # 计算指标
            if not self.trainer.calculate_metrics():
                logger.error("指标计算失败")
                return False
            
            # 计算IC
            if not self.trainer.calculate_ic():
                logger.error("IC计算失败")
                return False
            
            # 保存结果
            if not self.trainer.save_results():
                logger.error("结果保存失败")
                return False
            
            logger.info("模型训练完成!")
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def generate_final_report(self):
        """生成最终报告"""
        logger.info("="*60)
        logger.info("步骤3: 生成最终报告")
        logger.info("="*60)
        
        try:
            # 计算总耗时
            end_time = datetime.now()
            total_time = end_time - self.start_time
            
            # 生成报告
            report = {
                'pipeline_info': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_duration_seconds': total_time.total_seconds(),
                    'total_duration_minutes': total_time.total_seconds() / 60
                },
                'data_info': {
                    'original_data_shape': self.preprocessor.training_data.shape if self.preprocessor else None,
                    'feature_columns': list(self.preprocessor.features.columns) if self.preprocessor else [],
                    'target_column': 'return_15d',
                    'train_samples': len(self.preprocessor.train_data['X']) if self.preprocessor else 0,
                    'val_samples': len(self.preprocessor.val_data['X']) if self.preprocessor else 0,
                    'test_samples': len(self.preprocessor.test_data['X']) if self.preprocessor else 0
                },
                'model_info': {
                    'model_type': 'LightGBM Regression',
                    'best_iteration': self.trainer.model.best_iteration if self.trainer else None,
                    'feature_importance': self.trainer.feature_importance.to_dict('records') if self.trainer else []
                },
                'performance_metrics': self.trainer.metrics if self.trainer else {},
                'ic_analysis': self.trainer.ic_results if self.trainer else {}
            }
            
            # 保存报告
            import json
            with open('training_pipeline_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 生成文本报告
            with open('training_pipeline_report.txt', 'w', encoding='utf-8') as f:
                f.write("训练管道执行报告\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"执行时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总耗时: {total_time.total_seconds():.2f} 秒 ({total_time.total_seconds()/60:.2f} 分钟)\n\n")
                
                if self.preprocessor:
                    f.write("数据信息:\n")
                    f.write(f"  原始数据形状: {self.preprocessor.training_data.shape}\n")
                    f.write(f"  特征列: {list(self.preprocessor.features.columns)}\n")
                    f.write(f"  目标列: return_15d\n")
                    f.write(f"  训练集样本数: {len(self.preprocessor.train_data['X'])}\n")
                    f.write(f"  验证集样本数: {len(self.preprocessor.val_data['X'])}\n")
                    f.write(f"  测试集样本数: {len(self.preprocessor.test_data['X'])}\n\n")
                
                if self.trainer:
                    f.write("模型信息:\n")
                    f.write(f"  模型类型: LightGBM回归\n")
                    f.write(f"  最佳迭代次数: {self.trainer.model.best_iteration}\n\n")
                    
                    f.write("模型性能:\n")
                    for dataset, metrics in self.trainer.metrics.items():
                        f.write(f"  {dataset.upper()} - RMSE: {metrics['rmse']:.6f}, NRMSE: {metrics['nrmse']:.6f}, R²: {metrics['r2']:.6f}\n")
                    f.write("\n")
                    
                    f.write("IC和IR分析结果（测试集）:\n")
                    test_ic = self.trainer.ic_results['test']
                    f.write(f"  测试集 - 皮尔逊IC: {test_ic['pearson_ic']:.6f}, 皮尔逊IR: {test_ic['pearson_ir']:.6f}\n")
                    f.write(f"  测试集 - 斯皮尔曼IC: {test_ic['spearman_ic']:.6f}, 斯皮尔曼IR: {test_ic['spearman_ir']:.6f}\n")
                    f.write(f"  测试集样本数: {test_ic['sample_size']}, 滚动窗口数: {test_ic['rolling_windows']}\n")
                    f.write("\n")
                    
                    f.write("特征重要性:\n")
                    for _, row in self.trainer.feature_importance.iterrows():
                        f.write(f"  {row['feature']}: {row['importance']:.6f}\n")
            
            logger.info("最终报告生成完成!")
            return True
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            return False
    
    def run_pipeline(self):
        """运行完整的训练管道"""
        logger.info("开始执行训练管道...")
        
        try:
            # 步骤1: 数据预处理
            if not self.run_data_preprocessing():
                logger.error("数据预处理步骤失败")
                return False
            
            # 步骤2: 模型训练
            if not self.run_model_training():
                logger.error("模型训练步骤失败")
                return False
            
            # 步骤3: 生成最终报告
            if not self.generate_final_report():
                logger.error("报告生成步骤失败")
                return False
            
            logger.info("训练管道执行完成!")
            return True
            
        except Exception as e:
            logger.error(f"训练管道执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    print("LightGBM训练管道")
    print("="*50)
    print("目标: 预测return_15d")
    print("特征: CUSTOM_PRICE_SCORE_FACTOR, CUSTOM_ZHANGTING_SCORE_FACTOR, CUSTOM_VOLUME_SCORE_FACTOR")
    print("="*50)
    
    # 创建并运行训练管道
    pipeline = TrainingPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n" + "="*50)
        print("训练管道执行成功!")
        print("="*50)
        print("输出文件:")
        print("  - processed_data/ (处理后的数据)")
        print("  - model_results/ (模型结果)")
        print("  - training_pipeline_report.json (JSON报告)")
        print("  - training_pipeline_report.txt (文本报告)")
        print("  - *.log (日志文件)")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("训练管道执行失败!")
        print("请查看日志文件了解详细错误信息。")
        print("="*50)

if __name__ == "__main__":
    main()
