# -*- coding: utf-8 -*-
"""
进度跟踪模块
实时显示工作流执行进度、耗时统计和剩余时间估算
"""

import time
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressTracker:
    """进度跟踪类 - 实时显示工作流执行进度、耗时统计和剩余时间估算"""
    
    def __init__(self, total_steps):
        """
        初始化进度跟踪器
        
        Args:
            total_steps (int): 总步骤数
        """
        self.total_steps = total_steps  # 总步骤数
        self.current_step = 0  # 当前步骤
        self.start_time = time.time()  # 开始时间
        self.checkpoints = []  # 检查点列表（用于错误恢复）
        
    def update(self, step_name, additional_info=""):
        """更新进度信息
        
        Args:
            step_name (str): 当前步骤名称
            additional_info (str): 附加信息（可选）
            
        Returns:
            dict: 进度信息字典
        """
        self.current_step += 1
        elapsed = time.time() - self.start_time  # 已用时间
        remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)  # 估算剩余时间
        
        progress_info = {
            'step': self.current_step,
            'total': self.total_steps,
            'step_name': step_name,
            'elapsed': round(elapsed, 2),
            'remaining': round(remaining, 2),
            'percentage': round((self.current_step / self.total_steps) * 100, 1),
            'additional_info': additional_info,
            'timestamp': datetime.now().isoformat()
        }
        
        self.checkpoints.append(progress_info)  # 保存检查点
        
        logger.info(f"进度: {self.current_step}/{self.total_steps} ({progress_info['percentage']}%) - {step_name}")
        if additional_info:
            logger.info(f"  详细信息: {additional_info}")
            
        return progress_info
    
    def save_checkpoint(self, data, checkpoint_file):
        """保存检查点
        
        Args:
            data: 要保存的数据
            checkpoint_file (str): 检查点文件路径
        """
        checkpoint_data = {
            'progress': {
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'checkpoints': self.checkpoints
            },
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"检查点已保存: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_file):
        """加载检查点
        
        Args:
            checkpoint_file (str): 检查点文件路径
            
        Returns:
            tuple: (数据, 进度信息) 或 (None, None)
        """
        if not os.path.exists(checkpoint_file):
            return None, None
            
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.current_step = checkpoint_data['progress']['current_step']
            self.checkpoints = checkpoint_data['progress']['checkpoints']
            
            logger.info(f"从检查点恢复: 步骤 {self.current_step}/{self.total_steps}")
            return checkpoint_data['data'], checkpoint_data['progress']
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None, None
    
    def get_elapsed_time(self):
        """获取已用时间
        
        Returns:
            float: 已用时间（秒）
        """
        return time.time() - self.start_time
    
    def get_remaining_time(self):
        """获取估算剩余时间
        
        Returns:
            float: 估算剩余时间（秒）
        """
        if self.current_step == 0:
            return 0
        elapsed = self.get_elapsed_time()
        return (elapsed / self.current_step) * (self.total_steps - self.current_step)
    
    def get_progress_summary(self):
        """获取进度摘要
        
        Returns:
            dict: 进度摘要信息
        """
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'percentage': round((self.current_step / self.total_steps) * 100, 1),
            'elapsed_time': round(self.get_elapsed_time(), 2),
            'remaining_time': round(self.get_remaining_time(), 2)
        }
