# -*- coding: utf-8 -*-
"""
内存监控模块
实时监控系统内存使用情况，防止内存溢出
"""

import psutil
import gc
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控类 - 实时监控系统内存使用情况，防止内存溢出"""
    
    def __init__(self, max_memory_gb=4):
        """
        初始化内存监控器
        
        Args:
            max_memory_gb (int): 最大内存使用限制(GB)
        """
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024  # 转换为字节
        self.initial_memory = psutil.virtual_memory().used  # 记录初始内存使用量
        
    def check_memory(self):
        """检查当前内存使用情况
        
        Returns:
            dict: 包含内存使用信息的字典
                - used_gb: 程序使用的内存(GB)
                - total_gb: 系统总内存使用(GB)
                - percentage: 内存使用百分比
                - is_safe: 是否安全（未超过限制）
        """
        current_memory = psutil.virtual_memory().used
        used_gb = (current_memory - self.initial_memory) / (1024**3)  # 计算程序使用的内存
        total_gb = current_memory / (1024**3)  # 系统总内存使用
        
        return {
            'used_gb': round(used_gb, 2),
            'total_gb': round(total_gb, 2),
            'percentage': round((current_memory / psutil.virtual_memory().total) * 100, 1),
            'is_safe': current_memory < self.max_memory_bytes
        }
    
    def force_gc(self):
        """强制垃圾回收 - 释放未使用的内存"""
        gc.collect()  # 执行Python垃圾回收
        logger.info("执行垃圾回收")
    
    def get_memory_info(self):
        """获取详细内存信息
        
        Returns:
            dict: 详细内存信息
        """
        memory_info = self.check_memory()
        system_memory = psutil.virtual_memory()
        
        return {
            **memory_info,
            'system_total_gb': round(system_memory.total / (1024**3), 2),
            'system_available_gb': round(system_memory.available / (1024**3), 2),
            'system_free_gb': round(system_memory.free / (1024**3), 2)
        }
