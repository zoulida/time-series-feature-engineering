# -*- coding: utf-8 -*-
"""
工具函数模块
提供通用的工具函数和辅助方法
"""

import os
import glob
import logging
import psutil
from datetime import datetime

logger = logging.getLogger(__name__)


def cleanup_old_files(data_dir, file_patterns):
    """清理旧文件
    
    Args:
        data_dir (str): 数据目录路径
        file_patterns (list): 要清理的文件模式列表
    """
    try:
        cleaned_files = []
        for pattern in file_patterns:
            files = glob.glob(os.path.join(data_dir, pattern))
            for file_path in files:
                try:
                    os.remove(file_path)
                    cleaned_files.append(os.path.basename(file_path))
                except Exception as e:
                    logger.warning(f"无法删除文件 {file_path}: {e}")
        
        if cleaned_files:
            logger.info(f"已清理 {len(cleaned_files)} 个旧文件: {', '.join(cleaned_files[:5])}{'...' if len(cleaned_files) > 5 else ''}")
        else:
            logger.info("没有找到需要清理的旧文件")
            
    except Exception as e:
        logger.warning(f"清理旧文件时出错: {e}")


def check_dependencies(memory_monitor):
    """检查系统依赖项
    
    Args:
        memory_monitor: 内存监控器实例
        
    Returns:
        bool: 依赖项检查是否通过
    """
    logger.info("检查依赖项...")
    
    # 检查内存
    memory_info = memory_monitor.check_memory()
    logger.info(f"内存使用: {memory_info['used_gb']}GB / {memory_info['total_gb']}GB ({memory_info['percentage']}%)")
    
    if not memory_info['is_safe']:
        logger.warning("内存使用接近限制，建议减少批次大小")
    
    # 检查磁盘空间
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    logger.info(f"可用磁盘空间: {free_gb:.1f}GB")
    
    if free_gb < 2:
        logger.warning("磁盘空间不足，可能影响文件保存")
    
    logger.info("依赖项检查完成")
    return True


def get_timestamp():
    """获取当前时间戳字符串
    
    Returns:
        str: 格式化的时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_directory(directory_path):
    """确保目录存在
    
    Args:
        directory_path (str): 目录路径
    """
    os.makedirs(directory_path, exist_ok=True)


def format_file_size(size_bytes):
    """格式化文件大小
    
    Args:
        size_bytes (int): 文件大小（字节）
        
    Returns:
        str: 格式化后的文件大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"


def log_performance_stats(stage_name, start_time, end_time, additional_info=""):
    """记录性能统计信息
    
    Args:
        stage_name (str): 阶段名称
        start_time (float): 开始时间
        end_time (float): 结束时间
        additional_info (str): 附加信息
    """
    duration = end_time - start_time
    logger.info(f"{stage_name}完成，耗时: {duration:.2f}秒")
    if additional_info:
        logger.info(f"  详细信息: {additional_info}")
