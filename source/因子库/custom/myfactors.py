# -*- coding: utf-8 -*-
"""
自定义因子库 - MyFactors
实现用户自定义的量化因子

作者: 用户自定义
创建时间: 2024年
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import sys
import os
from decimal import Decimal

# 添加涨停判断模块路径
sys.path.append(r'd:\pythonProject\JQKA\indicator')
try:
    from zhangtingCalculation import limitUp, getAmplitude
except ImportError:
    print("警告: 无法导入涨停判断模块，请检查路径")
    limitUp = None
    getAmplitude = None


class MyFactors:
    """自定义因子库类"""
    
    def __init__(self):
        """初始化自定义因子库"""
        self.factors = {}
        self._initialize_factors()
    
    def _initialize_factors(self):
        """初始化所有自定义因子"""
        self.factors = {
            "ZHANGTING_VOLUME_PRICE_FACTOR": {
                "expression": "ZHANGTING_VOLUME_PRICE_FACTOR($close, $high, $volume, $stock_code)",
                "function_name": "zhangting_volume_price_factor",
                "description": "涨停+成交量+价格复合因子（连续评分）：7日内涨停按次数评分，成交量按超过比例评分，价格按比例评分",
                "category": "复合因子",
                "formula": "等权组合(涨停评分, 成交量评分, 价格评分)",
                "parameters": {
                    "涨停判断周期": 7,
                    "成交量均线周期": 20,
                    "成交量比较周期": 7,
                    "成交量倍数": 1.5,
                    "价格均线周期": 20,
                    "价格上限比例": 0.1
                }
            },
            "ZHANGTING_SCORE_FACTOR": {
                "expression": "ZHANGTING_SCORE_FACTOR($close, $high, $stock_code)",
                "function_name": "zhangting_score_factor",
                "description": "涨停评分因子：7个交易日内涨停次数评分（连续涨停算一次）",
                "category": "涨停因子",
                "formula": "连续涨停算一次，按涨停次数评分",
                "parameters": {
                    "涨停判断周期": 7,
                    "评分范围": "[0, 1]"
                }
            },
            "VOLUME_SCORE_FACTOR": {
                "expression": "VOLUME_SCORE_FACTOR($volume)",
                "function_name": "volume_score_factor",
                "description": "成交量评分因子：成交量20日均线当日大于7天前1.5倍的超过比例评分",
                "category": "成交量因子",
                "formula": "按超过比例评分，使用对数函数平滑",
                "parameters": {
                    "成交量均线周期": 20,
                    "比较周期": 7,
                    "倍数阈值": 1.5,
                    "评分范围": "[0, 1]"
                }
            },
            "PRICE_SCORE_FACTOR": {
                "expression": "PRICE_SCORE_FACTOR($close)",
                "function_name": "price_score_factor",
                "description": "价格评分因子：价格不高于close20日均线10%的比例评分",
                "category": "价格因子",
                "formula": "价格越低，评分越高",
                "parameters": {
                    "价格均线周期": 20,
                    "最大比例": 0.1,
                    "评分范围": "[0, 1]"
                }
            }
        }
    
    def zhangting_volume_price_factor(self, 
                                    close: pd.Series, 
                                    high: pd.Series, 
                                    volume: pd.Series, 
                                    stock_code: str = '000001.SZ') -> pd.Series:
        """
        涨停+成交量+价格复合因子（连续评分）
        
        条件1: 7个交易日内有涨停 - 连续涨停算一次，按涨停次数评分
        条件2: 成交量20日均线当日大于7天前1.5倍 - 按超过比例评分
        条件3: 价格不高于close20日均线10% - 按比例评分
        
        三个条件等权组合生成连续因子值
        
        Parameters:
        -----------
        close : pd.Series
            收盘价序列
        high : pd.Series
            最高价序列
        volume : pd.Series
            成交量序列
        stock_code : str
            股票代码，用于涨停判断
            
        Returns:
        --------
        pd.Series
            因子值序列，连续值，范围[0, 3]，值越大表示条件满足程度越高
        """
        if limitUp is None:
            raise ImportError("涨停判断模块不可用，请检查zhangtingCalculation.py路径")
        
        # 确保数据长度足够
        min_length = max(20, 7)  # 至少需要20天数据
        if len(close) < min_length:
            return pd.Series([0] * len(close), index=close.index)
        
        # 条件1: 7个交易日内涨停评分
        zhangting_score = self._calculate_zhangting_score(close, high, stock_code, period=7)
        
        # 条件2: 成交量条件评分
        volume_score = self._calculate_volume_score(volume, ma_period=20, compare_period=7, multiplier=1.5)
        
        # 条件3: 价格条件评分
        price_score = self._calculate_price_score(close, ma_period=20, max_ratio=0.1)
        
        # 等权组合三个条件（每个条件最高1分）
        result = zhangting_score * ( volume_score + price_score )
        
        return result
    
    def zhangting_score_factor(self, 
                              close: pd.Series, 
                              high: pd.Series, 
                              volume: pd.Series = None,
                              stock_code: str = '000001.SZ') -> pd.Series:
        """
        涨停评分因子：7个交易日内涨停次数评分（连续涨停算一次）
        
        评分规则：
        - 没有涨停：0分
        - 涨停1次：1分
        - 涨停2次：2分
        - 涨停3次及以上：2.5分
        - 最高评分：1分（标准化到0-1范围）
        
        Parameters:
        -----------
        close : pd.Series
            收盘价序列
        high : pd.Series
            最高价序列
        stock_code : str
            股票代码
        period : int
            检查周期，默认7天
            
        Returns:
        --------
        pd.Series
            涨停评分，范围[0, 1]
        """
        if limitUp is None:
            raise ImportError("涨停判断模块不可用，请检查zhangtingCalculation.py路径")
        
        # 验证必需参数
        if close is None or high is None:
            raise ValueError("涨停评分因子需要close和high参数")
        
        # 确保数据长度足够
        min_length = max(20, 7)  # 至少需要20天数据
        if len(close) < min_length:
            return pd.Series([0] * len(close), index=close.index)
        
        return self._calculate_zhangting_score(close, high, stock_code, period=7)
    
    def volume_score_factor(self, 
                           close: pd.Series = None,
                           high: pd.Series = None,
                           volume: pd.Series = None,
                           stock_code: str = '000001.SZ') -> pd.Series:
        """
        成交量评分因子：成交量20日均线当日大于7天前1.5倍的超过比例评分
        
        评分规则：
        - 没有超过1.5倍：0分
        - 超过1.5倍：按超过比例评分，最高1分
        
        Parameters:
        -----------
        volume : pd.Series
            成交量序列
            
        Returns:
        --------
        pd.Series
            成交量评分，范围[0, 1]
        """
        # 验证必需参数
        if volume is None:
            raise ValueError("成交量评分因子需要volume参数")
        
        # 确保数据长度足够
        min_length = 27  # 20天均线 + 7天比较
        if len(volume) < min_length:
            return pd.Series([0] * len(volume), index=volume.index)
        
        return self._calculate_volume_score(volume, ma_period=20, compare_period=7, multiplier=1.5)
    
    def price_score_factor(self, 
                          close: pd.Series = None,
                          high: pd.Series = None,
                          volume: pd.Series = None,
                          stock_code: str = '000001.SZ') -> pd.Series:
        """
        价格评分因子：价格不高于close20日均线10%的比例评分
        
        评分规则：
        - 价格高于20日均线10%：0分
        - 价格在20日均线10%以内：按比例评分，最高1分
        
        Parameters:
        -----------
        close : pd.Series
            收盘价序列
            
        Returns:
        --------
        pd.Series
            价格评分，范围[0, 1]
        """
        # 验证必需参数
        if close is None:
            raise ValueError("价格评分因子需要close参数")
        
        # 确保数据长度足够
        min_length = 20  # 20天均线
        if len(close) < min_length:
            return pd.Series([0] * len(close), index=close.index)
        
        return self._calculate_price_score(close, ma_period=20, max_ratio=0.1)
    
    def _calculate_zhangting_score(self, close: pd.Series, high: pd.Series, 
                                  stock_code: str, period: int = 7) -> pd.Series:
        """
        计算涨停评分：连续涨停算一次，按涨停次数评分
        
        评分规则：
        - 没有涨停：0分
        - 涨停1次：1分
        - 涨停2次：2分
        - 涨停3次及以上：2.5分
        - 最高评分：1分（标准化到0-1范围）
        
        Parameters:
        -----------
        close : pd.Series
            收盘价序列
        high : pd.Series
            最高价序列
        stock_code : str
            股票代码
        period : int
            检查周期
            
        Returns:
        --------
        pd.Series
            涨停评分，范围[0, 1]
        """
        result = pd.Series(0.0, index=close.index)
        
        for i in range(period, len(close)):
            # 检查过去period天的涨停情况
            window_close = close.iloc[i-period:i]
            window_high = high.iloc[i-period:i]
            
            # 统计涨停次数（连续涨停算一次）
            zhangting_count = 0
            in_zhangting_sequence = False
            
            for j in range(len(window_close)):
                # 计算涨停价
                zhangting_price = limitUp(window_close.iloc[j], stock_code)
                # 检查最高价是否达到涨停价
                is_zhangting = window_high.iloc[j] >= float(zhangting_price)
                
                if is_zhangting and not in_zhangting_sequence:
                    # 开始新的涨停序列
                    zhangting_count += 1
                    in_zhangting_sequence = True
                elif not is_zhangting:
                    # 涨停序列结束
                    in_zhangting_sequence = False
            
            # 根据涨停次数计算评分
            if zhangting_count == 0:
                score = 0.0
            elif zhangting_count == 1:
                score = 1.0
            elif zhangting_count == 2:
                score = 2.0
            else:  # 3次及以上
                score = 2.5
            
            # 标准化到0-1范围（最高2.5分对应1.0）
            result.iloc[i] = min(score / 2.5, 1.0)
        
        return result
    
    def _calculate_volume_score(self, volume: pd.Series, ma_period: int = 20, 
                               compare_period: int = 7, multiplier: float = 1.5) -> pd.Series:
        """
        计算成交量评分：按超过比例评分
        
        评分规则：
        - 没有超过1.5倍：0分
        - 超过1.5倍：按超过比例评分，最高1分
        
        Parameters:
        -----------
        volume : pd.Series
            成交量序列
        ma_period : int
            成交量均线周期
        compare_period : int
            比较周期
        multiplier : float
            倍数阈值
            
        Returns:
        --------
        pd.Series
            成交量评分，范围[0, 1]
        """
        result = pd.Series(0.0, index=volume.index)
        
        # 计算成交量20日均线
        volume_ma = volume.rolling(window=ma_period).mean()
        
        for i in range(ma_period + compare_period, len(volume)):
            current_volume_ma = volume_ma.iloc[i]
            compare_volume_ma = volume_ma.iloc[i - compare_period]
            
            # 计算倍数
            ratio = current_volume_ma / compare_volume_ma
            
            if ratio > multiplier:
                # 按超过比例评分，使用对数函数平滑评分
                excess_ratio = (ratio - multiplier) / multiplier  # 超过的比例
                score = min(np.log(1 + excess_ratio) / np.log(2), 1.0)  # 对数函数，最高1分
                result.iloc[i] = score
            else:
                result.iloc[i] = 0.0
        
        return result
    
    def _calculate_price_score(self, close: pd.Series, ma_period: int = 20, 
                              max_ratio: float = 0.1) -> pd.Series:
        """
        计算价格评分：按比例评分
        
        评分规则：
        - 价格高于20日均线10%：0分
        - 价格在20日均线10%以内：按比例评分，最高1分
        
        Parameters:
        -----------
        close : pd.Series
            收盘价序列
        ma_period : int
            价格均线周期
        max_ratio : float
            最大比例阈值
            
        Returns:
        --------
        pd.Series
            价格评分，范围[0, 1]
        """
        result = pd.Series(0.0, index=close.index)
        
        # 计算收盘价20日均线
        close_ma = close.rolling(window=ma_period).mean()
        
        for i in range(ma_period, len(close)):
            current_close = close.iloc[i]
            current_close_ma = close_ma.iloc[i]
            
            # 计算价格相对于均线的比例
            price_ratio = current_close / current_close_ma
            max_price_ratio = 1 + max_ratio
            
            if price_ratio <= max_price_ratio:
                # 在阈值内，按比例评分
                # 价格越低，评分越高
                score = (max_price_ratio - price_ratio) / max_ratio
                result.iloc[i] = min(score, 1.0)
            else:
                # 超过阈值，0分
                result.iloc[i] = 0.0
        
        return result
    
    def get_all_factors(self) -> Dict[str, Dict]:
        """获取所有自定义因子"""
        return self.factors
    
    def get_factor_info(self, factor_name: str) -> Optional[Dict]:
        """获取指定因子的详细信息"""
        return self.factors.get(factor_name)
    
    def get_factor_function_name(self, factor_name: str) -> Optional[str]:
        """获取因子函数名"""
        factor_info = self.get_factor_info(factor_name)
        return factor_info.get('function_name') if factor_info else None
    
    def get_factor_description(self, factor_name: str) -> Optional[str]:
        """获取因子描述"""
        factor_info = self.get_factor_info(factor_name)
        return factor_info.get('description') if factor_info else None
    
    def list_factors(self) -> List[str]:
        """列出所有因子名称"""
        return list(self.factors.keys())
    
    def calculate_factor(self, factor_name: str, **kwargs) -> pd.Series:
        """
        计算指定因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        **kwargs
            因子计算所需的参数
            
        Returns:
        --------
        pd.Series
            因子值序列
        """
        if factor_name not in self.factors:
            raise ValueError(f"因子 {factor_name} 不存在")
        
        function_name = self.get_factor_function_name(factor_name)
        if function_name is None:
            raise ValueError(f"因子 {factor_name} 没有对应的计算函数")
        
        # 获取计算函数
        factor_func = getattr(self, function_name)
        
        # 调用计算函数
        return factor_func(**kwargs)
    
    def add_custom_factor(self, name: str, expression: str, function_name: str, 
                         description: str, category: str = "自定义因子", **kwargs):
        """
        添加自定义因子
        
        Parameters:
        -----------
        name : str
            因子名称
        expression : str
            因子表达式
        function_name : str
            函数名称
        description : str
            因子描述
        category : str
            因子类别
        **kwargs
            其他因子信息
        """
        self.factors[name] = {
            "expression": expression,
            "function_name": function_name,
            "description": description,
            "category": category,
            **kwargs
        }
        print(f"已添加自定义因子: {name}")
    
    def export_factors(self, filename: str = "my_factors_export.csv"):
        """导出因子库到CSV文件"""
        data = []
        for factor_name, info in self.factors.items():
            data.append({
                '因子名称': factor_name,
                '表达式': info.get('expression', ''),
                '函数名': info.get('function_name', ''),
                '描述': info.get('description', ''),
                '类别': info.get('category', ''),
                '公式': info.get('formula', ''),
                '参数': str(info.get('parameters', {}))
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"因子库已导出到: {filename}")
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取因子库统计信息"""
        return {
            "总因子数量": len(self.factors),
            "因子列表": list(self.factors.keys()),
            "按类别统计": self._get_category_stats()
        }
    
    def _get_category_stats(self) -> Dict[str, int]:
        """获取按类别统计的因子数量"""
        category_stats = {}
        for info in self.factors.values():
            category = info.get('category', '未知')
            category_stats[category] = category_stats.get(category, 0) + 1
        return category_stats


def create_my_factors() -> MyFactors:
    """创建自定义因子库的便捷函数"""
    return MyFactors()


def main():
    """主函数，演示自定义因子库的使用"""
    print("=" * 60)
    print("自定义因子库演示")
    print("=" * 60)
    
    # 创建自定义因子库
    try:
        my_factors = MyFactors()
    except Exception as e:
        print(f"创建因子库失败: {e}")
        return
    
    # 获取统计信息
    stats = my_factors.get_statistics()
    print(f"\n总因子数量: {stats['总因子数量']}")
    print(f"因子列表: {stats['因子列表']}")
    print(f"按类别统计: {stats['按类别统计']}")
    
    # 显示因子详情
    print("\n因子详情:")
    for factor_name in my_factors.list_factors():
        info = my_factors.get_factor_info(factor_name)
        print(f"\n{factor_name}:")
        print(f"  描述: {info['description']}")
        print(f"  表达式: {info['expression']}")
        print(f"  类别: {info['category']}")
        if 'parameters' in info:
            print(f"  参数: {info['parameters']}")
    
    # 导出因子库
    df = my_factors.export_factors()
    
    print("\n" + "=" * 60)
    print("自定义因子库演示完成！")
    print("=" * 60)
    
    return my_factors


if __name__ == "__main__":
    main()
