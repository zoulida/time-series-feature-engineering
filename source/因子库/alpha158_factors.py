# -*- coding: utf-8 -*-
"""
Alpha158因子库
从qlib.contrib.data.handler.Alpha158和qlib.contrib.data.loader.Alpha158DL提取的因子定义

包含158个技术分析因子，分为以下几类：
1. K线形态因子 (K-Bar Factors)
2. 价格因子 (Price Factors) 
3. 滚动统计因子 (Rolling Statistical Factors)
4. 成交量因子 (Volume Factors)

采用直接计算函数的方式，避免复杂的表达式解析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class Alpha158Factors:
    """Alpha158因子库类 - 提供直接计算函数"""
    
    def __init__(self):
        """初始化因子库"""
        self.factors = self._initialize_factors()
    
    def _initialize_factors(self) -> Dict[str, Dict]:
        """初始化所有因子定义"""
        factors = {}
        
        # 1. K线形态因子
        kbar_factors = self._get_kbar_factors()
        factors.update(kbar_factors)
        
        # 2. 价格因子
        price_factors = self._get_price_factors()
        factors.update(price_factors)
        
        # 3. 滚动统计因子
        rolling_factors = self._get_rolling_factors()
        factors.update(rolling_factors)
        
        return factors
    
    def _get_kbar_factors(self) -> Dict[str, Dict]:
        """获取K线形态因子"""
        return {
            "KMID": {
                "function_name": "kbar_mid_ratio",
                "description": "K线实体相对开盘价的比例，衡量收盘价与开盘价的相对关系",
                "category": "K线形态",
                "formula": "(收盘价 - 开盘价) / 开盘价",
                "parameters": ["close", "open"]
            },
            "KLEN": {
                "function_name": "kbar_length_ratio",
                "description": "K线长度相对开盘价的比例，衡量当日价格波动幅度",
                "category": "K线形态",
                "formula": "(最高价 - 最低价) / 开盘价",
                "parameters": ["high", "low", "open"]
            },
            "KMID2": {
                "function_name": "kbar_mid_body_ratio",
                "description": "K线实体占整个K线长度的比例，衡量实体相对影线的强度",
                "category": "K线形态", 
                "formula": "(收盘价 - 开盘价) / (最高价 - 最低价 + 1e-12)",
                "parameters": ["close", "open", "high", "low"]
            },
            "KUP": {
                "function_name": "kbar_upper_shadow_ratio",
                "description": "上影线相对开盘价的比例，衡量上方压力",
                "category": "K线形态",
                "formula": "(最高价 - max(开盘价, 收盘价)) / 开盘价",
                "parameters": ["high", "open", "close"]
            },
            "KUP2": {
                "function_name": "kbar_upper_shadow_body_ratio", 
                "description": "上影线占整个K线长度的比例",
                "category": "K线形态",
                "formula": "(最高价 - max(开盘价, 收盘价)) / (最高价 - 最低价 + 1e-12)",
                "parameters": ["high", "open", "close", "low"]
            },
            "KLOW": {
                "function_name": "kbar_lower_shadow_ratio",
                "description": "下影线相对开盘价的比例，衡量下方支撑",
                "category": "K线形态",
                "formula": "(min(开盘价, 收盘价) - 最低价) / 开盘价",
                "parameters": ["open", "close", "low"]
            },
            "KLOW2": {
                "function_name": "kbar_lower_shadow_body_ratio",
                "description": "下影线占整个K线长度的比例", 
                "category": "K线形态",
                "formula": "(min(开盘价, 收盘价) - 最低价) / (最高价 - 最低价 + 1e-12)",
                "parameters": ["open", "close", "low", "high"]
            },
            "KSFT": {
                "function_name": "kbar_soft_ratio",
                "description": "K线软度指标，衡量收盘价在当日价格区间中的位置",
                "category": "K线形态",
                "formula": "(2*收盘价 - 最高价 - 最低价) / 开盘价",
                "parameters": ["close", "high", "low", "open"]
            },
            "KSFT2": {
                "function_name": "kbar_soft_body_ratio",
                "description": "K线软度占整个K线长度的比例",
                "category": "K线形态", 
                "formula": "(2*收盘价 - 最高价 - 最低价) / (最高价 - 最低价 + 1e-12)",
                "parameters": ["close", "high", "low"]
            }
        }
    
    # ==================== K线形态因子计算函数 ====================
    
    def kbar_mid_ratio(self, close: pd.Series, open: pd.Series) -> pd.Series:
        """K线实体相对开盘价的比例"""
        return (close - open) / (open + 1e-12)
    
    def kbar_length_ratio(self, high: pd.Series, low: pd.Series, open: pd.Series) -> pd.Series:
        """K线长度相对开盘价的比例"""
        return (high - low) / (open + 1e-12)
    
    def kbar_mid_body_ratio(self, close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """K线实体占整个K线长度的比例"""
        return (close - open) / (high - low + 1e-12)
    
    def kbar_upper_shadow_ratio(self, high: pd.Series, open: pd.Series, close: pd.Series) -> pd.Series:
        """上影线相对开盘价的比例"""
        return (high - np.maximum(open, close)) / (open + 1e-12)
    
    def kbar_upper_shadow_body_ratio(self, high: pd.Series, open: pd.Series, close: pd.Series, low: pd.Series) -> pd.Series:
        """上影线占整个K线长度的比例"""
        return (high - np.maximum(open, close)) / (high - low + 1e-12)
    
    def kbar_lower_shadow_ratio(self, open: pd.Series, close: pd.Series, low: pd.Series) -> pd.Series:
        """下影线相对开盘价的比例"""
        return (np.minimum(open, close) - low) / (open + 1e-12)
    
    def kbar_lower_shadow_body_ratio(self, open: pd.Series, close: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
        """下影线占整个K线长度的比例"""
        return (np.minimum(open, close) - low) / (high - low + 1e-12)
    
    def kbar_soft_ratio(self, close: pd.Series, high: pd.Series, low: pd.Series, open: pd.Series) -> pd.Series:
        """K线软度指标"""
        return (2 * close - high - low) / (open + 1e-12)
    
    def kbar_soft_body_ratio(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """K线软度占整个K线长度的比例"""
        return (2 * close - high - low) / (high - low + 1e-12)
    
    def _get_price_factors(self) -> Dict[str, Dict]:
        """获取价格因子"""
        factors = {}
        windows = [0]  # Alpha158默认只使用当前价格
        features = ["OPEN", "HIGH", "LOW", "VWAP"]
        
        for feature in features:
            for window in windows:
                if window == 0:
                    func_name = f"price_{feature.lower()}_current_ratio"
                    desc = f"当前{feature}相对收盘价的比例"
                    params = [feature.lower(), "close"]
                else:
                    func_name = f"price_{feature.lower()}_{window}d_ratio"
                    desc = f"{window}天前{feature}相对当前收盘价的比例"
                    params = [feature.lower(), "close"]
                
                factors[f"{feature}{window}"] = {
                    "function_name": func_name,
                    "description": desc,
                    "category": "价格因子",
                    "formula": f"{feature} / 收盘价" if window == 0 else f"{window}天前{feature} / 当前收盘价",
                    "parameters": params
                }
        
        return factors
    
    # ==================== 价格因子计算函数 ====================
    
    def price_open_current_ratio(self, open: pd.Series, close: pd.Series) -> pd.Series:
        """当前开盘价相对收盘价的比例"""
        return open / (close + 1e-12)
    
    def price_high_current_ratio(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """当前最高价相对收盘价的比例"""
        return high / (close + 1e-12)
    
    def price_low_current_ratio(self, low: pd.Series, close: pd.Series) -> pd.Series:
        """当前最低价相对收盘价的比例"""
        return low / (close + 1e-12)
    
    def price_vwap_current_ratio(self, vwap: pd.Series, close: pd.Series) -> pd.Series:
        """当前VWAP相对收盘价的比例"""
        return vwap / (close + 1e-12)
    
    # ==================== 滚动统计因子计算函数 ====================
    
    def rate_of_change_5d(self, close: pd.Series) -> pd.Series:
        """5天价格变化率"""
        return close.shift(5) / (close + 1e-12)
    
    def rate_of_change_10d(self, close: pd.Series) -> pd.Series:
        """10天价格变化率"""
        return close.shift(10) / (close + 1e-12)
    
    def rate_of_change_20d(self, close: pd.Series) -> pd.Series:
        """20天价格变化率"""
        return close.shift(20) / (close + 1e-12)
    
    def rate_of_change_30d(self, close: pd.Series) -> pd.Series:
        """30天价格变化率"""
        return close.shift(30) / (close + 1e-12)
    
    def rate_of_change_60d(self, close: pd.Series) -> pd.Series:
        """60天价格变化率"""
        return close.shift(60) / (close + 1e-12)
    
    def moving_average_5d(self, close: pd.Series) -> pd.Series:
        """5天移动平均相对当前价格"""
        return close.rolling(5).mean() / (close + 1e-12)
    
    def moving_average_10d(self, close: pd.Series) -> pd.Series:
        """10天移动平均相对当前价格"""
        return close.rolling(10).mean() / (close + 1e-12)
    
    def moving_average_20d(self, close: pd.Series) -> pd.Series:
        """20天移动平均相对当前价格"""
        return close.rolling(20).mean() / (close + 1e-12)
    
    def moving_average_30d(self, close: pd.Series) -> pd.Series:
        """30天移动平均相对当前价格"""
        return close.rolling(30).mean() / (close + 1e-12)
    
    def moving_average_60d(self, close: pd.Series) -> pd.Series:
        """60天移动平均相对当前价格"""
        return close.rolling(60).mean() / (close + 1e-12)
    
    def price_std_5d(self, close: pd.Series) -> pd.Series:
        """5天价格标准差相对当前价格"""
        return close.rolling(5).std() / (close + 1e-12)
    
    def price_std_10d(self, close: pd.Series) -> pd.Series:
        """10天价格标准差相对当前价格"""
        return close.rolling(10).std() / (close + 1e-12)
    
    def price_std_20d(self, close: pd.Series) -> pd.Series:
        """20天价格标准差相对当前价格"""
        return close.rolling(20).std() / (close + 1e-12)
    
    def price_std_30d(self, close: pd.Series) -> pd.Series:
        """30天价格标准差相对当前价格"""
        return close.rolling(30).std() / (close + 1e-12)
    
    def price_std_60d(self, close: pd.Series) -> pd.Series:
        """60天价格标准差相对当前价格"""
        return close.rolling(60).std() / (close + 1e-12)
    
    # ==================== 更多滚动统计因子计算函数 ====================
    
    def price_slope_5d(self, close: pd.Series) -> pd.Series:
        """5天价格线性回归斜率"""
        return close.rolling(5).apply(lambda x: self._calculate_slope(x)) / (close + 1e-12)
    
    def price_slope_10d(self, close: pd.Series) -> pd.Series:
        """10天价格线性回归斜率"""
        return close.rolling(10).apply(lambda x: self._calculate_slope(x)) / (close + 1e-12)
    
    def price_slope_20d(self, close: pd.Series) -> pd.Series:
        """20天价格线性回归斜率"""
        return close.rolling(20).apply(lambda x: self._calculate_slope(x)) / (close + 1e-12)
    
    def price_slope_30d(self, close: pd.Series) -> pd.Series:
        """30天价格线性回归斜率"""
        return close.rolling(30).apply(lambda x: self._calculate_slope(x)) / (close + 1e-12)
    
    def price_slope_60d(self, close: pd.Series) -> pd.Series:
        """60天价格线性回归斜率"""
        return close.rolling(60).apply(lambda x: self._calculate_slope(x)) / (close + 1e-12)
    
    def price_rsquare_5d(self, close: pd.Series) -> pd.Series:
        """5天价格线性回归R平方值"""
        return close.rolling(5).apply(lambda x: self._calculate_rsquare(x))
    
    def price_rsquare_10d(self, close: pd.Series) -> pd.Series:
        """10天价格线性回归R平方值"""
        return close.rolling(10).apply(lambda x: self._calculate_rsquare(x))
    
    def price_rsquare_20d(self, close: pd.Series) -> pd.Series:
        """20天价格线性回归R平方值"""
        return close.rolling(20).apply(lambda x: self._calculate_rsquare(x))
    
    def price_rsquare_30d(self, close: pd.Series) -> pd.Series:
        """30天价格线性回归R平方值"""
        return close.rolling(30).apply(lambda x: self._calculate_rsquare(x))
    
    def price_rsquare_60d(self, close: pd.Series) -> pd.Series:
        """60天价格线性回归R平方值"""
        return close.rolling(60).apply(lambda x: self._calculate_rsquare(x))
    
    def price_residual_5d(self, close: pd.Series) -> pd.Series:
        """5天价格线性回归残差"""
        return close.rolling(5).apply(lambda x: self._calculate_resi(x)) / (close + 1e-12)
    
    def price_residual_10d(self, close: pd.Series) -> pd.Series:
        """10天价格线性回归残差"""
        return close.rolling(10).apply(lambda x: self._calculate_resi(x)) / (close + 1e-12)
    
    def price_residual_20d(self, close: pd.Series) -> pd.Series:
        """20天价格线性回归残差"""
        return close.rolling(20).apply(lambda x: self._calculate_resi(x)) / (close + 1e-12)
    
    def price_residual_30d(self, close: pd.Series) -> pd.Series:
        """30天价格线性回归残差"""
        return close.rolling(30).apply(lambda x: self._calculate_resi(x)) / (close + 1e-12)
    
    def price_residual_60d(self, close: pd.Series) -> pd.Series:
        """60天价格线性回归残差"""
        return close.rolling(60).apply(lambda x: self._calculate_resi(x)) / (close + 1e-12)
    
    def _calculate_slope(self, x):
        """计算线性回归斜率"""
        if len(x) < 2:
            return np.nan
        try:
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            return slope
        except:
            return np.nan
    
    def _calculate_rsquare(self, x):
        """计算线性回归R平方值"""
        if len(x) < 2:
            return np.nan
        try:
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            y_pred = slope * y + intercept
            ss_res = np.sum((x - y_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            if ss_tot == 0:
                return np.nan
            return 1 - (ss_res / ss_tot)
        except:
            return np.nan
    
    def _calculate_resi(self, x):
        """计算线性回归残差"""
        if len(x) < 2:
            return np.nan
        try:
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            y_pred = slope * y + intercept
            return np.mean((x - y_pred) ** 2)
        except:
            return np.nan
    
    # ==================== 更多缺失的因子计算函数 ====================
    
    # MAX/MIN因子
    def price_max_5d(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """5天最高价相对当前收盘价"""
        return high.rolling(5).max() / (close + 1e-12)
    
    def price_max_10d(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """10天最高价相对当前收盘价"""
        return high.rolling(10).max() / (close + 1e-12)
    
    def price_max_20d(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """20天最高价相对当前收盘价"""
        return high.rolling(20).max() / (close + 1e-12)
    
    def price_max_30d(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """30天最高价相对当前收盘价"""
        return high.rolling(30).max() / (close + 1e-12)
    
    def price_max_60d(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """60天最高价相对当前收盘价"""
        return high.rolling(60).max() / (close + 1e-12)
    
    def price_min_5d(self, low: pd.Series, close: pd.Series) -> pd.Series:
        """5天最低价相对当前收盘价"""
        return low.rolling(5).min() / (close + 1e-12)
    
    def price_min_10d(self, low: pd.Series, close: pd.Series) -> pd.Series:
        """10天最低价相对当前收盘价"""
        return low.rolling(10).min() / (close + 1e-12)
    
    def price_min_20d(self, low: pd.Series, close: pd.Series) -> pd.Series:
        """20天最低价相对当前收盘价"""
        return low.rolling(20).min() / (close + 1e-12)
    
    def price_min_30d(self, low: pd.Series, close: pd.Series) -> pd.Series:
        """30天最低价相对当前收盘价"""
        return low.rolling(30).min() / (close + 1e-12)
    
    def price_min_60d(self, low: pd.Series, close: pd.Series) -> pd.Series:
        """60天最低价相对当前收盘价"""
        return low.rolling(60).min() / (close + 1e-12)
    
    # 分位数因子
    def price_quantile_upper_5d(self, close: pd.Series) -> pd.Series:
        """5天收盘价上分位数(80%)相对当前价格"""
        return close.rolling(5).quantile(0.8) / (close + 1e-12)
    
    def price_quantile_upper_10d(self, close: pd.Series) -> pd.Series:
        """10天收盘价上分位数(80%)相对当前价格"""
        return close.rolling(10).quantile(0.8) / (close + 1e-12)
    
    def price_quantile_upper_20d(self, close: pd.Series) -> pd.Series:
        """20天收盘价上分位数(80%)相对当前价格"""
        return close.rolling(20).quantile(0.8) / (close + 1e-12)
    
    def price_quantile_upper_30d(self, close: pd.Series) -> pd.Series:
        """30天收盘价上分位数(80%)相对当前价格"""
        return close.rolling(30).quantile(0.8) / (close + 1e-12)
    
    def price_quantile_upper_60d(self, close: pd.Series) -> pd.Series:
        """60天收盘价上分位数(80%)相对当前价格"""
        return close.rolling(60).quantile(0.8) / (close + 1e-12)
    
    def price_quantile_lower_5d(self, close: pd.Series) -> pd.Series:
        """5天收盘价下分位数(20%)相对当前价格"""
        return close.rolling(5).quantile(0.2) / (close + 1e-12)
    
    def price_quantile_lower_10d(self, close: pd.Series) -> pd.Series:
        """10天收盘价下分位数(20%)相对当前价格"""
        return close.rolling(10).quantile(0.2) / (close + 1e-12)
    
    def price_quantile_lower_20d(self, close: pd.Series) -> pd.Series:
        """20天收盘价下分位数(20%)相对当前价格"""
        return close.rolling(20).quantile(0.2) / (close + 1e-12)
    
    def price_quantile_lower_30d(self, close: pd.Series) -> pd.Series:
        """30天收盘价下分位数(20%)相对当前价格"""
        return close.rolling(30).quantile(0.2) / (close + 1e-12)
    
    def price_quantile_lower_60d(self, close: pd.Series) -> pd.Series:
        """60天收盘价下分位数(20%)相对当前价格"""
        return close.rolling(60).quantile(0.2) / (close + 1e-12)
    
    # ==================== 排名因子计算函数 ====================
    
    def price_rank_5d(self, close: pd.Series) -> pd.Series:
        """5天收盘价排名"""
        return close.rolling(5).rank(pct=True)
    
    def price_rank_10d(self, close: pd.Series) -> pd.Series:
        """10天收盘价排名"""
        return close.rolling(10).rank(pct=True)
    
    def price_rank_20d(self, close: pd.Series) -> pd.Series:
        """20天收盘价排名"""
        return close.rolling(20).rank(pct=True)
    
    def price_rank_30d(self, close: pd.Series) -> pd.Series:
        """30天收盘价排名"""
        return close.rolling(30).rank(pct=True)
    
    def price_rank_60d(self, close: pd.Series) -> pd.Series:
        """60天收盘价排名"""
        return close.rolling(60).rank(pct=True)
    
    # ==================== 随机指标计算函数 ====================
    
    def stochastic_5d(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """5天随机指标KDJ的K值"""
        lowest_low = low.rolling(5).min()
        highest_high = high.rolling(5).max()
        return (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    
    def stochastic_10d(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """10天随机指标KDJ的K值"""
        lowest_low = low.rolling(10).min()
        highest_high = high.rolling(10).max()
        return (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    
    def stochastic_20d(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """20天随机指标KDJ的K值"""
        lowest_low = low.rolling(20).min()
        highest_high = high.rolling(20).max()
        return (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    
    def stochastic_30d(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """30天随机指标KDJ的K值"""
        lowest_low = low.rolling(30).min()
        highest_high = high.rolling(30).max()
        return (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    
    def stochastic_60d(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """60天随机指标KDJ的K值"""
        lowest_low = low.rolling(60).min()
        highest_high = high.rolling(60).max()
        return (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    
    # ==================== 指数位置计算函数 ====================
    
    def max_price_index_5d(self, high: pd.Series) -> pd.Series:
        """5天最高价出现位置"""
        return high.rolling(5).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan) / 5
    
    def max_price_index_10d(self, high: pd.Series) -> pd.Series:
        """10天最高价出现位置"""
        return high.rolling(10).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan) / 10
    
    def max_price_index_20d(self, high: pd.Series) -> pd.Series:
        """20天最高价出现位置"""
        return high.rolling(20).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan) / 20
    
    def max_price_index_30d(self, high: pd.Series) -> pd.Series:
        """30天最高价出现位置"""
        return high.rolling(30).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan) / 30
    
    def max_price_index_60d(self, high: pd.Series) -> pd.Series:
        """60天最高价出现位置"""
        return high.rolling(60).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan) / 60
    
    def min_price_index_5d(self, low: pd.Series) -> pd.Series:
        """5天最低价出现位置"""
        return low.rolling(5).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan) / 5
    
    def min_price_index_10d(self, low: pd.Series) -> pd.Series:
        """10天最低价出现位置"""
        return low.rolling(10).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan) / 10
    
    def min_price_index_20d(self, low: pd.Series) -> pd.Series:
        """20天最低价出现位置"""
        return low.rolling(20).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan) / 20
    
    def min_price_index_30d(self, low: pd.Series) -> pd.Series:
        """30天最低价出现位置"""
        return low.rolling(30).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan) / 30
    
    def min_price_index_60d(self, low: pd.Series) -> pd.Series:
        """60天最低价出现位置"""
        return low.rolling(60).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan) / 60
    
    def max_min_index_diff_5d(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """5天最高价和最低价位置差"""
        max_idx = high.rolling(5).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)
        min_idx = low.rolling(5).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)
        return (max_idx - min_idx) / 5
    
    def max_min_index_diff_10d(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """10天最高价和最低价位置差"""
        max_idx = high.rolling(10).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)
        min_idx = low.rolling(10).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)
        return (max_idx - min_idx) / 10
    
    def max_min_index_diff_20d(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """20天最高价和最低价位置差"""
        max_idx = high.rolling(20).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)
        min_idx = low.rolling(20).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)
        return (max_idx - min_idx) / 20
    
    def max_min_index_diff_30d(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """30天最高价和最低价位置差"""
        max_idx = high.rolling(30).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)
        min_idx = low.rolling(30).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)
        return (max_idx - min_idx) / 30
    
    def max_min_index_diff_60d(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """60天最高价和最低价位置差"""
        max_idx = high.rolling(60).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)
        min_idx = low.rolling(60).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)
        return (max_idx - min_idx) / 60
    
    # ==================== 相关性计算函数 ====================
    
    def price_volume_corr_5d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """5天价格与成交量相关性"""
        return close.rolling(5).corr(volume)
    
    def price_volume_corr_10d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """10天价格与成交量相关性"""
        return close.rolling(10).corr(volume)
    
    def price_volume_corr_20d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """20天价格与成交量相关性"""
        return close.rolling(20).corr(volume)
    
    def price_volume_corr_30d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """30天价格与成交量相关性"""
        return close.rolling(30).corr(volume)
    
    def price_volume_corr_60d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """60天价格与成交量相关性"""
        return close.rolling(60).corr(volume)
    
    def price_change_volume_corr_5d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """5天价格变化与成交量变化相关性"""
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        return price_change.rolling(5).corr(volume_change)
    
    def price_change_volume_corr_10d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """10天价格变化与成交量变化相关性"""
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        return price_change.rolling(10).corr(volume_change)
    
    def price_change_volume_corr_20d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """20天价格变化与成交量变化相关性"""
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        return price_change.rolling(20).corr(volume_change)
    
    def price_change_volume_corr_30d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """30天价格变化与成交量变化相关性"""
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        return price_change.rolling(30).corr(volume_change)
    
    def price_change_volume_corr_60d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """60天价格变化与成交量变化相关性"""
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        return price_change.rolling(60).corr(volume_change)
    
    # ==================== 比率因子计算函数 ====================
    
    def up_days_ratio_5d(self, close: pd.Series) -> pd.Series:
        """5天上涨天数比例"""
        return (close > close.shift(1)).rolling(5).mean()
    
    def up_days_ratio_10d(self, close: pd.Series) -> pd.Series:
        """10天上涨天数比例"""
        return (close > close.shift(1)).rolling(10).mean()
    
    def up_days_ratio_20d(self, close: pd.Series) -> pd.Series:
        """20天上涨天数比例"""
        return (close > close.shift(1)).rolling(20).mean()
    
    def up_days_ratio_30d(self, close: pd.Series) -> pd.Series:
        """30天上涨天数比例"""
        return (close > close.shift(1)).rolling(30).mean()
    
    def up_days_ratio_60d(self, close: pd.Series) -> pd.Series:
        """60天上涨天数比例"""
        return (close > close.shift(1)).rolling(60).mean()
    
    def down_days_ratio_5d(self, close: pd.Series) -> pd.Series:
        """5天下跌天数比例"""
        return (close < close.shift(1)).rolling(5).mean()
    
    def down_days_ratio_10d(self, close: pd.Series) -> pd.Series:
        """10天下跌天数比例"""
        return (close < close.shift(1)).rolling(10).mean()
    
    def down_days_ratio_20d(self, close: pd.Series) -> pd.Series:
        """20天下跌天数比例"""
        return (close < close.shift(1)).rolling(20).mean()
    
    def down_days_ratio_30d(self, close: pd.Series) -> pd.Series:
        """30天下跌天数比例"""
        return (close < close.shift(1)).rolling(30).mean()
    
    def down_days_ratio_60d(self, close: pd.Series) -> pd.Series:
        """60天下跌天数比例"""
        return (close < close.shift(1)).rolling(60).mean()
    
    def up_down_days_diff_5d(self, close: pd.Series) -> pd.Series:
        """5天上涨下跌天数差"""
        up_days = (close > close.shift(1)).rolling(5).sum()
        down_days = (close < close.shift(1)).rolling(5).sum()
        return (up_days - down_days) / 5
    
    def up_down_days_diff_10d(self, close: pd.Series) -> pd.Series:
        """10天上涨下跌天数差"""
        up_days = (close > close.shift(1)).rolling(10).sum()
        down_days = (close < close.shift(1)).rolling(10).sum()
        return (up_days - down_days) / 10
    
    def up_down_days_diff_20d(self, close: pd.Series) -> pd.Series:
        """20天上涨下跌天数差"""
        up_days = (close > close.shift(1)).rolling(20).sum()
        down_days = (close < close.shift(1)).rolling(20).sum()
        return (up_days - down_days) / 20
    
    def up_down_days_diff_30d(self, close: pd.Series) -> pd.Series:
        """30天上涨下跌天数差"""
        up_days = (close > close.shift(1)).rolling(30).sum()
        down_days = (close < close.shift(1)).rolling(30).sum()
        return (up_days - down_days) / 30
    
    def up_down_days_diff_60d(self, close: pd.Series) -> pd.Series:
        """60天上涨下跌天数差"""
        up_days = (close > close.shift(1)).rolling(60).sum()
        down_days = (close < close.shift(1)).rolling(60).sum()
        return (up_days - down_days) / 60
    
    def positive_return_ratio_5d(self, close: pd.Series) -> pd.Series:
        """5天正收益比例"""
        returns = close.pct_change()
        return (returns > 0).rolling(5).mean()
    
    def positive_return_ratio_10d(self, close: pd.Series) -> pd.Series:
        """10天正收益比例"""
        returns = close.pct_change()
        return (returns > 0).rolling(10).mean()
    
    def positive_return_ratio_20d(self, close: pd.Series) -> pd.Series:
        """20天正收益比例"""
        returns = close.pct_change()
        return (returns > 0).rolling(20).mean()
    
    def positive_return_ratio_30d(self, close: pd.Series) -> pd.Series:
        """30天正收益比例"""
        returns = close.pct_change()
        return (returns > 0).rolling(30).mean()
    
    def positive_return_ratio_60d(self, close: pd.Series) -> pd.Series:
        """60天正收益比例"""
        returns = close.pct_change()
        return (returns > 0).rolling(60).mean()
    
    def negative_return_ratio_5d(self, close: pd.Series) -> pd.Series:
        """5天负收益比例"""
        returns = close.pct_change()
        return (returns < 0).rolling(5).mean()
    
    def negative_return_ratio_10d(self, close: pd.Series) -> pd.Series:
        """10天负收益比例"""
        returns = close.pct_change()
        return (returns < 0).rolling(10).mean()
    
    def negative_return_ratio_20d(self, close: pd.Series) -> pd.Series:
        """20天负收益比例"""
        returns = close.pct_change()
        return (returns < 0).rolling(20).mean()
    
    def negative_return_ratio_30d(self, close: pd.Series) -> pd.Series:
        """30天负收益比例"""
        returns = close.pct_change()
        return (returns < 0).rolling(30).mean()
    
    def negative_return_ratio_60d(self, close: pd.Series) -> pd.Series:
        """60天负收益比例"""
        returns = close.pct_change()
        return (returns < 0).rolling(60).mean()
    
    def return_diff_ratio_5d(self, close: pd.Series) -> pd.Series:
        """5天收益差比例"""
        returns = close.pct_change()
        positive_returns = (returns > 0).rolling(5).sum()
        total_returns = (returns != 0).rolling(5).sum()
        return positive_returns / (total_returns + 1e-12)
    
    def return_diff_ratio_10d(self, close: pd.Series) -> pd.Series:
        """10天收益差比例"""
        returns = close.pct_change()
        positive_returns = (returns > 0).rolling(10).sum()
        total_returns = (returns != 0).rolling(10).sum()
        return positive_returns / (total_returns + 1e-12)
    
    def return_diff_ratio_20d(self, close: pd.Series) -> pd.Series:
        """20天收益差比例"""
        returns = close.pct_change()
        positive_returns = (returns > 0).rolling(20).sum()
        total_returns = (returns != 0).rolling(20).sum()
        return positive_returns / (total_returns + 1e-12)
    
    def return_diff_ratio_30d(self, close: pd.Series) -> pd.Series:
        """30天收益差比例"""
        returns = close.pct_change()
        positive_returns = (returns > 0).rolling(30).sum()
        total_returns = (returns != 0).rolling(30).sum()
        return positive_returns / (total_returns + 1e-12)
    
    def return_diff_ratio_60d(self, close: pd.Series) -> pd.Series:
        """60天收益差比例"""
        returns = close.pct_change()
        positive_returns = (returns > 0).rolling(60).sum()
        total_returns = (returns != 0).rolling(60).sum()
        return positive_returns / (total_returns + 1e-12)
    
    # ==================== 成交量因子计算函数 ====================
    
    def volume_ma_5d(self, volume: pd.Series) -> pd.Series:
        """5天成交量移动平均相对当前成交量"""
        return volume.rolling(5).mean() / (volume + 1e-12)
    
    def volume_ma_10d(self, volume: pd.Series) -> pd.Series:
        """10天成交量移动平均相对当前成交量"""
        return volume.rolling(10).mean() / (volume + 1e-12)
    
    def volume_ma_20d(self, volume: pd.Series) -> pd.Series:
        """20天成交量移动平均相对当前成交量"""
        return volume.rolling(20).mean() / (volume + 1e-12)
    
    def volume_ma_30d(self, volume: pd.Series) -> pd.Series:
        """30天成交量移动平均相对当前成交量"""
        return volume.rolling(30).mean() / (volume + 1e-12)
    
    def volume_ma_60d(self, volume: pd.Series) -> pd.Series:
        """60天成交量移动平均相对当前成交量"""
        return volume.rolling(60).mean() / (volume + 1e-12)
    
    def volume_std_5d(self, volume: pd.Series) -> pd.Series:
        """5天成交量标准差相对当前成交量"""
        return volume.rolling(5).std() / (volume + 1e-12)
    
    def volume_std_10d(self, volume: pd.Series) -> pd.Series:
        """10天成交量标准差相对当前成交量"""
        return volume.rolling(10).std() / (volume + 1e-12)
    
    def volume_std_20d(self, volume: pd.Series) -> pd.Series:
        """20天成交量标准差相对当前成交量"""
        return volume.rolling(20).std() / (volume + 1e-12)
    
    def volume_std_30d(self, volume: pd.Series) -> pd.Series:
        """30天成交量标准差相对当前成交量"""
        return volume.rolling(30).std() / (volume + 1e-12)
    
    def volume_std_60d(self, volume: pd.Series) -> pd.Series:
        """60天成交量标准差相对当前成交量"""
        return volume.rolling(60).std() / (volume + 1e-12)
    
    def volume_increase_ratio_5d(self, volume: pd.Series) -> pd.Series:
        """5天成交量增加比例"""
        volume_increase = (volume > volume.shift(1)).rolling(5).sum()
        volume_total = (volume != volume.shift(1)).rolling(5).sum()
        return volume_increase / (volume_total + 1e-12)
    
    def volume_increase_ratio_10d(self, volume: pd.Series) -> pd.Series:
        """10天成交量增加比例"""
        volume_increase = (volume > volume.shift(1)).rolling(10).sum()
        volume_total = (volume != volume.shift(1)).rolling(10).sum()
        return volume_increase / (volume_total + 1e-12)
    
    def volume_increase_ratio_20d(self, volume: pd.Series) -> pd.Series:
        """20天成交量增加比例"""
        volume_increase = (volume > volume.shift(1)).rolling(20).sum()
        volume_total = (volume != volume.shift(1)).rolling(20).sum()
        return volume_increase / (volume_total + 1e-12)
    
    def volume_increase_ratio_30d(self, volume: pd.Series) -> pd.Series:
        """30天成交量增加比例"""
        volume_increase = (volume > volume.shift(1)).rolling(30).sum()
        volume_total = (volume != volume.shift(1)).rolling(30).sum()
        return volume_increase / (volume_total + 1e-12)
    
    def volume_increase_ratio_60d(self, volume: pd.Series) -> pd.Series:
        """60天成交量增加比例"""
        volume_increase = (volume > volume.shift(1)).rolling(60).sum()
        volume_total = (volume != volume.shift(1)).rolling(60).sum()
        return volume_increase / (volume_total + 1e-12)
    
    def volume_decrease_ratio_5d(self, volume: pd.Series) -> pd.Series:
        """5天成交量减少比例"""
        volume_decrease = (volume < volume.shift(1)).rolling(5).sum()
        volume_total = (volume != volume.shift(1)).rolling(5).sum()
        return volume_decrease / (volume_total + 1e-12)
    
    def volume_decrease_ratio_10d(self, volume: pd.Series) -> pd.Series:
        """10天成交量减少比例"""
        volume_decrease = (volume < volume.shift(1)).rolling(10).sum()
        volume_total = (volume != volume.shift(1)).rolling(10).sum()
        return volume_decrease / (volume_total + 1e-12)
    
    def volume_decrease_ratio_20d(self, volume: pd.Series) -> pd.Series:
        """20天成交量减少比例"""
        volume_decrease = (volume < volume.shift(1)).rolling(20).sum()
        volume_total = (volume != volume.shift(1)).rolling(20).sum()
        return volume_decrease / (volume_total + 1e-12)
    
    def volume_decrease_ratio_30d(self, volume: pd.Series) -> pd.Series:
        """30天成交量减少比例"""
        volume_decrease = (volume < volume.shift(1)).rolling(30).sum()
        volume_total = (volume != volume.shift(1)).rolling(30).sum()
        return volume_decrease / (volume_total + 1e-12)
    
    def volume_decrease_ratio_60d(self, volume: pd.Series) -> pd.Series:
        """60天成交量减少比例"""
        volume_decrease = (volume < volume.shift(1)).rolling(60).sum()
        volume_total = (volume != volume.shift(1)).rolling(60).sum()
        return volume_decrease / (volume_total + 1e-12)
    
    def volume_diff_ratio_5d(self, volume: pd.Series) -> pd.Series:
        """5天成交量差比例"""
        volume_increase = (volume > volume.shift(1)).rolling(5).sum()
        volume_decrease = (volume < volume.shift(1)).rolling(5).sum()
        volume_total = (volume != volume.shift(1)).rolling(5).sum()
        return (volume_increase - volume_decrease) / (volume_total + 1e-12)
    
    def volume_diff_ratio_10d(self, volume: pd.Series) -> pd.Series:
        """10天成交量差比例"""
        volume_increase = (volume > volume.shift(1)).rolling(10).sum()
        volume_decrease = (volume < volume.shift(1)).rolling(10).sum()
        volume_total = (volume != volume.shift(1)).rolling(10).sum()
        return (volume_increase - volume_decrease) / (volume_total + 1e-12)
    
    def volume_diff_ratio_20d(self, volume: pd.Series) -> pd.Series:
        """20天成交量差比例"""
        volume_increase = (volume > volume.shift(1)).rolling(20).sum()
        volume_decrease = (volume < volume.shift(1)).rolling(20).sum()
        volume_total = (volume != volume.shift(1)).rolling(20).sum()
        return (volume_increase - volume_decrease) / (volume_total + 1e-12)
    
    def volume_diff_ratio_30d(self, volume: pd.Series) -> pd.Series:
        """30天成交量差比例"""
        volume_increase = (volume > volume.shift(1)).rolling(30).sum()
        volume_decrease = (volume < volume.shift(1)).rolling(30).sum()
        volume_total = (volume != volume.shift(1)).rolling(30).sum()
        return (volume_increase - volume_decrease) / (volume_total + 1e-12)
    
    def volume_diff_ratio_60d(self, volume: pd.Series) -> pd.Series:
        """60天成交量差比例"""
        volume_increase = (volume > volume.shift(1)).rolling(60).sum()
        volume_decrease = (volume < volume.shift(1)).rolling(60).sum()
        volume_total = (volume != volume.shift(1)).rolling(60).sum()
        return (volume_increase - volume_decrease) / (volume_total + 1e-12)
    
    def volume_weighted_volatility_5d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """5天成交量加权波动率"""
        returns = close.pct_change()
        weighted_vol = (returns.abs() * volume).rolling(5).sum() / volume.rolling(5).sum()
        return weighted_vol / (returns.abs().rolling(5).mean() + 1e-12)
    
    def volume_weighted_volatility_10d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """10天成交量加权波动率"""
        returns = close.pct_change()
        weighted_vol = (returns.abs() * volume).rolling(10).sum() / volume.rolling(10).sum()
        return weighted_vol / (returns.abs().rolling(10).mean() + 1e-12)
    
    def volume_weighted_volatility_20d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """20天成交量加权波动率"""
        returns = close.pct_change()
        weighted_vol = (returns.abs() * volume).rolling(20).sum() / volume.rolling(20).sum()
        return weighted_vol / (returns.abs().rolling(20).mean() + 1e-12)
    
    def volume_weighted_volatility_30d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """30天成交量加权波动率"""
        returns = close.pct_change()
        weighted_vol = (returns.abs() * volume).rolling(30).sum() / volume.rolling(30).sum()
        return weighted_vol / (returns.abs().rolling(30).mean() + 1e-12)
    
    def volume_weighted_volatility_60d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """60天成交量加权波动率"""
        returns = close.pct_change()
        weighted_vol = (returns.abs() * volume).rolling(60).sum() / volume.rolling(60).sum()
        return weighted_vol / (returns.abs().rolling(60).mean() + 1e-12)
    
    def _get_rolling_factors(self) -> Dict[str, Dict]:
        """获取滚动统计因子"""
        factors = {}
        windows = [5, 10, 20, 30, 60]  # Alpha158默认窗口
        
        # ROC - 变化率因子
        for window in windows:
            factors[f"ROC{window}"] = {
                "function_name": f"rate_of_change_{window}d",
                "description": f"{window}天价格变化率，衡量价格动量",
                "category": "滚动统计",
                "formula": f"{window}天前收盘价 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # MA - 移动平均因子
        for window in windows:
            factors[f"MA{window}"] = {
                "function_name": f"moving_average_{window}d",
                "description": f"{window}天简单移动平均相对当前价格",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价均值 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # STD - 标准差因子
        for window in windows:
            factors[f"STD{window}"] = {
                "function_name": f"price_std_{window}d",
                "description": f"{window}天价格标准差相对当前价格，衡量价格波动性",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价标准差 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # BETA - 斜率因子
        for window in windows:
            factors[f"BETA{window}"] = {
                "function_name": f"price_slope_{window}d",
                "description": f"{window}天价格线性回归斜率，衡量价格趋势强度",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价线性回归斜率 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # RSQR - R平方因子
        for window in windows:
            factors[f"RSQR{window}"] = {
                "function_name": f"price_rsquare_{window}d",
                "description": f"{window}天价格线性回归R平方值，衡量趋势线性度",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价线性回归R平方值",
                "parameters": ["close"]
            }
        
        # RESI - 残差因子
        for window in windows:
            factors[f"RESI{window}"] = {
                "function_name": f"price_residual_{window}d",
                "description": f"{window}天价格线性回归残差，衡量偏离趋势线的程度",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价线性回归残差 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # MAX - 最大值因子
        for window in windows:
            factors[f"MAX{window}"] = {
                "function_name": f"price_max_{window}d",
                "description": f"{window}天最高价相对当前收盘价，衡量价格阻力位",
                "category": "滚动统计",
                "formula": f"过去{window}天最高价 / 当前收盘价",
                "parameters": ["high", "close"]
            }
        
        # MIN - 最小值因子
        for window in windows:
            factors[f"MIN{window}"] = {
                "function_name": f"price_min_{window}d",
                "description": f"{window}天最低价相对当前收盘价，衡量价格支撑位",
                "category": "滚动统计",
                "formula": f"过去{window}天最低价 / 当前收盘价",
                "parameters": ["low", "close"]
            }
        
        # QTLU - 上分位数因子
        for window in windows:
            factors[f"QTLU{window}"] = {
                "function_name": f"price_quantile_upper_{window}d",
                "description": f"{window}天收盘价80%分位数相对当前价格",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价80%分位数 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # QTLD - 下分位数因子
        for window in windows:
            factors[f"QTLD{window}"] = {
                "function_name": f"price_quantile_lower_{window}d",
                "description": f"{window}天收盘价20%分位数相对当前价格",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价20%分位数 / 当前收盘价",
                "parameters": ["close"]
            }
        
        # RANK - 排名因子
        for window in windows:
            factors[f"RANK{window}"] = {
                "function_name": f"price_rank_{window}d",
                "description": f"{window}天收盘价排名百分位，衡量当前价格相对历史位置",
                "category": "滚动统计",
                "formula": f"当前收盘价在过去{window}天中的排名百分位",
                "parameters": ["close"]
            }
        
        # RSV - 随机指标因子
        for window in windows:
            factors[f"RSV{window}"] = {
                "function_name": f"stochastic_{window}d",
                "description": f"{window}天随机指标，衡量当前价格在区间中的相对位置",
                "category": "滚动统计",
                "formula": f"(当前收盘价 - 过去{window}天最低价) / (过去{window}天最高价 - 过去{window}天最低价 + 1e-12)",
                "parameters": ["high", "low", "close"]
            }
        
        # IMAX - 最高价位置因子
        for window in windows:
            factors[f"IMAX{window}"] = {
                "function_name": f"max_price_index_{window}d",
                "description": f"{window}天最高价出现位置，阿隆指标的一部分",
                "category": "滚动统计",
                "formula": f"过去{window}天最高价出现天数 / {window}",
                "parameters": ["high"]
            }
        
        # IMIN - 最低价位置因子
        for window in windows:
            factors[f"IMIN{window}"] = {
                "function_name": f"min_price_index_{window}d",
                "description": f"{window}天最低价出现位置，阿隆指标的一部分",
                "category": "滚动统计",
                "formula": f"过去{window}天最低价出现天数 / {window}",
                "parameters": ["low"]
            }
        
        # IMXD - 最高最低价位置差因子
        for window in windows:
            factors[f"IMXD{window}"] = {
                "function_name": f"max_min_index_diff_{window}d",
                "description": f"{window}天最高价与最低价出现位置差，衡量价格动量方向",
                "category": "滚动统计",
                "formula": f"(过去{window}天最高价出现天数 - 过去{window}天最低价出现天数) / {window}",
                "parameters": ["high", "low"]
            }
        
        # CORR - 价格成交量相关性因子
        for window in windows:
            factors[f"CORR{window}"] = {
                "function_name": f"price_volume_corr_{window}d",
                "description": f"{window}天收盘价与成交量对数相关性",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价与成交量对数的相关系数",
                "parameters": ["close", "volume"]
            }
        
        # CORD - 价格变化与成交量变化相关性因子
        for window in windows:
            factors[f"CORD{window}"] = {
                "function_name": f"price_change_volume_corr_{window}d",
                "description": f"{window}天价格变化率与成交量变化率相关性",
                "category": "滚动统计",
                "formula": f"过去{window}天价格变化率与成交量变化率对数的相关系数",
                "parameters": ["close", "volume"]
            }
        
        # CNTP - 上涨天数比例因子
        for window in windows:
            factors[f"CNTP{window}"] = {
                "function_name": f"up_days_ratio_{window}d",
                "description": f"{window}天上涨天数比例，衡量上涨概率",
                "category": "滚动统计",
                "formula": f"过去{window}天上涨天数 / {window}",
                "parameters": ["close"]
            }
        
        # CNTN - 下跌天数比例因子
        for window in windows:
            factors[f"CNTN{window}"] = {
                "function_name": f"down_days_ratio_{window}d",
                "description": f"{window}天下跌天数比例，衡量下跌概率",
                "category": "滚动统计",
                "formula": f"过去{window}天下跌天数 / {window}",
                "parameters": ["close"]
            }
        
        # CNTD - 涨跌天数差因子
        for window in windows:
            factors[f"CNTD{window}"] = {
                "function_name": f"up_down_days_diff_{window}d",
                "description": f"{window}天涨跌天数差，衡量多空力量对比",
                "category": "滚动统计",
                "formula": f"过去{window}天上涨天数比例 - 过去{window}天下跌天数比例",
                "parameters": ["close"]
            }
        
        # SUMP - 上涨收益比例因子
        for window in windows:
            factors[f"SUMP{window}"] = {
                "function_name": f"positive_return_ratio_{window}d",
                "description": f"{window}天上涨收益占总收益比例，类似RSI指标",
                "category": "滚动统计",
                "formula": f"过去{window}天上涨收益总和 / (过去{window}天绝对收益总和 + 1e-12)",
                "parameters": ["close"]
            }
        
        # SUMN - 下跌收益比例因子
        for window in windows:
            factors[f"SUMN{window}"] = {
                "function_name": f"negative_return_ratio_{window}d",
                "description": f"{window}天下跌收益占总收益比例，类似RSI指标",
                "category": "滚动统计",
                "formula": f"过去{window}天下跌收益总和 / (过去{window}天绝对收益总和 + 1e-12)",
                "parameters": ["close"]
            }
        
        # SUMD - 涨跌收益差因子
        for window in windows:
            factors[f"SUMD{window}"] = {
                "function_name": f"return_diff_ratio_{window}d",
                "description": f"{window}天涨跌收益差比例，类似RSI指标",
                "category": "滚动统计",
                "formula": f"(过去{window}天上涨收益总和 - 过去{window}天下跌收益总和) / (过去{window}天绝对收益总和 + 1e-12)",
                "parameters": ["close"]
            }
        
        # VMA - 成交量移动平均因子
        for window in windows:
            factors[f"VMA{window}"] = {
                "function_name": f"volume_ma_{window}d",
                "description": f"{window}天成交量移动平均相对当前成交量",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量均值 / (当前成交量 + 1e-12)",
                "parameters": ["volume"]
            }
        
        # VSTD - 成交量标准差因子
        for window in windows:
            factors[f"VSTD{window}"] = {
                "function_name": f"volume_std_{window}d",
                "description": f"{window}天成交量标准差相对当前成交量，衡量成交量波动性",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量标准差 / (当前成交量 + 1e-12)",
                "parameters": ["volume"]
            }
        
        # WVMA - 成交量加权价格波动因子
        for window in windows:
            factors[f"WVMA{window}"] = {
                "function_name": f"volume_weighted_volatility_{window}d",
                "description": f"{window}天成交量加权价格波动率",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量加权价格变化标准差 / (过去{window}天成交量加权价格变化均值 + 1e-12)",
                "parameters": ["close", "volume"]
            }
        
        # VSUMP - 成交量上涨比例因子
        for window in windows:
            factors[f"VSUMP{window}"] = {
                "function_name": f"volume_increase_ratio_{window}d",
                "description": f"{window}天成交量上涨比例，类似RSI指标",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量上涨总和 / (过去{window}天成交量绝对变化总和 + 1e-12)",
                "parameters": ["volume"]
            }
        
        # VSUMN - 成交量下跌比例因子
        for window in windows:
            factors[f"VSUMN{window}"] = {
                "function_name": f"volume_decrease_ratio_{window}d",
                "description": f"{window}天成交量下跌比例，类似RSI指标",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量下跌总和 / (过去{window}天成交量绝对变化总和 + 1e-12)",
                "parameters": ["volume"]
            }
        
        # VSUMD - 成交量涨跌差因子
        for window in windows:
            factors[f"VSUMD{window}"] = {
                "function_name": f"volume_diff_ratio_{window}d",
                "description": f"{window}天成交量涨跌差比例，类似RSI指标",
                "category": "成交量因子",
                "formula": f"(过去{window}天成交量上涨总和 - 过去{window}天成交量下跌总和) / (过去{window}天成交量绝对变化总和 + 1e-12)",
                "parameters": ["volume"]
            }
        
        return factors
    
    def get_factor_info(self, factor_name: str) -> Optional[Dict]:
        """获取指定因子的详细信息"""
        return self.factors.get(factor_name)
    
    def get_factors_by_category(self, category: str) -> Dict[str, Dict]:
        """按类别获取因子"""
        return {name: info for name, info in self.factors.items() 
                if info.get('category') == category}
    
    def get_all_factors(self) -> Dict[str, Dict]:
        """获取所有因子"""
        return self.factors
    
    def get_factor_expression(self, factor_name: str) -> Optional[str]:
        """获取因子表达式"""
        factor_info = self.get_factor_info(factor_name)
        return factor_info.get('expression') if factor_info else None
    
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
    
    def get_factor_summary(self) -> pd.DataFrame:
        """获取因子摘要信息"""
        data = []
        for name, info in self.factors.items():
            data.append({
                '因子名称': name,
                '函数名': info.get('function_name', ''),
                '类别': info.get('category', ''),
                '描述': info.get('description', ''),
                '表达式': info.get('expression', ''),
                '公式': info.get('formula', '')
            })
        return pd.DataFrame(data)


def main():
    """主函数，演示因子库的使用"""
    # 创建因子库实例
    factor_lib = Alpha158Factors()
    
    # 获取因子摘要
    summary_df = factor_lib.get_factor_summary()
    print("Alpha158因子库摘要:")
    print(f"总因子数量: {len(summary_df)}")
    print("\n按类别统计:")
    print(summary_df['类别'].value_counts())
    
    # 显示前10个因子
    print("\n前10个因子:")
    print(summary_df[['因子名称', '函数名', '类别', '描述']].head(10).to_string(index=False))
    
    # 获取特定因子信息
    print("\nK线形态因子:")
    kbar_factors = factor_lib.get_factors_by_category('K线形态')
    for name, info in kbar_factors.items():
        print(f"{name}: {info['description']}")
    
    # 保存因子库到文件
    summary_df.to_csv('source/因子库/alpha158_factors_summary.csv', 
                     index=False, encoding='utf-8-sig')
    print(f"\n因子摘要已保存到: source/因子库/alpha158_factors_summary.csv")


if __name__ == "__main__":
    main()
