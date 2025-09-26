# -*- coding: utf-8 -*-
"""
Alpha158因子库
从qlib.contrib.data.handler.Alpha158和qlib.contrib.data.loader.Alpha158DL提取的因子定义

包含158个技术分析因子，分为以下几类：
1. K线形态因子 (K-Bar Factors)
2. 价格因子 (Price Factors) 
3. 滚动统计因子 (Rolling Statistical Factors)
4. 成交量因子 (Volume Factors)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class Alpha158Factors:
    """Alpha158因子库类"""
    
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
                "expression": "($close-$open)/$open",
                "function_name": "kbar_mid_ratio",
                "description": "K线实体相对开盘价的比例，衡量收盘价与开盘价的相对关系",
                "category": "K线形态",
                "formula": "(收盘价 - 开盘价) / 开盘价"
            },
            "KLEN": {
                "expression": "($high-$low)/$open", 
                "function_name": "kbar_length_ratio",
                "description": "K线长度相对开盘价的比例，衡量当日价格波动幅度",
                "category": "K线形态",
                "formula": "(最高价 - 最低价) / 开盘价"
            },
            "KMID2": {
                "expression": "($close-$open)/($high-$low+1e-12)",
                "function_name": "kbar_mid_body_ratio",
                "description": "K线实体占整个K线长度的比例，衡量实体相对影线的强度",
                "category": "K线形态", 
                "formula": "(收盘价 - 开盘价) / (最高价 - 最低价 + 1e-12)"
            },
            "KUP": {
                "expression": "($high-Greater($open, $close))/$open",
                "function_name": "kbar_upper_shadow_ratio",
                "description": "上影线相对开盘价的比例，衡量上方压力",
                "category": "K线形态",
                "formula": "(最高价 - max(开盘价, 收盘价)) / 开盘价"
            },
            "KUP2": {
                "expression": "($high-Greater($open, $close))/($high-$low+1e-12)",
                "function_name": "kbar_upper_shadow_body_ratio", 
                "description": "上影线占整个K线长度的比例",
                "category": "K线形态",
                "formula": "(最高价 - max(开盘价, 收盘价)) / (最高价 - 最低价 + 1e-12)"
            },
            "KLOW": {
                "expression": "(Less($open, $close)-$low)/$open",
                "function_name": "kbar_lower_shadow_ratio",
                "description": "下影线相对开盘价的比例，衡量下方支撑",
                "category": "K线形态",
                "formula": "(min(开盘价, 收盘价) - 最低价) / 开盘价"
            },
            "KLOW2": {
                "expression": "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "function_name": "kbar_lower_shadow_body_ratio",
                "description": "下影线占整个K线长度的比例", 
                "category": "K线形态",
                "formula": "(min(开盘价, 收盘价) - 最低价) / (最高价 - 最低价 + 1e-12)"
            },
            "KSFT": {
                "expression": "(2*$close-$high-$low)/$open",
                "function_name": "kbar_soft_ratio",
                "description": "K线软度指标，衡量收盘价在当日价格区间中的位置",
                "category": "K线形态",
                "formula": "(2*收盘价 - 最高价 - 最低价) / 开盘价"
            },
            "KSFT2": {
                "expression": "(2*$close-$high-$low)/($high-$low+1e-12)",
                "function_name": "kbar_soft_body_ratio",
                "description": "K线软度占整个K线长度的比例",
                "category": "K线形态", 
                "formula": "(2*收盘价 - 最高价 - 最低价) / (最高价 - 最低价 + 1e-12)"
            }
        }
    
    def _get_price_factors(self) -> Dict[str, Dict]:
        """获取价格因子"""
        factors = {}
        windows = [0]  # Alpha158默认只使用当前价格
        features = ["OPEN", "HIGH", "LOW", "VWAP"]
        
        for feature in features:
            for window in windows:
                if window == 0:
                    expression = f"${feature.lower()}/$close"
                    func_name = f"price_{feature.lower()}_current_ratio"
                    desc = f"当前{feature}相对收盘价的比例"
                else:
                    expression = f"Ref(${feature.lower()}, {window})/$close"
                    func_name = f"price_{feature.lower()}_{window}d_ratio"
                    desc = f"{window}天前{feature}相对当前收盘价的比例"
                
                factors[f"{feature}{window}"] = {
                    "expression": expression,
                    "function_name": func_name,
                    "description": desc,
                    "category": "价格因子",
                    "formula": f"{feature} / 收盘价" if window == 0 else f"{window}天前{feature} / 当前收盘价"
                }
        
        return factors
    
    def _get_rolling_factors(self) -> Dict[str, Dict]:
        """获取滚动统计因子"""
        factors = {}
        windows = [5, 10, 20, 30, 60]  # Alpha158默认窗口
        
        # ROC - 变化率因子
        for window in windows:
            factors[f"ROC{window}"] = {
                "expression": f"Ref($close, {window})/$close",
                "function_name": f"rate_of_change_{window}d",
                "description": f"{window}天价格变化率，衡量价格动量",
                "category": "滚动统计",
                "formula": f"{window}天前收盘价 / 当前收盘价"
            }
        
        # MA - 移动平均因子
        for window in windows:
            factors[f"MA{window}"] = {
                "expression": f"Mean($close, {window})/$close",
                "function_name": f"moving_average_{window}d",
                "description": f"{window}天简单移动平均相对当前价格",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价均值 / 当前收盘价"
            }
        
        # STD - 标准差因子
        for window in windows:
            factors[f"STD{window}"] = {
                "expression": f"Std($close, {window})/$close",
                "function_name": f"price_std_{window}d",
                "description": f"{window}天价格标准差相对当前价格，衡量价格波动性",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价标准差 / 当前收盘价"
            }
        
        # BETA - 斜率因子
        for window in windows:
            factors[f"BETA{window}"] = {
                "expression": f"Slope($close, {window})/$close",
                "function_name": f"price_slope_{window}d",
                "description": f"{window}天价格线性回归斜率，衡量价格趋势强度",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价线性回归斜率 / 当前收盘价"
            }
        
        # RSQR - R平方因子
        for window in windows:
            factors[f"RSQR{window}"] = {
                "expression": f"Rsquare($close, {window})",
                "function_name": f"price_rsquare_{window}d",
                "description": f"{window}天价格线性回归R平方值，衡量趋势线性度",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价线性回归R平方值"
            }
        
        # RESI - 残差因子
        for window in windows:
            factors[f"RESI{window}"] = {
                "expression": f"Resi($close, {window})/$close",
                "function_name": f"price_residual_{window}d",
                "description": f"{window}天价格线性回归残差，衡量偏离趋势线的程度",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价线性回归残差 / 当前收盘价"
            }
        
        # MAX - 最大值因子
        for window in windows:
            factors[f"MAX{window}"] = {
                "expression": f"Max($high, {window})/$close",
                "function_name": f"price_max_{window}d",
                "description": f"{window}天最高价相对当前收盘价，衡量价格阻力位",
                "category": "滚动统计",
                "formula": f"过去{window}天最高价 / 当前收盘价"
            }
        
        # MIN - 最小值因子
        for window in windows:
            factors[f"MIN{window}"] = {
                "expression": f"Min($low, {window})/$close",
                "function_name": f"price_min_{window}d",
                "description": f"{window}天最低价相对当前收盘价，衡量价格支撑位",
                "category": "滚动统计",
                "formula": f"过去{window}天最低价 / 当前收盘价"
            }
        
        # QTLU - 上分位数因子
        for window in windows:
            factors[f"QTLU{window}"] = {
                "expression": f"Quantile($close, {window}, 0.8)/$close",
                "function_name": f"price_quantile_upper_{window}d",
                "description": f"{window}天收盘价80%分位数相对当前价格",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价80%分位数 / 当前收盘价"
            }
        
        # QTLD - 下分位数因子
        for window in windows:
            factors[f"QTLD{window}"] = {
                "expression": f"Quantile($close, {window}, 0.2)/$close",
                "function_name": f"price_quantile_lower_{window}d",
                "description": f"{window}天收盘价20%分位数相对当前价格",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价20%分位数 / 当前收盘价"
            }
        
        # RANK - 排名因子
        for window in windows:
            factors[f"RANK{window}"] = {
                "expression": f"Rank($close, {window})",
                "function_name": f"price_rank_{window}d",
                "description": f"{window}天收盘价排名百分位，衡量当前价格相对历史位置",
                "category": "滚动统计",
                "formula": f"当前收盘价在过去{window}天中的排名百分位"
            }
        
        # RSV - 随机指标因子
        for window in windows:
            factors[f"RSV{window}"] = {
                "expression": f"($close-Min($low, {window}))/(Max($high, {window})-Min($low, {window})+1e-12)",
                "function_name": f"stochastic_{window}d",
                "description": f"{window}天随机指标，衡量当前价格在区间中的相对位置",
                "category": "滚动统计",
                "formula": f"(当前收盘价 - 过去{window}天最低价) / (过去{window}天最高价 - 过去{window}天最低价 + 1e-12)"
            }
        
        # IMAX - 最高价位置因子
        for window in windows:
            factors[f"IMAX{window}"] = {
                "expression": f"IdxMax($high, {window})/{window}",
                "function_name": f"max_price_index_{window}d",
                "description": f"{window}天最高价出现位置，阿隆指标的一部分",
                "category": "滚动统计",
                "formula": f"过去{window}天最高价出现天数 / {window}"
            }
        
        # IMIN - 最低价位置因子
        for window in windows:
            factors[f"IMIN{window}"] = {
                "expression": f"IdxMin($low, {window})/{window}",
                "function_name": f"min_price_index_{window}d",
                "description": f"{window}天最低价出现位置，阿隆指标的一部分",
                "category": "滚动统计",
                "formula": f"过去{window}天最低价出现天数 / {window}"
            }
        
        # IMXD - 最高最低价位置差因子
        for window in windows:
            factors[f"IMXD{window}"] = {
                "expression": f"(IdxMax($high, {window})-IdxMin($low, {window}))/{window}",
                "function_name": f"max_min_index_diff_{window}d",
                "description": f"{window}天最高价与最低价出现位置差，衡量价格动量方向",
                "category": "滚动统计",
                "formula": f"(过去{window}天最高价出现天数 - 过去{window}天最低价出现天数) / {window}"
            }
        
        # CORR - 价格成交量相关性因子
        for window in windows:
            factors[f"CORR{window}"] = {
                "expression": f"Corr($close, Log($volume+1), {window})",
                "function_name": f"price_volume_corr_{window}d",
                "description": f"{window}天收盘价与成交量对数相关性",
                "category": "滚动统计",
                "formula": f"过去{window}天收盘价与成交量对数的相关系数"
            }
        
        # CORD - 价格变化与成交量变化相关性因子
        for window in windows:
            factors[f"CORD{window}"] = {
                "expression": f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {window})",
                "function_name": f"price_change_volume_corr_{window}d",
                "description": f"{window}天价格变化率与成交量变化率相关性",
                "category": "滚动统计",
                "formula": f"过去{window}天价格变化率与成交量变化率对数的相关系数"
            }
        
        # CNTP - 上涨天数比例因子
        for window in windows:
            factors[f"CNTP{window}"] = {
                "expression": f"Mean($close>Ref($close, 1), {window})",
                "function_name": f"up_days_ratio_{window}d",
                "description": f"{window}天上涨天数比例，衡量上涨概率",
                "category": "滚动统计",
                "formula": f"过去{window}天上涨天数 / {window}"
            }
        
        # CNTN - 下跌天数比例因子
        for window in windows:
            factors[f"CNTN{window}"] = {
                "expression": f"Mean($close<Ref($close, 1), {window})",
                "function_name": f"down_days_ratio_{window}d",
                "description": f"{window}天下跌天数比例，衡量下跌概率",
                "category": "滚动统计",
                "formula": f"过去{window}天下跌天数 / {window}"
            }
        
        # CNTD - 涨跌天数差因子
        for window in windows:
            factors[f"CNTD{window}"] = {
                "expression": f"Mean($close>Ref($close, 1), {window})-Mean($close<Ref($close, 1), {window})",
                "function_name": f"up_down_days_diff_{window}d",
                "description": f"{window}天涨跌天数差，衡量多空力量对比",
                "category": "滚动统计",
                "formula": f"过去{window}天上涨天数比例 - 过去{window}天下跌天数比例"
            }
        
        # SUMP - 上涨收益比例因子
        for window in windows:
            factors[f"SUMP{window}"] = {
                "expression": f"Sum(Greater($close-Ref($close, 1), 0), {window})/(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
                "function_name": f"positive_return_ratio_{window}d",
                "description": f"{window}天上涨收益占总收益比例，类似RSI指标",
                "category": "滚动统计",
                "formula": f"过去{window}天上涨收益总和 / (过去{window}天绝对收益总和 + 1e-12)"
            }
        
        # SUMN - 下跌收益比例因子
        for window in windows:
            factors[f"SUMN{window}"] = {
                "expression": f"Sum(Greater(Ref($close, 1)-$close, 0), {window})/(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
                "function_name": f"negative_return_ratio_{window}d",
                "description": f"{window}天下跌收益占总收益比例，类似RSI指标",
                "category": "滚动统计",
                "formula": f"过去{window}天下跌收益总和 / (过去{window}天绝对收益总和 + 1e-12)"
            }
        
        # SUMD - 涨跌收益差因子
        for window in windows:
            factors[f"SUMD{window}"] = {
                "expression": f"(Sum(Greater($close-Ref($close, 1), 0), {window})-Sum(Greater(Ref($close, 1)-$close, 0), {window}))/(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
                "function_name": f"return_diff_ratio_{window}d",
                "description": f"{window}天涨跌收益差比例，类似RSI指标",
                "category": "滚动统计",
                "formula": f"(过去{window}天上涨收益总和 - 过去{window}天下跌收益总和) / (过去{window}天绝对收益总和 + 1e-12)"
            }
        
        # VMA - 成交量移动平均因子
        for window in windows:
            factors[f"VMA{window}"] = {
                "expression": f"Mean($volume, {window})/($volume+1e-12)",
                "function_name": f"volume_ma_{window}d",
                "description": f"{window}天成交量移动平均相对当前成交量",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量均值 / (当前成交量 + 1e-12)"
            }
        
        # VSTD - 成交量标准差因子
        for window in windows:
            factors[f"VSTD{window}"] = {
                "expression": f"Std($volume, {window})/($volume+1e-12)",
                "function_name": f"volume_std_{window}d",
                "description": f"{window}天成交量标准差相对当前成交量，衡量成交量波动性",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量标准差 / (当前成交量 + 1e-12)"
            }
        
        # WVMA - 成交量加权价格波动因子
        for window in windows:
            factors[f"WVMA{window}"] = {
                "expression": f"Std(Abs($close/Ref($close, 1)-1)*$volume, {window})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {window})+1e-12)",
                "function_name": f"volume_weighted_volatility_{window}d",
                "description": f"{window}天成交量加权价格波动率",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量加权价格变化标准差 / (过去{window}天成交量加权价格变化均值 + 1e-12)"
            }
        
        # VSUMP - 成交量上涨比例因子
        for window in windows:
            factors[f"VSUMP{window}"] = {
                "expression": f"Sum(Greater($volume-Ref($volume, 1), 0), {window})/(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
                "function_name": f"volume_increase_ratio_{window}d",
                "description": f"{window}天成交量上涨比例，类似RSI指标",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量上涨总和 / (过去{window}天成交量绝对变化总和 + 1e-12)"
            }
        
        # VSUMN - 成交量下跌比例因子
        for window in windows:
            factors[f"VSUMN{window}"] = {
                "expression": f"Sum(Greater(Ref($volume, 1)-$volume, 0), {window})/(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
                "function_name": f"volume_decrease_ratio_{window}d",
                "description": f"{window}天成交量下跌比例，类似RSI指标",
                "category": "成交量因子",
                "formula": f"过去{window}天成交量下跌总和 / (过去{window}天成交量绝对变化总和 + 1e-12)"
            }
        
        # VSUMD - 成交量涨跌差因子
        for window in windows:
            factors[f"VSUMD{window}"] = {
                "expression": f"(Sum(Greater($volume-Ref($volume, 1), 0), {window})-Sum(Greater(Ref($volume, 1)-$volume, 0), {window}))/(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
                "function_name": f"volume_diff_ratio_{window}d",
                "description": f"{window}天成交量涨跌差比例，类似RSI指标",
                "category": "成交量因子",
                "formula": f"(过去{window}天成交量上涨总和 - 过去{window}天成交量下跌总和) / (过去{window}天成交量绝对变化总和 + 1e-12)"
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
