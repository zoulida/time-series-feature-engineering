# -*- coding: utf-8 -*-
"""
因子库配置文件
从Alpha158和Alpha158DL提取的因子定义，包含表达式、函数名和注释
"""

# K线形态因子
KBAR_FACTORS = {
    "KMID": {
        "expression": "($close-$open)/$open",
        "function_name": "kbar_mid_ratio",
        "description": "K线实体相对开盘价的比例"
    },
    "KLEN": {
        "expression": "($high-$low)/$open",
        "function_name": "kbar_length_ratio", 
        "description": "K线长度相对开盘价的比例"
    },
    "KMID2": {
        "expression": "($close-$open)/($high-$low+1e-12)",
        "function_name": "kbar_mid_body_ratio",
        "description": "K线实体占整个K线长度的比例"
    },
    "KUP": {
        "expression": "($high-Greater($open, $close))/$open",
        "function_name": "kbar_upper_shadow_ratio",
        "description": "上影线相对开盘价的比例"
    },
    "KUP2": {
        "expression": "($high-Greater($open, $close))/($high-$low+1e-12)",
        "function_name": "kbar_upper_shadow_body_ratio",
        "description": "上影线占整个K线长度的比例"
    },
    "KLOW": {
        "expression": "(Less($open, $close)-$low)/$open",
        "function_name": "kbar_lower_shadow_ratio",
        "description": "下影线相对开盘价的比例"
    },
    "KLOW2": {
        "expression": "(Less($open, $close)-$low)/($high-$low+1e-12)",
        "function_name": "kbar_lower_shadow_body_ratio",
        "description": "下影线占整个K线长度的比例"
    },
    "KSFT": {
        "expression": "(2*$close-$high-$low)/$open",
        "function_name": "kbar_soft_ratio",
        "description": "K线软度指标"
    },
    "KSFT2": {
        "expression": "(2*$close-$high-$low)/($high-$low+1e-12)",
        "function_name": "kbar_soft_body_ratio",
        "description": "K线软度占整个K线长度的比例"
    }
}

# 价格因子
PRICE_FACTORS = {
    "OPEN0": {
        "expression": "$open/$close",
        "function_name": "price_open_current_ratio",
        "description": "当前开盘价相对收盘价的比例"
    },
    "HIGH0": {
        "expression": "$high/$close",
        "function_name": "price_high_current_ratio",
        "description": "当前最高价相对收盘价的比例"
    },
    "LOW0": {
        "expression": "$low/$close",
        "function_name": "price_low_current_ratio",
        "description": "当前最低价相对收盘价的比例"
    },
    "VWAP0": {
        "expression": "$vwap/$close",
        "function_name": "price_vwap_current_ratio",
        "description": "当前VWAP相对收盘价的比例"
    }
}

# 滚动统计因子 - 变化率
ROC_FACTORS = {
    f"ROC{window}": {
        "expression": f"Ref($close, {window})/$close",
        "function_name": f"rate_of_change_{window}d",
        "description": f"{window}天价格变化率"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 移动平均
MA_FACTORS = {
    f"MA{window}": {
        "expression": f"Mean($close, {window})/$close",
        "function_name": f"moving_average_{window}d",
        "description": f"{window}天简单移动平均"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 标准差
STD_FACTORS = {
    f"STD{window}": {
        "expression": f"Std($close, {window})/$close",
        "function_name": f"price_std_{window}d",
        "description": f"{window}天价格标准差"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 斜率
BETA_FACTORS = {
    f"BETA{window}": {
        "expression": f"Slope($close, {window})/$close",
        "function_name": f"price_slope_{window}d",
        "description": f"{window}天价格线性回归斜率"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - R平方
RSQR_FACTORS = {
    f"RSQR{window}": {
        "expression": f"Rsquare($close, {window})",
        "function_name": f"price_rsquare_{window}d",
        "description": f"{window}天价格线性回归R平方值"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 残差
RESI_FACTORS = {
    f"RESI{window}": {
        "expression": f"Resi($close, {window})/$close",
        "function_name": f"price_residual_{window}d",
        "description": f"{window}天价格线性回归残差"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 最大值
MAX_FACTORS = {
    f"MAX{window}": {
        "expression": f"Max($high, {window})/$close",
        "function_name": f"price_max_{window}d",
        "description": f"{window}天最高价相对当前收盘价"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 最小值
MIN_FACTORS = {
    f"MIN{window}": {
        "expression": f"Min($low, {window})/$close",
        "function_name": f"price_min_{window}d",
        "description": f"{window}天最低价相对当前收盘价"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 分位数
QTLU_FACTORS = {
    f"QTLU{window}": {
        "expression": f"Quantile($close, {window}, 0.8)/$close",
        "function_name": f"price_quantile_upper_{window}d",
        "description": f"{window}天收盘价80%分位数"
    }
    for window in [5, 10, 20, 30, 60]
}

QTLD_FACTORS = {
    f"QTLD{window}": {
        "expression": f"Quantile($close, {window}, 0.2)/$close",
        "function_name": f"price_quantile_lower_{window}d",
        "description": f"{window}天收盘价20%分位数"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 排名
RANK_FACTORS = {
    f"RANK{window}": {
        "expression": f"Rank($close, {window})",
        "function_name": f"price_rank_{window}d",
        "description": f"{window}天收盘价排名百分位"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 随机指标
RSV_FACTORS = {
    f"RSV{window}": {
        "expression": f"($close-Min($low, {window}))/(Max($high, {window})-Min($low, {window})+1e-12)",
        "function_name": f"stochastic_{window}d",
        "description": f"{window}天随机指标"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 阿隆指标
IMAX_FACTORS = {
    f"IMAX{window}": {
        "expression": f"IdxMax($high, {window})/{window}",
        "function_name": f"max_price_index_{window}d",
        "description": f"{window}天最高价出现位置"
    }
    for window in [5, 10, 20, 30, 60]
}

IMIN_FACTORS = {
    f"IMIN{window}": {
        "expression": f"IdxMin($low, {window})/{window}",
        "function_name": f"min_price_index_{window}d",
        "description": f"{window}天最低价出现位置"
    }
    for window in [5, 10, 20, 30, 60]
}

IMXD_FACTORS = {
    f"IMXD{window}": {
        "expression": f"(IdxMax($high, {window})-IdxMin($low, {window}))/{window}",
        "function_name": f"max_min_index_diff_{window}d",
        "description": f"{window}天最高价与最低价出现位置差"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 相关性
CORR_FACTORS = {
    f"CORR{window}": {
        "expression": f"Corr($close, Log($volume+1), {window})",
        "function_name": f"price_volume_corr_{window}d",
        "description": f"{window}天收盘价与成交量对数相关性"
    }
    for window in [5, 10, 20, 30, 60]
}

CORD_FACTORS = {
    f"CORD{window}": {
        "expression": f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {window})",
        "function_name": f"price_change_volume_corr_{window}d",
        "description": f"{window}天价格变化率与成交量变化率相关性"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 涨跌统计
CNTP_FACTORS = {
    f"CNTP{window}": {
        "expression": f"Mean($close>Ref($close, 1), {window})",
        "function_name": f"up_days_ratio_{window}d",
        "description": f"{window}天上涨天数比例"
    }
    for window in [5, 10, 20, 30, 60]
}

CNTN_FACTORS = {
    f"CNTN{window}": {
        "expression": f"Mean($close<Ref($close, 1), {window})",
        "function_name": f"down_days_ratio_{window}d",
        "description": f"{window}天下跌天数比例"
    }
    for window in [5, 10, 20, 30, 60]
}

CNTD_FACTORS = {
    f"CNTD{window}": {
        "expression": f"Mean($close>Ref($close, 1), {window})-Mean($close<Ref($close, 1), {window})",
        "function_name": f"up_down_days_diff_{window}d",
        "description": f"{window}天涨跌天数差"
    }
    for window in [5, 10, 20, 30, 60]
}

# 滚动统计因子 - 收益统计
SUMP_FACTORS = {
    f"SUMP{window}": {
        "expression": f"Sum(Greater($close-Ref($close, 1), 0), {window})/(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
        "function_name": f"positive_return_ratio_{window}d",
        "description": f"{window}天上涨收益占总收益比例"
    }
    for window in [5, 10, 20, 30, 60]
}

SUMN_FACTORS = {
    f"SUMN{window}": {
        "expression": f"Sum(Greater(Ref($close, 1)-$close, 0), {window})/(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
        "function_name": f"negative_return_ratio_{window}d",
        "description": f"{window}天下跌收益占总收益比例"
    }
    for window in [5, 10, 20, 30, 60]
}

SUMD_FACTORS = {
    f"SUMD{window}": {
        "expression": f"(Sum(Greater($close-Ref($close, 1), 0), {window})-Sum(Greater(Ref($close, 1)-$close, 0), {window}))/(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
        "function_name": f"return_diff_ratio_{window}d",
        "description": f"{window}天涨跌收益差比例"
    }
    for window in [5, 10, 20, 30, 60]
}

# 成交量因子
VMA_FACTORS = {
    f"VMA{window}": {
        "expression": f"Mean($volume, {window})/($volume+1e-12)",
        "function_name": f"volume_ma_{window}d",
        "description": f"{window}天成交量移动平均"
    }
    for window in [5, 10, 20, 30, 60]
}

VSTD_FACTORS = {
    f"VSTD{window}": {
        "expression": f"Std($volume, {window})/($volume+1e-12)",
        "function_name": f"volume_std_{window}d",
        "description": f"{window}天成交量标准差"
    }
    for window in [5, 10, 20, 30, 60]
}

WVMA_FACTORS = {
    f"WVMA{window}": {
        "expression": f"Std(Abs($close/Ref($close, 1)-1)*$volume, {window})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {window})+1e-12)",
        "function_name": f"volume_weighted_volatility_{window}d",
        "description": f"{window}天成交量加权价格波动率"
    }
    for window in [5, 10, 20, 30, 60]
}

VSUMP_FACTORS = {
    f"VSUMP{window}": {
        "expression": f"Sum(Greater($volume-Ref($volume, 1), 0), {window})/(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
        "function_name": f"volume_increase_ratio_{window}d",
        "description": f"{window}天成交量上涨比例"
    }
    for window in [5, 10, 20, 30, 60]
}

VSUMN_FACTORS = {
    f"VSUMN{window}": {
        "expression": f"Sum(Greater(Ref($volume, 1)-$volume, 0), {window})/(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
        "function_name": f"volume_decrease_ratio_{window}d",
        "description": f"{window}天成交量下跌比例"
    }
    for window in [5, 10, 20, 30, 60]
}

VSUMD_FACTORS = {
    f"VSUMD{window}": {
        "expression": f"(Sum(Greater($volume-Ref($volume, 1), 0), {window})-Sum(Greater(Ref($volume, 1)-$volume, 0), {window}))/(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
        "function_name": f"volume_diff_ratio_{window}d",
        "description": f"{window}天成交量涨跌差比例"
    }
    for window in [5, 10, 20, 30, 60]
}

# 合并所有因子
ALL_FACTORS = {
    **KBAR_FACTORS,
    **PRICE_FACTORS,
    **ROC_FACTORS,
    **MA_FACTORS,
    **STD_FACTORS,
    **BETA_FACTORS,
    **RSQR_FACTORS,
    **RESI_FACTORS,
    **MAX_FACTORS,
    **MIN_FACTORS,
    **QTLU_FACTORS,
    **QTLD_FACTORS,
    **RANK_FACTORS,
    **RSV_FACTORS,
    **IMAX_FACTORS,
    **IMIN_FACTORS,
    **IMXD_FACTORS,
    **CORR_FACTORS,
    **CORD_FACTORS,
    **CNTP_FACTORS,
    **CNTN_FACTORS,
    **CNTD_FACTORS,
    **SUMP_FACTORS,
    **SUMN_FACTORS,
    **SUMD_FACTORS,
    **VMA_FACTORS,
    **VSTD_FACTORS,
    **WVMA_FACTORS,
    **VSUMP_FACTORS,
    **VSUMN_FACTORS,
    **VSUMD_FACTORS
}

def get_factor_expression(factor_name: str) -> str:
    """获取因子表达式"""
    return ALL_FACTORS.get(factor_name, {}).get("expression", "")

def get_factor_function_name(factor_name: str) -> str:
    """获取因子函数名"""
    return ALL_FACTORS.get(factor_name, {}).get("function_name", "")

def get_factor_description(factor_name: str) -> str:
    """获取因子描述"""
    return ALL_FACTORS.get(factor_name, {}).get("description", "")

def list_all_factors() -> list:
    """列出所有因子名称"""
    return list(ALL_FACTORS.keys())

def get_factors_by_type(factor_type: str) -> dict:
    """按类型获取因子"""
    type_mapping = {
        "kbar": KBAR_FACTORS,
        "price": PRICE_FACTORS,
        "roc": ROC_FACTORS,
        "ma": MA_FACTORS,
        "std": STD_FACTORS,
        "beta": BETA_FACTORS,
        "rsqr": RSQR_FACTORS,
        "resi": RESI_FACTORS,
        "max": MAX_FACTORS,
        "min": MIN_FACTORS,
        "qtlu": QTLU_FACTORS,
        "qtld": QTLD_FACTORS,
        "rank": RANK_FACTORS,
        "rsv": RSV_FACTORS,
        "imax": IMAX_FACTORS,
        "imin": IMIN_FACTORS,
        "imxd": IMXD_FACTORS,
        "corr": CORR_FACTORS,
        "cord": CORD_FACTORS,
        "cntp": CNTP_FACTORS,
        "cntn": CNTN_FACTORS,
        "cntd": CNTD_FACTORS,
        "sump": SUMP_FACTORS,
        "sumn": SUMN_FACTORS,
        "sumd": SUMD_FACTORS,
        "vma": VMA_FACTORS,
        "vstd": VSTD_FACTORS,
        "wvma": WVMA_FACTORS,
        "vsump": VSUMP_FACTORS,
        "vsumn": VSUMN_FACTORS,
        "vsumd": VSUMD_FACTORS
    }
    return type_mapping.get(factor_type.lower(), {})
