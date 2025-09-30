#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复剩余的Alpha158因子定义中的parameters字段
"""

import re

def fix_remaining_parameters():
    """修复alpha158_factors.py中剩余的因子定义"""
    
    # 读取文件
    with open('source/因子库/alpha158_factors.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义需要修复的因子类型和对应的参数
    factor_fixes = {
        # 比率因子
        'CNTP': ['close'],  # 上涨天数比例
        'CNTN': ['close'],  # 下跌天数比例
        'CNTD': ['close'],  # 上涨下跌天数差
        'SUMP': ['close'],  # 正收益比例
        'SUMN': ['close'],  # 负收益比例
        'SUMD': ['close'],  # 收益差比例
        # 成交量因子
        'VMA': ['volume'],  # 成交量移动平均
        'VSTD': ['volume'],  # 成交量标准差
        'VSUMP': ['volume'],  # 成交量增加比例
        'VSUMN': ['volume'],  # 成交量减少比例
        'VSUMD': ['volume'],  # 成交量差比例
        'WVMA': ['close', 'volume'],  # 成交量加权波动率
    }
    
    # 修复每个因子类型
    for factor_type, params in factor_fixes.items():
        # 查找所有该类型的因子定义
        pattern = rf'factors\[f"{factor_type}(\d+)"\] = \{{(.*?)\}}'
        
        def replace_factor(match):
            window = match.group(1)
            factor_content = match.group(2)
            
            # 移除expression字段
            factor_content = re.sub(r'"expression": [^,}]+,\s*', '', factor_content)
            
            # 添加parameters字段
            params_str = '["' + '", "'.join(params) + '"]'
            factor_content = factor_content.rstrip().rstrip(',') + f',\n                "parameters": {params_str}'
            
            return f'factors[f"{factor_type}{window}"] = {{{factor_content}}}'
        
        content = re.sub(pattern, replace_factor, content, flags=re.DOTALL)
    
    # 写回文件
    with open('source/因子库/alpha158_factors.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("剩余Alpha158因子参数修复完成！")

if __name__ == "__main__":
    fix_remaining_parameters()
