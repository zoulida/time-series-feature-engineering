#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复Alpha158因子定义中的parameters字段
"""

import re

def fix_alpha158_parameters():
    """修复alpha158_factors.py中的因子定义"""
    
    # 读取文件
    with open('source/因子库/alpha158_factors.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义需要修复的因子类型和对应的参数
    factor_fixes = {
        # 排名因子
        'RANK': ['close'],
        # 随机指标因子
        'RSV': ['high', 'low', 'close'],
        # 指数位置因子
        'IMAX': ['high'],
        'IMIN': ['low'],
        'IMXD': ['high', 'low'],
        # 相关性因子
        'CORR': ['close', 'volume'],
        'CORD': ['close', 'volume'],
        # 比率因子
        'CNTP': ['close'],
        'CNTN': ['close'],
        'CNTD': ['close'],
        'SUMP': ['close'],
        'SUMN': ['close'],
        'SUMD': ['close'],
        # 成交量因子
        'VMA': ['volume'],
        'VSTD': ['volume'],
        'VSUMP': ['volume'],
        'VSUMN': ['volume'],
        'VSUMD': ['volume'],
        'WVMA': ['close', 'volume'],
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
    
    print("Alpha158因子参数修复完成！")

if __name__ == "__main__":
    fix_alpha158_parameters()
