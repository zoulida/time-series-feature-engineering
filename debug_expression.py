# -*- coding: utf-8 -*-
"""
调试表达式解析
"""

import pandas as pd
import numpy as np

def debug_expression_parsing():
    """调试表达式解析"""
    
    # 创建测试数据
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105],
        'open': [99, 100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105, 106],
        'low': [98, 99, 100, 101, 102, 103]
    })
    
    print("测试数据:")
    print(data)
    print()
    
    # 测试表达式
    expression = "Mean($close, 5)/$close"
    print(f"原始表达式: {expression}")
    
    # 替换变量
    expr = expression
    if '$close' in expr and 'close' in data.columns:
        expr = expr.replace('$close', 'data["close"]')
    
    print(f"替换变量后: {expr}")
    
    # 替换函数名 - 使用正则表达式更精确地替换
    import re
    
    # 替换Mean函数
    expr = re.sub(r'Mean\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).mean()', expr)
    print(f"替换函数名后: {expr}")
    
    # 尝试执行
    try:
        result = eval(expr)
        print(f"执行结果: {result}")
        print(f"结果类型: {type(result)}")
        if hasattr(result, 'values'):
            print(f"结果值: {result.values}")
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    debug_expression_parsing()
