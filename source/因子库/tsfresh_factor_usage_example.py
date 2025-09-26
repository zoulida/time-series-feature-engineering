# -*- coding: utf-8 -*-
"""
因子库使用示例
演示如何使用从Alpha158提取的因子库
"""

import pandas as pd
import numpy as np
from factor_library import (
    ALL_FACTORS, 
    get_factor_expression, 
    get_factor_function_name, 
    get_factor_description,
    list_all_factors,
    get_factors_by_type
)


def demonstrate_factor_library():
    """演示因子库的使用"""
    
    print("=" * 60)
    print("Alpha158因子库使用演示")
    print("=" * 60)
    
    # 1. 列出所有因子
    print(f"\n1. 总因子数量: {len(list_all_factors())}")
    print("前10个因子:")
    for i, factor in enumerate(list_all_factors()[:10]):
        print(f"  {i+1:2d}. {factor}")
    
    # 2. 查看特定因子的详细信息
    print("\n2. 特定因子详细信息:")
    sample_factors = ["KMID", "MA5", "STD20", "ROC10", "VMA30"]
    for factor in sample_factors:
        print(f"\n因子: {factor}")
        print(f"  表达式: {get_factor_expression(factor)}")
        print(f"  函数名: {get_factor_function_name(factor)}")
        print(f"  描述: {get_factor_description(factor)}")
    
    # 3. 按类型查看因子
    print("\n3. 按类型查看因子:")
    factor_types = ["kbar", "price", "roc", "ma", "std"]
    for factor_type in factor_types:
        factors = get_factors_by_type(factor_type)
        print(f"\n{factor_type.upper()}类型因子 ({len(factors)}个):")
        for factor_name in list(factors.keys())[:5]:  # 只显示前5个
            print(f"  - {factor_name}: {get_factor_description(factor_name)}")
        if len(factors) > 5:
            print(f"  ... 还有{len(factors)-5}个因子")
    
    # 4. 生成因子表达式列表
    print("\n4. 生成因子表达式列表:")
    expressions = []
    for factor_name in list_all_factors()[:10]:  # 只显示前10个
        expr = get_factor_expression(factor_name)
        expressions.append(f"{factor_name}: {expr}")
    
    print("前10个因子的表达式:")
    for expr in expressions:
        print(f"  {expr}")
    
    # 5. 生成因子函数名列表
    print("\n5. 生成因子函数名列表:")
    function_names = []
    for factor_name in list_all_factors()[:10]:  # 只显示前10个
        func_name = get_factor_function_name(factor_name)
        function_names.append(f"{factor_name}: {func_name}")
    
    print("前10个因子的函数名:")
    for func in function_names:
        print(f"  {func}")


def create_factor_dataframe():
    """创建因子信息DataFrame"""
    
    data = []
    for factor_name in list_all_factors():
        data.append({
            '因子名称': factor_name,
            '表达式': get_factor_expression(factor_name),
            '函数名': get_factor_function_name(factor_name),
            '描述': get_factor_description(factor_name)
        })
    
    df = pd.DataFrame(data)
    return df


def export_factor_library():
    """导出因子库到CSV文件"""
    
    df = create_factor_dataframe()
    
    # 保存到CSV
    output_file = 'alpha158_factors_export.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n因子库已导出到: {output_file}")
    
    # 显示统计信息
    print(f"\n导出统计:")
    print(f"  总因子数量: {len(df)}")
    print(f"  文件大小: {len(df.to_csv())} 字符")
    
    return df


def generate_factor_code():
    """生成因子计算代码模板"""
    
    print("\n6. 生成因子计算代码模板:")
    
    # 选择几个代表性因子
    sample_factors = ["KMID", "MA5", "STD20", "ROC10", "VMA30"]
    
    code_template = """
# 因子计算函数模板
import pandas as pd
import numpy as np

def calculate_factors(data):
    '''
    计算Alpha158因子
    
    Parameters:
    data: DataFrame, 包含OHLCV数据的DataFrame
          列名: ['open', 'high', 'low', 'close', 'volume', 'vwap']
    
    Returns:
    DataFrame: 包含所有因子值的DataFrame
    '''
    result = data.copy()
    
"""
    
    for factor in sample_factors:
        expr = get_factor_expression(factor)
        func_name = get_factor_function_name(factor)
        desc = get_factor_description(factor)
        
        # 将表达式转换为Python代码
        python_expr = expr.replace('$', 'data[').replace('(', '[').replace(')', ']')
        python_expr = python_expr.replace('data[close]', 'data["close"]')
        python_expr = python_expr.replace('data[open]', 'data["open"]')
        python_expr = python_expr.replace('data[high]', 'data["high"]')
        python_expr = python_expr.replace('data[low]', 'data["low"]')
        python_expr = python_expr.replace('data[volume]', 'data["volume"]')
        python_expr = python_expr.replace('data[vwap]', 'data["vwap"]')
        
        code_template += f"""
    # {desc}
    result['{factor}'] = {python_expr}
"""
    
    code_template += """
    return result
"""
    
    print(code_template)
    
    # 保存代码模板
    with open('factor_calculation_template.py', 'w', encoding='utf-8') as f:
        f.write(code_template)
    
    print("代码模板已保存到: factor_calculation_template.py")


def main():
    """主函数"""
    
    # 演示因子库使用
    demonstrate_factor_library()
    
    # 导出因子库
    df = export_factor_library()
    
    # 生成代码模板
    generate_factor_code()
    
    print("\n" + "=" * 60)
    print("因子库演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
