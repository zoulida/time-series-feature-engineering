# -*- coding: utf-8 -*-
"""
自定义因子库使用示例
演示如何使用MyFactors库计算自定义因子

作者: 用户自定义
创建时间: 2024年
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# 导入自定义因子库
from myfactors import MyFactors, create_my_factors


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """
    创建示例股票数据
    
    Parameters:
    -----------
    n_days : int
        数据天数
        
    Returns:
    --------
    pd.DataFrame
        包含OHLCV数据的DataFrame
    """
    # 生成日期序列
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 生成模拟价格数据
    np.random.seed(42)  # 确保结果可重现
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLCV数据
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # 生成开盘价、最高价、最低价
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price
        
        # 生成成交量
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df


def test_zhangting_factor():
    """测试涨停+成交量+价格复合因子"""
    print("=" * 60)
    print("测试涨停+成交量+价格复合因子")
    print("=" * 60)
    
    # 创建示例数据
    print("创建示例数据...")
    data = create_sample_data(100)
    print(f"数据形状: {data.shape}")
    print(f"数据列: {list(data.columns)}")
    print(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 创建自定义因子库
    print("\n创建自定义因子库...")
    my_factors = create_my_factors()
    
    # 显示可用因子
    print(f"可用因子: {my_factors.list_factors()}")
    
    # 计算涨停+成交量+价格复合因子
    print("\n计算涨停+成交量+价格复合因子...")
    try:
        factor_values = my_factors.calculate_factor(
            'ZHANGTING_VOLUME_PRICE_FACTOR',
            close=data['close'],
            high=data['high'],
            volume=data['volume'],
            stock_code='000001.SZ'
        )
        
        print(f"因子值范围: {factor_values.min():.2f} 到 {factor_values.max():.2f}")
        print(f"因子值统计:")
        print(f"  均值: {factor_values.mean():.4f}")
        print(f"  标准差: {factor_values.std():.4f}")
        print(f"  非零值数量: {(factor_values > 0).sum()}")
        
        # 分析因子值分布
        print(f"\n因子值分布:")
        for i in range(4):
            count = (factor_values == i).sum()
            percentage = count / len(factor_values) * 100
            print(f"  值={i}: {count}次 ({percentage:.1f}%)")
        
        # 显示最近几天的因子值
        print(f"\n最近10天的因子值:")
        recent_values = factor_values.tail(10)
        for date, value in recent_values.items():
            print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")
        
        # 保存结果
        result_df = data.copy()
        result_df['zhangting_volume_price_factor'] = factor_values
        result_df.to_csv('myfactors_test_result.csv')
        print(f"\n结果已保存到: myfactors_test_result.csv")
        
    except Exception as e:
        print(f"计算因子时出错: {e}")
        import traceback
        traceback.print_exc()


def test_factor_components():
    """测试因子的各个组成部分"""
    print("\n" + "=" * 60)
    print("测试因子组成部分")
    print("=" * 60)
    
    # 创建示例数据
    data = create_sample_data(100)
    my_factors = create_my_factors()
    
    # 分别测试各个条件
    print("1. 测试涨停条件...")
    zhangting_condition = my_factors._check_zhangting_in_period(
        data['close'], data['high'], '000001.SZ', period=7
    )
    print(f"   涨停条件满足次数: {zhangting_condition.sum()}")
    
    print("2. 测试成交量条件...")
    volume_condition = my_factors._check_volume_condition(
        data['volume'], ma_period=20, compare_period=7, multiplier=1.5
    )
    print(f"   成交量条件满足次数: {volume_condition.sum()}")
    
    print("3. 测试价格条件...")
    price_condition = my_factors._check_price_condition(
        data['close'], ma_period=20, max_ratio=0.1
    )
    print(f"   价格条件满足次数: {price_condition.sum()}")
    
    # 组合条件
    combined = zhangting_condition + volume_condition + price_condition
    print(f"\n组合条件统计:")
    for i in range(4):
        count = (combined == i).sum()
        percentage = count / len(combined) * 100
        print(f"  满足{i}个条件: {count}次 ({percentage:.1f}%)")


def test_with_real_data():
    """使用真实数据测试（如果可用）"""
    print("\n" + "=" * 60)
    print("使用真实数据测试")
    print("=" * 60)
    
    # 尝试加载真实数据
    data_path = "../../output/raw_data/000001.SZ_20250624_20250922.csv"
    
    if os.path.exists(data_path):
        print(f"加载真实数据: {data_path}")
        try:
            data = pd.read_csv(data_path)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            print(f"数据形状: {data.shape}")
            print(f"数据列: {list(data.columns)}")
            
            # 创建因子库并计算因子
            my_factors = create_my_factors()
            
            # 确保数据列名正确
            required_columns = ['close', 'high', 'volume']
            if all(col in data.columns for col in required_columns):
                factor_values = my_factors.calculate_factor(
                    'ZHANGTING_VOLUME_PRICE_FACTOR',
                    close=data['close'],
                    high=data['high'],
                    volume=data['volume'],
                    stock_code='000001.SZ'
                )
                
                print(f"真实数据因子值统计:")
                print(f"  范围: {factor_values.min():.2f} 到 {factor_values.max():.2f}")
                print(f"  均值: {factor_values.mean():.4f}")
                print(f"  非零值: {(factor_values > 0).sum()}")
                
                # 保存结果
                result_df = data.copy()
                result_df['zhangting_volume_price_factor'] = factor_values
                result_df.to_csv('myfactors_real_data_result.csv')
                print(f"真实数据结果已保存到: myfactors_real_data_result.csv")
                
            else:
                print(f"数据缺少必要列，需要: {required_columns}")
                print(f"实际列: {list(data.columns)}")
                
        except Exception as e:
            print(f"处理真实数据时出错: {e}")
    else:
        print(f"真实数据文件不存在: {data_path}")
        print("使用模拟数据继续测试...")


def demonstrate_factor_library_features():
    """演示因子库的各种功能"""
    print("\n" + "=" * 60)
    print("演示因子库功能")
    print("=" * 60)
    
    # 创建因子库
    my_factors = create_my_factors()
    
    # 显示因子信息
    print("1. 因子库统计信息:")
    stats = my_factors.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 显示因子详情
    print("\n2. 因子详情:")
    for factor_name in my_factors.list_factors():
        info = my_factors.get_factor_info(factor_name)
        print(f"\n   {factor_name}:")
        print(f"     描述: {info['description']}")
        print(f"     表达式: {info['expression']}")
        print(f"     类别: {info['category']}")
        if 'parameters' in info:
            print(f"     参数: {info['parameters']}")
    
    # 导出因子库
    print("\n3. 导出因子库:")
    df = my_factors.export_factors("my_factors_demo.csv")
    print(f"   已导出到: my_factors_demo.csv")
    print(f"   导出记录数: {len(df)}")
    
    # 添加自定义因子示例
    print("\n4. 添加自定义因子示例:")
    my_factors.add_custom_factor(
        name="SIMPLE_MA_FACTOR",
        expression="SIMPLE_MA_FACTOR($close, 20)",
        function_name="simple_ma_factor",
        description="简单移动平均因子",
        category="技术指标",
        parameters={"周期": 20}
    )
    
    print(f"   添加后因子数量: {len(my_factors.list_factors())}")


def main():
    """主函数"""
    print("自定义因子库使用示例")
    print("=" * 60)
    
    try:
        # 测试涨停+成交量+价格复合因子
        test_zhangting_factor()
        
        # 测试因子组成部分
        test_factor_components()
        
        # 使用真实数据测试
        test_with_real_data()
        
        # 演示因子库功能
        demonstrate_factor_library_features()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
