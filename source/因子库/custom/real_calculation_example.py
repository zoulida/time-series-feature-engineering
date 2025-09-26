# -*- coding: utf-8 -*-
"""
真实计算示例
使用真实股票数据演示ZHANGTING_VOLUME_PRICE_FACTOR因子的计算过程

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


def load_real_data():
    """加载真实股票数据"""
    # 尝试加载真实数据
    data_paths = [
        "../../output/raw_data/000001.SZ_20250624_20250922.csv",
        "../output/raw_data/000001.SZ_20250624_20250922.csv",
        "output/raw_data/000001.SZ_20250624_20250922.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            try:
                data = pd.read_csv(path)
                print(f"原始数据形状: {data.shape}")
                print(f"原始数据列: {list(data.columns)}")
                
                # 处理数据格式
                if 'time' in data.columns:
                    data['date'] = pd.to_datetime(data['time'])
                    data.set_index('date', inplace=True)
                elif 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)
                
                # 确保列名正确
                column_mapping = {
                    'close': 'close',
                    'high': 'high', 
                    'volume': 'volume'
                }
                
                for old_name, new_name in column_mapping.items():
                    if old_name in data.columns:
                        data[new_name] = data[old_name]
                
                # 选择需要的列
                required_columns = ['close', 'high', 'volume']
                if all(col in data.columns for col in required_columns):
                    result_data = data[required_columns].copy()
                    print(f"处理后数据形状: {result_data.shape}")
                    print(f"处理后数据列: {list(result_data.columns)}")
                    print(f"数据时间范围: {result_data.index[0]} 到 {result_data.index[-1]}")
                    return result_data
                else:
                    print(f"数据缺少必要列，需要: {required_columns}")
                    print(f"实际列: {list(data.columns)}")
                    
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue
    
    print("未找到真实数据文件，使用模拟数据...")
    return create_simulated_data()


def create_simulated_data():
    """创建模拟数据用于演示"""
    print("创建模拟数据用于演示...")
    
    # 生成日期序列
    n_days = 50
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 生成更真实的股票数据
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
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
        
        # 生成成交量（模拟真实波动）
        base_volume = 5000000
        volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
        
        data.append({
            'close': close_price,
            'high': high_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    print(f"模拟数据形状: {df.shape}")
    print(f"模拟数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    return df


def demonstrate_factor_calculation():
    """演示因子计算过程"""
    print("=" * 80)
    print("ZHANGTING_VOLUME_PRICE_FACTOR 真实计算示例")
    print("=" * 80)
    
    # 加载数据
    data = load_real_data()
    
    # 创建因子库
    my_factors = create_my_factors()
    
    # 计算因子
    print("\n1. 计算因子...")
    factor_values = my_factors.calculate_factor(
        'ZHANGTING_VOLUME_PRICE_FACTOR',
        close=data['close'],
        high=data['high'],
        volume=data['volume'],
        stock_code='000001.SZ'
    )
    
    # 基本统计
    print(f"\n2. 因子值统计:")
    print(f"   数据长度: {len(factor_values)}")
    print(f"   因子值范围: {factor_values.min():.2f} 到 {factor_values.max():.2f}")
    print(f"   均值: {factor_values.mean():.4f}")
    print(f"   标准差: {factor_values.std():.4f}")
    
    # 因子值分布
    print(f"\n3. 因子值分布:")
    for i in range(4):
        count = (factor_values == i).sum()
        percentage = count / len(factor_values) * 100
        print(f"   值={i}: {count}次 ({percentage:.1f}%)")
    
    # 显示满足条件的日期
    print(f"\n4. 满足条件的日期:")
    condition_dates = factor_values[factor_values > 0]
    if len(condition_dates) > 0:
        print(f"   找到 {len(condition_dates)} 个满足条件的日期:")
        for date, value in condition_dates.items():
            print(f"     {date.strftime('%Y-%m-%d')}: 因子值={value:.0f}")
    else:
        print("   没有找到满足条件的日期")
    
    # 详细分析最近几天的数据
    print(f"\n5. 最近10天详细分析:")
    recent_data = data.tail(10)
    recent_factors = factor_values.tail(10)
    
    for i, (date, row) in enumerate(recent_data.iterrows()):
        factor_val = recent_factors.iloc[i]
        print(f"\n   日期: {date.strftime('%Y-%m-%d')}")
        print(f"   收盘价: {row['close']:.2f}")
        print(f"   最高价: {row['high']:.2f}")
        print(f"   成交量: {row['volume']:,}")
        print(f"   因子值: {factor_val:.0f}")
        
        # 分析各个条件
        if i >= 20:  # 确保有足够的历史数据
            print(f"   分析:")
            
            # 涨停条件分析
            zhangting_condition = my_factors._check_zhangting_in_period(
                data['close'], data['high'], '000001.SZ', period=7
            )
            zhangting_val = zhangting_condition.iloc[i]
            print(f"     - 涨停条件(7日内): {zhangting_val:.0f}")
            
            # 成交量条件分析
            volume_condition = my_factors._check_volume_condition(
                data['volume'], ma_period=20, compare_period=7, multiplier=1.5
            )
            volume_val = volume_condition.iloc[i]
            print(f"     - 成交量条件: {volume_val:.0f}")
            
            # 价格条件分析
            price_condition = my_factors._check_price_condition(
                data['close'], ma_period=20, max_ratio=0.1
            )
            price_val = price_condition.iloc[i]
            print(f"     - 价格条件: {price_val:.0f}")
    
    # 保存结果
    result_df = data.copy()
    result_df['zhangting_volume_price_factor'] = factor_values
    result_df.to_csv('real_calculation_result.csv')
    print(f"\n6. 结果已保存到: real_calculation_result.csv")
    
    return result_df


def analyze_factor_components():
    """分析因子的各个组成部分"""
    print("\n" + "=" * 80)
    print("因子组成部分详细分析")
    print("=" * 80)
    
    # 加载数据
    data = load_real_data()
    my_factors = create_my_factors()
    
    # 分别计算各个条件
    print("\n1. 涨停条件分析:")
    zhangting_condition = my_factors._check_zhangting_in_period(
        data['close'], data['high'], '000001.SZ', period=7
    )
    print(f"   涨停条件满足次数: {zhangting_condition.sum()}")
    print(f"   涨停条件满足比例: {zhangting_condition.mean()*100:.1f}%")
    
    print("\n2. 成交量条件分析:")
    volume_condition = my_factors._check_volume_condition(
        data['volume'], ma_period=20, compare_period=7, multiplier=1.5
    )
    print(f"   成交量条件满足次数: {volume_condition.sum()}")
    print(f"   成交量条件满足比例: {volume_condition.mean()*100:.1f}%")
    
    print("\n3. 价格条件分析:")
    price_condition = my_factors._check_price_condition(
        data['close'], ma_period=20, max_ratio=0.1
    )
    print(f"   价格条件满足次数: {price_condition.sum()}")
    print(f"   价格条件满足比例: {price_condition.mean()*100:.1f}%")
    
    # 组合条件分析
    print("\n4. 组合条件分析:")
    combined = zhangting_condition + volume_condition + price_condition
    for i in range(4):
        count = (combined == i).sum()
        percentage = count / len(combined) * 100
        print(f"   满足{i}个条件: {count}次 ({percentage:.1f}%)")
    
    # 相关性分析
    print("\n5. 条件间相关性分析:")
    conditions_df = pd.DataFrame({
        '涨停条件': zhangting_condition,
        '成交量条件': volume_condition,
        '价格条件': price_condition
    })
    correlation = conditions_df.corr()
    print("   条件间相关系数矩阵:")
    print(correlation.round(3))


def main():
    """主函数"""
    print("ZHANGTING_VOLUME_PRICE_FACTOR 真实计算示例")
    print("=" * 80)
    
    try:
        # 演示因子计算
        result_df = demonstrate_factor_calculation()
        
        # 分析因子组成部分
        analyze_factor_components()
        
        print("\n" + "=" * 80)
        print("真实计算示例完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"计算过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
