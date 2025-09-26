# -*- coding: utf-8 -*-
"""
手动计算示例
详细演示ZHANGTING_VOLUME_PRICE_FACTOR因子的手动计算过程

作者: 用户自定义
创建时间: 2024年
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# 添加涨停判断模块路径
sys.path.append(r'd:\pythonProject\JQKA\indicator')
try:
    from zhangtingCalculation import limitUp, getAmplitude
except ImportError:
    print("警告: 无法导入涨停判断模块，使用模拟函数")
    def limitUp(close, stock_code='000001.SZ'):
        return close * 1.1  # 模拟涨停价计算
    def getAmplitude(stock_code):
        return 0.1


def create_detailed_sample_data():
    """创建详细的示例数据用于手动计算演示"""
    print("创建详细示例数据...")
    
    # 生成30天的数据
    n_days = 30
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 生成价格数据，包含一些特殊情况的模拟
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    # 模拟价格走势，包含一些涨停情况
    for i in range(1, n_days):
        if i == 10:  # 第10天模拟涨停
            change = 0.095  # 接近涨停
        elif i == 15:  # 第15天模拟涨停
            change = 0.08   # 接近涨停
        else:
            change = np.random.normal(0.001, 0.02)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # 生成开盘价、最高价、最低价
        open_price = price * (1 + np.random.normal(0, 0.005))
        
        # 如果是涨停日，确保最高价达到涨停价
        if i in [10, 15]:
            zhangting_price = limitUp(price, '000001.SZ')
            high_price = float(zhangting_price)
        else:
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price
        
        # 生成成交量，模拟真实波动
        base_volume = 5000000
        if i >= 20:  # 后期成交量放大
            volume = int(base_volume * (1 + np.random.normal(0.5, 0.3)))
        else:
            volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
        
        data.append({
            'close': close_price,
            'high': high_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    print(f"示例数据形状: {df.shape}")
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    return df


def manual_zhangting_check(close_series, high_series, stock_code, period=7):
    """手动检查涨停条件"""
    print(f"\n=== 手动检查涨停条件 (周期={period}) ===")
    
    result = []
    for i in range(len(close_series)):
        if i < period:
            result.append(0.0)
            continue
        
        # 检查过去period天是否有涨停
        window_close = close_series.iloc[i-period:i]
        window_high = high_series.iloc[i-period:i]
        
        has_zhangting = False
        for j in range(len(window_close)):
            zhangting_price = limitUp(window_close.iloc[j], stock_code)
            if window_high.iloc[j] >= float(zhangting_price):
                has_zhangting = True
                print(f"  第{i}天: 发现涨停 (第{i-period+j}天, 收盘价={window_close.iloc[j]:.2f}, 涨停价={float(zhangting_price):.2f}, 最高价={window_high.iloc[j]:.2f})")
                break
        
        result.append(1.0 if has_zhangting else 0.0)
    
    return pd.Series(result, index=close_series.index)


def manual_volume_check(volume_series, ma_period=20, compare_period=7, multiplier=1.5):
    """手动检查成交量条件"""
    print(f"\n=== 手动检查成交量条件 (均线周期={ma_period}, 比较周期={compare_period}, 倍数={multiplier}) ===")
    
    # 计算成交量20日均线
    volume_ma = volume_series.rolling(window=ma_period).mean()
    
    result = []
    for i in range(len(volume_series)):
        if i < ma_period + compare_period:
            result.append(0.0)
            continue
        
        current_volume_ma = volume_ma.iloc[i]
        compare_volume_ma = volume_ma.iloc[i - compare_period]
        
        condition_met = current_volume_ma > compare_volume_ma * multiplier
        
        if condition_met:
            print(f"  第{i}天: 成交量条件满足 (当前均线={current_volume_ma:,.0f}, {compare_period}天前均线={compare_volume_ma:,.0f}, 倍数={current_volume_ma/compare_volume_ma:.2f})")
        
        result.append(1.0 if condition_met else 0.0)
    
    return pd.Series(result, index=volume_series.index)


def manual_price_check(close_series, ma_period=20, max_ratio=0.1):
    """手动检查价格条件"""
    print(f"\n=== 手动检查价格条件 (均线周期={ma_period}, 最大比例={max_ratio}) ===")
    
    # 计算收盘价20日均线
    close_ma = close_series.rolling(window=ma_period).mean()
    
    result = []
    for i in range(len(close_series)):
        if i < ma_period:
            result.append(0.0)
            continue
        
        current_close = close_series.iloc[i]
        current_close_ma = close_ma.iloc[i]
        max_price = current_close_ma * (1 + max_ratio)
        
        condition_met = current_close <= max_price
        
        if condition_met:
            print(f"  第{i}天: 价格条件满足 (当前价格={current_close:.2f}, 20日均线={current_close_ma:.2f}, 上限={max_price:.2f})")
        
        result.append(1.0 if condition_met else 0.0)
    
    return pd.Series(result, index=close_series.index)


def demonstrate_manual_calculation():
    """演示手动计算过程"""
    print("=" * 80)
    print("ZHANGTING_VOLUME_PRICE_FACTOR 手动计算演示")
    print("=" * 80)
    
    # 创建示例数据
    data = create_detailed_sample_data()
    
    # 显示数据概览
    print(f"\n数据概览:")
    print(data.head(10))
    
    # 手动计算各个条件
    print(f"\n开始手动计算各个条件...")
    
    # 1. 涨停条件
    zhangting_condition = manual_zhangting_check(data['close'], data['high'], '000001.SZ', period=7)
    
    # 2. 成交量条件
    volume_condition = manual_volume_check(data['volume'], ma_period=20, compare_period=7, multiplier=1.5)
    
    # 3. 价格条件
    price_condition = manual_price_check(data['close'], ma_period=20, max_ratio=0.1)
    
    # 4. 组合因子
    print(f"\n=== 组合因子计算 ===")
    factor_values = zhangting_condition + volume_condition + price_condition
    
    # 显示结果
    print(f"\n计算结果:")
    print(f"数据长度: {len(factor_values)}")
    print(f"因子值范围: {factor_values.min():.0f} 到 {factor_values.max():.0f}")
    print(f"均值: {factor_values.mean():.2f}")
    
    # 显示因子值分布
    print(f"\n因子值分布:")
    for i in range(4):
        count = (factor_values == i).sum()
        percentage = count / len(factor_values) * 100
        print(f"  值={i}: {count}次 ({percentage:.1f}%)")
    
    # 显示满足条件的日期
    print(f"\n满足条件的日期:")
    condition_dates = factor_values[factor_values > 0]
    if len(condition_dates) > 0:
        for date, value in condition_dates.items():
            print(f"  {date.strftime('%Y-%m-%d')}: 因子值={value:.0f}")
    else:
        print("  没有找到满足条件的日期")
    
    # 详细分析最后几天的计算过程
    print(f"\n=== 最后5天详细计算过程 ===")
    for i in range(-5, 0):
        idx = len(factor_values) + i
        date = factor_values.index[idx]
        
        print(f"\n日期: {date.strftime('%Y-%m-%d')}")
        print(f"  收盘价: {data['close'].iloc[idx]:.2f}")
        print(f"  最高价: {data['high'].iloc[idx]:.2f}")
        print(f"  成交量: {data['volume'].iloc[idx]:,}")
        
        print(f"  条件1 (涨停): {zhangting_condition.iloc[idx]:.0f}")
        print(f"  条件2 (成交量): {volume_condition.iloc[idx]:.0f}")
        print(f"  条件3 (价格): {price_condition.iloc[idx]:.0f}")
        print(f"  组合因子: {factor_values.iloc[idx]:.0f}")
    
    # 保存结果
    result_df = data.copy()
    result_df['zhangting_condition'] = zhangting_condition
    result_df['volume_condition'] = volume_condition
    result_df['price_condition'] = price_condition
    result_df['factor_value'] = factor_values
    result_df.to_csv('manual_calculation_result.csv')
    print(f"\n详细结果已保存到: manual_calculation_result.csv")
    
    return result_df


def main():
    """主函数"""
    print("ZHANGTING_VOLUME_PRICE_FACTOR 手动计算演示")
    print("=" * 80)
    
    try:
        # 演示手动计算
        result_df = demonstrate_manual_calculation()
        
        print("\n" + "=" * 80)
        print("手动计算演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"计算过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
