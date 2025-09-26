# -*- coding: utf-8 -*-
"""
连续评分示例
详细演示ZHANGTING_VOLUME_PRICE_FACTOR因子的连续评分计算过程

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

# 导入自定义因子库
from myfactors import MyFactors, create_my_factors


def create_detailed_sample_data():
    """创建包含涨停情况的详细示例数据"""
    print("创建包含涨停情况的详细示例数据...")
    
    # 生成30天的数据
    n_days = 30
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 生成价格数据，包含涨停情况
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    # 模拟价格走势，包含涨停情况
    for i in range(1, n_days):
        if i == 5:  # 第5天涨停
            change = 0.095  # 接近涨停
        elif i == 6:  # 第6天连续涨停
            change = 0.095  # 连续涨停
        elif i == 12:  # 第12天涨停
            change = 0.08   # 涨停
        elif i == 18:  # 第18天涨停
            change = 0.09   # 涨停
        elif i == 19:  # 第19天连续涨停
            change = 0.09   # 连续涨停
        elif i == 20:  # 第20天连续涨停
            change = 0.09   # 连续涨停
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
        if i in [5, 6, 12, 18, 19, 20]:
            zhangting_price = limitUp(price, '000001.SZ')
            high_price = float(zhangting_price)
        else:
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price
        
        # 生成成交量，模拟真实波动
        base_volume = 5000000
        if i >= 15:  # 后期成交量放大
            volume = int(base_volume * (1 + np.random.normal(0.8, 0.4)))
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


def demonstrate_continuous_scoring():
    """演示连续评分计算过程"""
    print("=" * 80)
    print("ZHANGTING_VOLUME_PRICE_FACTOR 连续评分演示")
    print("=" * 80)
    
    # 创建示例数据
    data = create_detailed_sample_data()
    
    # 创建因子库
    my_factors = create_my_factors()
    
    # 计算各个评分
    print("\n1. 计算各个评分...")
    zhangting_score = my_factors._calculate_zhangting_score(
        data['close'], data['high'], '000001.SZ', period=7
    )
    volume_score = my_factors._calculate_volume_score(
        data['volume'], ma_period=20, compare_period=7, multiplier=1.5
    )
    price_score = my_factors._calculate_price_score(
        data['close'], ma_period=20, max_ratio=0.1
    )
    
    # 计算组合因子
    factor_values = zhangting_score + volume_score + price_score
    
    # 显示评分统计
    print(f"\n2. 评分统计:")
    print(f"   涨停评分: 均值={zhangting_score.mean():.3f}, 范围={zhangting_score.min():.3f}-{zhangting_score.max():.3f}")
    print(f"   成交量评分: 均值={volume_score.mean():.3f}, 范围={volume_score.min():.3f}-{volume_score.max():.3f}")
    print(f"   价格评分: 均值={price_score.mean():.3f}, 范围={price_score.min():.3f}-{price_score.max():.3f}")
    print(f"   组合因子: 均值={factor_values.mean():.3f}, 范围={factor_values.min():.3f}-{factor_values.max():.3f}")
    
    # 详细分析最后10天
    print(f"\n3. 最后10天详细评分分析:")
    recent_data = data.tail(10)
    recent_zhangting = zhangting_score.tail(10)
    recent_volume = volume_score.tail(10)
    recent_price = price_score.tail(10)
    recent_factor = factor_values.tail(10)
    
    for i, (date, row) in enumerate(recent_data.iterrows()):
        idx = len(data) - 10 + i
        print(f"\n   日期: {date.strftime('%Y-%m-%d')}")
        print(f"     收盘价: {row['close']:.2f}")
        print(f"     最高价: {row['high']:.2f}")
        print(f"     成交量: {row['volume']:,}")
        
        print(f"     涨停评分: {recent_zhangting.iloc[i]:.3f}")
        print(f"     成交量评分: {recent_volume.iloc[i]:.3f}")
        print(f"     价格评分: {recent_price.iloc[i]:.3f}")
        print(f"     组合因子: {recent_factor.iloc[i]:.3f}")
        
        # 分析涨停情况
        if idx >= 7:
            window_close = data['close'].iloc[idx-7:idx]
            window_high = data['high'].iloc[idx-7:idx]
            
            zhangting_count = 0
            in_zhangting_sequence = False
            
            for j in range(len(window_close)):
                zhangting_price = limitUp(window_close.iloc[j], '000001.SZ')
                is_zhangting = window_high.iloc[j] >= float(zhangting_price)
                
                if is_zhangting and not in_zhangting_sequence:
                    zhangting_count += 1
                    in_zhangting_sequence = True
                elif not is_zhangting:
                    in_zhangting_sequence = False
            
            print(f"     7日内涨停次数: {zhangting_count}")
            
            # 分析成交量情况
            if idx >= 27:
                current_volume_ma = data['volume'].rolling(20).mean().iloc[idx]
                compare_volume_ma = data['volume'].rolling(20).mean().iloc[idx-7]
                ratio = current_volume_ma / compare_volume_ma
                print(f"     成交量倍数: {ratio:.3f}")
            
            # 分析价格情况
            if idx >= 20:
                current_close = row['close']
                close_ma = data['close'].rolling(20).mean().iloc[idx]
                price_ratio = current_close / close_ma
                print(f"     价格比例: {price_ratio:.3f}")
    
    # 保存详细结果
    result_df = data.copy()
    result_df['zhangting_score'] = zhangting_score
    result_df['volume_score'] = volume_score
    result_df['price_score'] = price_score
    result_df['factor_value'] = factor_values
    result_df.to_csv('continuous_scoring_result.csv')
    print(f"\n4. 详细结果已保存到: continuous_scoring_result.csv")
    
    return result_df


def analyze_scoring_distribution():
    """分析评分分布"""
    print("\n" + "=" * 80)
    print("评分分布分析")
    print("=" * 80)
    
    # 创建示例数据
    data = create_detailed_sample_data()
    my_factors = create_my_factors()
    
    # 计算评分
    zhangting_score = my_factors._calculate_zhangting_score(
        data['close'], data['high'], '000001.SZ', period=7
    )
    volume_score = my_factors._calculate_volume_score(
        data['volume'], ma_period=20, compare_period=7, multiplier=1.5
    )
    price_score = my_factors._calculate_price_score(
        data['close'], ma_period=20, max_ratio=0.1
    )
    factor_values = zhangting_score + volume_score + price_score
    
    # 评分分布
    print("\n1. 评分分布:")
    score_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    for name, scores in [("涨停评分", zhangting_score), ("成交量评分", volume_score), 
                        ("价格评分", price_score), ("组合因子", factor_values)]:
        print(f"\n   {name}:")
        for low, high in score_ranges:
            count = ((scores >= low) & (scores < high)).sum()
            percentage = count / len(scores) * 100
            print(f"     {low:.1f}-{high:.1f}: {count}次 ({percentage:.1f}%)")
    
    # 相关性分析
    print("\n2. 评分相关性分析:")
    scores_df = pd.DataFrame({
        '涨停评分': zhangting_score,
        '成交量评分': volume_score,
        '价格评分': price_score
    })
    correlation = scores_df.corr()
    print(correlation.round(3))
    
    # 高分情况分析
    print("\n3. 高分情况分析:")
    high_score_threshold = 0.8
    high_score_dates = factor_values[factor_values >= high_score_threshold]
    
    if len(high_score_dates) > 0:
        print(f"   高分日期 (因子值 >= {high_score_threshold}):")
        for date, score in high_score_dates.items():
            print(f"     {date.strftime('%Y-%m-%d')}: {score:.3f}")
    else:
        print(f"   没有找到高分日期 (因子值 >= {high_score_threshold})")


def main():
    """主函数"""
    print("ZHANGTING_VOLUME_PRICE_FACTOR 连续评分演示")
    print("=" * 80)
    
    try:
        # 演示连续评分
        result_df = demonstrate_continuous_scoring()
        
        # 分析评分分布
        analyze_scoring_distribution()
        
        print("\n" + "=" * 80)
        print("连续评分演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
