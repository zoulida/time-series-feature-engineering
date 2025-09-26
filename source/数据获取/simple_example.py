"""
股票数据获取模块简单使用示例
"""

from stock_data_fetcher import StockDataFetcher

def main():
    """简单使用示例"""
    print("股票数据获取模块简单示例")
    print("=" * 40)
    
    # 1. 创建数据获取器
    fetcher = StockDataFetcher()
    
    # 2. 获取可用股票
    stocks = fetcher.get_available_stocks()
    print(f"可用股票数量: {len(stocks)}")
    
    # 3. 选择几只股票
    sample_stocks = stocks[:3]
    print(f"示例股票: {sample_stocks}")
    
    # 4. 设置日期范围
    start_date = "20250101"
    end_date = "20250110"
    print(f"日期范围: {start_date} 到 {end_date}")
    
    # 5. 加载数据
    print("\n加载股票数据...")
    stock_data = fetcher.load_multiple_stocks_data(sample_stocks, start_date, end_date)
    
    # 6. 显示数据摘要
    summary = fetcher.get_data_summary(stock_data)
    print("\n数据摘要:")
    print(summary[['stock_code', 'start_date', 'end_date', 'total_days']].to_string(index=False))
    
    # 7. 显示单只股票数据
    first_stock = list(stock_data.keys())[0]
    df = stock_data[first_stock]
    print(f"\n{first_stock} 数据预览:")
    print(df[['date', 'open', 'high', 'low', 'close', 'volume']].head())
    
    # 8. 创建面板数据
    print("\n创建面板数据...")
    panel_data = fetcher.create_panel_data(stock_data, 'close')
    print(f"面板数据形状: {panel_data.shape}")
    print(panel_data.head())
    
    # 9. 保存数据
    print("\n保存数据...")
    fetcher.save_data(df, 'output/sample_data.csv', 'csv')
    fetcher.save_data(panel_data, 'output/panel_data.csv', 'csv')
    print("数据已保存到 output/ 目录")
    
    print("\n✅ 示例完成！")

if __name__ == "__main__":
    main()
