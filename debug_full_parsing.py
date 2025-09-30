import re
import pandas as pd
import numpy as np

# 模拟完整的表达式解析过程
def debug_expression_parsing(expression, data):
    print(f"原始表达式: {expression}")
    
    # 第一步：变量替换
    expr = expression
    if '$close' in expr and 'close' in data.columns:
        expr = expr.replace('$close', 'data["close"]')
    if '$high' in expr and 'high' in data.columns:
        expr = expr.replace('$high', 'data["high"]')
    if '$low' in expr and 'low' in data.columns:
        expr = expr.replace('$low', 'data["low"]')
    if '$open' in expr and 'open' in data.columns:
        expr = expr.replace('$open', 'data["open"]')
    if '$volume' in expr and 'volume' in data.columns:
        expr = expr.replace('$volume', 'data["volume"]')
    
    print(f"变量替换后: {expr}")
    
    # 第二步：替换Ref函数
    expr = re.sub(r'Ref\(([^,]+),\s*(\d+)\)', r'\1.shift(\2)', expr)
    print(f"Ref函数替换后: {expr}")
    
    # 第三步：替换Mean函数
    expr = re.sub(r'Mean\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).mean()', expr)
    print(f"Mean函数替换后: {expr}")
    
    # 第四步：替换Std函数
    expr = re.sub(r'Std\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).std()', expr)
    print(f"Std函数替换后: {expr}")
    
    # 第五步：替换Max函数
    expr = re.sub(r'Max\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).max()', expr)
    print(f"Max函数替换后: {expr}")
    
    # 第六步：替换Min函数
    expr = re.sub(r'Min\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).min()', expr)
    print(f"Min函数替换后: {expr}")
    
    # 第七步：替换Sum函数
    expr = re.sub(r'Sum\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).sum()', expr)
    print(f"Sum函数替换后: {expr}")
    
    # 第八步：替换其他函数
    expr = expr.replace('Greater(', 'np.maximum(')
    expr = expr.replace('Less(', 'np.minimum(')
    expr = expr.replace('Abs(', 'np.abs(')
    print(f"其他函数替换后: {expr}")
    
    # 第九步：处理IdxMax和IdxMin函数
    print(f"处理IdxMax前: {expr}")
    expr = re.sub(r'IdxMax\(data\["([^"]+)"\],\s*(\d+)\)', r'data["\1"].rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
    print(f"IdxMax函数替换后: {expr}")
    
    print(f"处理IdxMin前: {expr}")
    expr = re.sub(r'IdxMin\(data\["([^"]+)"\],\s*(\d+)\)', r'data["\1"].rolling(\2).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)', expr)
    print(f"IdxMin函数替换后: {expr}")
    
    # 第十步：处理Quantile函数
    expr = re.sub(r'Quantile\(([^,]+),\s*(\d+),\s*([0-9.]+)\)', r'\1.rolling(\2).quantile(\3)', expr)
    print(f"Quantile函数替换后: {expr}")
    
    # 第十一步：处理Rank函数
    expr = re.sub(r'Rank\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).rank(pct=True)', expr)
    print(f"Rank函数替换后: {expr}")
    
    # 第十二步：处理Corr函数
    expr = re.sub(r'Corr\(([^,]+),\s*([^,]+),\s*(\d+)\)', r'\1.rolling(\3).corr(\2)', expr)
    print(f"Corr函数替换后: {expr}")
    
    # 第十三步：处理Rsquare函数
    expr = re.sub(r'Rsquare\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).apply(lambda x: x.corr(pd.Series(range(len(x))))**2 if len(x) > 1 else np.nan)', expr)
    print(f"Rsquare函数替换后: {expr}")
    
    # 第十四步：处理Resi函数
    expr = re.sub(r'Resi\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).apply(lambda x: x.iloc[-1] - x.mean() if len(x) > 0 else np.nan)', expr)
    print(f"Resi函数替换后: {expr}")
    
    # 第十五步：处理Slope函数
    expr = re.sub(r'Slope\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan)', expr)
    print(f"Slope函数替换后: {expr}")
    
    # 第十六步：处理Log函数
    expr = re.sub(r'Log\(([^)]+)\)', r'np.log(\1)', expr)
    print(f"Log函数替换后: {expr}")
    
    # 第十七步：处理条件表达式
    expr = re.sub(r'Mean\(([^>]+)>([^,]+),\s*(\d+)\)', r'(\1 > \2).rolling(\3).mean()', expr)
    expr = re.sub(r'Mean\(([^<]+)<([^,]+),\s*(\d+)\)', r'(\1 < \2).rolling(\3).mean()', expr)
    print(f"条件表达式替换后: {expr}")
    
    # 第十八步：处理vwap变量
    if '$vwap' in expr:
        expr = expr.replace('$vwap', '((data["high"] + data["low"] + data["close"]) / 3)')
    print(f"vwap变量替换后: {expr}")
    
    print(f"最终表达式: {expr}")
    
    return expr

# 测试数据
data = pd.DataFrame({
    'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

# 测试IdxMax函数
print("=" * 80)
print("测试IdxMax函数")
print("=" * 80)
debug_expression_parsing("IdxMax($high, 5)", data)

print("\n" + "=" * 80)
print("测试IdxMin函数")
print("=" * 80)
debug_expression_parsing("IdxMin($low, 5)", data)
