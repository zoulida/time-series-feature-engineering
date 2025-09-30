import re
import pandas as pd
import numpy as np

# 模拟完整的表达式解析过程
def test_idxmax_parsing():
    # 测试数据
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    # 测试表达式
    expression = "IdxMax($high, 5)"
    print(f"原始表达式: {expression}")
    
    # 第一步：变量替换
    expr = expression
    if '$high' in expr and 'high' in data.columns:
        expr = expr.replace('$high', 'data["high"]')
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
    
    # 第五步：先处理IdxMax和IdxMin函数（避免被Max和Min函数误匹配）
    expr = re.sub(r'IdxMax\(data\["([^"]+)"\],\s*(\d+)\)', r'data["\1"].rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
    print(f"IdxMax函数替换后: {expr}")
    
    expr = re.sub(r'IdxMin\(data\["([^"]+)"\],\s*(\d+)\)', r'data["\1"].rolling(\2).apply(lambda x: x.idxmin() - x.index[0] if len(x) > 0 else np.nan)', expr)
    print(f"IdxMin函数替换后: {expr}")
    
    # 第六步：替换Max函数
    expr = re.sub(r'Max\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).max()', expr)
    print(f"Max函数替换后: {expr}")
    
    # 第七步：替换Min函数
    expr = re.sub(r'Min\(([^,]+),\s*(\d+)\)', r'\1.rolling(\2).min()', expr)
    print(f"Min函数替换后: {expr}")
    
    print(f"最终表达式: {expr}")
    
    # 尝试执行表达式
    try:
        result = eval(expr)
        print(f"执行成功！结果: {result}")
        return True
    except Exception as e:
        print(f"执行失败: {e}")
        return False

if __name__ == "__main__":
    print("测试IdxMax函数解析修复")
    print("=" * 50)
    success = test_idxmax_parsing()
    print(f"测试结果: {'成功' if success else '失败'}")
