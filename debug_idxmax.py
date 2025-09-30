import re

# 模拟变量替换过程
expr = "IdxMax($high, 5)"
print(f"原始表达式: {expr}")

# 第一步：变量替换
expr = expr.replace('$high', 'data["high"]')
print(f"变量替换后: {expr}")

# 第二步：尝试不同的正则表达式
patterns = [
    r'IdxMax\(data\["([^"]+)"\],\s*(\d+)\)',
    r'IdxMax\(([^,)]+),\s*(\d+)\)',
    r'IdxMax\(\$([^,)]+),\s*(\d+)\)'
]

for i, pattern in enumerate(patterns):
    match = re.search(pattern, expr)
    if match:
        print(f"模式 {i+1} 匹配成功: {pattern}")
        print(f"  匹配组: {match.groups()}")
        result = re.sub(pattern, r'data["\1"].rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
        print(f"  替换结果: {result}")
    else:
        print(f"模式 {i+1} 匹配失败: {pattern}")

# 测试完整的替换过程
expr = "IdxMax($high, 5)"
expr = expr.replace('$high', 'data["high"]')
pattern = r'IdxMax\(data\["([^"]+)"\],\s*(\d+)\)'
result = re.sub(pattern, r'data["\1"].rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
print(f"\n完整替换结果: {result}")
