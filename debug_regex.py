import re

# 测试正则表达式
expr = "IdxMax($high, 5)"

# 当前的正则表达式
pattern1 = r'IdxMax\(([^,)]+),\s*(\d+)\)'
result1 = re.sub(pattern1, r'\1.rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
print(f"原始表达式: {expr}")
print(f"当前正则: {pattern1}")
print(f"结果: {result1}")

# 改进的正则表达式
pattern2 = r'IdxMax\(([^,)]+),\s*(\d+)\)'
result2 = re.sub(pattern2, r'\1.rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
print(f"改进正则: {pattern2}")
print(f"结果: {result2}")

# 更精确的正则表达式
pattern3 = r'IdxMax\(\$([^,)]+),\s*(\d+)\)'
result3 = re.sub(pattern3, r'data["\1"].rolling(\2).apply(lambda x: x.idxmax() - x.index[0] if len(x) > 0 else np.nan)', expr)
print(f"精确正则: {pattern3}")
print(f"结果: {result3}")

# 测试匹配
match = re.search(pattern1, expr)
if match:
    print(f"匹配组1: '{match.group(1)}'")
    print(f"匹配组2: '{match.group(2)}'")
else:
    print("没有匹配")
