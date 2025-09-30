with open('step3_generate_training_data_batch.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复第49行的缩进（索引为48）
lines[48] = '            self.factor_lib = UnifiedFactorLibrary()\n'

with open('step3_generate_training_data_batch.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("修复完成！")
