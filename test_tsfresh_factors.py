# -*- coding: utf-8 -*-
"""
测试新的tsfresh因子库
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加路径
sys.path.append('source/因子库')

def test_tsfresh_factors():
    """测试tsfresh因子库"""
    try:
        from tsfresh_factors import TSFreshFactors
        
        print("=" * 60)
        print("测试tsfresh因子库")
        print("=" * 60)
        
        # 创建因子库实例
        tsfresh_lib = TSFreshFactors()
        
        # 获取所有因子
        factors = tsfresh_lib.get_all_factors()
        print(f"总因子数量: {len(factors)}")
        
        # 显示前10个因子
        print("\n前10个因子:")
        for i, (name, info) in enumerate(list(factors.items())[:10]):
            print(f"  {i+1:2d}. {name}: {info['description']}")
        
        # 创建测试数据
        np.random.seed(42)
        test_data = pd.Series(np.random.randn(50).cumsum() + 100, 
                             index=pd.date_range('2023-01-01', periods=50))
        
        print(f"\n测试数据长度: {len(test_data)}")
        print(f"测试数据前5个值: {test_data.head().values}")
        
        # 测试几个基础因子
        test_factors = ['tsfresh_mean', 'tsfresh_std', 'tsfresh_max', 'tsfresh_min']
        
        print("\n测试因子计算:")
        for factor_name in test_factors:
            try:
                if hasattr(tsfresh_lib, factor_name):
                    result = getattr(tsfresh_lib, factor_name)(test_data)
                    print(f"  {factor_name}: {result.iloc[0] if len(result) > 0 else 'N/A'}")
                else:
                    print(f"  {factor_name}: 函数不存在")
            except Exception as e:
                print(f"  {factor_name}: 计算失败 - {str(e)}")
        
        print("\n✅ tsfresh因子库测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tsfresh_factors()
