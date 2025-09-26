# -*- coding: utf-8 -*-
"""
统一因子库
整合Alpha158、tsfresh和其他因子源的统一接口

支持多种因子源：
1. Alpha158因子 - 量化投资专用因子
2. tsfresh因子 - 通用时间序列特征
3. 自定义因子 - 用户自定义因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings

# 导入各个因子库
try:
    from alpha158_factors import Alpha158Factors
except ImportError:
    Alpha158Factors = None
    warnings.warn("Alpha158因子库导入失败，请检查alpha158_factors.py文件")

try:
    from extract_tsfresh_features import create_tsfresh_factor_library
except ImportError:
    create_tsfresh_factor_library = None
    warnings.warn("tsfresh因子库导入失败，请检查extract_tsfresh_features.py文件")
except Exception as e:
    create_tsfresh_factor_library = None
    warnings.warn(f"tsfresh因子库导入失败: {e}")


class FactorSource(ABC):
    """因子源抽象基类"""
    
    @abstractmethod
    def get_factors(self) -> Dict[str, Dict]:
        """获取所有因子"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """获取因子源名称"""
        pass
    
    @abstractmethod
    def get_factor_count(self) -> int:
        """获取因子数量"""
        pass


class Alpha158FactorSource(FactorSource):
    """Alpha158因子源"""
    
    def __init__(self):
        if Alpha158Factors is None:
            raise ImportError("Alpha158因子库不可用")
        self.factor_lib = Alpha158Factors()
    
    def get_factors(self) -> Dict[str, Dict]:
        """获取Alpha158因子"""
        factors = {}
        for name, info in self.factor_lib.get_all_factors().items():
            factors[f"ALPHA_{name}"] = {
                **info,
                "source": "Alpha158",
                "category": f"Alpha158_{info.get('category', '未知')}"
            }
        return factors
    
    def get_source_name(self) -> str:
        return "Alpha158"
    
    def get_factor_count(self) -> int:
        return len(self.factor_lib.get_all_factors())


class TSFreshFactorSource(FactorSource):
    """tsfresh因子源"""
    
    def __init__(self):
        if create_tsfresh_factor_library is None:
            raise ImportError("tsfresh因子库不可用")
        self.factor_lib, _ = create_tsfresh_factor_library()
    
    def get_factors(self) -> Dict[str, Dict]:
        """获取tsfresh因子"""
        factors = {}
        for name, info in self.factor_lib.items():
            factors[f"TSFRESH_{name}"] = {
                **info,
                "source": "tsfresh",
                "category": info.get('category', 'tsfresh_其他特征')
            }
        return factors
    
    def get_source_name(self) -> str:
        return "tsfresh"
    
    def get_factor_count(self) -> int:
        return len(self.factor_lib)


class CustomFactorSource(FactorSource):
    """自定义因子源"""
    
    def __init__(self, factors: Dict[str, Dict]):
        """
        初始化自定义因子源
        
        Parameters:
        factors: 自定义因子字典，格式为 {因子名: {因子信息}}
        """
        self.factors = factors
    
    def get_factors(self) -> Dict[str, Dict]:
        """获取自定义因子"""
        factors = {}
        for name, info in self.factors.items():
            factors[f"CUSTOM_{name}"] = {
                **info,
                "source": "custom",
                "category": info.get('category', '自定义因子')
            }
        return factors
    
    def get_source_name(self) -> str:
        return "custom"
    
    def get_factor_count(self) -> int:
        return len(self.factors)


class UnifiedFactorLibrary:
    """统一因子库类"""
    
    def __init__(self, sources: Optional[List[str]] = None):
        """
        初始化统一因子库
        
        Parameters:
        sources: 要加载的因子源列表，可选值: ['alpha158', 'tsfresh', 'custom']
                如果为None，则加载所有可用的因子源
        """
        self.sources = {}
        self.factors = {}
        self._initialize_sources(sources)
        self._load_factors()
    
    def _initialize_sources(self, sources: Optional[List[str]]):
        """初始化因子源"""
        available_sources = []
        
        # 尝试加载Alpha158
        if sources is None or 'alpha158' in sources:
            try:
                self.sources['alpha158'] = Alpha158FactorSource()
                available_sources.append('alpha158')
            except ImportError as e:
                warnings.warn(f"无法加载Alpha158因子源: {e}")
        
        # 尝试加载tsfresh
        if sources is None or 'tsfresh' in sources:
            try:
                self.sources['tsfresh'] = TSFreshFactorSource()
                available_sources.append('tsfresh')
            except ImportError as e:
                warnings.warn(f"无法加载tsfresh因子源: {e}")
        
        if not available_sources:
            raise RuntimeError("没有可用的因子源")
        
        print(f"成功加载因子源: {', '.join(available_sources)}")
    
    def _load_factors(self):
        """加载所有因子"""
        self.factors = {}
        for source_name, source in self.sources.items():
            try:
                source_factors = source.get_factors()
                self.factors.update(source_factors)
                print(f"从 {source_name} 加载了 {len(source_factors)} 个因子")
            except Exception as e:
                warnings.warn(f"从 {source_name} 加载因子时出错: {e}")
    
    def add_custom_factors(self, factors: Dict[str, Dict]):
        """添加自定义因子"""
        custom_source = CustomFactorSource(factors)
        custom_factors = custom_source.get_factors()
        self.factors.update(custom_factors)
        print(f"添加了 {len(custom_factors)} 个自定义因子")
    
    def get_all_factors(self) -> Dict[str, Dict]:
        """获取所有因子"""
        return self.factors
    
    def get_factors_by_source(self, source: str) -> Dict[str, Dict]:
        """按来源获取因子"""
        return {name: info for name, info in self.factors.items() 
                if info.get('source') == source}
    
    def get_factors_by_category(self, category: str) -> Dict[str, Dict]:
        """按类别获取因子"""
        return {name: info for name, info in self.factors.items() 
                if category in info.get('category', '')}
    
    def get_factor_info(self, factor_name: str) -> Optional[Dict]:
        """获取指定因子的详细信息"""
        return self.factors.get(factor_name)
    
    def get_factor_expression(self, factor_name: str) -> Optional[str]:
        """获取因子表达式"""
        factor_info = self.get_factor_info(factor_name)
        return factor_info.get('expression') if factor_info else None
    
    def get_factor_function_name(self, factor_name: str) -> Optional[str]:
        """获取因子函数名"""
        factor_info = self.get_factor_info(factor_name)
        return factor_info.get('function_name') if factor_info else None
    
    def get_factor_description(self, factor_name: str) -> Optional[str]:
        """获取因子描述"""
        factor_info = self.get_factor_info(factor_name)
        return factor_info.get('description') if factor_info else None
    
    def list_factors(self) -> List[str]:
        """列出所有因子名称"""
        return list(self.factors.keys())
    
    def search_factors(self, keyword: str) -> Dict[str, Dict]:
        """搜索包含关键词的因子"""
        keyword = keyword.lower()
        return {name: info for name, info in self.factors.items() 
                if keyword in name.lower() or keyword in info.get('description', '').lower()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取因子库统计信息"""
        stats = {
            "总因子数量": len(self.factors),
            "按来源统计": {},
            "按类别统计": {},
            "可用因子源": list(self.sources.keys())
        }
        
        for name, info in self.factors.items():
            source = info.get('source', '未知')
            category = info.get('category', '未知')
            
            # 统计来源
            if source not in stats["按来源统计"]:
                stats["按来源统计"][source] = 0
            stats["按来源统计"][source] += 1
            
            # 统计类别
            if category not in stats["按类别统计"]:
                stats["按类别统计"][category] = 0
            stats["按类别统计"][category] += 1
        
        return stats
    
    def export_to_csv(self, filename: str = "unified_factors_export.csv"):
        """导出因子库到CSV文件"""
        data = []
        for factor_name, info in self.factors.items():
            data.append({
                '因子名称': factor_name,
                '原始名称': factor_name.replace('ALPHA_', '').replace('TSFRESH_', '').replace('CUSTOM_', ''),
                '表达式': info.get('expression', ''),
                '函数名': info.get('function_name', ''),
                '描述': info.get('description', ''),
                '类别': info.get('category', ''),
                '来源': info.get('source', ''),
                '模块': info.get('module', '')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"因子库已导出到: {filename}")
        return df
    
    def get_factor_summary(self) -> pd.DataFrame:
        """获取因子摘要信息"""
        data = []
        for name, info in self.factors.items():
            data.append({
                '因子名称': name,
                '函数名': info.get('function_name', ''),
                '类别': info.get('category', ''),
                '来源': info.get('source', ''),
                '描述': info.get('description', ''),
                '表达式': info.get('expression', '')
            })
        return pd.DataFrame(data)
    
    def filter_factors(self, 
                      sources: Optional[List[str]] = None,
                      categories: Optional[List[str]] = None,
                      keywords: Optional[List[str]] = None) -> Dict[str, Dict]:
        """过滤因子"""
        filtered = self.factors.copy()
        
        # 按来源过滤
        if sources:
            filtered = {name: info for name, info in filtered.items() 
                       if info.get('source') in sources}
        
        # 按类别过滤
        if categories:
            filtered = {name: info for name, info in filtered.items() 
                       if any(cat in info.get('category', '') for cat in categories)}
        
        # 按关键词过滤
        if keywords:
            keyword_filtered = {}
            for name, info in filtered.items():
                if any(keyword.lower() in name.lower() or 
                      keyword.lower() in info.get('description', '').lower() 
                      for keyword in keywords):
                    keyword_filtered[name] = info
            filtered = keyword_filtered
        
        return filtered


def create_unified_factor_library(sources: Optional[List[str]] = None) -> UnifiedFactorLibrary:
    """创建统一因子库的便捷函数"""
    return UnifiedFactorLibrary(sources)


def main():
    """主函数，演示统一因子库的使用"""
    print("=" * 60)
    print("统一因子库演示")
    print("=" * 60)
    
    # 创建统一因子库
    try:
        factor_lib = UnifiedFactorLibrary()
    except Exception as e:
        print(f"创建因子库失败: {e}")
        return
    
    # 获取统计信息
    stats = factor_lib.get_statistics()
    print(f"\n总因子数量: {stats['总因子数量']}")
    print(f"可用因子源: {', '.join(stats['可用因子源'])}")
    
    print("\n按来源统计:")
    for source, count in stats["按来源统计"].items():
        print(f"  {source}: {count}个")
    
    print("\n按类别统计 (前10个):")
    sorted_categories = sorted(stats["按类别统计"].items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:10]:
        print(f"  {category}: {count}个")
    
    # 演示因子查询
    print("\n因子查询演示:")
    
    # 查询Alpha158因子
    alpha_factors = factor_lib.get_factors_by_source('Alpha158')
    print(f"Alpha158因子数量: {len(alpha_factors)}")
    
    # 查询tsfresh因子
    tsfresh_factors = factor_lib.get_factors_by_source('tsfresh')
    print(f"tsfresh因子数量: {len(tsfresh_factors)}")
    
    # 搜索包含"mean"的因子
    mean_factors = factor_lib.search_factors("mean")
    print(f"包含'mean'的因子数量: {len(mean_factors)}")
    
    # 显示示例因子
    print("\n示例因子:")
    sample_factors = list(factor_lib.factors.keys())[:10]
    for factor in sample_factors:
        info = factor_lib.factors[factor]
        print(f"  {factor}: {info['description']} ({info['source']})")
    
    # 导出因子库
    df = factor_lib.export_to_csv()
    
    print("\n" + "=" * 60)
    print("统一因子库演示完成！")
    print("=" * 60)
    
    return factor_lib


if __name__ == "__main__":
    main()
