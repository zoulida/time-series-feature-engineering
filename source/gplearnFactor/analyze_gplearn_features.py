#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析GPLearn生成的特征，尝试推断可能的表达式
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def analyze_gplearn_features():
    """分析GPLearn生成的特征"""
    print("="*60)
    print("🔍 GPLearn特征表达式分析")
    print("="*60)
    
    # 加载数据
    print("加载数据...")
    X = pd.read_csv('source/结果文件/cleaned_X_data.csv')
    y = pd.read_csv('source/结果文件/cleaned_y_data.csv')
    gp_features = pd.read_csv('source/结果文件/gplearn_features.csv')
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"GPLearn特征数量: {gp_features.shape[1]}")
    print(f"数据样本数: {X.shape[0]}")
    
    # 标准化原始特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("\n📊 原始特征列表:")
    for i, col in enumerate(X.columns):
        print(f"  {i:2d}: {col}")
    
    print(f"\n🎯 GPLearn特征列表:")
    for i, col in enumerate(gp_features.columns):
        print(f"  {i}: {col}")
    
    # 分析每个GPLearn特征与原始特征的关系
    print("\n" + "="*60)
    print("🔍 特征表达式推断分析")
    print("="*60)
    
    for i, gp_col in enumerate(gp_features.columns):
        print(f"\n📈 分析 {gp_col}:")
        
        # 获取当前GPLearn特征
        gp_feature = gp_features[gp_col].values
        
        # 计算与原始特征的相关性
        correlations = []
        for j, orig_col in enumerate(X_scaled_df.columns):
            corr = np.corrcoef(X_scaled_df[orig_col], gp_feature)[0, 1]
            if not np.isnan(corr):
                correlations.append((j, orig_col, corr))
        
        # 按相关性绝对值排序
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"  与原始特征的相关性 (Top 5):")
        for j, (idx, col, corr) in enumerate(correlations[:5]):
            print(f"    {idx:2d}: {col:15s} -> {corr:8.4f}")
        
        # 尝试线性回归拟合
        print(f"  线性回归拟合分析:")
        try:
            # 使用相关性最高的特征进行拟合
            top_features = [X_scaled_df[corr[1]] for corr in correlations[:3]]
            X_fit = np.column_stack(top_features)
            
            lr = LinearRegression()
            lr.fit(X_fit, gp_feature)
            
            print(f"    使用Top 3特征拟合:")
            for j, (idx, col, corr) in enumerate(correlations[:3]):
                coef = lr.coef_[j]
                print(f"      {col:15s} * {coef:8.4f}")
            print(f"    截距: {lr.intercept_:8.4f}")
            print(f"    R²: {lr.score(X_fit, gp_feature):8.4f}")
            
            # 推断可能的表达式
            if lr.score(X_fit, gp_feature) > 0.5:
                print(f"    🎯 推断表达式:")
                expr_parts = []
                for j, (idx, col, corr) in enumerate(correlations[:3]):
                    coef = lr.coef_[j]
                    if abs(coef) > 0.01:  # 只显示重要系数
                        if coef > 0:
                            expr_parts.append(f"+{coef:.4f}*{col}")
                        else:
                            expr_parts.append(f"{coef:.4f}*{col}")
                
                if lr.intercept_ > 0.01:
                    expr_parts.insert(0, f"+{lr.intercept_:.4f}")
                elif lr.intercept_ < -0.01:
                    expr_parts.insert(0, f"{lr.intercept_:.4f}")
                
                expression = "".join(expr_parts)
                if expression.startswith("+"):
                    expression = expression[1:]
                print(f"      {gp_col} ≈ {expression}")
            
        except Exception as e:
            print(f"    拟合失败: {e}")
        
        # 特征统计信息
        print(f"  统计信息:")
        print(f"    均值: {np.mean(gp_feature):8.4f}")
        print(f"    标准差: {np.std(gp_feature):8.4f}")
        print(f"    最小值: {np.min(gp_feature):8.4f}")
        print(f"    最大值: {np.max(gp_feature):8.4f}")
    
    print("\n" + "="*60)
    print("📋 总结")
    print("="*60)
    print("由于GPLearn 0.4.1版本的限制，无法直接获取特征表达式。")
    print("但通过相关性分析和线性回归拟合，我们可以推断出可能的表达式。")
    print("这些推断的表达式可以帮助理解GPLearn生成的特征组合方式。")
    
    # 保存分析结果
    print("\n💾 保存分析结果...")
    analysis_results = []
    
    for i, gp_col in enumerate(gp_features.columns):
        gp_feature = gp_features[gp_col].values
        
        # 计算与所有原始特征的相关性
        correlations = []
        for j, orig_col in enumerate(X_scaled_df.columns):
            corr = np.corrcoef(X_scaled_df[orig_col], gp_feature)[0, 1]
            if not np.isnan(corr):
                correlations.append((j, orig_col, corr))
        
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 记录分析结果
        analysis_results.append({
            'gp_feature': gp_col,
            'top_correlation_feature': correlations[0][1] if correlations else 'N/A',
            'top_correlation_value': correlations[0][2] if correlations else 0,
            'mean': np.mean(gp_feature),
            'std': np.std(gp_feature),
            'min': np.min(gp_feature),
            'max': np.max(gp_feature)
        })
    
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df.to_csv('source/结果文件/gplearn_feature_analysis.csv', index=False)
    print("✅ 分析结果已保存到: source/结果文件/gplearn_feature_analysis.csv")

if __name__ == "__main__":
    analyze_gplearn_features()
