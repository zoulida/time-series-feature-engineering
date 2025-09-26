#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版特征工程 - 手动创建复杂特征组合
替代GPLearn的功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedFeatureExtractor:
    """增强版特征提取器"""
    
    def __init__(self):
        self.features = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def load_data(self, target_file, long_file, target_id='000001.SZ'):
        """加载数据文件"""
        print("正在加载数据...")
        
        # 加载目标数据
        self.target_df = pd.read_csv(target_file)
        self.target_df['time'] = pd.to_datetime(self.target_df['time'])
        
        # 加载时序数据
        self.long_df = pd.read_csv(long_file)
        self.long_df['time'] = pd.to_datetime(self.long_df['time'])
        
        # 只保留指定ID的数据
        self.target_df = self.target_df[self.target_df['id'] == target_id].copy()
        self.long_df = self.long_df[self.long_df['id'] == target_id].copy()
        
        print(f"目标数据形状: {self.target_df.shape}")
        print(f"时序数据形状: {self.long_df.shape}")
        print(f"处理目标ID: {target_id}")
        
        return self
    
    def prepare_features(self, window_sizes=[5, 10]):
        """准备时序特征"""
        print("正在准备时序特征...")
        
        feature_dfs = []
        total_windows = len(window_sizes)
        
        for i, window in enumerate(window_sizes, 1):
            print(f"处理窗口大小: {window} ({i}/{total_windows})")
            
            # 计算滚动统计量
            rolling_stats = self.long_df.groupby('id')['value'].rolling(
                window=window, min_periods=1
            ).agg([
                'mean', 'std', 'min', 'max', 'median',
                'skew', 'kurt', 'var'
            ])
            
            # 单独计算分位数
            q25 = self.long_df.groupby('id')['value'].rolling(
                window=window, min_periods=1
            ).quantile(0.25)
            
            q75 = self.long_df.groupby('id')['value'].rolling(
                window=window, min_periods=1
            ).quantile(0.75)
            
            # 计算价格变化
            pct_change = self.long_df.groupby('id')['value'].pct_change()
            pct_change_rolling = self.long_df.groupby('id')['value'].pct_change().rolling(
                window=window, min_periods=1
            ).mean()
            
            # 计算技术指标
            sma = self.long_df.groupby('id')['value'].rolling(
                window=window, min_periods=1
            ).mean()
            
            ema = self.long_df.groupby('id')['value'].ewm(
                span=window, min_periods=1
            ).mean()
            
            rsi = self._calculate_rsi(window)
            
            # 创建特征DataFrame
            rolling_features = pd.DataFrame({
                'id': self.long_df['id'],
                'time': self.long_df['time'],
                'mean': rolling_stats['mean'].reset_index(0, drop=True),
                'std': rolling_stats['std'].reset_index(0, drop=True),
                'min': rolling_stats['min'].reset_index(0, drop=True),
                'max': rolling_stats['max'].reset_index(0, drop=True),
                'median': rolling_stats['median'].reset_index(0, drop=True),
                'skew': rolling_stats['skew'].reset_index(0, drop=True),
                'kurt': rolling_stats['kurt'].reset_index(0, drop=True),
                'var': rolling_stats['var'].reset_index(0, drop=True),
                'q25': q25.reset_index(0, drop=True),
                'q75': q75.reset_index(0, drop=True),
                'pct_change': pct_change.reset_index(0, drop=True),
                'pct_change_rolling': pct_change_rolling.reset_index(0, drop=True),
                'sma': sma.reset_index(0, drop=True),
                'ema': ema.reset_index(0, drop=True),
                'rsi': rsi,
                'window': window
            })
            
            feature_dfs.append(rolling_features)
        
        # 合并所有特征
        self.features_df = pd.concat(feature_dfs, ignore_index=True)
        
        # 确保时间列格式一致
        self.features_df['time'] = pd.to_datetime(self.features_df['time'])
        
        # 与目标数据合并
        self.merged_df = pd.merge(
            self.target_df,
            self.features_df,
            on=['id', 'time'],
            how='left'
        )
        
        print(f"合并后数据形状: {self.merged_df.shape}")
        return self
    
    def _calculate_rsi(self, window):
        """计算RSI指标"""
        rsi_values = []
        
        for id_val in self.long_df['id'].unique():
            id_data = self.long_df[self.long_df['id'] == id_val]['value'].values
            
            if len(id_data) < 2:
                rsi_values.extend([np.nan] * len(id_data))
                continue
            
            pct_changes = np.diff(id_data) / id_data[:-1]
            gains = np.where(pct_changes > 0, pct_changes, 0)
            losses = np.where(pct_changes < 0, -pct_changes, 0)
            
            avg_gains = []
            avg_losses = []
            
            for i in range(len(pct_changes)):
                start_idx = max(0, i - window + 1)
                avg_gains.append(np.mean(gains[start_idx:i+1]))
                avg_losses.append(np.mean(losses[start_idx:i+1]))
            
            rsi = []
            for i in range(len(avg_gains)):
                if avg_losses[i] == 0:
                    rsi.append(100)
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi.append(100 - (100 / (1 + rs)))
            
            rsi_values.extend([np.nan] + rsi)
        
        return pd.Series(rsi_values, index=self.long_df.index)
    
    def create_enhanced_features(self):
        """创建增强特征组合"""
        print("正在创建增强特征组合...")
        
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in ['id', 'time', 'target', 'window']]
        
        X = self.merged_df[feature_cols].fillna(0)
        y = self.merged_df['target'].fillna(0)
        
        # 移除包含NaN的行
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"数据清理后: X={X.shape}, y={y.shape}")
        
        # 创建增强特征
        enhanced_features = {}
        
        # 1. 价格相关特征组合
        if 'mean' in X.columns and 'std' in X.columns:
            enhanced_features['price_volatility_ratio'] = X['mean'] / (X['std'] + 1e-8)
            enhanced_features['price_range_ratio'] = (X['max'] - X['min']) / (X['mean'] + 1e-8)
            enhanced_features['price_skewness_abs'] = np.abs(X['skew'])
            enhanced_features['price_kurtosis_norm'] = X['kurt'] / (X['std'] + 1e-8)
        
        # 2. 移动平均相关特征
        if 'sma' in X.columns and 'ema' in X.columns:
            enhanced_features['trend_strength'] = (X['sma'] - X['ema']) / (X['ema'] + 1e-8)
            enhanced_features['trend_consistency'] = np.abs(X['sma'] - X['ema']) / (X['std'] + 1e-8)
            enhanced_features['ma_cross_signal'] = np.where(X['sma'] > X['ema'], 1, -1)
        
        # 3. RSI相关特征
        if 'rsi' in X.columns:
            enhanced_features['rsi_signal'] = np.where(X['rsi'] > 70, 1, np.where(X['rsi'] < 30, -1, 0))
            enhanced_features['rsi_distance'] = np.abs(X['rsi'] - 50) / 50
            enhanced_features['rsi_trend'] = np.where(X['rsi'] > 50, 1, -1)
        
        # 4. 分位数相关特征
        if 'q25' in X.columns and 'q75' in X.columns:
            enhanced_features['iqr_ratio'] = (X['q75'] - X['q25']) / (X['mean'] + 1e-8)
            enhanced_features['price_position'] = (X['mean'] - X['q25']) / (X['q75'] - X['q25'] + 1e-8)
        
        # 5. 价格变化相关特征
        if 'pct_change' in X.columns and 'pct_change_rolling' in X.columns:
            enhanced_features['momentum_strength'] = X['pct_change'] / (X['pct_change_rolling'] + 1e-8)
            enhanced_features['momentum_consistency'] = np.abs(X['pct_change'] - X['pct_change_rolling'])
        
        # 6. 复合特征
        if 'mean' in X.columns and 'std' in X.columns and 'rsi' in X.columns:
            enhanced_features['volatility_rsi_interaction'] = X['std'] * (X['rsi'] / 100)
            enhanced_features['mean_rsi_ratio'] = X['mean'] * (X['rsi'] / 100)
        
        if 'sma' in X.columns and 'ema' in X.columns and 'rsi' in X.columns:
            enhanced_features['trend_rsi_signal'] = ((X['sma'] - X['ema']) / (X['ema'] + 1e-8)) * (X['rsi'] / 100)
        
        # 7. 统计特征组合
        if 'skew' in X.columns and 'kurt' in X.columns:
            enhanced_features['distribution_score'] = np.abs(X['skew']) + np.abs(X['kurt'])
            enhanced_features['skew_kurt_ratio'] = X['skew'] / (X['kurt'] + 1e-8)
        
        # 8. 价格位置特征
        if 'min' in X.columns and 'max' in X.columns and 'mean' in X.columns:
            enhanced_features['price_position_relative'] = (X['mean'] - X['min']) / (X['max'] - X['min'] + 1e-8)
            enhanced_features['price_range_efficiency'] = (X['max'] - X['min']) / (X['std'] + 1e-8)
        
        # 创建特征DataFrame
        if enhanced_features:
            self.features = pd.DataFrame(enhanced_features)
            self.feature_names = list(enhanced_features.keys())
        else:
            self.features = X
            self.feature_names = feature_cols
        
        print(f"创建了 {len(self.feature_names)} 个增强特征")
        
        # 打印特征表达式
        self._print_enhanced_features(enhanced_features)
        
        return self
    
    def _print_enhanced_features(self, enhanced_features):
        """打印增强特征表达式"""
        print("\n" + "="*60)
        print("🎯 增强特征表达式")
        print("="*60)
        
        for i, (name, expression) in enumerate(enhanced_features.items(), 1):
            print(f"特征{i}: {name}")
            print(f"   表达式: {expression}")
            print()
        
        print("="*60)
    
    def calculate_ic(self):
        """计算特征与目标变量的信息系数"""
        print("正在计算IC值...")
        
        if self.features is None:
            print("请先创建特征！")
            return None
        
        # 合并特征和目标
        ic_data = pd.concat([
            self.features,
            self.merged_df[['id', 'time', 'target']]
        ], axis=1)
        
        # 按时间和ID分组计算IC
        ic_results = []
        
        for time_group in ic_data.groupby('time'):
            time = time_group[0]
            group_data = time_group[1]
            
            for feature in self.feature_names:
                if feature in group_data.columns:
                    try:
                        # 计算相关系数
                        corr = group_data[feature].corr(group_data['target'])
                        
                        # 计算互信息
                        mi = mutual_info_score(
                            group_data[feature].fillna(0),
                            group_data['target']
                        )
                        
                        ic_results.append({
                            'time': time,
                            'feature': feature,
                            'correlation': corr,
                            'mutual_info': mi,
                            'ic_score': abs(corr) if not pd.isna(corr) else 0
                        })
                    except Exception as e:
                        print(f"警告：特征 {feature} 在时间 {time} 的IC计算失败: {e}")
                        continue
        
        self.ic_df = pd.DataFrame(ic_results)
        print(f"IC计算完成，共 {len(ic_results)} 条记录")
        return self.ic_df
    
    def analyze_ic(self):
        """分析IC结果"""
        if not hasattr(self, 'ic_df') or self.ic_df is None:
            print("请先计算IC值！")
            return None
        
        print("正在分析IC结果...")
        
        # 计算特征的平均IC值
        feature_ic = self.ic_df.groupby('feature').agg({
            'ic_score': ['mean', 'std', 'count'],
            'correlation': ['mean', 'std'],
            'mutual_info': ['mean', 'std']
        }).round(4)
        
        feature_ic.columns = [
            'IC_mean', 'IC_std', 'IC_count',
            'Corr_mean', 'Corr_std',
            'MI_mean', 'MI_std'
        ]
        
        # 按平均IC值排序
        feature_ic = feature_ic.sort_values('IC_mean', ascending=False)
        
        self.feature_ic_summary = feature_ic
        print("IC分析完成")
        return feature_ic
    
    def visualize_results(self):
        """可视化结果"""
        if not hasattr(self, 'feature_ic_summary') or self.feature_ic_summary is None:
            print("请先分析IC结果！")
            return
        
        if not hasattr(self, 'ic_df') or self.ic_df is None or len(self.ic_df) == 0:
            print("没有IC数据可供可视化！")
            return
        
        print("正在生成可视化图表...")
        
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('增强特征提取结果分析', fontsize=16)
            
            # 1. 特征IC值排名
            top_features = self.feature_ic_summary.head(20)
            if len(top_features) > 0:
                axes[0, 0].barh(range(len(top_features)), top_features['IC_mean'])
                axes[0, 0].set_yticks(range(len(top_features)))
                axes[0, 0].set_yticklabels(top_features.index)
                axes[0, 0].set_xlabel('平均IC值')
                axes[0, 0].set_title('Top 20特征IC值排名')
                axes[0, 0].invert_yaxis()
            else:
                axes[0, 0].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Top 20特征IC值排名')
            
            # 2. IC值分布
            if len(self.ic_df) > 0:
                axes[0, 1].hist(self.ic_df['ic_score'], bins=30, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('IC值')
                axes[0, 1].set_ylabel('频次')
                axes[0, 1].set_title('IC值分布')
            else:
                axes[0, 1].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('IC值分布')
            
            # 3. 相关性vs互信息散点图
            if len(self.ic_df) > 0:
                axes[1, 0].scatter(
                    self.ic_df['correlation'].abs(),
                    self.ic_df['mutual_info'],
                    alpha=0.6
                )
                axes[1, 0].set_xlabel('|相关系数|')
                axes[1, 0].set_ylabel('互信息')
                axes[1, 0].set_title('相关性 vs 互信息')
            else:
                axes[1, 0].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('相关性 vs 互信息')
            
            # 4. 时间序列IC值
            if len(self.ic_df) > 0:
                time_ic = self.ic_df.groupby('time')['ic_score'].mean()
                if len(time_ic) > 0:
                    axes[1, 1].plot(time_ic.index, time_ic.values, marker='o')
                    axes[1, 1].set_xlabel('时间')
                    axes[1, 1].set_ylabel('平均IC值')
                    axes[1, 1].set_title('时间序列IC值变化')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 1].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('时间序列IC值变化')
            else:
                axes[1, 1].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('时间序列IC值变化')
            
            plt.tight_layout()
            plt.savefig('source/可视化文件/enhanced_ic_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("可视化图表已保存到 source/可视化文件/enhanced_ic_analysis.png")
            
        except Exception as e:
            print(f"可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self):
        """保存结果"""
        print("正在保存结果...")
        
        # 保存特征
        if self.features is not None:
            self.features.to_csv('source/结果文件/enhanced_features.csv', index=False)
            print("特征已保存到: source/结果文件/enhanced_features.csv")
        
        # 保存IC结果
        if hasattr(self, 'ic_df') and self.ic_df is not None:
            self.ic_df.to_csv('source/结果文件/enhanced_ic_results.csv', index=False)
            print("IC结果已保存到: source/结果文件/enhanced_ic_results.csv")
        
        # 保存IC分析摘要
        if hasattr(self, 'feature_ic_summary') and self.feature_ic_summary is not None:
            self.feature_ic_summary.to_csv('source/结果文件/enhanced_ic_summary.csv')
            print("IC分析摘要已保存到: source/结果文件/enhanced_ic_summary.csv")
        
        print("所有结果保存完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("增强版特征提取和IC测试程序")
    print("=" * 60)
    
    # 创建特征提取器
    extractor = EnhancedFeatureExtractor()
    
    # 加载数据
    extractor.load_data(
        target_file='source/数据文件/tsfresh_target_panel.csv',
        long_file='source/数据文件/tsfresh_long.csv',
        target_id='000001.SZ'
    )
    
    # 准备特征
    extractor.prepare_features(window_sizes=[5, 10])
    
    # 创建增强特征
    extractor.create_enhanced_features()
    
    # 计算IC值
    extractor.calculate_ic()
    
    # 分析IC结果
    extractor.analyze_ic()
    
    # 可视化结果
    extractor.visualize_results()
    
    # 保存结果
    extractor.save_results()
    
    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
