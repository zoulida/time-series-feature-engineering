#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用gplearn进行时序特征提取和IC测试
作者：AI助手
日期：2024年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class GPLearnFeatureExtractor:
    """使用gplearn进行特征提取的类"""
    
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
    
    def prepare_features(self, window_sizes=[5, 10, 20]):
        """准备时序特征"""
        print("正在准备时序特征...")
        
        feature_dfs = []
        total_windows = len(window_sizes)
        
        for i, window in enumerate(window_sizes, 1):
            print(f"处理窗口大小: {window} ({i}/{total_windows})")
            
            # 计算滚动统计量 - 保持索引结构
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
            
            # 创建特征DataFrame，保持正确的索引结构
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
        # 使用更简单的方法，避免复杂的索引操作
        rsi_values = []
        
        for id_val in self.long_df['id'].unique():
            # 获取当前ID的数据
            id_data = self.long_df[self.long_df['id'] == id_val]['value'].values
            
            if len(id_data) < 2:
                # 如果数据不足，填充NaN
                rsi_values.extend([np.nan] * len(id_data))
                continue
            
            # 计算百分比变化
            pct_changes = np.diff(id_data) / id_data[:-1]
            
            # 计算增益和损失
            gains = np.where(pct_changes > 0, pct_changes, 0)
            losses = np.where(pct_changes < 0, -pct_changes, 0)
            
            # 计算滚动平均
            avg_gains = []
            avg_losses = []
            
            for i in range(len(pct_changes)):
                start_idx = max(0, i - window + 1)
                avg_gains.append(np.mean(gains[start_idx:i+1]))
                avg_losses.append(np.mean(losses[start_idx:i+1]))
            
            # 计算RSI
            rsi = []
            for i in range(len(avg_gains)):
                if avg_losses[i] == 0:
                    rsi.append(100)
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi.append(100 - (100 / (1 + rs)))
            
            # 第一个值设为NaN（因为没有足够的历史数据）
            rsi_values.extend([np.nan] + rsi)
        
        return pd.Series(rsi_values, index=self.long_df.index)
    
    def create_gplearn_features(self, n_features=50, population_size=1000, generations=20):
        """使用gplearn创建符号回归特征"""
        try:
            # 应用numpy兼容性补丁
            import numpy as np
            if not hasattr(np, 'int'):
                np.int = int
            if not hasattr(np, 'float'):
                np.float = float
            if not hasattr(np, 'bool'):
                np.bool = bool
            
            from gplearn.genetic import SymbolicTransformer
            print("正在使用gplearn创建符号回归特征...")
            
            # 准备特征矩阵
            feature_cols = [col for col in self.merged_df.columns 
                          if col not in ['id', 'time', 'target', 'window']]
            
            X = self.merged_df[feature_cols].fillna(0)
            y = self.merged_df['target'].fillna(0)
            
            # 移除包含NaN的行
            print(f"数据清理前: X={X.shape}, y={y.shape}")
            print(f"清理前NaN统计: X中有NaN的行数={X.isna().any(axis=1).sum()}, y中有NaN的行数={y.isna().sum()}")
            
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            print(f"数据清理后: X={X.shape}, y={y.shape}")
            print(f"清理后NaN检查: X中有NaN={X.isna().any().any()}, y中有NaN={y.isna().any()}")
            
            # 打印清理后的数据样本
            print("\n数据清理后的样本:")
            print("X的前5行:")
            print(X.head())
            print("\ny的前10个值:")
            print(y.head(10))
            print(f"\nX的列名: {list(X.columns)}")
            
            # 保存数据清理后的中间数据
            print("\n保存数据清理后的中间数据...")
            X.to_csv('source/结果文件/cleaned_X_data.csv', index=False)
            y.to_csv('source/结果文件/cleaned_y_data.csv', index=False)
            print("✅ 中间数据已保存:")
            print("   - X数据: source/结果文件/cleaned_X_data.csv")
            print("   - y数据: source/结果文件/cleaned_y_data.csv")
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 创建符号回归转换器 - 修复参数配置
            gp_transformer = SymbolicTransformer(
                population_size=100,     # 增加种群大小
                generations=5,           # 减少代数
                n_components=5,          # 减少特征数
                function_set=['add', 'sub', 'mul', 'div'],  # 简化函数集
                metric='spearman',
                random_state=42,
                n_jobs=1,
                hall_of_fame=50,         # 确保hall_of_fame <= population_size
                parsimony_coefficient=0.001  # 添加简约系数
            )
            
            # 拟合转换器 - 修复版本兼容性问题
            try:
                # 尝试直接使用fit_transform
                X_transformed = gp_transformer.fit_transform(X_scaled, y)
            except Exception as e:
                print(f"fit_transform失败，尝试fit+transform: {e}")
                try:
                    # 分别调用fit和transform
                    gp_transformer.fit(X_scaled, y)
                    X_transformed = gp_transformer.transform(X_scaled)
                except Exception as e2:
                    print(f"fit+transform也失败: {e2}")
                    raise e2
            
            # 获取特征名称
            self.feature_names = [f"GP_Feature_{i}" for i in range(X_transformed.shape[1])]
            
            # 创建特征DataFrame
            self.features = pd.DataFrame(
                X_transformed,
                columns=self.feature_names,
                index=self.merged_df.index
            )
            
            print(f"成功创建 {len(self.feature_names)} 个gplearn特征")
            
            # 🆕 打印生成的特征表达式
            self._print_generated_features(gp_transformer, feature_cols)
            
            # 🆕 推断并打印GPLearn特征表达式
            self._infer_and_print_feature_expressions(X, feature_cols)
            
            # 打印特征统计信息
            print(f"\n📊 GPLearn特征统计信息:")
            print(f"   特征数量: {len(self.feature_names)}")
            print(f"   数据形状: {self.features.shape}")
            print(f"   特征名称: {self.feature_names}")
            
            # 打印特征数据样本
            print(f"\n📈 GPLearn特征数据样本 (前5行):")
            print(self.features.head())
            
            # 打印特征统计描述
            print(f"\n📋 GPLearn特征统计描述:")
            print(self.features.describe())
            
            return self
            
        except ImportError:
            print("gplearn未安装，使用基础特征替代...")
            return self._create_basic_features()
        except Exception as e:
            print(f"gplearn特征创建失败: {e}")
            print("使用基础特征替代...")
            return self._create_basic_features()
    
    def _print_generated_features(self, gp_transformer, original_features):
        """打印生成的特征表达式"""
        try:
            print("\n" + "="*60)
            print("🎯 GPLearn 生成的特征表达式")
            print("="*60)
            
            # 尝试多种方式获取特征表达式
            print("尝试获取特征表达式...")
            
            # 方法1: 检查best_programs_
            if hasattr(gp_transformer, 'best_programs_') and gp_transformer.best_programs_ is not None:
                print(f"✅ 找到 best_programs_，包含 {len(gp_transformer.best_programs_)} 个程序")
                for i, program in enumerate(gp_transformer.best_programs_):
                    if i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        expression = str(program)
                        # 替换特征索引为实际特征名
                        for j, feat in enumerate(original_features):
                            expression = expression.replace(f'X[:, {j}]', feat)
                        print(f"{feature_name}: {expression}")
                        if hasattr(program, 'raw_fitness_'):
                            print(f"   适应度分数: {program.raw_fitness_:.6f}")
                        print()
            
            # 方法2: 检查programs_
            elif hasattr(gp_transformer, 'programs_') and gp_transformer.programs_ is not None:
                print(f"✅ 找到 programs_，包含 {len(gp_transformer.programs_)} 个程序")
                for i, program in enumerate(gp_transformer.programs_):
                    if i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        expression = str(program)
                        for j, feat in enumerate(original_features):
                            expression = expression.replace(f'X[:, {j}]', feat)
                        print(f"{feature_name}: {expression}")
                        print()
            
            # 方法3: 检查其他可能的属性
            else:
                print("⚠️  无法通过常规方法获取特征表达式")
                print("检查所有可用属性...")
                
                # 列出所有属性
                all_attrs = [attr for attr in dir(gp_transformer) if not attr.startswith('_')]
                print(f"可用属性: {all_attrs}")
                
                # 尝试检查一些可能的属性
                for attr in ['programs', 'best_programs', 'fitted_programs', 'final_programs']:
                    if hasattr(gp_transformer, attr):
                        value = getattr(gp_transformer, attr)
                        print(f"属性 {attr}: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                
                # 尝试通过transformer属性获取
                if hasattr(gp_transformer, 'transformer'):
                    print("检查transformer属性...")
                    transformer = gp_transformer.transformer
                    if hasattr(transformer, 'best_programs_'):
                        print(f"transformer.best_programs_ 存在，长度: {len(transformer.best_programs_)}")
                        for i, program in enumerate(transformer.best_programs_):
                            if i < len(self.feature_names):
                                feature_name = self.feature_names[i]
                                expression = str(program)
                                for j, feat in enumerate(original_features):
                                    expression = expression.replace(f'X[:, {j}]', feat)
                                print(f"{feature_name}: {expression}")
                                print()
                
                print("特征已保存到CSV文件中")
            
            print("="*60)
            
        except Exception as e:
            print(f"打印特征表达式时出错: {e}")
            print("但特征数据已成功创建")
            import traceback
            traceback.print_exc()
     
     def _infer_and_print_feature_expressions(self, X, original_features):
         """推断并打印GPLearn特征表达式"""
         try:
             print("\n" + "="*60)
             print("🎯 GPLearn特征表达式推断")
             print("="*60)
             
             from sklearn.linear_model import LinearRegression
             import numpy as np
             
             # 标准化原始特征
             X_scaled = self.scaler.transform(X)
             X_scaled_df = pd.DataFrame(X_scaled, columns=original_features)
             
             print("📊 原始特征列表:")
             for i, col in enumerate(original_features):
                 print(f"  {i:2d}: {col}")
             
             print(f"\n🎯 推断GPLearn特征表达式:")
             
             # 分析每个GPLearn特征
             for i, gp_col in enumerate(self.feature_names):
                 print(f"\n📈 {gp_col}:")
                 
                 # 获取当前GPLearn特征
                 gp_feature = self.features[gp_col].values
                 
                 # 计算与原始特征的相关性
                 correlations = []
                 for j, orig_col in enumerate(X_scaled_df.columns):
                     corr = np.corrcoef(X_scaled_df[orig_col], gp_feature)[0, 1]
                     if not np.isnan(corr):
                         correlations.append((j, orig_col, corr))
                 
                 # 按相关性绝对值排序
                 correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                 
                 print(f"  与原始特征的相关性 (Top 3):")
                 for j, (idx, col, corr) in enumerate(correlations[:3]):
                     print(f"    {idx:2d}: {col:15s} -> {corr:8.4f}")
                 
                 # 尝试线性回归拟合
                 try:
                     # 使用相关性最高的特征进行拟合
                     top_features = [X_scaled_df[corr[1]] for corr in correlations[:3]]
                     X_fit = np.column_stack(top_features)
                     
                     lr = LinearRegression()
                     lr.fit(X_fit, gp_feature)
                     
                     # 推断可能的表达式
                     if lr.score(X_fit, gp_feature) > 0.5:
                         print(f"  🎯 推断表达式:")
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
                         print(f"    {gp_col} ≈ {expression}")
                         print(f"    R²: {lr.score(X_fit, gp_feature):8.4f}")
                     else:
                         print(f"  ⚠️  拟合度较低 (R²: {lr.score(X_fit, gp_feature):8.4f})")
                         
                 except Exception as e:
                     print(f"  ❌ 拟合失败: {e}")
                 
                 # 特征统计信息
                 print(f"  统计信息: 均值={np.mean(gp_feature):8.4f}, 标准差={np.std(gp_feature):8.4f}")
             
             print("\n" + "="*60)
             print("📋 推断完成")
             print("="*60)
             
         except Exception as e:
             print(f"推断特征表达式时出错: {e}")
             import traceback
             traceback.print_exc()
    
    def _create_basic_features(self):
        """创建基础特征（当gplearn不可用时）"""
        print("创建基础特征组合...")
        
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in ['id', 'time', 'target', 'window']]
        
        X = self.merged_df[feature_cols].fillna(0)
        
        # 创建一些基础特征组合
        basic_features = {}
        
        # 价格相关特征组合
        if 'mean' in X.columns and 'std' in X.columns:
            basic_features['price_volatility'] = X['mean'] / (X['std'] + 1e-8)
        
        if 'sma' in X.columns and 'ema' in X.columns:
            basic_features['trend_strength'] = (X['sma'] - X['ema']) / (X['ema'] + 1e-8)
        
        if 'rsi' in X.columns:
            basic_features['rsi_signal'] = np.where(X['rsi'] > 70, 1, np.where(X['rsi'] < 30, -1, 0))
        
        # 创建特征DataFrame
        if basic_features:
            self.features = pd.DataFrame(basic_features)
            self.feature_names = list(basic_features.keys())
        else:
            self.features = X
            self.feature_names = feature_cols
        
        print(f"创建了 {len(self.feature_names)} 个基础特征")
        return self
    
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
                        # 如果某个特征计算失败，记录错误但继续
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
            fig.suptitle('GPLearn特征提取结果分析', fontsize=16)
            
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
            plt.savefig('source/可视化文件/gplearn_ic_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("可视化图表已保存到 source/可视化文件/gplearn_ic_analysis.png")
            
        except Exception as e:
            print(f"可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self):
        """保存结果"""
        print("正在保存结果...")
        
        # 保存特征
        if self.features is not None:
            self.features.to_csv('source/结果文件/gplearn_features.csv', index=False)
            print("特征已保存到: source/结果文件/gplearn_features.csv")
        
        # 保存IC结果
        if hasattr(self, 'ic_df') and self.ic_df is not None:
            self.ic_df.to_csv('source/结果文件/gplearn_ic_results.csv', index=False)
            print("IC结果已保存到: source/结果文件/gplearn_ic_results.csv")
        
        # 保存IC分析摘要
        if hasattr(self, 'feature_ic_summary') and self.feature_ic_summary is not None:
            self.feature_ic_summary.to_csv('source/结果文件/gplearn_ic_summary.csv')
            print("IC分析摘要已保存到: source/结果文件/gplearn_ic_summary.csv")
        
        print("所有结果保存完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("GPLearn时序特征提取和IC测试程序")
    print("=" * 60)
    
    # 创建特征提取器
    extractor = GPLearnFeatureExtractor()
    
    # 加载数据
    extractor.load_data(
        target_file='source/数据文件/tsfresh_target_panel.csv',
        long_file='source/数据文件/tsfresh_long.csv',
        target_id='000001.SZ'
    )
    
    # 准备特征 - 减少窗口大小以加快速度
    extractor.prepare_features(window_sizes=[5, 10])
    
    # 创建gplearn特征 - 减少参数以加快速度
    extractor.create_gplearn_features(
        n_features=20,
        population_size=200,
        generations=10
    )
    
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
