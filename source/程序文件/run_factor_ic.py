# -*- coding: utf-8 -*-
"""
基于 prepare_tsfresh_input.py 产出的长表与 panel target：
- 构建按 (id,time) 滚动窗口样本
- 优先使用 tsfresh 提取每个窗口的特征（因子），若失败则回退到增强版特征提取器
- 复用 tsfresh_ic_analysis.py 的 IC 计算与报告

用法示例：
python run_factor_ic.py \
  --long-csv tsfresh_long.csv \
  --target-csv tsfresh_target_panel.csv \
  --window 60 \
  --out-features window_features.csv \
  --out-ic ic_scores.csv
"""

import os
import argparse
import pandas as pd
from pathlib import Path

# 禁用GPU相关路径，避免stumpy/numba触发CUDA
os.environ.setdefault("STUMPY_DISABLE_GPU", "1")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")


def build_window_samples(long_df: pd.DataFrame, window: int):
    """从单变量长表 [id,time,value] 构建滚动窗口样本。

    返回：
    - windows_df: 适配tsfresh的长表 (id=window_id, time, value)
    - mapping_df: 映射表 (window_id, stock_id, anchor_time)
    window_id = "{stock_id}|{YYYYMMDD}" 对应窗口末端时间(anchor_time)
    """
    required_cols = {"id", "time", "value"}
    if not required_cols.issubset(long_df.columns):
        raise ValueError(f"长表缺少列: {required_cols - set(long_df.columns)}，请使用单变量格式[id,time,value]")

    df = long_df.copy()
    df["time"] = pd.to_datetime(df["time"])  # 兼容字符串/时间戳

    windows = []
    mapping = []

    for stock_id, g in df.groupby("id"):
        g = g.sort_values("time").reset_index(drop=True)
        n = len(g)
        for i in range(window - 1, n):
            anchor_time = g.loc[i, "time"]
            start_idx = i - window + 1
            if start_idx < 0:
                continue
            window_id = f"{stock_id}|{anchor_time.strftime('%Y%m%d')}"
            win_slice = g.loc[start_idx:i, ["time", "value"]].copy()
            win_slice.insert(0, "id", window_id)
            windows.append(win_slice)
            mapping.append({
                "window_id": window_id,
                "stock_id": stock_id,
                "anchor_time": anchor_time,
            })

    windows_df = pd.concat(windows, ignore_index=True)
    mapping_df = pd.DataFrame(mapping)
    return windows_df, mapping_df


def extract_window_features(windows_df: pd.DataFrame, feature_level: str = "efficient") -> pd.DataFrame:
    """优先用 tsfresh 提取窗口因子；失败则回退增强版特征提取器。返回每个 window_id 一行的特征表。

    参数：
    - windows_df: tsfresh长表 (id=window_id, time, value)
    - feature_level: 特征集级别，可选 minimal/efficient/comprehensive（默认efficient）
    """
    try:
        import time
        from tsfresh import extract_features
        from tsfresh.feature_extraction import (
            MinimalFCParameters,
            EfficientFCParameters,
            ComprehensiveFCParameters,
        )
        # 选择特征集
        if feature_level == "minimal":
            fc_params = MinimalFCParameters()
            eta_per_1k_min = (1.0, 3.0)
        elif feature_level == "comprehensive":
            fc_params = ComprehensiveFCParameters()
            eta_per_1k_min = (60.0, 180.0)
        else:
            # 默认 efficient
            fc_params = EfficientFCParameters()
            eta_per_1k_min = (5.0, 15.0)

        num_windows = windows_df["id"].nunique()
        est_min_lo = num_windows / 1000.0 * eta_per_1k_min[0]
        est_min_hi = num_windows / 1000.0 * eta_per_1k_min[1]
        print(f"[INFO] 待提取窗口数={num_windows}，特征级别={feature_level}，预计耗时≈{est_min_lo:.1f}~{est_min_hi:.1f} 分钟（经验估算）")

        t0 = time.time()
        X = extract_features(
            windows_df,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=fc_params,
            disable_progressbar=False,
            n_jobs=0,
        )
        elapsed_min = (time.time() - t0) / 60.0
        rate_per_1k = (elapsed_min / max(num_windows, 1)) * 1000.0
        print(f"[OK] tsfresh完成，耗时≈{elapsed_min:.2f} 分钟，速度≈{rate_per_1k:.2f} 分钟/每千窗口")

        X = X.fillna(0)
        features_df = X.reset_index().rename(columns={X.index.name or "index": "window_id"})
        return features_df
    except Exception as e:
        print(f"[WARN] tsfresh特征提取失败，将回退到增强版特征提取器: {e}")
        # 回退方案
        try:
            import time
            from enhanced_feature_extractor import EnhancedFeatureExtractor
        except Exception as ee:
            raise RuntimeError(f"增强版特征提取器不可用: {ee}")
        extractor = EnhancedFeatureExtractor()
        num_windows = windows_df["id"].nunique()
        print(f"[INFO] 使用增强版提取器，窗口数={num_windows}")
        t0 = time.time()
        # 直接对windows_df提取（id=window_id）
        feat = extractor.extract_features(windows_df, column_id="id", column_sort="time", column_value="value")
        elapsed_min = (time.time() - t0) / 60.0
        print(f"[OK] 增强版特征提取完成，耗时≈{elapsed_min:.2f} 分钟")
        feat = feat.fillna(0)
        return feat.rename(columns={"id": "window_id"})


def ensure_parent_dir(path_str: str) -> None:
    out_dir = Path(path_str).parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="滚动窗口因子提取与IC评估")
    
    # 修复输出路径，使用绝对路径
    script_dir = Path(__file__).parent
    source_dir = script_dir.parent
    
    parser.add_argument("--long-csv", type=str, 
                        default=str(source_dir / "数据文件" / "tsfresh_long.csv"), 
                        help="prepare脚本生成的单变量长表(id,time,value)")
    parser.add_argument("--target-csv", type=str, 
                        default=str(source_dir / "数据文件" / "tsfresh_target_panel.csv"), 
                        help="panel目标(id,time,target)")
    parser.add_argument("--window", type=int, default=60, help="滚动窗口长度")
    parser.add_argument("--out-features", type=str, 
                        default=str(source_dir / "数据文件" / "window_features.csv"), 
                        help="窗口因子表输出")
    parser.add_argument("--out-ic", type=str, 
                        default=str(source_dir / "结果文件" / "ic_scores.csv"), 
                        help="IC结果输出")
    parser.add_argument("--feature-level", type=str, default="efficient",
                        choices=["minimal", "efficient", "comprehensive", "custom"],
                        help="特征提取级别：minimal(快速), efficient(平衡), comprehensive(全面), custom(自定义)")

    args = parser.parse_args()

    # 读入数据
    long_df = pd.read_csv(args.long_csv)
    target_df = pd.read_csv(args.target_csv)
    target_df["time"] = pd.to_datetime(target_df["time"])  # 确保时间类型

    # 若长表为多变量(kind,value)，这里简单选取close作为value（若存在）
    if "kind" in long_df.columns:
        if (long_df["kind"] == "close").any():
            long_df = long_df[long_df["kind"] == "close"]["id time value".split()].copy()
        else:
            # 退而求其次：选择出现频次最高的 kind
            top_kind = long_df["kind"].value_counts().idxmax()
            print(f"[WARN] 未找到close列，改用kind={top_kind}")
            long_df = long_df[long_df["kind"] == top_kind]["id time value".split()].copy()

    # 构建窗口
    windows_df, mapping_df = build_window_samples(long_df, args.window)

    # 提取因子
    features_df = extract_window_features(windows_df, feature_level=args.feature_level)

    # 关联原股票与锚点时间
    mapping_df["anchor_time"] = pd.to_datetime(mapping_df["anchor_time"])  # 类型安全
    features_df = features_df.merge(mapping_df, on="window_id", how="left")

    # 合并 target（对齐到anchor_time）
    merged = features_df.merge(
        target_df.rename(columns={"id": "stock_id", "time": "anchor_time"}),
        on=["stock_id", "anchor_time"], how="left"
    )
    merged = merged.dropna(subset=["target"])  # 丢弃无目标的窗口

    # 保存窗口因子
    ensure_parent_dir(args.out_features)
    merged.to_csv(args.out_features, index=False, encoding="utf-8")
    print(f"[OK] 已保存窗口因子: {args.out_features}  形状={merged.shape}")

    # 计算IC
    from tsfresh_ic_analysis import calculate_ic_scores, visualize_ic_scores, generate_feature_importance_report

    # 仅保留因子列+target；排除标识列
    drop_cols = {"window_id", "stock_id", "anchor_time"}
    factor_df = merged[[c for c in merged.columns if c not in drop_cols]].copy()

    ic_df = calculate_ic_scores(factor_df)
    ensure_parent_dir(args.out_ic)
    ic_df.to_csv(args.out_ic, index=False, encoding="utf-8")
    print(f"[OK] 已保存IC结果: {args.out_ic}  形状={ic_df.shape}")

    # 可视化与报告
    visualize_ic_scores(ic_df, top_n=min(20, len(ic_df)))
    report_path = str(source_dir / "结果文件" / "window_feature_ic_report.txt")
    ensure_parent_dir(report_path)
    generate_feature_importance_report(ic_df, output_file=report_path)

    print("完成：")
    print(f"- 窗口因子: {args.out_features}")
    print(f"- IC结果: {args.out_ic}")
    print(f"- 图表: {source_dir}/可视化文件/ic_analysis_results.png")
    print(f"- 报告: {report_path}")
    print(f"- 特征提取级别: {args.feature_level}")
    print(f"- 总特征数: {len(ic_df)}")


if __name__ == "__main__":
    main() 