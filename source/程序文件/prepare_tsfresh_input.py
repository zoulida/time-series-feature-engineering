# -*- coding: utf-8 -*-
"""
将目录中的A股日线CSV整理为 tsfresh 输入的长表格式，并可选执行特征提取。

- 方案A（单变量）：输出列 [id, time, value]（value=close）
- 方案B（多变量）：输出列 [id, time, kind, value]

使用示例：
python prepare_tsfresh_input.py \
  --data-dir F:\\stockdata\\getDayKlineData\\20241101-20250818-front \
  --multi-kind \
  --use-date \
  --out-long tsfresh_long.csv \
  --out-target tsfresh_target_panel.csv \
  --extract \
  --out-feat tsfresh_features.csv
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# 保存前创建父目录
from pathlib import Path as _PathAlias

def ensure_parent_dir(path_str: str) -> None:
    out_dir = _PathAlias(path_str).parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 核心清洗函数
# -----------------------------

def load_to_tsfresh_long(
    data_dir: str,
    use_multi_kind: bool = True,
    use_date_column: bool = True,
    value_cols = ("open", "high", "low", "close", "volume", "amount"),
    tickers: list | None = None,
) -> pd.DataFrame:
    """将目录中的 *.csv 拼接为 tsfresh 长表格式。

    参数：
    - data_dir: CSV目录（每只股票一文件）
    - use_multi_kind: True 输出 [id,time,kind,value]；False 输出 [id,time,value]
    - use_date_column: True 用 date(YYYYMMDD) 作为时间；False 用 time(毫秒时间戳)
    - value_cols: 多变量方案下参与的列
    - tickers: 需要处理的股票代码列表（如 ["000001.SZ"]），None 表示全部
    """
    frames = []
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    # 代码过滤（大小写不敏感）
    if tickers:
        tickers_upper = {t.upper() for t in tickers}
        csv_files = [fp for fp in csv_files if fp.stem.upper() in tickers_upper]
    if not csv_files:
        raise FileNotFoundError(f"目录中未找到CSV文件: {data_dir}")

    for fp in csv_files:
        ticker = fp.stem  # 例如 000001.SZ
        df = pd.read_csv(fp)

        # 选择时间来源
        if use_date_column and "date" in df.columns:
            t = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        elif "time" in df.columns:
            # 原始为毫秒时间戳
            t = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
        else:
            raise ValueError(f"{fp} 缺少 date/time 列")

        # 可选：过滤停牌记录
        if "suspendFlag" in df.columns:
            df = df[df["suspendFlag"] == 0]

        # 排序
        sort_key = "date" if (use_date_column and "date" in df.columns) else ("time" if "time" in df.columns else None)
        if sort_key is not None:
            df = df.sort_values(by=sort_key).reset_index(drop=True)

        df["id"] = str(ticker)
        df["time"] = t

        if use_multi_kind:
            use_cols = [c for c in value_cols if c in df.columns]
            if not use_cols:
                continue
            sub = df[["id", "time"] + use_cols].copy()
            long_df = sub.melt(id_vars=["id", "time"], var_name="kind", value_name="value")
        else:
            if "close" not in df.columns:
                continue
            long_df = df[["id", "time", "close"]].rename(columns={"close": "value"})

        frames.append(long_df)

    if not frames:
        raise ValueError("未能从CSV中拼出任何数据，请检查列名是否匹配。")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["id", "time"]).reset_index(drop=True)
    return out

# -----------------------------
# 构建 panel target（未来20日最高收盘价，排除当天）
# -----------------------------

def build_panel_target(data_dir: str, use_date_column: bool = True, tickers: list | None = None) -> pd.DataFrame:
    """生成逐日 panel 目标表 (id,time,target)，
    target = max(收益率[t+1..t+20])，收益率 = (close[t+k] - close[t]) / close[t]，
    无未来窗口则为 NaN。
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if tickers:
        tickers_upper = {t.upper() for t in tickers}
        csv_files = [fp for fp in csv_files if fp.stem.upper() in tickers_upper]
    if not csv_files:
        raise FileNotFoundError(f"目录中未找到CSV文件: {data_dir}")

    targets = []
    for fp in csv_files:
        sid = fp.stem
        df = pd.read_csv(fp)
        if use_date_column and "date" in df.columns:
            t = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        elif "time" in df.columns:
            t = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
        else:
            raise ValueError(f"{fp} 缺少 date/time 列")
        df = df.sort_values(by="date" if (use_date_column and "date" in df.columns) else "time").reset_index(drop=True)
        close = df["close"].astype(float)
        
        # 计算未来1-20日的收益率，然后取最大值
        future_returns = []
        for k in range(1, 21):  # 从t+1到t+20
            future_close = close.shift(-k)
            # 收益率 = (未来价格 - 当前价格) / 当前价格
            returns = (future_close - close) / close
            future_returns.append(returns)
        
        # 取未来20日收益率的最大值
        future_max_return = pd.concat(future_returns, axis=1).max(axis=1)
        
        tmp = pd.DataFrame({"id": sid, "time": t, "target": future_max_return})
        targets.append(tmp)
    panel = pd.concat(targets, ignore_index=True)
    panel = panel.sort_values(["id", "time"]).reset_index(drop=True)
    return panel

# -----------------------------
# tsfresh 特征提取（可选）
# -----------------------------

def maybe_extract_features(long_df: pd.DataFrame, multi_kind: bool, out_path: str) -> None:
    """尝试使用 tsfresh 进行特征提取。若未安装则给出提示。"""
    try:
        os.environ["NUMBA_DISABLE_CUDA"] = "1"  # 禁用CUDA以避免GPU依赖
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
        params = MinimalFCParameters()

        if multi_kind:
            X = extract_features(
                long_df,
                column_id="id",
                column_sort="time",
                column_kind="kind",
                column_value="value",
                default_fc_parameters=params,
                disable_progressbar=True,
                n_jobs=0,
            )
        else:
            X = extract_features(
                long_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=params,
                disable_progressbar=True,
                n_jobs=0,
            )
        ensure_parent_dir(out_path)
        X.to_csv(out_path, encoding="utf-8", index=True)
        print(f"[OK] 已保存特征矩阵: {out_path}  形状={X.shape}")
    except ImportError:
        print("[WARN] 未安装 tsfresh，已跳过特征提取。可先执行: pip install tsfresh")
    except Exception as e:
        print(f"[WARN] tsfresh 提取失败: {e}")

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="整理A股CSV为tsfresh长表并可选提取特征，同时生成 panel target")
    parser.add_argument("--data-dir", type=str, required=False,
                        default=r"F:\\stockdata\\getDayKlineData\\20241101-20250818-front",
                        help="CSV目录路径")
    parser.add_argument("--multi-kind", action="store_true", help="输出多变量长表 [id,time,kind,value]")
    parser.add_argument("--single", dest="multi_kind", action="store_false", help="输出单变量长表 [id,time,value]")
    parser.set_defaults(multi_kind=False)

    parser.add_argument("--use-date", action="store_true", help="使用date(YYYYMMDD)作为时间")
    parser.add_argument("--use-time", dest="use_date", action="store_false", help="使用time(毫秒)作为时间")
    parser.set_defaults(use_date=True)

    # 调试期默认仅处理 000001.SZ；正式跑全量可使用 --all-tickers
    parser.add_argument("--tickers", type=str, default="000001.SZ",
                        help="要处理的股票代码，逗号分隔，如: 000001.SZ,000002.SZ；默认仅处理000001.SZ")
    parser.add_argument("--all-tickers", action="store_true",
                        help="处理目录下全部股票，忽略 --tickers 配置")

    # 修复输出路径，使用绝对路径
    script_dir = Path(__file__).parent
    source_dir = script_dir.parent
    
    parser.add_argument("--out-long", type=str, 
                        default=str(source_dir / "数据文件" / "tsfresh_long.csv"), 
                        help="长表输出CSV路径")
    parser.add_argument("--out-target", type=str, 
                        default=str(source_dir / "数据文件" / "tsfresh_target_panel.csv"), 
                        help="panel目标输出CSV路径(id,time,target)")
    parser.add_argument("--extract", action="store_true", help="是否执行tsfresh特征提取")
    parser.add_argument("--out-feat", type=str, 
                        default=str(source_dir / "结果文件" / "tsfresh_features.csv"), 
                        help="特征矩阵输出CSV路径")

    args = parser.parse_args()

    args.all_tickers = True###################################重要设置
    tickers_list = None if args.all_tickers else [s.strip().upper() for s in args.tickers.split(',') if s.strip()]

    # 生成长表
    long_df = load_to_tsfresh_long(
        data_dir=args.data_dir,
        use_multi_kind=args.multi_kind,
        use_date_column=args.use_date,
        tickers=tickers_list,
    )
    ensure_parent_dir(args.out_long)
    long_df.to_csv(args.out_long, index=False, encoding="utf-8")
    print(f"[OK] 已保存长表: {args.out_long}  形状={long_df.shape}")
    print(long_df.head(10))

    # 生成 panel target（未来20日最高收益率）
    panel_target = build_panel_target(args.data_dir, use_date_column=args.use_date, tickers=tickers_list)
    ensure_parent_dir(args.out_target)
    panel_target.to_csv(args.out_target, index=False, encoding="utf-8")
    print(f"[OK] 已保存panel目标: {args.out_target}  形状={panel_target.shape}")
    print(panel_target.dropna().head(10))

    # 可选：提取特征（按完整长表聚合为每个id一行）
    if args.extract:
        maybe_extract_features(long_df, args.multi_kind, args.out_feat)


if __name__ == "__main__":
    main() 