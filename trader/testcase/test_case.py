from __future__ import annotations

import yaml
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Any, Dict, List

from trader.indicators.feature_computer import FeatureComputer
from trader.predictor.predictor import Predictor


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    1. 說明:
        讀取 YAML 檔案並回傳 dict。
    2. inputs:
        - path: 檔案路徑
    3. return:
        - dict: 解析後內容
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _side_is_long(x: Any) -> bool:
    """
    1. 說明:
        將 side 欄位判斷是否為 long。
    2. inputs:
        - x: 任意 side 值
    3. return:
        - bool: True 表示 long
    """
    s = str(x).strip().lower()
    if s in {"long", "buy", "l", "+1", "1"}:
        return True
    try:
        return float(s) == 1.0
    except Exception:
        return False


def _to_utc_ts(ts_like: Any) -> pd.Timestamp:
    """
    1. 說明:
        將任意時間轉為 UTC tz-aware Timestamp。
    2. inputs:
        - ts_like: 可轉為 Timestamp 的物件
    3. return:
        - pd.Timestamp (UTC)
    """
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _prepare_feat_cfg(
    cfg: Dict[str, Any],
    *,
    side: str | None = None,
    selected_path_override: str | None = None,
) -> Dict[str, Any]:
    """Prepare feature config, optionally filtering to one side.

    Args:
        cfg: Base feature config dict.
        side: ``"long"`` / ``"short"`` / ``None``.
        selected_path_override: Force this path for the given side.
    Returns:
        Adjusted config dict.
    """
    cfg = dict(cfg)
    sel = dict(cfg.get("selected_feat_path", {}) or {})
    if selected_path_override:
        if side:
            sel = {side: selected_path_override}
        else:
            sel = {"long": selected_path_override, "short": selected_path_override}
    if side:
        sel = {side: sel.get(side, "")}
    cfg["selected_feat_path"] = sel
    return cfg


def _side_label(x: Any) -> str | None:
    """
    1. 說明:
        將 side 轉成 'long' / 'short'，無法判斷則回傳 None。
    """
    if _side_is_long(x):
        return "long"
    s = str(x).strip().lower()
    if s in {"short", "sell", "s", "-1"}:
        return "short"
    try:
        return "short" if float(s) == -1.0 else None
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    """
    1. 說明:
        解析 CLI 參數：start/end 日期可選。
    2. inputs:
        - None
    3. return:
        - argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Trader indicator + predictor test case")
    parser.add_argument("--start", type=str, default=None, help="Filter label t0 >= start (inclusive)")
    parser.add_argument("--end", type=str, default=None, help="Filter label t0 <= end (inclusive)")
    return parser.parse_args()


def main() -> None:
    """
    1. 說明:
        測試流程：讀取 raw OHLCV+FNG，計算 long 特徵，對 long TBM 事件取 seq_len 視窗並用多模型投票推論。
    2. inputs:
        - None
    3. return:
        - None
    """
    base_dir = Path(__file__).resolve().parent
    trader_dir = base_dir.parent

    # 1) 讀取配置
    base_feat_cfg = _load_yaml(trader_dir / "indicators" / "feature.yaml")
    compute_cfg_long = _prepare_feat_cfg(base_feat_cfg, side="long")
    compute_cfg_short = _prepare_feat_cfg(base_feat_cfg, side="short")
    predictor_cfg = _load_yaml(trader_dir / "predictor" / "predictor.yaml")

    feat_engines = {
        "long": FeatureComputer(compute_cfg_long),
        "short": FeatureComputer(compute_cfg_short),
    }
    predictors = {
        "long": Predictor(predictor_cfg, "long"),
        "short": Predictor(predictor_cfg, "short"),
    }
    # 2) 準備原始資料與特徵計算器
    raw_path = Path("data/derived/ohlcv_fng_15m.csv")
    raw_df = pd.read_csv(raw_path)
    time_cols = compute_cfg_long.get("time", {}).get("columns", ["datetime", "timestamp"])
    raw_df = feat_engines["long"]._normalize_time_index(raw_df, time_cols)  # 轉成 UTC DatetimeIndex 以便切片

    # 3) 讀取 label，篩 long 事件
    label_path = Path("data/TBM_label/3candle/3candle_label_15min.csv")
    labels_df = pd.read_csv(label_path, parse_dates=["t0"])
    labels_df["t0"] = pd.to_datetime(labels_df["t0"], utc=True)
    labels_df = labels_df.sort_values("t0")
    # 同一 t0/side 若有多筆，先在這裡去重，避免重複推論/輸出重複列
    before = len(labels_df)
    labels_df = labels_df.drop_duplicates(subset=["t0", "side"], keep="last")
    if len(labels_df) != before:
        print(f"[Info] dedup labels: {before} -> {len(labels_df)} rows (by t0,side)")
    args = _parse_args()
    if args.start:
        start_utc = _to_utc_ts(args.start)
        labels_df = labels_df[labels_df["t0"] >= start_utc]
    if args.end:
        end_utc = _to_utc_ts(args.end)
        labels_df = labels_df[labels_df["t0"] <= end_utc]

    # 4) 逐事件切片 (warmup_len + seq_len) 計算特徵，再取末段 seq_len 推論
    seq_len = int(predictor_cfg["seq_len"])
    norm_cfg = compute_cfg_long.get("feat_normalization", {}) or {}
    warmup_len = int(norm_cfg.get("rolling_window", 0) or 0) if norm_cfg.get("enabled", False) else 0
    context_len = warmup_len + seq_len
    freq = pd.Timedelta(compute_cfg_long.get("time", {}).get("freq", "15min"))

    pred_rows: List[Dict[str, Any]] = []
    for _, row in labels_df.iterrows():
        side = _side_label(row.get("side"))
        if side not in {"long", "short"}:
            continue
        fc = feat_engines[side]
        predictor = predictors[side]
        t0 = row["t0"]
        end_time = t0 - freq  # 使用 t0 前一根為序列終點
        start_time = end_time - freq * (context_len - 1)
        raw_window = raw_df.loc[(raw_df.index >= start_time) & (raw_df.index <= end_time)]
        if len(raw_window) < context_len:
            continue

        feat_full = fc.compute(raw_window, side=side)
        feat_window = feat_full.tail(seq_len)
        if len(feat_window) != seq_len:
            continue
        if not np.isfinite(feat_window.to_numpy()).all():
            continue

        inf_time, pred_bool = predictor.predict_vote(feat_window)
        win_start = feat_window.index.min()
        win_end = feat_window.index.max()
        pred_rows.append(
            {
                "t0": t0,
                "side": row.get("side"),
                "label": row.get("label", None),
                "pred": int(pred_bool),
                "inference_time": inf_time,
            }
        )
        print(
            f"[Predict] window={win_start} ~ {win_end}, t0={t0}, side={row.get('side')}, "
            f"pred={int(pred_bool)}, inference_time={inf_time:.6f}s"
        )

    pred_out = pd.DataFrame(
        pred_rows,
        columns=["t0", "side", "label", "pred", "inference_time"],
    )
    out_path = base_dir / "test_case_pred.csv"
    pred_out.to_csv(out_path, index=False)
    print(f"[OK] saved predictions -> {out_path}")


if __name__ == "__main__":
    main()

"""
python trader/testcase/test_case.py\
    --start 2025-05-01\
    --end 2025-10-20
"""
