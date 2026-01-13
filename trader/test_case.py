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


def _prepare_feat_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    1. 說明:
        調整特徵配置中的路徑，確保存在本地。
    2. inputs:
        - cfg: 原始 compute_config dict
    3. return:
        - dict: 調整後的配置
    """
    cfg = dict(cfg)
    plan = cfg.get("feat_plan", {}) or {}
    lp_default = Path("trader/indicators/config/long_feat.yaml")
    sp_default = Path("trader/indicators/config/short_config.yaml")

    lp = Path(plan.get("long_feat_path", lp_default))
    sp = Path(plan.get("short_feat_path", sp_default))

    if not lp.exists():
        lp = lp_default
    if not sp.exists():
        sp = sp_default
    plan["long_feat_path"] = str(lp)
    plan["short_feat_path"] = str(sp)
    cfg["feat_plan"] = plan
    return cfg


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

    # 1) 讀取配置
    compute_cfg = _prepare_feat_cfg(_load_yaml(base_dir / "indicators" / "config" / "compute_config.yaml"))
    predictor_cfg = _load_yaml(base_dir / "predictor" / "predictor.yaml")
    side = "long"
    # 2) 準備原始資料與特徵計算器
    raw_path = Path("data/derived/ohlcv_fng_15m.csv")
    raw_df = pd.read_csv(raw_path)
    fc = FeatureComputer(compute_cfg)
    time_cols = compute_cfg.get("time", {}).get("columns", ["datetime", "timestamp"])
    raw_df = fc._normalize_time_index(raw_df, time_cols)  # 轉成 UTC DatetimeIndex 以便切片

    # 3) 讀取 label，篩 long 事件
    label_path = Path("data/TBM_label/win_rate/BTC-USDT_1h_ewma_up8_dn8_lookback108_label.csv")
    labels_df = pd.read_csv(label_path, parse_dates=["t0"])
    labels_df["t0"] = pd.to_datetime(labels_df["t0"], utc=True)
    labels_df = labels_df[labels_df["side"].apply(_side_is_long)].copy()
    labels_df = labels_df.sort_values("t0")
    args = _parse_args()
    if args.start:
        start_utc = _to_utc_ts(args.start)
        labels_df = labels_df[labels_df["t0"] >= start_utc]
    if args.end:
        end_utc = _to_utc_ts(args.end)
        labels_df = labels_df[labels_df["t0"] <= end_utc]

    # 4) 逐事件切片 (warmup_len + seq_len) 計算特徵，再取末段 seq_len 推論
    seq_len = int(predictor_cfg["seq_len"])
    norm_cfg = compute_cfg.get("feat_normalization", {}) or {}
    warmup_len = int(norm_cfg.get("rolling_window", 0) or 0) if norm_cfg.get("enabled", False) else 0
    context_len = warmup_len + seq_len
    freq = pd.Timedelta(compute_cfg.get("time", {}).get("freq", "15min"))
    predictor = Predictor(predictor_cfg, side)

    pred_rows: List[Dict[str, Any]] = []
    for _, row in labels_df.iterrows():
        t0 = row["t0"]
        end_time = t0 - freq  # 使用 t0 前一根為序列終點
        start_time = end_time - freq * (context_len - 1)
        raw_window = raw_df.loc[(raw_df.index >= start_time) & (raw_df.index <= end_time)]
        if len(raw_window) < context_len:
            continue

        feat_full = fc.compute(raw_window, side="long")
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

    pred_out = pd.DataFrame(pred_rows)
    out_path = base_dir / "test_case_pred.csv"
    pred_out.to_csv(out_path, index=False)
    print(f"[OK] saved predictions -> {out_path}")


if __name__ == "__main__":
    main()

"""
python trader/test_case.py\
    --start 2024-01-01\
    --end 2024-06-30
"""
