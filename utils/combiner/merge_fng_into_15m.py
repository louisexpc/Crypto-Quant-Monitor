# merge_fng_into_15m.py
# ------------------------------------------------------------
# 將 FNG（15m 序列）對齊/廣播到你的 15m 主特徵表，新增：
#   sent_fng, sent_fng_diff1, sent_fng_z7d
# 不在此處 shift(1)；統一在 pre-dataloader 做全域 shift。
# ------------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------
# 讀取與規格化
# --------------------------
def _parse_datetime_series(s: pd.Series) -> pd.DatetimeIndex:
    """
    將各種格式的時間欄統一轉為 UTC tz-aware DatetimeIndex。
    支援：
      - ISO 字串（含或不含 +00:00）
      - 整數/字串 Unix 秒（10 位）
    """
    # 先嘗試當作 Unix 秒
    try:
        as_int = pd.to_numeric(s, errors="raise")
        dt = pd.to_datetime(as_int.astype("int64"), unit="s", utc=True)
        return pd.DatetimeIndex(dt)
    except Exception:
        # 再用一般字串解析
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if dt.isna().any():
            bad = s[dt.isna()]
            raise ValueError(f"[時間解析失敗] 有無法解析的時間值，例如：{bad.iloc[:3].tolist()}")
        return pd.DatetimeIndex(dt)

def _load_base_15m(base_csv: str) -> pd.DataFrame:
    df = pd.read_csv(base_csv)
    if "datetime" not in df.columns:
        raise KeyError("主表缺少 'datetime' 欄位。請確認你的 15m 主表含 datetime（UTC）。")
    # 轉 UTC + 設為索引
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError("主表 'datetime' 有無法解析的時間字串。")
    df = df.drop(columns=["datetime"])
    df.index = pd.DatetimeIndex(dt)
    df.index.name = "datetime"
    # 排序 & 去重
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _load_fng_15m(fng_csv: str) -> pd.DataFrame:
    df = pd.read_csv(fng_csv)
    # 自動辨識時間欄位
    time_col = None
    for cand in ("datetime", "timestamp", "time"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise KeyError("FNG 檔找不到時間欄位（需含 'datetime' 或 'timestamp' 或 'time'）。")

    # 解析時間 → UTC 索引
    dt_index = _parse_datetime_series(df[time_col])
    df = df.set_index(pd.DatetimeIndex(dt_index))
    df.index.name = "datetime"

    # 欄位正名（允許不同大小寫）
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "fng":         rename_map[c] = "fng"
        elif lc == "fng_diff1": rename_map[c] = "fng_diff1"
        elif lc == "fng_z7d":   rename_map[c] = "fng_z7d"
    df = df.rename(columns=rename_map)

    # 只保留需要欄位，缺者補 NaN
    need = ["fng", "fng_diff1", "fng_z7d"]
    for n in need:
        if n not in df.columns:
            df[n] = np.nan
    df = df[need].sort_index()

    # 一律對齊到等距 15m → ffill（等距時不改、稀疏時補缺）
    df = df.resample("15T").ffill()

    # dtype 壓成 float32；加前綴 sent_
    df = df.astype(np.float32).rename(columns={
        "fng": "sent_fng",
        "fng_diff1": "sent_fng_diff1",
        "fng_z7d": "sent_fng_z7d",
    })

    # 去重 & 排序
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

# --------------------------
# 合併與輸出
# --------------------------
def merge_and_save(base_csv: str, fng_csv: str, out_csv: str, trim_to_intersection: bool) -> None:
    base = _load_base_15m(base_csv)
    fng  = _load_fng_15m(fng_csv)

    # 左連接：以主表時間為準
    merged = base.join(fng, how="left")

    # 可選：限制在交集區間（避免前段全 NaN）
    if trim_to_intersection and len(fng):
        idx = merged.index
        lo = max(idx.min(), fng.index.min())
        hi = min(idx.max(), fng.index.max())
        merged = merged.loc[lo:hi]

    # 將 datetime 作為第一欄輸出（保留 UTC tz-aware）
    out = merged.reset_index()

    # 確保新欄位保持 float32（reset_index 可能改成 float64）
    for c in ("sent_fng", "sent_fng_diff1", "sent_fng_z7d"):
        if c in out.columns:
            out[c] = out[c].astype("float32")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[OK] merged saved → {out_path.resolve()}")
    print(f"rows={len(out):,}, cols={len(out.columns):,}")
    print("new columns:", [c for c in out.columns if c.startswith("sent_")])

# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge 15m FNG into 15m master indicators (no shift here).")
    ap.add_argument("--base_csv", default="data/derived/btcusdt_15m_with_flat_1m.csv",
                    help="你的 15m 主表 CSV（含 datetime 欄，UTC）")
    ap.add_argument("--fng_csv",  default="data/FNG/fng_15m_utc.csv",
                    help="FNG 15m CSV（含 fng/fng_diff1/fng_z7d 與 datetime 或 timestamp）")
    ap.add_argument("--out_csv",  default="data/FNG/btcusdt_15m_with_fng_with_flat_1m.csv",
                    help="輸出路徑")
    ap.add_argument("--trim_to_intersection", action="store_true",
                    help="僅保留與 FNG 有交集的時間段")
    return ap.parse_args()

def main():
    args = parse_args()
    merge_and_save(
        base_csv=args.base_csv,
        fng_csv=args.fng_csv,
        out_csv=args.out_csv,
        trim_to_intersection=bool(args.trim_to_intersection),
    )

if __name__ == "__main__":
    main()
