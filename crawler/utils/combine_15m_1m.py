import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_15M = "data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_15m.csv"
DEFAULT_1M = "data/derived/btcusdt_trades_1min_features.csv"
DEFAULT_OUT = "data/derived/btcusdt_15m_with_flat_1m.csv"
DEFAULT_MINUTES = 15
DEFAULT_TZ = "Asia/Taipei"
DEFAULT_FILL = "zero"


def combine(
    ohlcv_csv: str,
    trades_1m_csv: str,
    out_csv: str,
    minute_steps: int,
    bar_is_end: bool,
    tz_15m: str,
    fill_policy: str,
) -> Path:
    p_15m = Path(ohlcv_csv)
    p_1m = Path(trades_1m_csv)
    p_out = Path(out_csv)

    df15 = pd.read_csv(p_15m)

    if "timestamp" in df15.columns:
        ts_raw = pd.to_numeric(df15["timestamp"], errors="coerce")
        if ts_raw.isna().any():
            raise ValueError("15m 檔案的 timestamp 欄位含有非數值內容。")
        unit = "ms" if ts_raw.abs().max() > 1e12 else "s"
        t_utc = pd.to_datetime(ts_raw.astype("int64"), unit=unit, utc=True)
    else:
        dt_local = pd.to_datetime(df15["datetime"], errors="coerce")
        if dt_local.isna().any():
            raise ValueError("15m 檔案的 datetime 欄位有無法解析的值。")
        if dt_local.dt.tz is None:
            dt_local = dt_local.dt.tz_localize(tz_15m)
        else:
            dt_local = dt_local.dt.tz_convert(tz_15m)
        t_utc = dt_local.dt.tz_convert("UTC")

    if not bar_is_end:
        t_utc = t_utc + pd.Timedelta(minutes=15)

    timestamp_sec = (t_utc.astype("int64") // 10**9).astype("int64")
    df15 = df15.assign(
        datetime=t_utc.map(lambda ts: ts.isoformat()),
        timestamp=timestamp_sec,
    )
    df15 = df15.set_index("timestamp").sort_index()
    df15.index.name = "timestamp"

    base_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df15.columns]
    for c in base_cols:
        df15[c] = pd.to_numeric(df15[c], errors="coerce")
    df15_features = df15[base_cols]

    df1m = pd.read_csv(p_1m)

    if "timestamp" not in df1m.columns:
        if "time_utc" in df1m.columns:
            t1 = pd.to_datetime(df1m["time_utc"], utc=True, errors="coerce")
            if t1.isna().any():
                raise ValueError("1m 檔的 time_utc 欄位含有無法解析的值。")
            df1m["timestamp"] = (t1.astype("int64") // 10**9).astype("int64")
        elif "datetime" in df1m.columns:
            t1 = pd.to_datetime(df1m["datetime"], utc=True, errors="coerce")
            if t1.isna().any():
                raise ValueError("1m 檔的 datetime 欄位含有無法解析的值。")
            df1m["timestamp"] = (t1.astype("int64") // 10**9).astype("int64")
        else:
            raise ValueError("1m 檔缺少 timestamp 欄位。請先在匯出時加入 timestamp。")

    ts1 = pd.to_numeric(df1m["timestamp"], errors="coerce")
    if ts1.isna().any():
        raise ValueError("1m 檔的 timestamp 欄位含有非數值內容。")
    df1m["timestamp"] = ts1.astype("int64")

    minute_cols = [c for c in df1m.columns if c not in {"timestamp", "datetime", "time_utc"}]
    if not minute_cols:
        raise ValueError("1m 檔案沒有可用的分鐘特徵欄位。")

    for c in minute_cols:
        df1m[c] = pd.to_numeric(df1m[c], errors="coerce")

    df1m = df1m[["timestamp"] + minute_cols].set_index("timestamp").sort_index()

    full_idx = np.arange(df1m.index.min(), df1m.index.max() + 60, 60, dtype="int64")
    df1m = df1m.reindex(full_idx)

    if fill_policy == "ffill":
        df1m = df1m.fillna(method="ffill").fillna(0.0)
    elif fill_policy == "zero":
        df1m = df1m.fillna(0.0)
    elif fill_policy == "drop":
        pass
    else:
        raise ValueError("未知 FILL_POLICY")

    rows = []
    times = []
    flat_colnames = []
    for k in range(minute_steps):
        offset = -(minute_steps - 1 - k)
        flat_colnames += [f"m_{offset}_{c}" for c in minute_cols]

    for t, base_row in df15_features.iterrows():
        idx = np.arange(t - 60 * (minute_steps - 1), t + 60, 60, dtype="int64")
        block = df1m.reindex(idx)
        if fill_policy == "drop" and block.isna().any().any():
            continue
        block = block.fillna(0.0)
        flat = block.to_numpy().reshape(-1)
        feat = np.concatenate([base_row.to_numpy(dtype=float), flat], axis=0)
        rows.append(feat)
        times.append(t)

    out_cols = base_cols + flat_colnames
    df_out = pd.DataFrame(rows, index=pd.Index(times, name="timestamp"), columns=out_cols).sort_index()

    if "close" in df15.columns:
        future_close = df15["close"].reindex(df_out.index).shift(-4)
        ret_1h = (future_close - df15["close"].reindex(df_out.index)) / df15["close"].reindex(df_out.index)
        df_out["y_ret_1h"] = ret_1h
        thr = 0.001
        conds = [ret_1h < -thr, ret_1h.abs() <= thr, ret_1h > thr]
        df_out["y_class_3"] = np.select(conds, [0, 1, 2], default=np.nan).astype("float")

    ts_out = df_out.index.to_numpy(dtype="int64")
    dt_out = pd.to_datetime(ts_out, unit="s", utc=True).map(lambda ts: ts.isoformat())
    df_out.insert(0, "datetime", dt_out)
    df_out.insert(1, "timestamp", ts_out)
    df_out = df_out.reset_index(drop=True)

    p_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(p_out, index=False)
    print(f"Saved: {p_out}  shape={df_out.shape}")
    print("Minute feature count =", len(minute_cols), "| flattened part =", minute_steps * len(minute_cols))
    return p_out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine 15m OHLCV with flattened 1m trade features")
    ap.add_argument("--ohlcv_csv", default=DEFAULT_15M)
    ap.add_argument("--trades_1m_csv", default=DEFAULT_1M)
    ap.add_argument("--out_csv", default=DEFAULT_OUT)
    ap.add_argument("--minute_steps", type=int, default=DEFAULT_MINUTES)
    ap.add_argument("--bar_is_end", action="store_true", help="15m timestamps already represent bar end")
    ap.add_argument("--time_zone", default=DEFAULT_TZ, help="Timezone of 15m datetime column when no timestamp column")
    ap.add_argument("--fill_policy", choices=["zero", "ffill", "drop"], default=DEFAULT_FILL)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    combine(
        ohlcv_csv=args.ohlcv_csv,
        trades_1m_csv=args.trades_1m_csv,
        out_csv=args.out_csv,
        minute_steps=args.minute_steps,
        bar_is_end=args.bar_is_end,
        tz_15m=args.time_zone,
        fill_policy=args.fill_policy,
    )


if __name__ == "__main__":
    main()
