# data_stastic.py
import os
import math
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

# -----------------------------
# utils
# -----------------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

def _resolve_default_csv() -> Optional[str]:
    """
    嘗試幾個相對路徑，不用 cd 也能找到：
    - CWD/data/ohlcv/...
    - <script_dir>/data/ohlcv/...
    - <script_dir>/../data/ohlcv/...
    """
    rel = os.path.join("data", "ohlcv", "mexc_swap_BTC-USDT-USDT_1h.csv")
    cwd_candidate = os.path.abspath(rel)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    here_candidate = os.path.abspath(os.path.join(script_dir, rel))
    parent_candidate = os.path.abspath(os.path.join(script_dir, "..", rel))

    for p in [cwd_candidate, here_candidate, parent_candidate]:
        if os.path.isfile(p):
            return p
    return None

def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # 找時間欄
    time_col = _find_col(df, ["timestamp", "time", "datetime", "date", "open_time"])
    if time_col is None:
        raise ValueError("找不到時間欄（試過: timestamp/time/datetime/date/open_time）")

    # 轉時間
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.sort_values(time_col).reset_index(drop=True)

    # 找 OHLCV 欄位（不分大小寫）
    o_col = _find_col(df, ["open", "o"])
    h_col = _find_col(df, ["high", "h"])
    l_col = _find_col(df, ["low",  "l"])
    c_col = _find_col(df, ["close","c"])
    v_col = _find_col(df, ["volume","vol","v"])

    for name, col in zip(["open","high","low","close","volume"], [o_col,h_col,l_col,c_col,v_col]):
        if col is None and name != "volume":  # 沒有成交量也勉強可畫K線，但OHLC缺一不可
            raise ValueError(f"找不到必要欄位: {name}")

    return df, time_col, o_col, h_col, l_col, c_col, v_col

def plot_candles(
    df: pd.DataFrame,
    time_col: str,
    o_col: str,
    h_col: str,
    l_col: str,
    c_col: str,
    v_col: Optional[str] = None,
    window: int = 300,
    out_path: str = "stats_output/kline.png",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sub = df.tail(window).copy()
    x = np.arange(len(sub))

    opens  = sub[o_col].astype(float).values
    highs  = sub[h_col].astype(float).values
    lows   = sub[l_col].astype(float).values
    closes = sub[c_col].astype(float).values
    times  = sub[time_col].values

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])

    # 畫影線
    ax.vlines(x, lows, highs, linewidth=1)

    # 畫實體（矩形）
    width = 0.7
    for i in range(len(sub)):
        o, c = opens[i], closes[i]
        lower = min(o, c)
        height = abs(c - o)
        # 上漲用上色，下跌空心（或反過來都可）
        color = "#2ca02c" if c >= o else "#d62728"  # 綠漲紅跌
        edgec = color
        if height == 0:  # 平收，畫一條細線
            ax.hlines(o, i - width/2, i + width/2, linewidth=1, color=color)
        else:
            rect = Rectangle((i - width/2, lower), width, height, facecolor=color, edgecolor=edgec, linewidth=1)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_ylabel("Price")

    # X 軸顯示時間（減少刻度密度）
    step = max(1, len(sub)//8)
    ax.set_xticks(np.arange(0, len(sub), step))
    ax.set_xticklabels([pd.Timestamp(t).strftime("%Y-%m-%d\n%H:%M") for t in times[::step]])
    ax.grid(True, linestyle="--", alpha=0.3)

    # 成交量（如果有）
    if v_col is not None and v_col in sub.columns:
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        vols = sub[v_col].astype(float).values
        ax2.bar(x, vols, width=0.7)
        ax2.set_ylabel("Vol")
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax2.grid(True, linestyle="--", alpha=0.2)
        plt.setp(ax2.get_xticklabels(), rotation=0)
        # 隱藏上面子圖的 x tick labels（避免重疊）
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.suptitle(f"K-line: last {len(sub)} bars", y=0.96)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] K 線已輸出: {out_path}")

def plot_label_counts(
    df: pd.DataFrame,
    label_col: str,
    out_path_img: str = "stats_output/label_counts.png",
    out_path_csv: str = "stats_output/label_counts.csv",
):
    os.makedirs(os.path.dirname(out_path_img), exist_ok=True)
    counts = df[label_col].value_counts(dropna=False).sort_index()
    # 存 CSV
    counts.to_csv(out_path_csv, header=["count"])
    print(f"[OK] 標籤計數 CSV: {out_path_csv}")

    # 畫圖
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Label counts ({label_col})")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_img, dpi=150)
    plt.close(fig)
    print(f"[OK] 標籤計數圖: {out_path_img}")

# -----------------------------
# main
# -----------------------------
def main():
    # 你說「直接給 data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv」
    # 這裡就把它當成預設，不需要 cd
    csv_path = _resolve_default_csv()
    if csv_path is None:
        raise FileNotFoundError(
            "找不到 CSV：data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv\n"
            "請確認檔案存在於專案的 data/ohlcv/ 底下。"
        )
    print(f"[INFO] 使用檔案: {csv_path}")

    # 讀資料 + 欄位自動對應
    df, t_col, o_col, h_col, l_col, c_col, v_col = load_ohlcv(csv_path)

    # 畫 K 線（最近 300 根，可自行調）
    plot_candles(
        df=df,
        time_col=t_col,
        o_col=o_col, h_col=h_col, l_col=l_col, c_col=c_col,
        v_col=v_col,
        window=300,
        out_path="stats_output/kline.png",
    )

    # 統計 label
    lbl_col = _find_col(df, ["label", "target", "y"])
    if lbl_col is None:
        print("[WARN] 找不到 label/target/y 欄位，略過標籤統計。")
    else:
        plot_label_counts(
            df=df,
            label_col=lbl_col,
            out_path_img="stats_output/label_counts.png",
            out_path_csv="stats_output/label_counts.csv",
        )

if __name__ == "__main__":
    main()
