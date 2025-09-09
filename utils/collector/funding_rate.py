# funding_fetch_binance.py
import time, math, requests, pandas as pd

BINANCE_FAPI = "https://fapi.binance.com"

def fetch_funding_history(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """
    歷史 funding rate（實際結算）：
      GET /fapi/v1/fundingRate?symbol=BTCUSDT&startTime=...&endTime=...&limit=1000
    備註：Binance 單次最多 1000 筆，超過要用游標/時間分段拉。
    """
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    rows, params = [], {"symbol": symbol, "limit": limit, "startTime": start_ms}
    while True:
        if "startTime" in params and params["startTime"] > end_ms:
            break
        if "endTime" not in params:
            params["endTime"] = end_ms
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        # 依文件：若資料量 > limit，下一頁從上一筆 fundingTime+1 開始
        last_t = batch[-1]["fundingTime"]
        next_start = last_t + 1
        if next_start > end_ms or len(batch) < limit:
            break
        params["startTime"] = next_start
        time.sleep(0.2)  # 禮貌性間隔
    if not rows:
        return pd.DataFrame(columns=["symbol","fundingRate","fundingTime","markPrice"])
    df = pd.DataFrame(rows)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["markPrice"]   = pd.to_numeric(df.get("markPrice"), errors="coerce")
    # 對齊成台北時區索引（UTC→Asia/Taipei）
    df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    return df.set_index("datetime").sort_index()


# funding_main.py
import os, time
import pandas as pd
from pathlib import Path
import yaml


def load_cfg(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def to_utc_ms(ts: str, tz='Asia/Taipei') -> int:
    return int(pd.Timestamp(ts, tz=tz).tz_convert('UTC').timestamp() * 1000)

def main():
    # === 1) Load YAML config ===
    cfg_path = Path("utils/collector/collector_config.yaml")
    cfg = load_cfg(cfg_path)
    ex_id  = cfg["exchange"]["id"]           # e.g., "binanceusdm"
    start  = cfg["range"]["start"]
    end    = cfg["range"]["end"]
    outdir = Path(cfg["io"]["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    fmt    = cfg["io"]["fmt"]
    append = bool(cfg["io"]["append"])
    limit  = int(cfg["fetch"]["limit"])
    sleep  = float(cfg["fetch"]["sleep"])
    retries = int(cfg["fetch"]["max_retries"])

    monitoring = cfg["monitoring"]
    if not monitoring:
        raise SystemExit("❌ YAML 中缺少 monitoring.symbols 設定")

    # === 2) 時間範圍處理 ===
    start_ms = to_utc_ms(start)
    end_ms = to_utc_ms(end) if end else int(pd.Timestamp.utcnow().timestamp() * 1000)

    print(f"=== Funding Rate Downloader ===\nStart: {start}  End: {end or 'NOW'}  Format: {fmt}\n")

    for item in monitoring:
        sym = item["symbol"]        # e.g., "BTC/USDT:USDT"
        sym_api = sym.replace("/", "").replace(":USDT", "")  # Binance 要 "BTCUSDT"
        fname = f"{ex_id}_{sym_api}.{fmt}"
        fpath = outdir / fname

        since_ms = start_ms

        if append and fpath.exists():
            # 續接：從最後一筆時間之後開始
            df_old = pd.read_csv(fpath, index_col=0, parse_dates=True) if fmt == "csv" else pd.read_parquet(fpath)
            if len(df_old):
                last_ts = df_old.index[-1]
                if last_ts.tz is None:
                    last_ts = last_ts.tz_localize("Asia/Taipei")
                since_ms = int(last_ts.tz_convert("UTC").timestamp() * 1000) + 1
                print(f"[append] {sym_api}: resume from {last_ts}")

        # === 3) Fetch funding rate ===
        print(f"\n📥 Fetching {sym_api} funding rate from {pd.to_datetime(since_ms, unit='ms')} to {pd.to_datetime(end_ms, unit='ms')}")

        df = fetch_funding_history(sym_api, since_ms, end_ms, limit=limit)
        print(f"✔️  Downloaded: {len(df)} rows")

        # === 4) 合併舊資料（如需）===
        if append and fpath.exists():
            df_old = pd.read_csv(fpath, index_col=0, parse_dates=True) if fmt == "csv" else pd.read_parquet(fpath)
            df_old.index = pd.to_datetime(df_old.index).tz_localize("Asia/Taipei") if df_old.index.tz is None else df_old.index
            df = pd.concat([df_old, df], axis=0)
            df = df[~df.index.duplicated(keep="last")].sort_index()

        # === 5) Save ===
        if fmt == "csv":
            df.to_csv(fpath)
        else:
            df.to_parquet(fpath)
        print(f"✅ Saved to: {fpath}")

if __name__ == "__main__":
    main()
