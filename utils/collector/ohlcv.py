#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YAML-driven OHLCV backfill (ccxt)
---------------------------------
- Multiple symbols/timeframes
- Start/End (Asia/Taipei -> UTC)
- Incremental append (resume from last saved candle)
- Closed candles only (drop forming bar)
- Dedup by index
- Parquet or CSV output
- Config via YAML (CLI overrides YAML)

Research use only. Not investment advice.
"""
import os, sys, time, math, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
import yaml

# --------------- Utils ---------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")

def to_utc_ms(ts, tz='Asia/Taipei'):
    t = pd.Timestamp(ts, tz=tz)
    return int(t.tz_convert('UTC').timestamp() * 1000)

def closed_cutoff_ms(tf_ms: int) -> int:
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    current_open = now_ms - (now_ms % tf_ms)  # open time of the forming candle
    return current_open  # any candle whose timestamp >= this is partial

def fetch_ohlcv_paginated(exchange, symbol, timeframe, since_ms, until_ms,
                          limit=1000, sleep_s=0.8, max_retries=5):
    all_rows = []
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = since_ms
    cutoff = closed_cutoff_ms(tf_ms)
    while True:
        print(f"[dbg] {symbol} {timeframe} "
            f"since={pd.to_datetime(since, unit='ms', utc=True)} "
            f"until={pd.to_datetime(min(until_ms, cutoff), unit='ms', utc=True)}")
        if since >= min(until_ms, cutoff):
            break
        tries = 0
        while True:
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                tries += 1
                if tries > max_retries:
                    raise
                wait = min(60.0, (2 ** tries) * 0.5)
                print(f"[{symbol} {timeframe}] Network issue: {e}. Retry in {wait:.1f}s ...")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                raise
        if not batch:
            break
        # drop forming bar(s)
        batch = [row for row in batch if row and row[0] < cutoff]
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + tf_ms  # move to the next candle
        time.sleep(sleep_s)
    if not all_rows:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Taipei')
    df = df.set_index('datetime').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df

def save_frame(df: pd.DataFrame, out_path: Path, fmt='parquet'):
    ensure_dir(out_path.parent)
    if fmt == 'parquet':
        try:
            df.to_parquet(out_path)
        except Exception as e:
            print(f"[warn] Parquet failed ({e}), falling back to CSV.")
            df.to_csv(out_path.with_suffix('.csv'))
            return out_path.with_suffix('.csv')
    else:
        df.to_csv(out_path)
    return out_path

def load_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)

def run_backfill(exchange_id: str, default_type: str, symbols, timeframes,
                 start: str, end: str, outdir: Path, fmt='parquet', append=False,
                 limit=1000, sleep=0.8, max_retries=5):
    # init exchange
    ex_class = getattr(ccxt, exchange_id)
    exchange = ex_class({'options': {'defaultType': default_type}})
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"[warn] load_markets failed: {e}")

    start_ms = to_utc_ms(start)
    until_ms = to_utc_ms(end) if end else int(pd.Timestamp.utcnow().timestamp() * 1000)
    print(f"range: {pd.to_datetime(start_ms, unit='ms', utc=True)} "
      f"→ {pd.to_datetime(until_ms, unit='ms', utc=True)} (UTC)")

    for symbol in symbols:
        for tf in timeframes:
            print(f"\n=== {exchange_id}/{default_type} {symbol} {tf} ===")
            sym_tag = sanitize_symbol(symbol)
            fname = f"{exchange_id}_{default_type}_{sym_tag}_{tf}.{fmt}"
            out_path = outdir / fname

            # incremental append
            since_ms = start_ms
            if append and out_path.exists():
                old = load_frame(out_path)
                if old is not None and len(old):
                    last_idx = old.index[-1]
                    if getattr(last_idx, 'tz', None) is None:
                        last_idx = last_idx.tz_localize('Asia/Taipei')

                    tf_ms = exchange.parse_timeframe(tf) * 1000
                    since_ms = int(last_idx.tz_convert('UTC').timestamp() * 1000) + tf_ms
                    print(f"[append] continue from {last_idx} -> since_ms={since_ms}")

            df = fetch_ohlcv_paginated(exchange, symbol, tf, since_ms, until_ms,
                                       limit=limit, sleep_s=sleep, max_retries=max_retries)

            # merge with old
            if append and out_path.exists():
                old = load_frame(out_path)
                if old is not None and len(old):
                    if old.index.tz is None:
                        old.index = old.index.tz_localize('Asia/Taipei')
                    df = pd.concat([old, df], axis=0)
                    df = df[~df.index.duplicated(keep='last')].sort_index()

            # final save
            use_fmt = 'parquet' if fmt == 'parquet' else 'csv'
            save_target = out_path if use_fmt == 'parquet' else out_path.with_suffix('.csv')
            saved = save_frame(df, save_target, fmt=use_fmt)
            print(f"Saved {len(df)} rows -> {saved}")

def load_cfg_from_yaml(path: Path):
    if yaml is None:
        raise RuntimeError("pyyaml is not installed. pip install pyyaml")

    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    ex  = cfg.get('exchange', {}) or {}
    rng = cfg.get('range', {}) or {}
    io  = cfg.get('io', {}) or {}
    fet = cfg.get('fetch', {}) or {}
    mon = cfg.get('monitoring', []) or []
    return ex, rng, io, fet, mon

def pick(cli, cfg, default=None):
    # merge rule: CLI > YAML > default; preserve falsey booleans
    if cli is not None:
        return cli
    if cfg is not None:
        return cfg
    return default

DEFAULT_CFG = r"utils/collector/collector_config.yaml"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None, help='YAML config path')
    ap.add_argument('--exchange_id', default=None)
    ap.add_argument('--default_type', default=None, choices=['spot','swap'])
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--timeframes', nargs='*', default=None)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--fmt', default=None, choices=['parquet','csv'])
    ap.add_argument('--append', action='store_true', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--sleep', type=float, default=None)
    ap.add_argument('--max_retries', type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    ex, rng, io_cfg, fet, mon = load_cfg_from_yaml(cfg_path)
    print(f"Loaded config from {cfg_path}.")


    exchange_id  = pick(args.exchange_id,  ex.get('id'),            'mexc')
    default_type = pick(args.default_type, ex.get('default_type'),  'swap')

    start = pick(args.start, rng.get('start'), None)
    end   = pick(args.end,   rng.get('end'),   None)
    if start is None:
        raise SystemExit("start is required (provide --start or range.start in YAML)")

    outdir = Path(pick(args.outdir, io_cfg.get('outdir'), 'data/ohlcv')); ensure_dir(outdir)
    fmt    = pick(args.fmt,   io_cfg.get('fmt'),   'parquet')
    append = pick(args.append, io_cfg.get('append'), False)

    limit       = pick(args.limit,       fet.get('limit'),       1000)
    sleep       = pick(args.sleep,       fet.get('sleep'),       0.8)
    max_retries = pick(args.max_retries, fet.get('max_retries'), 5)

    # symbols/timeframes source
    if mon:  # YAML
        pairs = [(item['symbol'], item['timeframes']) for item in mon]
    else:
        if not args.symbols or not args.timeframes:
            raise SystemExit("Provide --symbols and --timeframes when no --config is given.")
        pairs = [(s, args.timeframes) for s in args.symbols]

    for symbol, tfs in pairs:
        run_backfill(
            exchange_id=exchange_id,
            default_type=default_type,
            symbols=[symbol],
            timeframes=tfs,
            start=start,
            end=end,
            outdir=outdir,
            fmt=fmt,
            append=append,
            limit=limit,
            sleep=sleep,
            max_retries=max_retries
        )

if __name__ == '__main__':
    main()
