# Crawler Pipeline README

本資料夾提供一條龍資料蒐集與特徵製程，從交易所 OHLCV、FNG 情緒指標到 Binance 逐筆交易，最後產出 15 分鐘 + 打平成 1 分鐘特徵的資料集。

```
┌──────────────────┐    ┌────────────────────┐
│  ohlcv/ohlcv.py  │    │  fng/fng_index.py  │
│  (15m OHLCV)     │    │  (15m FNG features)│
└────────┬─────────┘    └──────────┬─────────┘
         │                          │
         │                          ▼
         │                 merge_fng_into_15m.py
         │                          │
         ▼                          │
trades/binance_trades.py            │
         │                          │
         ▼                          │
utils/trades_to_1min_features.py    │
         │                          │
         └──────────────┬───────────┘
                        ▼
             utils/combine_15m_1m.py
                        │
                        ▼
             Final dataset (15m+1m)

All stages can be串接 via `craw_ohlcv_trades.sh`.
```

## 檔案功能

| 檔案 | 說明 |
| --- | --- |
| `craw_ohlcv_trades.sh` | 全流程 orchestrator，依序執行 OHLCV → FNG → trades → FNG merge → 1m 特徵 → 15m+1m combine，支援大量 CLI 參數與 `PYTHON_BIN`。 |
| `ohlcv/ohlcv.py` | 使用 ccxt 依 YAML/CLI 參數回補指定 symbol/timeframe 的 OHLCV，支援增量 append、Parquet/CSV 輸出。 |
| `fng/fng_index.py` | 從 alternative.me 下載 Fear & Greed Index，補成 15m 序列並輸出 `fng`, `fng_diff1`, `fng_z7d` 欄位。 |
| `trades/binance_trades.py` | 下載/續傳 Binance UM daily trades zip，並合併成單一 CSV（可指定 symbol、日期、輸出路徑）。 |
| `utils/merge_fng_into_15m.py` | 將 FNG 15m 欄位 left join 到主 15m 表，輸出 `sent_fng*` 欄。 |
| `utils/trades_to_1min_features.py` | 讀取每日 trades zip，聚合成 1 分鐘統計與高階特徵（volume spike、price jump、trend slope 等）。 |
| `utils/combine_15m_1m.py` | 以 15 分鐘 bar 為 anchor，連接對應 15 個 1 分鐘特徵並打平成欄位，可選 1H 報酬/分類標籤。 |

## 使用範例

### 1. 全自動流程

```bash
# 以預設設定回補 2023-01-01 起的資料
bash crawler/craw_ohlcv_trades.sh \
  --timeframes "15m" \
  --start 2023-01-01 \
  --trades-symbol BTCUSDT \
  --ohlcv-outdir data/ohlcv_2023_new
```

常用參數：

- `--end 2025-01-01`：限制資料到指定日期。
- `--minute-steps 15`：調整 1 分鐘打平視窗。
- `--final-out data/derived/custom.csv`：自訂最終輸出路徑。

### 2. 單步執行 CLI 範例

```bash
# OHLCV（若忽略 --symbols，使用 YAML monitoring 列表）
python crawler/ohlcv/ohlcv.py \
  --config utils/collector/collector_config.yaml \
  --start 2023-01-01 --end 2025-01-01 \
  --timeframes 15m  \
  --outdir data/ohlcv_2023_new

# FNG 15m
python crawler/fng/fng_index.py --start 2023-01-01 --outdir data/FNG --outfile fng_15m_utc.csv

# Binance trades（自動續載）
python crawler/trades/binance_trades.py \
  --symbol BTCUSDT --start 2023-01-01 --outdir data/binance_trades/BTCUSDT \
  --output_csv data/binance_trades/BTCUSDT_trades_2023on.csv

# 1m 特徵
python crawler/utils/trades_to_1min_features.py \
  --input_dir data/binance_trades/BTCUSDT \
  --symbol BTCUSDT --start 2023-01-01 --end 2025-11-05 \
  --output_csv data/derived/btcusdt_trades_1min_features.csv

# FNG merge
python crawler/utils/merge_fng_into_15m.py \
  --base_csv data/ohlcv_2023_new/binanceusdm_swap_BTC-USDT-USDT_15m.csv \
  --fng_csv data/FNG/fng_15m_utc.csv \
  --out_csv data/derived/btcusdt_15m_with_fng.csv

# 15m + 1m combine
python crawler/utils/combine_15m_1m.py \
  --ohlcv_csv data/derived/btcusdt_15m_with_fng.csv \
  --trades_1m_csv data/derived/btcusdt_trades_1min_features.csv \
  --minute_steps 15 --fill_policy zero \
  --out_csv data/derived/btcusdt_15m_with_flat_1m.csv
```

若要使用特定 Python 版本，可在 shell 前設 `PYTHON_BIN=python3.10` 或於 `craw_ohlcv_trades.sh` 參數內配置。
