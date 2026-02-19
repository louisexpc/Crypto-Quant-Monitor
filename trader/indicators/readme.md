# Trader Indicators (新版)

## 1) 目的
- 這層是 **runtime 特徵計算器**，給交易端吃。
- 核心是 `FeatureComputer` + `FeatureLibPTA`：
  - `FeatureComputer`：時間/欄位清理、`shift(1)`、NaN 策略、可選 rolling z-score。
  - `FeatureLibPTA`：所有指標用 `pandas_ta` 計算，支援用 `feat txt` 指定要算哪些欄位。

## 2) 目前目錄
- `trader/indicators/feature_computer.py`
- `trader/indicators/feature_lib.py`
- `trader/indicators/feature.yaml`
- `trader/indicators/feat_cols/rank_biserial_40_long_feat_cols.txt`
- `trader/indicators/feat_cols/rank_biserial_40_short_feat_cols.txt`

## 3) Config（`feature.yaml`）
實際有用到的 key：

- `time.columns`
  - 候選時間欄位，依序找（預設 `["datetime", "timestamp"]`）。
- `ohlcv_required`
  - 必備欄位（通常 `open/high/low/close/volume/fng`）。
- `selected_feat_path.long / selected_feat_path.short`
  - long/short 對應的 `feat txt`。
- `nan_policy`
  - `raise | drop | linear_interp`。
- `feat_normalization.enabled`
  - 是否開 rolling z-score。
- `feat_normalization.rolling_window`
  - rolling window。
- `feat_normalization.skip_cols`
  - z-score 跳過欄位（可空）。
- `feat_normalization.std_floor`
  - 標準差下限，避免除以 0（預設 `1e-8`）。

目前保留但未用在 `FeatureComputer.compute()` 的 key：
- `time.freq`
- `trades.*`

## 4) FeatureComputer 流程
`FeatureComputer.compute(df_raw, side)`：

1. 標準化欄名（全轉小寫）。
2. 建 UTC `DatetimeIndex`（優先 `time.columns`，否則用原 index）。
3. 驗證 `ohlcv_required` 並轉 numeric。
4. 先對 raw 套 `nan_policy`。
5. 依 `side` 讀 `selected_feat_path.{side}`，呼叫 `FeatureLibPTA.compute_from_txt(...)`。
6. 對特徵統一 `shift(1)`（防洩漏）。
7. 特徵再做 inf/NaN 清理 + `nan_policy` + `float32`。
8. 若開啟，套 rolling z-score。

輸出：
- 只回傳特徵欄位（`pd.DataFrame`，UTC DatetimeIndex，`float32`）。
- 不含原始 OHLCV/time columns。

## 5) FeatureLibPTA（命名規則）
`feature_lib.py` 用 canonical 名稱解析，例如：

- raw：`open/high/low/close/volume/fng`
- single：`rsi_16`, `mom_4`, `entropy_96`, `atr_14`, `atrp_14`, `hl_range_32`, `kdj_9_3_3`
- multi family：
  - `macd_12_26_9`, `macds_12_26_9`, `macdh_12_26_9`
  - `adx_48`, `dmp_48`, `dmn_48`
  - `pvo_12_26_9`, `pvos_12_26_9`, `pvoh_12_26_9`
  - `kvo_34_55_13`, `kvos_34_55_13`, `kvoh_34_55_13`
  - `ewm_m_12`, `ewm_s_12`
- 其他：`dir_strength`, `pxv_lr_vchg`, `dirxvol`, `tod_sin`, `tod_cos`, `dow_sin`, `dow_cos`

三種入口：
- `compute_from_txt(txt_path, strict=...)`
- `compute_from_list(feature_names, strict=...)`
- `compute_all(...)`

## 6) 使用範例
```python
from pathlib import Path
import yaml
import pandas as pd

from trader.indicators.feature_computer import FeatureComputer

cfg = yaml.safe_load(Path("trader/indicators/feature.yaml").read_text())
fc = FeatureComputer(cfg)

# df_raw 需有 datetime/timestamp + ohlcv + fng
df_raw = pd.read_csv("data/derived/ohlcv_fng_15m.csv")

feat_long = fc.compute(df_raw, side="long")
feat_short = fc.compute(df_raw, side="short")
```

## 7) 行為重點
- `compute_from_txt(..., strict=False)`：未知欄位會 skip，並印 warning，不會中斷。
- runtime 預設會插值/補值（依 `nan_policy`）；如果你要嚴格 fail，設 `nan_policy: raise`。
- long/short 用不同 feat list；請確保和 checkpoint 的 `feature_columns` 對齊。
