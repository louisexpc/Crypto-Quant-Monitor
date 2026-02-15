# Trader Indicators

## 概要
- 純特徵計算器：吃原始 OHLCV(+FNG/Trades) DataFrame，依配置計算技術指標、shift(1)、可選 rolling z-score。
- 不做檔案 I/O；呼叫端自行讀寫 CSV/Parquet。
- 不依賴 `train/`，指標由本地 `indicators_lib` 提供。
- 支援白名單：先用白名單過濾 builder，避免計算不需要的特徵；計算後再保留白名單欄位。

## 配置
- `config/compute_config.yaml`：
  - `time.columns`：時間欄候選（例：["datetime", "timestamp"]）
  - `time.freq`：預期頻率（預設 15min，僅檢查不補齊）
  - `ohlcv_required`：必備欄位（含 fng）
  - `trades.enabled/window_len`：若計畫包含 1m 聚合特徵則開啟
  - `feat_plan.long_feat_path` / `short_feat_path`：多/空特徵計畫路徑
  - `selected_feat_path.long/short`：白名單 txt；留空則計算所有 enabled 特徵
  - `feat_normalization.enabled/rolling_window/skip_cols`：rolling z-score 設定
  - `feat_normalization.std_floor`：滾動標準差的下限（預設 1e-8，避免常數序列除以 0）
  - `manifest_path`：若啟用 `export.manifest_enabled`（預設 False），可指定 manifest 儲存路徑
  - `nan_policy`：raise | drop | linear_interp
- `feat_cfg/long_feat.yaml` / `short_feat.yaml`：特徵清單，格式 `{name, enabled, kwargs}`。

## 主要介面
- `FeatureComputer(cfg: dict)`  
  - cfg 需對應上方 compute_config 的鍵。
- `compute(df_raw: pd.DataFrame, side="long") -> pd.DataFrame`  
  - 輸入需含時間欄或 DatetimeIndex；輸出僅含計算後特徵（時間索引保留，未附原 OHLCV）。

## 流程概述
1. 時間正規化：UTC、排序、去重，檢查 freq 是否等距。
2. 檢查必備欄位並轉 float32（含 fng）；依 `nan_policy` 處理缺值。
3. 讀取白名單（若有），先過濾掉不會產生目標欄位的 builder，逐項計算並 shift(1)。
4. 合併特徵、檢查重複，若有白名單僅保留白名單欄位。
5. 可選 rolling z-score（skip cols 依設定，std_floor 避免除以 0）。
6. 若 `export.manifest_enabled=True`，輸出 manifest.json（欄名→family/kwargs/來源 spec）。

## 簡易用法
```python
from trader.indicators.feature_computer import FeatureComputer
import yaml, pandas as pd

cfg = yaml.safe_load(open("trader/indicators/config/compute_config.yaml"))
fc = FeatureComputer(cfg)
df_feat = fc.compute(df_raw, side="long")
```

## 輸入範例（15m OHLCV+FNG）

| datetime                  | timestamp   |   open |    high |     low |   close |   volume |   fng |
|:--------------------------|:------------|-------:|--------:|--------:|--------:|---------:|------:|
| 2022-12-31T16:00:00+00:00 | 1672502400  | 16586.1| 16591.7 | 16586.0 | 16588.4 |   873.158|  25.0 |
| 2022-12-31T16:15:00+00:00 | 1672503300  | 16588.4| 16599.3 | 16577.3 | 16599.1 |  1409.596|  25.0 |
| 2022-12-31T16:30:00+00:00 | 1672504200  | 16599.1| 16599.3 | 16590.7 | 16593.6 |   675.457|  25.0 |
| 2022-12-31T16:45:00+00:00 | 1672505100  | 16593.7| 16609.3 | 16592.3 | 16596.0 |   993.925|  25.0 |
| 2022-12-31T17:00:00+00:00 | 1672506000  | 16596.1| 16600.1 | 16576.2 | 16578.6 |  1496.217|  25.0 |
