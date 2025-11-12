# Feature Vetting 工作流程

本模組提供事件驅動的特徵篩選流程，從設定檔載入資料、對齊事件與特徵、建立時間序列交叉驗證折，最後以單變量評分器輸出特徵表現。

## 1. 執行方式

```bash
python -m feature_select.run_vetting --config feature_select/config.yaml \
       [--side all|long|short] [--debug]
```

- `--side` 可覆寫 YAML 的 `filter.side` 設定（預設 `all`）。
- `--debug` 會列印事件數量、折資訊與缺失率最高的特徵，便於診斷資料是否對齊。

## 2. 設定檔 (`config.yaml`)

主要參數如下：

- `paths`: 指向 `TBM_label.csv`、`precomputed.csv` 與輸出結果路徑。
- `index`: 事件與特徵欄位的時間欄名稱與時區，程式會統一轉為 UTC。
- `events`: 定義事件對特徵的窗口彙整方法，例如 `ewm_zscore`、`window_bars`、`strictly_previous`、`max_lag` 等。
- `cv`: 選擇 `purged_kfold` 或 `rolling` 模式、分割數以及 `embargo_hours` 等折疊參數。
- `filter.side`: 控制是否只保留 `long` 或 `short` 事件。
- `scoring`: 包含 `min_non_nan`（單折最少有效樣本數）、`fdr_q`（多重假設控制）。
- `columns`: 指定要排除的欄位或前綴，避免時間、價量等欄位被納入特徵。

## 3. 資料載入與前處理

`EventData` 類別負責以下工作：

1. 讀取 `TBM_label.csv` 與 `precomputed.csv`，並根據設定將時間欄轉為 UTC 時區。
2. 建立事件表 `evt_df`，包含 `y`、`side`、`entry_price`、`t1` 等欄位；`side` 會透過 `±1`、`long/short`、`buy/sell` 等字串自動正規化。
3. 依 `filter.side` 過濾事件；若指定方向後沒有資料會直接報錯。
4. 依 `exclude_exact` 與 `exclude_prefix` 篩選特徵欄。
5. 調用 `_lookback` 將 bar 級特徵做窗口彙整，再以 `merge_asof` 對齊到事件時刻，考慮 `strictly_previous` 與 `max_lag` 限制。

### 3.1 Lookback 窗口模式

對於 `events.reduce` 支援的模式，計算方式如下（窗口長度記為 `W`，`hl` 為半衰期，`q` 為分位數）：

- `last`: 直接取對齊前最後一筆值。
- `sma`: 滑動平均，`mean(window=W, min_periods=max(3, W/4))`。
- `ema`: 指數加權平均，`ewm(halflife=hl, adjust=False)`。
- `zscore`: 以滑動平均與標準差計算 `(x - mean) / (std + 1e-12)`。
- `ewm_zscore`: 指數加權平均與標準差計算 `(x - ewm_mean) / (ewm_std + 1e-12)`。
- `slope`: 對窗口內值做一階線性回歸，回傳斜率 `numpy.polyfit(range(W), values, 1)[0]`。
- `percentile`: 滑動分位數 `quantile(q)`，同樣使用 `min_periods=max(3, W/4)`。

所有結果在對齊事件時皆可選擇是否允許等於 `t0` (`strictly_previous`) 及最大可容忍落後時間 (`max_lag`)。

## 4. 交叉驗證折

`FoldGenerator` 透過 `train/data/folds.py` 建立時間序列折：

- `purged_kfold`: 依月切割並加入 `embargo_hours` 與 `min_train_days`，避免泄漏。
- `rolling`: 以 rolling window 模式產生折，並支援自訂測試頻率。

程式會把折結果傳遞給後續的評分器，並在 `--debug` 模式下列出各折的測試區間與事件數。

## 5. 特徵評分 (`VetScorer`)

對於每個特徵，`VetScorer` 執行以下步驟：

1. 依折逐一計算測試區間內的 ROC AUC，僅當有效樣本數 ≥ `scoring.min_non_nan` 時才記錄，並同步存下各折的樣本量供除錯。
2. 將跨折的 AUC 轉為 rank-biserial (`rrb = 2*AUC - 1`)，計算平均、標準差與 `icir`。
3. 使用所有事件做 Mann–Whitney U 檢定取得 `p` 值，並以 `fdr_q` 執行 Benjamini–Hochberg FDR 矯正。
4. 彙整結果後以 `rrb_mean` 由高至低排序，寫入 `feature_select/results/vetted_features.csv`。

### 5.1 指標定義

輸出欄位的計算方式：

- `auc_mean`: 所有有效折的 ROC AUC 平均值。
- `auc_std`: 以貝塞爾校正 (`ddof=1`) 計算的跨折 AUC 標準差；若僅有一個折則為 0。
- `rrb_mean`: 由 `rrb = 2*AUC - 1` 推得的平均 rank-biserial correlation。
- `rrb_std`: `rrb` 的跨折標準差，計算方式與 `auc_std` 相同。
- `icir`: `rrb_mean / (rrb_std + 1e-12)`，給出訊號強度與穩定度的比值。
- `pval`: 以全體樣本 (`y, score`) 執行 Mann–Whitney U 檢定的雙尾 p 值。
- `fdr_reject`: 對所有特徵的 `pval` 套用 Benjamini–Hochberg (`fdr_q`) 後是否拒絕虛無假設。

若所有特徵在任何折上都未達 `min_non_nan` 門檻，則回傳空表並提示可能的排錯方向（時區、缺失率、折設定等）。

## 6. 注意事項

- `ewm_zscore` 等窗口方法在初期樣本不足時會出現 NumPy 的 degrees-of-freedom 警告，屬正常情況；可視需要略過或調整窗口。
- `min_non_nan` 對單邊事件或高缺失特徵影響大，若只保留 `long` 或 `short` 事件，建議降低門檻或調整折數。
- 輸出 CSV 會保留四位小數，並印出前 20 筆結果於終端。
