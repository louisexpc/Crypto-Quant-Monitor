# Feature Computer PTA

基於 `pandas_ta` 的特徵工程 pipeline，從 OHLCV + FNG 資料產生機器學習特徵。

---

## 檔案結構

| 檔案 | 角色 |
|------|------|
| `cfg_pta.yaml` | Pipeline 設定檔（資料路徑、shift、NaN 填充、正規化） |
| `feat_computer_pta.py` | Pipeline 主控：載入資料 → 計算特徵 → shift → fill NaN → normalize → 輸出 |
| `feat_lib_pta.py` | 指標庫：封裝所有 `pandas_ta` 與手刻指標的計算邏輯 |

---

## 快速開始

### CLI 用法

```bash
# 使用預設設定（all mode）
python -m feature_selection.features_computer.feat_computer_pta

# 指定設定檔
python -m feature_selection.features_computer.feat_computer_pta --cfg path/to/cfg.yaml

# 使用 txt 特徵計劃
python -m feature_selection.features_computer.feat_computer_pta --feat_plan path/to/features.txt
```

### Python API

```python
from feature_selection.features_computer.feat_computer_pta import FeatureComputerPTA

# 方法一：使用 cfg_pta.yaml 設定
fc = FeatureComputerPTA()
df_feat = fc.compute()  # all mode

# 方法二：傳入 DataFrame + txt 計劃
fc = FeatureComputerPTA()
df_feat = fc.compute(df_raw=my_ohlcv_df, feat_plan="selected_features.txt")

# 方法三：直接使用底層 lib
from feature_selection.features_computer.feat_lib_pta import FeatureLibPTA

lib = FeatureLibPTA(ohlcv_df)
df_all = lib.compute_all()                        # 暴力全量
df_sel, skipped = lib.compute_from_txt("feat.txt") # 精選
df_rsi = lib.rsi(14)                               # 單一指標
```

---

## 設定檔 `cfg_pta.yaml`

```yaml
data:
  ohlcv_fng_path: data/derived/ohlcv_fng_15m.csv  # 輸入路徑

feat_plan: all                  # all | path/to/feat.txt
strict_txt: true                # txt 未知特徵是否報錯

# pipeline 順序: shift -> fill_nan -> normalization
shift_bars: 1                   # 防 lookahead，shift 1 bar
fill_nan: linear_interp         # last | zero | linear_interp

normalization:
  mode: z_rolling               # none | z_rolling
  rolling_window: 144           # 144 bars = 36h @15m
  min_periods: 1
  std_floor: 1.0e-8             # 防除零

output:
  feat_path: data/derived/ohlcv_fng_15m_feat.csv
```

### Pipeline 處理順序

```
載入 CSV → 欄名小寫化 → 驗證必要欄位 → 建立 UTC DatetimeIndex
→ 計算特徵 → shift(1) → fill NaN → z_rolling normalize → 輸出
```

---

## 三種特徵計劃模式

| 模式 | 設定 | 說明 |
|------|------|------|
| **All** | `feat_plan: all` | 呼叫 `_build_features()`，使用硬編碼的分組與參數 |
| **Txt** | `feat_plan: path/to/feat.txt` | 從 txt 讀特徵名，由 `_parse_feature_name()` 反解析 |
| **compute_all()** | Python API | `FeatureLibPTA.compute_all()` 全量產生所有指標 |

### Txt 格式範例

```text
rsi_14
macd_12_26_9
macds_12_26_9
stochk_16_3_3
ewm_m_12
tod_sin
dir_strength
```

每行一個 canonical feature name，空行與 `#` 開頭的行會被忽略。

---

## 長度分組（Length Groups）

`_build_features()` 和 `compute_all()` 使用三組長度參數：

| 分組 | 預設值 | 用途 | 對應時間（@15m） |
|------|--------|------|-----------------|
| `len_fast` | `[4, 16]` | 動能/震盪指標 | 1h, 4h |
| `len_trend` | `[16, 48, 96]` | 趨勢/強度指標 | 4h, 12h, 24h |
| `len_stats` | `[48, 96, 192]` | 統計量指標 | 12h, 24h, 48h |

---

## 特徵分組與說明

### 1. Raw（原始價量）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `open` | 原始開盤價 | 每根 K 線的第一筆成交價 |
| `high` | 原始最高價 | 該 bar 內最高成交價 |
| `low` | 原始最低價 | 該 bar 內最低成交價 |
| `close` | 原始收盤價 | 該 bar 內最後成交價，最常用的價格參考 |
| `volume` | 原始成交量 | 該 bar 成交的合約/幣量，衡量市場活躍度 |

### 2. FNG（恐懼貪婪指數）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `fng` | 原始 FNG 值 (0-100) | Alternative.me 恐懼貪婪指數，0=極度恐懼, 100=極度貪婪 |
| `fng_diff1` | `fng.diff()` | FNG 一階差分，衡量情緒變化速度 |
| `fng_z7d` | `(fng - roll_mean_672) / roll_std_672` | 以 7 天（672 bars @15m）為窗口的 z 分數，衡量當前情緒相對近一週的異常程度 |

### 3. Momentum / Oscillator（動能/震盪）— `len_fast`

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `rsi_{L}` | `100 - 100/(1 + avg_gain/avg_loss)` | Relative Strength Index。衡量漲跌力道比值，>70 超買 <30 超賣 |
| `willr_{L}` | `(highest_high - close) / (highest_high - lowest_low) × -100` | Williams %R。價格在過去 L bars 高低區間的相對位置，與 RSI 互補 |
| `cmo_{L}` | `(sum_up - sum_down) / (sum_up + sum_down) × 100` | Chande Momentum Oscillator。對稱化的動量震盪指標，-100~100 |
| `cfo_{L}` | 基於線性回歸的預測偏差 | Chande Forecast Oscillator。衡量實際價格偏離線性回歸預測的百分比 |
| `roc_{L}` | `(close - close_L) / close_L × 100` | Rate of Change。L-bar 百分比變化率，最直觀的動量指標 |
| `mom_{L}` | `close - close_L` | Momentum。L-bar 絕對價格變化量 |
| `rvi_{L}` | 基於 close 的相對波動指數 | Relative Vigor Index。衡量收盤價趨向 bar 高點還是低點的傾向 |
| `kdj_{K}_{D}_{SK}` | `J = 3×STOCHk - 2×STOCHd` | KDJ J 線。由隨機指標衍生，對超買超賣的反應比 K/D 更敏感 |

### 4. Stochastic（隨機指標）— `len_fast`

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `stochk_{K}_{D}_{SK}` | `SMA(SK, (close-lowest_low)/(highest_high-lowest_low)×100)` | Stochastic %K。價格在近 K bars 的區間位置，經 SK 平滑 |
| `stochd_{K}_{D}_{SK}` | `SMA(D, %K)` | Stochastic %D。%K 的 D 期平滑，作為信號線 |

### 5. Trend / Strength（趨勢/強度）— `len_trend`

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `ttm_trend_{L}` | 基於 6-bar high/low 均值的趨勢判定 | TTM Trend。+1 上升趨勢 / -1 下降趨勢，用於趨勢過濾 |
| `bias_{L}` | `(close - SMA(L)) / SMA(L) × 100` | 乖離率。價格偏離均線的百分比，衡量回歸壓力 |
| `slope_{L}` | 線性回歸斜率 | 價格的 L-bar 線性變化率，衡量趨勢速度 |
| `vhf_{L}` | `|max(C,L) - min(C,L)| / Σ|ΔC|` | Vertical Horizontal Filter。趨勢強度 vs. 盤整，>1 趨勢明顯 |
| `sma_{L}` | `mean(close, L)` | Simple Moving Average。L-bar 等權重均線 |
| `ema_{L}` | 指數加權移動均值 | Exponential MA。近期權重更高的均線，反應更快 |
| `tema_{L}` | 三重 EMA（3×EMA - 3×EMA² + EMA³） | Triple EMA。最低延遲的均線，適合快速跟蹤 |
| `dpo_{L}` | `close - SMA(L, shift=L/2+1)` | Detrended Price Oscillator。去除趨勢後的週期震盪 |
| `adx_{L}` | DI 差的平滑 | Average Directional Index。趨勢強度（不分方向），>25 表示強趨勢 |
| `dmp_{L}` | +DI（上升方向線） | 多頭方向強度，dmp > dmn → 上升趨勢 |
| `dmn_{L}` | -DI（下降方向線） | 空頭方向強度 |

### 6. MACD

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `macd_{F}_{S}_{SIG}` | `EMA(F) - EMA(S)` | MACD 主線。兩條 EMA 的差值，衡量短期動量變化 |
| `macds_{F}_{S}_{SIG}` | `EMA(SIG, macd)` | 信號線。MACD 的平滑，交叉產生買賣信號 |
| `macdh_{F}_{S}_{SIG}` | `macd - macds` | 柱狀圖。MACD 與信號線的差值，衡量動量加速/減速 |

預設參數：`(12,26,9)` 標準、`(4,12,6)` 加速。

### 7. AMAT LR（趨勢通道）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `amat_lr_{F}_{S}_{M}` | Archer Moving Average Trend LR | 長期趨勢方向信號，基於快慢均線排列判斷多空 |

預設：`(8, 21, 2)`。

### 8. Statistics（統計量）— `len_stats`

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `entropy_{L}` | 滾動熵值 | 價格序列在 L-bar 窗口的資訊熵。高熵 = 隨機/無序；低熵 = 有規律 |
| `skew_{L}` | 滾動偏度 | 報酬分佈的不對稱性。正偏 = 右尾較長（偶爾大漲）；負偏 = 左尾較長 |
| `kurtosis_{L}` | 滾動峰度 | 報酬分佈的尖峭度。高峰度 = 肥尾分佈，極端事件頻繁 |

### 9. Pattern / Filter（型態/過濾）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `decreasing_{L}` | 近 L bars 是否每根 close 遞減（0/1） | 連續下跌偵測，用於趨勢確認或 mean-reversion 觸發 |
| `decay_{L}` | 線性衰減函數 | 隨時間線性衰減的權重函數，用於近期加權 |

### 10. Volatility（波動）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `truerange` | `max(H-L, |H-C₋₁|, |L-C₋₁|)` | 單期真實區間。考慮跳空的實際波動幅度 |
| `atr_{L}` | `EMA(L, truerange)` | Average True Range。L-bar 平均真實波動，衡量市場活躍度 |
| `atrp_{L}` | `atr / |close|` | ATR 百分比。相對波動率，可跨價格水位比較 |
| `hl_range_{W}` | `rolling_mean(W, (H-L) / |C|)` | 高低區間均值（百分比）。W-bar 平均 bar 內波幅佔收盤價的比例 |
| `massi_9_25` | `EMA(9, H-L) / EMA(25, H-L)` | Mass Index。高低波幅的 EWM 比值，偵測區間收窄→擴張的 reversal bulge |
| `bbp_16_2.0` | `(close - BB_lower) / (BB_upper - BB_lower)` | Bollinger %B。價格在布林通道的相對位置，>1 突破上軌 <0 突破下軌 |

### 11. EWMRET（指數加權報酬統計）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `ewm_m_{HL}` | `EWM(halflife=HL, log_return).mean()` | 對數報酬的指數加權均值。HL 越小越貼近最近行為（短期趨勢） |
| `ewm_s_{HL}` | `EWM(halflife=HL, log_return).std()` | 對數報酬的指數加權標準差。衡量近期實現波動率 |

預設 halflife：`4` (1h)、`12` (3h)、`48` (12h)。

### 12. Volume（量能指標）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `pvo_{F}_{S}_{SIG}` | `(EMA(F,vol) - EMA(S,vol)) / EMA(S,vol) × 100` | Percentage Volume Oscillator。量版 MACD，衡量成交量動能 |
| `pvos_{F}_{S}_{SIG}` | PVO 信號線 | PVO 的平滑 |
| `pvoh_{F}_{S}_{SIG}` | PVO 柱狀圖 | PVO 與信號線之差 |
| `pvr` | `close_chg / volume_chg` | Price Volume Rank。價量同步性判斷 |
| `bop` | `(close - open) / (high - low)` | Balance of Power。衡量多空買賣力道，+1 多方控制 -1 空方控制 |
| `kvo_{F}_{S}_{SIG}` | Klinger Volume Oscillator | 結合價格趨勢與成交量的動能指標，偵測資金流向 |
| `kvos_{F}_{S}_{SIG}` | KVO 信號線 | |
| `kvoh_{F}_{S}_{SIG}` | KVO 柱狀圖 | |
| `efi_{L}` | `EMA(L, close_chg × volume)` | Elder's Force Index。結合價格變化與成交量的力道指標 |
| `eom_{L}` | `(midpoint_move × (H-L)) / volume` | Ease of Movement。衡量價格移動的「容易度」—量少價動大 = 容易 |

### 13. Price-Volume Interaction（價量融合）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `dir_strength` | `(close - open) / (high - low)` | 方向強度。K 線實體佔整根 K 線的比例，衡量單根 bar 的確定性 |
| `pxv_lr_vchg` | `log_return × volume_pct_change` | 報酬×量變。價量同向放大 → 正值（趨勢確認），反向 → 負值（背離） |
| `dirxvol` | `dir_strength × volume` | 方向強度 × 成交量。帶方向的成交力道，類似 VWAP 思想的簡化 |

### 14. Returns（報酬）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `logret_{L}` | `ln(close / close_L)` | L-bar 對數報酬。可加性讓多期報酬方便計算，常態近似更好 |
| `pctret_{L}` | `(close - close_L) / close_L` | L-bar 百分比報酬。直觀的漲跌幅 |

預設 lag：`1` (15m)、`4` (1h)、`8` (2h)、`16` (4h)。

### 15. Time Cycle（時間週期）

| 特徵名 | 公式 | 物理意義 |
|--------|------|---------|
| `tod_sin` | `sin(2π × hour / 24)` | 日內時間正弦編碼。讓模型學習每日週期性（如亞洲/歐洲/美洲盤時段） |
| `tod_cos` | `cos(2π × hour / 24)` | 日內時間餘弦編碼。與 sin 搭配提供完整的圓形表示 |
| `dow_sin` | `sin(2π × dayofweek / 7)` | 週內天數正弦編碼。捕捉週末效應等週期規律 |
| `dow_cos` | `cos(2π × dayofweek / 7)` | 週內天數餘弦編碼 |

預設時區：`Asia/Taipei`。

---

## Normalization

### z_rolling 正規化

```
z = (x - rolling_mean(window)) / max(rolling_std(window), std_floor)

window     = 144   (36h @15m)
min_periods = 1
std_floor   = 1e-8
```

在 `shift_bars=1` 之後執行，因此 rolling 窗口只看歷史值（無 lookahead）。

---

## 輸出格式

| 欄位 | 型別 | 說明 |
|------|------|------|
| `datetime` | datetime64 | UTC 時間戳 |
| `timestamp` | int64 | Unix timestamp (秒) |
| 所有特徵 | float32 | 已 shift + fill NaN + normalize |

---

## 開發指引

### 新增指標

1. 在 `feat_lib_pta.py` 的 `FeatureLibPTA` class 中新增方法
2. 若為 single-output（返回 `pd.Series`），加入 `_parse_feature_name()` 的 `scalar_roots` 或手寫 regex
3. 若為 multi-output（返回 `pd.DataFrame`），加入對應的 `kind: "multi"` 解析規則
4. 在 `compute_all()` 的對應分組中加入特徵名
5. 在 `feat_computer_pta.py` 的 `_build_features()` 對應 group 中加入呼叫

### 命名規則

- 全小寫 + 底線分隔
- 單參數：`{indicator}_{length}`，如 `rsi_14`
- 多參數：`{indicator}_{param1}_{param2}_...`，如 `macd_12_26_9`
- Multi-output 子欄位：`{sub}_{params}`，如 `macds_12_26_9`、`ewm_m_12`
