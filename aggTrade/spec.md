# Spec v1 — Trades (aggTrade) Feature Module

## 1) Goal / Non-Goal

### Goal

- 以 **`AggTradeRecord`** 作為 core input（已 parse、無檔案 I/O）
- 產出 **1m bar** 的 trade-derived feature row（header 對齊：Asia/Taipei 的 bar open）
- Store 維護 **per-symbol DataFrame**：
    - index：`bar_open_datetime_tpe`
    - key：`bar_open_timestamp_ms`
    - cols：header + trade features（固定欄位 + 可擴充欄位）

### Non-Goal（v1 不做）

- 不做 L2/L3 orderbook 的 local reconstruction（那是 diff/book 模組）
- 不做 tick-by-tick microprice simulation / impact 模型
- 不做 VPIN / Kyle lambda 這種需要更完整 microstructure 校正的估計（v2 再做）
- 不用 YAML 控制要算哪些 features（v1：預設全開，純 code 註冊）

---

## 2) Canonical Schema & Conventions（硬規範）

### 2.1 時間與對齊

- canonical join key：`(symbol, bar_interval, bar_open_timestamp_ms)`
- bar 對齊：`floor_to_bar_open_in_tz(..., tz=TZ_TPE)`
- bucket timestamp 選擇（trade 特別重要）：
    1. `trade.trade_ts_ms`（AggTradeRecord 的交易時間；應由來源 `msg.data.T` 映射）
    2. 若缺失才 fallback：`header.tx_ts_ms` → `header.event_ts_ms` → `header.recv_ts_ms`

### 2.2 Row 結構

- 輸出 row 必須遵循 `BarFrameHeader` 欄位語意
- trade features 欄位策略：
    - v1 固定一組「最小但夠用」欄位（見 5）
    - 擴充欄位走 `extra_features: dict`（與 snapshot 模組一致，最後展開成 columns）

### 2.3 版本

- `BarFrameHeader.schema_version = "barframe.v1"`
- trade feature schema 另加：
    - `feature_schema_version: str = "trade_features.v1"`
- 任何 trade feature 定義或計算方式變動 → bump `feature_schema_version`

---

## 3) Module Boundary（兩層架構）

### 3.1 Core（實盤 & 回測共用，無檔案 I/O）

1. `TradeFeatureEngine`（逐筆 trade → 累積統計，不做時間 bucket）
2. `TradeMinuteAggregator`（把逐筆 trade 累積成 1m bar row dict）
3. `FeatureStore`（沿用同一個 InMemoryFeatureStore 介面；或獨立 store 也可，但 contract 相同）

> 你現有的 Snapshot store/aggregator 是 “snapshot 驅動”；trade 是 “event-driven accumulation”，所以聚合器型態不同：trade aggregator 必須維護「當前分鐘的累積狀態」。
> 

### 3.2 Adapter

- `JsonlTradeAdapter`：`jsonl line -> AggTradeRecord` iterator
- file trades_<YYYYMMDD>.jsonl format :
    
    ```json
    {"type": "trade", "symbol": "BTCUSDT", "recv_ts": 1770912001.5185976, "msg": {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "E": 1770912001477, "a": 3133883427, "s": "BTCUSDT", "p": "67112.50", "q": "0.166", "nq": "0.166", "f": 7278176248, "l": 7278176256, "T": 1770912001475, "m": false}}}
    {"type": "trade", "symbol": "BTCUSDT", "recv_ts": 1770912001.518605, "msg": {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "E": 1770912001477, "a": 3133883428, "s": "BTCUSDT", "p": "67112.60", "q": "0.038", "nq": "0.038", "f": 7278176257, "l": 7278176257, "T": 1770912001475, "m": false}}}
    {"type": "trade", "symbol": "BTCUSDT", "recv_ts": 1770912001.5186083, "msg": {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "E": 1770912001477, "a": 3133883429, "s": "BTCUSDT", "p": "67112.80", "q": "0.002", "nq": "0.002", "f": 7278176258, "l": 7278176258, "T": 1770912001475, "m": false}}}
    ...etc
    ```
    

---

## 4) Public API（Trading bot pipeline 會直接呼叫）

### 4.1 Feature calculators 插拔協議（同 Snapshot）

```python
class TradeFeatureCalculator(Protocol):
    name: str
    version: str
    def update(self, state: "TradeBarState", trade: AggTradeRecord) -> None: ...
    def finalize(self, state: "TradeBarState") -> dict[str, float | int]: ...

```

> 設計 rationale：trade features 多為「累積型」：count/sum/imbalance/vwap…，用 update/finalize 比每筆 compute 更自然且更快。
> 

### 4.2 TradeBarState（分鐘內狀態容器）

```python
@dataclass
class TradeBarState:
    symbol: str
    bar_open_timestamp_ms: int
    bar_interval: BarInterval = "1m"
    # accumulators:
    trade_count: int
    volume: float
    notional: float
    buy_volume: float
    sell_volume: float
    buy_notional: float
    sell_notional: float
    vwap_num: float  # sum(price * qty)
    vwap_den: float  # sum(qty)
    # optional:
    last_trade_price: float | None
    first_trade_price: float | None
    high_price: float | None
    low_price: float | None
@dataclass
class TradeBarState:
    symbol: str
    bar_open_timestamp_ms: int
    bar_interval: BarInterval = "1m"
    # accumulators:
    trade_count: int
    volume: float
    notional: float
    buy_volume: float
    sell_volume: float
    buy_notional: float
    sell_notional: float
    vwap_num: float  # sum(price * qty)
    vwap_den: float  # sum(qty)
    # optional:
    last_trade_price: float | None
    first_trade_price: float | None
    high_price: float | None
    low_price: float | None

```

### 4.3 TradeFeatureEngine

```python
class TradeFeatureEngine:
    def __init__(self, calculators: list[TradeFeatureCalculator], feature_schema_version: str = "trade_features.v1"):
        ...

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        """Update accumulators for one trade event."""

    def finalize(self, state: TradeBarState) -> dict[str, float | int]:
        """Finalize a bar's features from state (pure, deterministic)."""

```

### 4.4 TradeMinuteAggregator（bar builder + rollover）

```python
class TradeMinuteAggregator:
    def ingest(self, trade: AggTradeRecord) -> list[dict[str, Any]]:
        """
        Ingest one trade event.
        Returns:
          - [] most of the time
          - [finalized_row] when a minute rolls over and the previous bar is completed
        """
    def flush(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Force-finalize the current open bar(s) (useful at shutdown/backfill end).
        """

```

- 行為：
    - 根據 trade timestamp 決定該 trade 屬於哪個 minute
    - 若 trade 來到新 minute：
        - finalize 舊 minute → 產出 1 row dict
        - 初始化新 minute state → update
    - 同 minute 多 trades：只更新 state，不立刻產出 row

### 4.5 FeatureStore（沿用 snapshot spec）

同一套 `append_row/get_df/get_window/latest`。

---

## 5) Default Calculators（v1 必做的 trade features）

> v1 目標：中低頻可用、穩定、可與 OHLCV/orderbook join。
> 

### 5.1 必做：基礎成交統計（count/volume/notional）

輸出欄位（columns）：

- `trade_count`
- `trade_volume`（sum qty）
- `trade_notional`（sum price*qty）

### 5.2 必做：方向性成交量（用 `is_buyer_maker`）

在 Binance aggTrade：

- `is_buyer_maker=True` 通常代表「主動賣（sell-initiated）」；反之為「主動買」
    
    （這是交易所 microstructure 常用方向判定，v1 先採用這個 convention。）
    

輸出欄位：

- `buy_volume`, `sell_volume`
- `buy_notional`, `sell_notional`
- `signed_volume = buy_volume - sell_volume`
- `volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + eps)`

### 5.3 必做：VWAP / trade return proxies

輸出欄位：

- `vwap = sum(p*q) / sum(q)`（若 sum(q)=0 → NaN/0 + flag）
- `first_trade_price`, `last_trade_price`
- `trade_high_price`, `trade_low_price`
- `trade_return = (last_trade_price / first_trade_price - 1)`（若缺 trade → NaN）
    - 注意：這不是 OHLCV return，它是「當分鐘成交序列的粗略 proxy」

### 5.4 必做：Quality flags

輸出欄位：

- `flag_no_trades`（該分鐘沒有 trade）
- `flag_non_positive_qty`（qty <= 0）
- `flag_non_positive_price`（price <= 0）
- `flag_out_of_order_ts`（trade_ts 逆序太多；用於診斷 feed）

> v1 不做丟資料：只打 flag，策略端自行過濾。
> 

---

## 6) DataFrame Contract（輸出形狀硬規範）

`store.get_df(symbol)` 回傳：

- index：`bar_open_datetime_tpe`（tz-aware，Asia/Taipei）
- 必備 header columns：
    - `symbol`, `bar_interval`
    - `bar_open_timestamp_ms`, `bar_close_timestamp_ms`
    - `schema_version`, `source`
- 必備 meta：
    - `feature_schema_version`
    - `last_trade_recv_ts_ms`（可選但建議；debug 用）
- 必備 trade features（5.1–5.4）

---

## 7) Failure Modes / Guardrails

- 若 trade timestamp 缺失 → fallback header 時間，但需打 `flag_missing_trade_ts=True`
- 若分鐘內 trade 缺失：
    - 仍可產出 row（對齊 OHLCV）或不產出 row（由策略決定）
    - v1 建議：**不主動補齊空分鐘**（避免假資料）；join 後 NaN 留給下游處理
- 若 out-of-order trades：
    - 允許更新，但若 trade_ts 落在「已 finalize 的 minute」：
        - v1 建議：丟到 `late_trade_count` 並打 flag（不回補舊分鐘，避免回寫 DataFrame）
        - v2 才考慮 watermark / allowed lateness 機制

---

## 8) Acceptance Criteria（驗收條件）

### 8.1 Unit-level

- bucket 對齊：
    - 同一分鐘內 trades → 同 `bar_open_timestamp_ms`
    - 跨分鐘 roll over → finalize 前一分鐘只產出一次 row
- VWAP 正確：
    - `vwap_num/vwap_den` 計算一致
- direction 正確：
    - `is_buyer_maker` 對應 buy/sell volume 的規則固定（寫在 docstring + tests）
- Store 行為：
    - append 後 `get_window(lookback)` 的 index 單調遞增
    - 同分鐘不會生成兩列（除非你刻意允許回補）

### 8.2 Integration-level

- 用 5–10 分鐘 aggTrade feed 跑完整 pipeline：
    - `store.get_df("BTCUSDT")` 有連續分鐘（有 trades 的分鐘）
    - 與 OHLCV / orderbook features merge（用 `bar_open_timestamp_ms`）後形狀正確

---

## 9) Implementation Order（ticket 順序）

1. `trade_state.py`：`TradeBarState`
2. `calculators/`：
    - `base_stats.py`（count/volume/notional）
    - `directional_flow.py`（buy/sell/signed/imbalance）
    - `vwap_prices.py`（vwap/first/last/high/low/return）
    - `quality.py`
3. `engine.py`：`TradeFeatureEngine`
4. `aggregator.py`：`TradeMinuteAggregator`（含 rollover + flush）
5. `store.py`：沿用你的 `InMemoryFeatureStore`（同 contract）
6. `adapters/jsonl_trade.py`

---

## 10) 與你現有 Snapshot module 的整合方式（關鍵點）

- 兩者輸出都必須帶同樣的 bar header 欄位（`bar_*`）
- 最終 join 建議用：
    - `["symbol", "bar_interval", "bar_open_timestamp_ms"]`
- OHLCV 也把 `kline_open_timestamp_ms` rename 成 `bar_open_timestamp_ms`，就能無痛合併