# Spec  — Orderbook Snapshot Feature Module

## 1) Goal / Non-Goal

### Goal

- 以 **`SnapshotRecord`** 作為 core input（已 parse、無檔案 I/O）
- 產出 **1m bar** 的 orderbook feature row（header 對齊 OHLCV：Asia/Taipei 的 bar open）
- Store 維護 **per-symbol DataFrame**：
    - index：`bar_open_datetime_tpe`
    - key：`bar_open_timestamp_ms`
    - cols：header + features（`extra_features` 展開成欄位）

### Non-Goal（v1 不做）

- 用 diff(depth=10) 合成 local orderbook1000
- 用 trade 做 order-flow / signed volume 等 features
- 任何 YAML 控制「算哪些 features」（v1：預設全開，純 code 註冊）

---

## 2) Canonical Schema & Conventions（硬規範）

### 2.1 時間與對齊

- canonical join key：`(symbol, bar_interval, bar_open_timestamp_ms)`
- bar 對齊：用 `floor_to_bar_open_in_tz(..., tz=TZ_TPE)`（Asia/Taipei）
- 每列都保留：
    - `bar_open_timestamp_ms`, `bar_open_datetime_tpe`
    - `snapshot_recv_ts_ms`（debug 延遲/補包）

### 2.2 Row 結構

- **輸出 row 必須符合** `OrderbookSnapshotFeatureRow` 的欄位語意
- 動態特徵一律放 `extra_features`（最後進 DataFrame 時展開成 columns）

### 2.3 版本

- `BarFrameHeader.schema_version = "barframe.v1"`
- Feature 集合另加欄位：`feature_schema_version: str`（例如 `"ob_snapshot_features.v1"`）
    - 任何特徵定義變更必須 bump

---

## 3) Module Boundary（兩層架構）

### 3.1 Core（實盤 & 回測共用，無檔案 I/O）

1. `SnapshotFeatureEngine`（計算層）
2. `MinuteAggregator`（bar 對齊與同分鐘覆蓋策略）
3. `FeatureStore`（DataFrame 為主體的 ring buffer + window query）

### 3.2 Adapter

- `JsonlSnapshotAdapter`：`jsonl line -> SnapshotRecord` iterator
    - Jsonl file name : `snapshot_<YYYYMMDD>T<HHMMSS>.jsonl`
    - Jsonl file format :
        
        ```tsx
        {
            "symbol": "BTCUSDT", 
            "snapshot_ts": 1770912044.838069, 
            "lastUpdateId": 85626389405, 
            "asks": [[88268.45, 2.35608], ...(depth = 1000 , [price,qty])], 
            "bids": [[88268.44, 1.19557], ...(depth = 1000 , [price,qty])]}
        ```
        
---

## 4) Public API（你 trading bot pipeline 會直接呼叫的）

### 4.1 Feature calculators 插拔協議

```python
class FeatureCalculator(Protocol):
    name: str
    version: str
    def compute(self, snap: SnapshotRecord) -> dict[str, float | int]:
        ...

```

### 4.2 SnapshotFeatureEngine

```python
class SnapshotFeatureEngine:
    def __init__(self, calculators: list[FeatureCalculator], feature_schema_version: str = "ob_snapshot_features.v1"):
        ...

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int]:
        """Pure function: snapshot -> features dict (no time bucketing, no IO)."""

```

- 規範：
    - 重複 key：後者覆蓋前者（建議 calculators 的命名避免 collision）
    - 必須輸出最少包含 TOB：best_bid/ask 與 spread/mid（若你放在必備 calculator）

### 4.3 MinuteAggregator（bar row builder）

```python
class MinuteAggregator:
    def ingest(self, snap: SnapshotRecord, features: dict[str, float | int]) -> dict[str, Any]:
        """
        Returns a single bar row dict:
        - includes BarFrameHeader fields
        - includes snapshot metadata fields
        - includes features columns (flattened)
        - includes feature_schema_version
        """

```

- 同分鐘多筆策略：**last-write-wins**
- bucket timestamp 選擇順序：
    1. `snap.header.event_ts_ms`
    2. `snap.header.tx_ts_ms`
    3. `snap.header.recv_ts_ms`

### 4.4 FeatureStore（DataFrame 為主）

```python
class FeatureStore(Protocol):
    def append_row(self, row: dict[str, Any]) -> None: ...
    def get_df(self, symbol: str) -> pd.DataFrame: ...
    def get_window(self, symbol: str, lookback: int) -> pd.DataFrame: ...
    def latest(self, symbol: str) -> pd.Series | None: ...
```

- per-symbol DataFrame
- ring buffer（max_rows 可配置）
- 同分鐘覆蓋：如果最後一列 key 相同，覆蓋（避免同分鐘多 snapshot 造成重複 index）

---

## 5) Default Calculators（v1 必做的最小特徵集合）

### 5.1 必做：Top-of-Book + micro state

輸出欄位（直接作為 columns）：

- `best_bid_p`, `best_bid_q`, `best_ask_p`, `best_ask_q`
- `spread`, `mid`
- `imbalance_l1`（可選，但建議 v1 做）

### 5.2 必做：Quality flags

輸出欄位：

- `flag_crossed_book`
- `flag_bad_sorting`
- `flag_depth_insufficient`（depth < 1000）
- `flag_non_positive_qty`
- depth-within-bps： 5/10/25/50 bps 的 bid/ask 累積量
    - `depth_bid_10bps`, `depth_ask_10bps` …
- bps bins（must included  0–5、5–10、10–25、25-50）
    - `bin_0_5bps_bid_qty`, `bin_0_5bps_ask_qty` …

> 最常用的 depth-within-bps 直接攤平成 columns，模型/策略比較直覺。
> 

---

## 6) DataFrame Contract（輸出形狀硬規範）

對任一 symbol，`store.get_df(symbol)` 必須回傳：

- index：`bar_open_datetime_tpe`（tz-aware，Asia/Taipei）
- 必備 columns（header）：
    - `symbol`, `bar_interval`
    - `bar_open_timestamp_ms`, `bar_close_timestamp_ms`
    - `bar_open_datetime_tpe`, `bar_close_datetime_tpe`（其中 open 被設 index）
    - `schema_version`, `source`
- 必備 columns（meta）：
    - `snapshot_depth`, `snapshot_last_update_id`, `snapshot_event_ts_ms`, `snapshot_recv_ts_ms`
    - `feature_schema_version`
- 必備 columns（features）：TOB + quality flags +（若你採用）depth/bins

---

## 7) Failure Modes / Guardrails（你會踩的坑，spec 直接規範）

- 若 snapshot bids/asks 空 → `SnapshotRecord` validator 直接 fail（上游要處理）
- 若 best_bid >= best_ask → `flag_crossed_book=True`，但仍允許輸出（策略端可濾掉）
- 若排序錯誤（bids 非遞減 / asks 非遞增）→ `flag_bad_sorting=True`
- 若 qty <= 0 → `flag_non_positive_qty=True`
- 所有 flags 只標記，不丟資料（避免 runtime 斷）

---

## 8) Acceptance Criteria（驗收條件）

### 8.1 Unit-level（可自動測）

- `floor_to_bar_open_in_tz` 對齊結果與 Asia/Taipei 分鐘邊界一致
- `make_bar_header` 生成的 close ts = open + 60s - 1ms（interval=1m）
- `SnapshotFeatureEngine.compute` 對同一 snapshot deterministic
- `MinuteAggregator.ingest`：
    - row 內含必備 header/meta/feature_schema_version
    - 同分鐘重複 ingest 時，store 最終只有一列（覆蓋生效）
- `FeatureStore.get_window`：
    - lookback=N 回傳 N 列（不足則回傳現有列數）
    - index 單調遞增

### 8.2 Integration-level（手動/簡單腳本）

- 以 5–10 分鐘 snapshot feed 跑完整 pipeline：
    - `store.get_df("BTCUSDT")` 有連續 1m index
    - 與 OHLCV 以 `bar_open_timestamp_ms` merge 後沒有大面積 NaN（允許缺分鐘）

---

## 9) Implementation Order（你可以直接照這個順序做 ticket）

1. `shared_types.py`（你已完成）
2. `calculators/`：
    - `tob.py`（best bid/ask + spread/mid + imbalance）
    - `quality.py`
    - （選）`depth_bps.py`、`bps_bins.py`
3. `engine.py`：`SnapshotFeatureEngine`
4. `aggregator.py`：`MinuteAggregator`
5. `store.py`：`InMemoryFeatureStore`
6. （選）`adapters/jsonl_snapshot.py`

---

## 10) 不以 YAML 控 feature set 的兼容

- v1：`default_calculators()` 全開、固定輸出
- 仍允許 trading bot 以 code 方式傳 `calculators=[...]`（for research A/B）
- 不允許 YAML 控制 calculators（避免 runtime 漂移）