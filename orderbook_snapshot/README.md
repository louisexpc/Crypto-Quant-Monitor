# Orderbook Snapshot Module

`orderbook_snapshot` 提供一個「無 I/O 的核心計算層 + 可選 adapter」來把 orderbook snapshot 轉成 1m bar 特徵列，並以 per-symbol DataFrame 形式保存。

對應 spec 請見：[`spec.md`](./spec.md)

---

## 1) Module 架構

```text
SnapshotRecord
   │
   ├─(FeatureCalculator x N)
   │      └─ SnapshotFeatureEngine.compute(snap) -> features dict
   │
   ├─ MinuteAggregator.ingest(snap, features) -> bar row dict
   │
   └─ InMemoryFeatureStore.append_row(row)
          ├─ per-symbol DataFrame
          ├─ index: bar_open_datetime_tpe (Asia/Taipei)
          └─ last-write-wins (同分鐘覆蓋)
```

### 目錄

- `calculators/`
  - `tob.py`: best bid/ask、spread、mid、imbalance、microprice
  - `quality.py`: crossed/sorting/depth/qty flags
  - `depth_bps.py`: depth within bps (5/10/25/50)
  - `bps_bins.py`: bps bins (0-5, 5-10, 10-25, 25-50)
- `engine.py`: `SnapshotFeatureEngine`
- `aggregator.py`: `MinuteAggregator`
- `store.py`: `FeatureStore` protocol + `InMemoryFeatureStore`
- `adapters/jsonl_snapshot.py`: `JsonlSnapshotAdapter`
- `domain_types.py`: 對 `shared_types.py` 的統一匯入介面
- `tests/`: 目前的 contract 測試

---

## 2) Public API

`orderbook_snapshot/__init__.py` 暴露：

- `SnapshotFeatureEngine`
- `MinuteAggregator`
- `FeatureStore`
- `InMemoryFeatureStore`
- `default_calculators`

---

## 3) 快速使用

## 3.1 最小端到端（單筆 snapshot）

```python
from orderbook_snapshot import (
    SnapshotFeatureEngine,
    MinuteAggregator,
    InMemoryFeatureStore,
    default_calculators,
)
from orderbook_snapshot.domain_types import SnapshotRecord

snap = SnapshotRecord(
    header={
        "symbol": "BTCUSDT",
        "event_type": "snapshot",
        "source": "binance_futures",
        "recv_ts_ms": 1771207265000,
        "event_ts_ms": 1771207265000,
    },
    depth=1000,
    last_update_id=123,
    bids=[(100.0, 2.0), (99.9, 3.0), (99.8, 1.5)],
    asks=[(100.1, 4.0), (100.2, 1.0), (100.3, 2.2)],
)

engine = SnapshotFeatureEngine(default_calculators())
aggregator = MinuteAggregator(bar_interval="1m", feature_schema_version="ob_snapshot_features.v1")
store = InMemoryFeatureStore(max_rows=10_000)

features = engine.compute(snap)
row = aggregator.ingest(snap, features)
store.append_row(row)

df = store.get_df("BTCUSDT")
print(df.tail(1))
```

## 3.2 使用 JSONL adapter

```python
from orderbook_snapshot import SnapshotFeatureEngine, MinuteAggregator, InMemoryFeatureStore, default_calculators
from orderbook_snapshot.adapters import JsonlSnapshotAdapter

adapter = JsonlSnapshotAdapter(source="binance_futures")
engine = SnapshotFeatureEngine(default_calculators())
aggregator = MinuteAggregator()
store = InMemoryFeatureStore(max_rows=5000)

for snap in adapter.iter_file("snapshot_20260216T100000.jsonl"):
    row = aggregator.ingest(snap, engine.compute(snap))
    store.append_row(row)

df = store.get_df("BTCUSDT")
window = store.get_window("BTCUSDT", lookback=200)
latest = store.latest("BTCUSDT")
```

---

## 4) DataFrame 契約重點

`store.get_df(symbol)`：

- index: `bar_open_datetime_tpe`（tz-aware, Asia/Taipei）
- canonical key: `(symbol, bar_interval, bar_open_timestamp_ms)`
- 同分鐘多筆：以最後一筆覆蓋（last-write-wins）

常見欄位：

- Header
  - `symbol`, `bar_interval`
  - `bar_open_timestamp_ms`, `bar_close_timestamp_ms`
  - `bar_open_datetime_tpe`, `bar_close_datetime_tpe`
  - `schema_version`, `source`
- Snapshot meta
  - `snapshot_depth`, `snapshot_last_update_id`
  - `snapshot_event_ts_ms`, `snapshot_recv_ts_ms`
  - `feature_schema_version`
- Features（預設 calculators）
  - TOB: `best_bid_p`, `best_bid_q`, `best_ask_p`, `best_ask_q`, `spread`, `mid`, `imbalance_l1`, `microprice`
  - Flags: `flag_crossed_book`, `flag_bad_sorting`, `flag_depth_insufficient`, `flag_non_positive_qty`
  - Depth/Bins: `depth_bid_5bps`...`depth_ask_50bps`, `bin_0_5bps_bid_qty`...`bin_25_50bps_ask_qty`

---

## 5) 客製 calculators

你可以替換預設 calculators（不走 YAML）：

```python
from orderbook_snapshot.engine import SnapshotFeatureEngine

class MyCalc:
    name = "my_custom"
    version = "1.0.0"
    def compute(self, snap):
        return {"my_feature": 1.0}

engine = SnapshotFeatureEngine(calculators=[MyCalc()])
```

注意：`MinuteAggregator` 會檢查 TOB 必備欄位（`best_bid_*`, `best_ask_*`, `spread`, `mid`），若缺少會拋 `ValueError`。

---

## 6) 測試

目前 contract 測試位於：`orderbook_snapshot/tests`

```bash
python -m pytest orderbook_snapshot/tests -q
```

已覆蓋：

- TPE 分鐘對齊
- bar close = open + 60s - 1ms
- engine deterministic
- 同分鐘覆蓋策略
- window lookback 與 index monotonic

---

## 7) 你應該知道的注意事項

- 目前專案已將原本根目錄 `types.py` 改名為 `shared_types.py`，避免遮蔽 Python stdlib `types`。
- `orderbook_snapshot/domain_types.py` 請維持從 `shared_types` 直接匯入，避免動態載入導致 Pydantic model rebuild 問題。
- `JsonlSnapshotAdapter` 會把 `snapshot_ts` 視為秒（float）或毫秒（int）自動轉成 ms。
- 若 snapshots 存在空 bids/asks，會由 `SnapshotRecord` 驗證失敗（上游需處理）。
- feature 定義有變更時，請同步 bump `feature_schema_version`（例如 `ob_snapshot_features.v2`）。

---

## 8) 典型整合方式（與 OHLCV merge）

```python
# ob_df: from store.get_df("BTCUSDT")
# ohlcv_df: your ohlcv bars (same bar_open_timestamp_ms key)

merged = ohlcv_df.merge(
    ob_df.reset_index(drop=False),
    on=["bar_open_timestamp_ms"],
    how="left",
)
```

建議 merge key 使用 `bar_open_timestamp_ms`，時間欄位僅用於可讀性與 debug。
