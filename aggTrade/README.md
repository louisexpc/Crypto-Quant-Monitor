# AggTrade Feature Module

`aggTrade` 模組把逐筆 `AggTradeRecord` 聚合為 1m trade features，輸出可直接與 OHLCV / orderbook features 以 `bar_open_timestamp_ms` 對齊。

對應規格：[`spec.md`](./spec.md)

## Module 架構

- `trade_state.py`: `TradeBarState`（分鐘內累積狀態）
- `calculators/`: update/finalize calculators
- `engine.py`: `TradeFeatureEngine`
- `aggregator.py`: `TradeMinuteAggregator`（ingest + rollover + flush）
- `store.py`: `InMemoryFeatureStore`
- `adapters/jsonl_trade.py`: `JsonlTradeAdapter`

## 快速使用

```python
from aggTrade import TradeFeatureEngine, TradeMinuteAggregator, InMemoryFeatureStore, default_calculators
from aggTrade.adapters import JsonlTradeAdapter

adapter = JsonlTradeAdapter(source="binance_futures")
engine = TradeFeatureEngine(default_calculators(), feature_schema_version="trade_features.v1")
aggregator = TradeMinuteAggregator(engine=engine)
store = InMemoryFeatureStore(max_rows=10000)

for trade in adapter.iter_file("trades_20260217.jsonl"):
    rows = aggregator.ingest(trade)
    for row in rows:
        store.append_row(row)

for row in aggregator.flush("BTCUSDT"):
    store.append_row(row)

df = store.get_df("BTCUSDT")
```

## 行為重點

- 時間分桶優先順序：`trade.trade_ts_ms -> header.tx_ts_ms -> header.event_ts_ms -> header.recv_ts_ms`
- 同分鐘多筆 trade：只更新狀態，不立刻產生 row
- 跨分鐘時：只 finalize 一次前一分鐘 row
- `flush()`：在回測尾端或程式關閉前輸出尚未 finalize 的當前分鐘
- out-of-order 且落在已過分鐘的 trade：v1 不回補歷史 row，改記錄 `late_trade_count`

## 測試

```bash
python -m pytest aggTrade/tests -q
```

## 多檔回填範例腳本

專案根目錄已提供 [`aggTradeJsonl.py`](../aggTradeJsonl.py) 範例，會：

- 掃描 `data/` 下 `trades_*.jsonl`
- 全域排序後 ingest（示範用）
- 自動 `flush(symbol)` 收尾
- 產出 `<SYMBOL>_trade_1m_features.csv`

執行方式：

```bash
python aggTradeJsonl.py
```
