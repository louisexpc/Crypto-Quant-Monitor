"""多個 aggTrade JSONL 檔案的 1m feature 聚合範例。

此範例示範如何：
1. 掃描 data/ 底下的多個 `trades_*.jsonl`
2. 解析成 AggTradeRecord
3. 依 timestamp 全域排序（降低 out-of-order 影響）
4. 透過 aggTrade module 聚合成 1m feature DataFrame
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggTrade import InMemoryFeatureStore, TradeFeatureEngine, TradeMinuteAggregator, default_calculators
from aggTrade.adapters import JsonlTradeAdapter
from aggTrade.domain_types import AggTradeRecord


def _resolve_trade_ts_ms(trade: AggTradeRecord) -> int:
    """取出 trade 的分桶基準時間。

    Args:
        trade: 單筆 aggTrade record。

    Returns:
        依序採用 `trade_ts_ms -> tx_ts_ms -> event_ts_ms -> recv_ts_ms` 的毫秒時間。
    """
    if trade.trade_ts_ms > 0:
        return trade.trade_ts_ms
    if trade.header.tx_ts_ms is not None:
        return trade.header.tx_ts_ms
    if trade.header.event_ts_ms is not None:
        return trade.header.event_ts_ms
    return trade.header.recv_ts_ms


def build_1m_trade_feature_df_from_jsonl_dir(
    data_dir: str | Path,
    symbol: str = "BTCUSDT",
    max_rows: int = 200_000,
) -> pd.DataFrame:
    """從資料夾內多個 trade JSONL 產生 1m feature DataFrame。

    Args:
        data_dir: 存放 trade JSONL 的資料夾路徑。
        symbol: 要輸出的交易對。
        max_rows: store ring buffer 上限。

    Returns:
        指定 symbol 的 1m trade feature DataFrame（index 為 bar_open_datetime_tpe）。

    Raises:
        FileNotFoundError: 找不到任何 `trades_*.jsonl` 時拋出。
    """
    adapter = JsonlTradeAdapter(source="binance_futures")
    engine = TradeFeatureEngine(default_calculators(), feature_schema_version="trade_features.v1")
    aggregator = TradeMinuteAggregator(engine=engine, bar_interval="1m", feature_schema_version="trade_features.v1")
    store = InMemoryFeatureStore(max_rows=max_rows)

    root = Path(data_dir)
    file_paths = sorted(root.rglob("trades_*.jsonl"))
    if not file_paths:
        raise FileNotFoundError(f"No trade jsonl files found under: {root}")

    # 全域排序（示範用）。
    # 若資料量很大，可改成檔名順序 + 流式 ingest。
    records: list[tuple[int, int, str, AggTradeRecord]] = []
    seq = 0
    for path in file_paths:
        for trade in adapter.iter_file(path):
            if trade.header.symbol != symbol:
                continue
            ts_ms = _resolve_trade_ts_ms(trade)
            records.append((ts_ms, seq, str(path), trade))
            seq += 1

    records.sort(key=lambda x: (x[0], x[1]))

    for _, _, _, trade in records:
        rows = aggregator.ingest(trade)
        for row in rows:
            store.append_row(row)

    # 收尾，把目前仍開啟的分鐘輸出。
    for row in aggregator.flush(symbol=symbol):
        store.append_row(row)

    return store.get_df(symbol)


def main() -> None:
    """執行多檔 trade 聚合範例。"""
    data_dir = Path("data")
    symbol = "BTCUSDT"

    df = build_1m_trade_feature_df_from_jsonl_dir(data_dir=data_dir, symbol=symbol, max_rows=200_000)

    print(f"[INFO] symbol={symbol}, rows={len(df)}, cols={len(df.columns)}")
    if not df.empty:
        print("\n[INFO] tail(3):")
        print(df.tail(3))
        print("\n[INFO] latest row:")
        print(df.iloc[-1])

    df.to_csv(f"{symbol}_trade_1m_features.csv", index=True)


if __name__ == "__main__":
    main()
