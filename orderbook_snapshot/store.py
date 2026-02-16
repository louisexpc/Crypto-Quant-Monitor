from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class FeatureStore(Protocol):
    """FeatureStore 行為契約。

    實作者需提供 append、查詢全量、查詢窗口與查詢最新列能力。
    """

    def append_row(self, row: dict[str, Any]) -> None:
        """追加或覆蓋一列特徵資料。

        Args:
            row: 由 aggregator 產生的單列 dict。
        """
        ...

    def get_df(self, symbol: str) -> pd.DataFrame:
        """取得某 symbol 的完整 DataFrame。

        Args:
            symbol: 交易標的。

        Returns:
            該 symbol 的資料表（通常以 bar_open_datetime_tpe 為 index）。
        """
        ...

    def get_window(self, symbol: str, lookback: int) -> pd.DataFrame:
        """取得某 symbol 的尾端窗口資料。

        Args:
            symbol: 交易標的。
            lookback: 回看列數。

        Returns:
            最多 `lookback` 列的 DataFrame。
        """
        ...

    def latest(self, symbol: str) -> pd.Series | None:
        """取得最新一列。

        Args:
            symbol: 交易標的。

        Returns:
            最新列 `pd.Series`；若不存在則回傳 `None`。
        """
        ...


class InMemoryFeatureStore:
    """以 in-memory DataFrame 實作的 FeatureStore。

    特性：
    - per-symbol 儲存
    - 以 `bar_open_timestamp_ms` 與 index 共同去重
    - 同 key 保留最後一筆（last-write-wins）
    - 支援 ring buffer（`max_rows`）
    """

    def __init__(self, max_rows: int = 10_000) -> None:
        """初始化 store。

        Args:
            max_rows: 每個 symbol 最多保留列數；小於等於 0 表示不裁切。
        """
        self.max_rows = max_rows
        self._frames: dict[str, pd.DataFrame] = {}

    def append_row(self, row: dict[str, Any]) -> None:
        """寫入單列資料，必要時覆蓋同分鐘舊列。

        Args:
            row: 單列特徵資料。
        """
        symbol = str(row["symbol"])

        row_df = pd.DataFrame([row])
        row_df["bar_open_datetime_tpe"] = pd.to_datetime(row_df["bar_open_datetime_tpe"], utc=True).dt.tz_convert(
            "Asia/Taipei"
        )
        row_df = row_df.set_index("bar_open_datetime_tpe", drop=True)
        row_df.index.name = "bar_open_datetime_tpe"

        current = self._frames.get(symbol)
        if current is None or current.empty:
            merged = row_df
        else:
            merged = pd.concat([current, row_df], axis=0, sort=False)

        if "bar_open_timestamp_ms" in merged.columns:
            merged = merged[~merged["bar_open_timestamp_ms"].duplicated(keep="last")]
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        if self.max_rows > 0 and len(merged) > self.max_rows:
            merged = merged.tail(self.max_rows)

        self._frames[symbol] = merged

    def get_df(self, symbol: str) -> pd.DataFrame:
        """回傳指定 symbol 的完整 DataFrame 複本。

        Args:
            symbol: 交易標的。

        Returns:
            DataFrame 複本；若 symbol 尚無資料則回傳空表。
        """
        df = self._frames.get(symbol)
        if df is None:
            return pd.DataFrame().rename_axis("bar_open_datetime_tpe")
        return df.copy()

    def get_window(self, symbol: str, lookback: int) -> pd.DataFrame:
        """回傳指定 symbol 的最後 `lookback` 列。

        Args:
            symbol: 交易標的。
            lookback: 回看列數。

        Returns:
            由舊到新排序的尾端窗口。
        """
        if lookback <= 0:
            return self.get_df(symbol).iloc[0:0]
        return self.get_df(symbol).tail(lookback)

    def latest(self, symbol: str) -> pd.Series | None:
        """回傳指定 symbol 的最新列。

        Args:
            symbol: 交易標的。

        Returns:
            最新列；若無資料回傳 `None`。
        """
        df = self._frames.get(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1].copy()
