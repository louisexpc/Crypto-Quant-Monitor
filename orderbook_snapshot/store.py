from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class FeatureStore(Protocol):
    def append_row(self, row: dict[str, Any]) -> None:
        ...

    def get_df(self, symbol: str) -> pd.DataFrame:
        ...

    def get_window(self, symbol: str, lookback: int) -> pd.DataFrame:
        ...

    def latest(self, symbol: str) -> pd.Series | None:
        ...


class InMemoryFeatureStore:
    def __init__(self, max_rows: int = 10_000) -> None:
        self.max_rows = max_rows
        self._frames: dict[str, pd.DataFrame] = {}

    def append_row(self, row: dict[str, Any]) -> None:
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
        df = self._frames.get(symbol)
        if df is None:
            return pd.DataFrame().rename_axis("bar_open_datetime_tpe")
        return df.copy()

    def get_window(self, symbol: str, lookback: int) -> pd.DataFrame:
        if lookback <= 0:
            return self.get_df(symbol).iloc[0:0]
        return self.get_df(symbol).tail(lookback)

    def latest(self, symbol: str) -> pd.Series | None:
        df = self._frames.get(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1].copy()
