# ./app/data_collector.py
from dataclasses import dataclass
import re
import ccxt
import pandas as pd
import os
import requests
import logging
from typing import Literal

import yaml
@dataclass(frozen=True)
class ExchangeConfig:
    id: str = 'binance'
    market_type: Literal['spot', 'future'] = 'future'
    sandbox: bool = True                  # true = testnet / false = live
    enable_rate_limit: bool = True
    timeout_ms: int = 30000
    recv_window: int = 10000
    adjust_for_time_difference: bool = True
    urls_override: dict | None = None     # e.g., {'api': 'https://testnet.binance.vision/api'}


class ExchangeDataCollector:
    def __init__(self,api_key, api_secret, exchange_config: ExchangeConfig):
        """
        通用數據採集器，根據傳入的配置初始化交易所。
        :param exchange_config: ExchangeConfig 實例。
        """
        self.tradeConfig = exchange_config
        self.apiKey = api_key
        self.apiSecret = api_secret

        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange = self._init_exchange()

    # -------------------------
    # Exchange initialization
    # -------------------------
    def _init_exchange(self):
        """
        - 建立 ccxt exchange instance 並套用常用參數（rate limit/timeout/recvWindow/options/defaultType）。
        - 若 exchange.sandbox=true，會呼叫 set_sandbox_mode(True)，並允許用 urls_override 覆寫 base api endpoint。

        Returns :
          - exchange: ccxt exchange instance（例如 ccxt.binance(...)）

        Hint : 
          - hint 2: defaultType 會根據 market_type = spot/future 設定。
          - hint 3: Spot testnet 常用 urls_override.api = "https://testnet.binance.vision/api"
        """

        options = {
            "defaultType": "spot" if self.tradeConfig.market_type == "spot" else "future",
            "adjustForTimeDifference": self.tradeConfig.adjust_for_time_difference,
        }

        exchange_class = getattr(ccxt, self.tradeConfig.id)
        exchange = exchange_class({
            "apiKey": self.apiKey,
            "secret": self.apiSecret,
            "enableRateLimit": self.tradeConfig.enable_rate_limit,
            "timeout": self.tradeConfig.timeout_ms,
            "recvWindow": self.tradeConfig.recv_window,
            "options": options,
        })

        if self.tradeConfig.sandbox:
            exchange.set_sandbox_mode(True)

            urls_override = self.tradeConfig.urls_override or {}
            if "api" in urls_override:
                exchange.urls = exchange.urls or {}
                exchange.urls["api"] = urls_override["api"]
                self.logger.info(f"Override exchange.urls['api'] => {urls_override['api']}")

        self.logger.info(
            f"Exchange initialized: id={self.tradeConfig.id}, market_type={self.tradeConfig.market_type}, sandbox={self.tradeConfig.sandbox}"
        )
        return exchange
    # -------------------------
    # Time helpers (DTO normalization)
    # -------------------------
    _TZ = "Asia/Taipei"

    @staticmethod
    def _timeframe_to_ms(timeframe: Literal['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']) -> int:
        """Convert ccxt timeframe string (e.g. '15m', '1h') into milliseconds."""
        tf = str(timeframe).strip()
        m = re.fullmatch(r"(\d+)([smhdw])", tf)
        if not m:
            raise ValueError(f"Unsupported timeframe format: {timeframe!r} (expected like '15m', '1h', '1d')")
        n = int(m.group(1))
        unit = m.group(2)
        mult = {
            "s": 1000,
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
        }[unit]
        return n * mult

    @classmethod
    def _to_taipei_dt(cls, ts_ms: pd.Series) -> pd.Series:
        """UTC ms -> tz-aware datetime (Asia/Taipei)."""
        return pd.to_datetime(ts_ms.astype("int64"), unit="ms", utc=True).dt.tz_convert(cls._TZ)
    # -------------------------
    # Data Fetching API
    # -------------------------
    def fetch_ohlcv_fng(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """
        同時獲取指定交易對和時間週期的 OHLCV 數據與 Fear & Greed Index 數據，並合併成一個 DataFrame。
        Args:
            - symbol (str): 交易對符號，例如 "BTCUSDT"
            - timeframe (str): K 線時間週期，例如 "15m"
            - limit (int): 要獲取的 K 線數量
        Returns:
            - pd.DataFrame | None: 合併後的 DataFrame，若失敗則回傳 None
                - timestamp (ms): open 時間戳記
                - datetime (Asia/Taipei): open 時間, index
                - open, high, low, close, volume: OHLCV 資料
                - fng: Fear & Greed Index 數值
        """
        ohlcv_df = self.fetch_ohlcv(symbol, timeframe, limit)
        if ohlcv_df is None:
            self.logger.error(f"Failed to fetch OHLCV data for {symbol} on {timeframe}.")
            return None
        range_days = (ohlcv_df.index[-1] - ohlcv_df.index[0]).days + 1
        fng_limit = max(10, range_days)  # 至少抓 10 筆 FNG，以涵蓋整個 OHLCV 範圍
        fng_df = self.fetch_FNG(limit=fng_limit)  # FNG 指數通常每日更新，這裡取最近 fng_limit 筆
        if fng_df is None:
            self.logger.error("Failed to fetch FNG data.")
            return None

        merged_df = self._merge_ohlcv_fng(ohlcv_df, fng_df, fill_method="ffill")
        return merged_df
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """獲取指定交易對和時間週期的 OHLCV 數據（DTO schema normalized）

        回傳 DataFrame（index 改為 **Kline open datetime (Asia/Taipei)**）：

        - index: datetime (Asia/Taipei) = kline_open_datetime
        - timestamp (ms): kline_open_timestamp_ms（作為 open ts）
        - kline_open_timestamp_ms: Kline 開盤 timestamp (ms)
        - kline_open_datetime: Kline 開盤 datetime (Asia/Taipei)
        - kline_close_timestamp_ms: Kline 收盤 timestamp (ms)
        - kline_close_datetime: Kline 收盤 datetime (Asia/Taipei)
        - open, high, low, close, volume: OHLCV

        Note:
        - ccxt.fetch_ohlcv 回傳的第一欄 timestamp 為「Kline open time」(ms)；此處會另外計算 close time。
        """
        try:
            self.logger.info(f"Fetching {limit} klines for {symbol} on {timeframe} timeframe.")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol} on {timeframe}.")
                return None

            # ccxt: [open_time_ms, open, high, low, close, volume]
            df = pd.DataFrame(ohlcv, columns=["open_time_ms", "open", "high", "low", "close", "volume"])

            # --- DTO fields ---
            df["kline_open_timestamp_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("Int64")
            if df["kline_open_timestamp_ms"].isna().any():
                raise ValueError("OHLCV contains invalid open_time_ms")

            tf_ms = self._timeframe_to_ms(timeframe)
            # Binance close time is (Open + Interval - 1ms)
            df["kline_close_timestamp_ms"] = (df["kline_open_timestamp_ms"].astype("int64") + tf_ms - 1).astype("int64")

            df["kline_open_datetime"] = self._to_taipei_dt(df["kline_open_timestamp_ms"].astype("int64"))
            df["kline_close_datetime"] = self._to_taipei_dt(df["kline_close_timestamp_ms"])

            # timestamp: use kline open time (ms)
            df["timestamp"] = df["kline_open_timestamp_ms"]

            # index = open datetime (Asia/Taipei)
            df = df.set_index("kline_open_datetime", drop=False)
            df.index = df.index.rename("datetime")

            # cleanup / stable column order (optional)
            df = df.drop(columns=["open_time_ms"])
            ordered = [
                "timestamp",
                "kline_open_timestamp_ms",
                "kline_open_datetime",
                "kline_close_timestamp_ms",
                "kline_close_datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            df = df[[c for c in ordered if c in df.columns]]

            self.logger.info(f"Successfully fetched {len(df)} records for {symbol} on {timeframe} in UTC+8.")
            return df

        except ccxt.NetworkError as e:
            self.logger.error(f"Network error while fetching {symbol}: {e}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error while fetching {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while fetching {symbol}: {e}", exc_info=True)

        return None
    def fetch_FNG(self, limit: int = 10) -> pd.DataFrame:
        """
        API Docs: https://alternative.me/crypto/fear-and-greed-index/
        獲取 Fear & Greed Index（FNG）數據（DTO schema normalized）

        回傳 DataFrame（index 與 timestamp 欄位皆為 **FNG datetime (Asia/Taipei)**）：

        - index: timestamp (Asia/Taipei)
        - timestamp: 同 index（Asia/Taipei datetime）
        - fng: numeric
        - fng_timestamp_ms: 將 API 的 timestamp 統一正規化為 ms（int64）
        - fng_value_classification: (optional) API 回傳的分類字串（若存在）
        - fng_time_until_update: (optional) API 回傳的下次更新倒數（若存在）
        """
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        j = requests.get(url, timeout=30).json()
        df = pd.DataFrame(j["data"])

        # timestamp: API 可能回傳秒或毫秒（此處統一轉成 ms）
        ts_num = pd.to_numeric(df.get("timestamp"), errors="coerce")
        if ts_num.isna().all():
            raise ValueError("No valid numeric timestamps returned from FNG API")

        max_val = ts_num.max()
        is_ms = bool(max_val > 1e11)
        ts_ms = ts_num.astype("int64") if is_ms else (ts_num.astype("int64") * 1000)

        df["fng_timestamp_ms"] = ts_ms
        df["timestamp"] = pd.to_datetime(df["fng_timestamp_ms"], unit="ms", utc=True).dt.tz_convert(self._TZ)

        # value -> fng
        df["fng"] = pd.to_numeric(df.get("value"), errors="coerce")

        # keep optional metadata if present (rename for clarity)
        if "value_classification" in df.columns:
            df = df.rename(columns={"value_classification": "fng_value_classification"})
        if "time_until_update" in df.columns:
            df = df.rename(columns={"time_until_update": "fng_time_until_update"})

        df = df.sort_values("timestamp")
        df = df.set_index("timestamp", drop=False)

        return df
    def _merge_ohlcv_fng(
        self,
        ohlcv_df: pd.DataFrame,
        fng_df: pd.DataFrame,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        將 OHLCV DataFrame 與 FNG DataFrame 合併，並處理同名欄位衝突與時區對齊。

        參數:
            - ohlcv_df: 以 datetime index 的 OHLCV（建議已為 Asia/Taipei）
            - fng_df: 以 datetime index 的 FNG（建議已為 Asia/Taipei）
            - fill_method: 若合併後 FNG 欄位有 NaN，使用何種 fill（"ffill"/None）

        回傳:
            - merged DataFrame（以 ohlcv_df 的 index 為基準）
        """
        # 1) 確保 copy（避免修改原參數）
        ohlcv = ohlcv_df.copy()
        fng = fng_df.copy()

        # 2) 確保兩邊 index 為 tz-aware 且同一時區 (Asia/Taipei)
        def _ensure_tz(df: pd.DataFrame) -> pd.DataFrame:
            if df.index.tz is None:
                return df.tz_localize("Asia/Taipei")
            return df.tz_convert("Asia/Taipei")

        ohlcv = _ensure_tz(ohlcv)
        fng = _ensure_tz(fng)

        # 3) 移除 fng 中會與 ohlcv 衝突的欄位（通常不需要 'timestamp' / 'datetime'）
        conflict_cols = [c for c in fng.columns if c in ohlcv.columns]
        if conflict_cols:
            fng = fng.drop(columns=conflict_cols, errors="ignore")
        for col in ["datetime", "timestamp"]:
            if col in fng.columns:
                fng = fng.drop(columns=[col])

        # 4) 僅保留你需要的 FNG 欄位（可自行擴充）
        keep_cols = [c for c in ["fng"] if c in fng.columns]
        fng = fng[keep_cols]

        # 5) 關鍵：把 fng 的 index 併入時間軸，再 ffill，最後切回 ohlcv index
        #    這樣 ohlcv 即使沒有對應到 fng 的 timestamp，也能吃到「上一筆已知值」
        fng = fng.sort_index()
        fng = fng[~fng.index.duplicated(keep="last")]  # 避免同一 timestamp 重複造成不確定性

        union_idx = ohlcv.index.union(fng.index).sort_values()
        fng_aligned = fng.reindex(union_idx)

        if fill_method == "ffill":
            fng_aligned = fng_aligned.ffill()

        # 切回 ohlcv 時間點：每根 K 線都會拿到當下(或最近過去)的 fng 值
        fng_aligned = fng_aligned.reindex(ohlcv.index)

        # 6) join 回去
        merged = ohlcv.join(fng_aligned, how="left")
        return merged

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test():
    # 測試用例
    config = ExchangeConfig()
    collector = ExchangeDataCollector(
        api_key=os.getenv("API_KEY", ""),
        api_secret=os.getenv("API_SECRET", ""),
        exchange_config=config
    )
    ohlcv_fng_df = collector.fetch_ohlcv_fng(symbol='BTC/USDT', timeframe='15m', limit=100)
    print(ohlcv_fng_df.head())
    ohlcv_fng_df.to_csv("ohlcv_fng_output.csv", index=False)
if __name__ == "__main__":
    test()

