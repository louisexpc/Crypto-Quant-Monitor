# ./app/data_collector.py
from dataclasses import dataclass
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
        """獲取指定交易對和時間週期的 OHLCV 數據"""
        try:
            self.logger.info(f"Fetching {limit} klines for {symbol} on {timeframe} timeframe.")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol} on {timeframe}.")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Taipei')
            df.set_index('datetime', inplace=True)
            
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol} on {timeframe} in UTC+8.")
            return df
        except ccxt.NetworkError as e:
            self.logger.error(f"Network error while fetching {symbol}: {e}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error while fetching {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while fetching {symbol}: {e}", exc_info=True)
        
        return None
    def fetch_FNG(self, limit: int = 10)->pd.DataFrame:
        """
        API Docs: https://alternative.me/crypto/fear-and-greed-index/
        獲取 Fear & Greed Index 數據
        Args:
            - limit (int): 要獲取的數據點數量(FNG 指數通常每日更新)
        Returns:
            - pd.DataFrame: 包含 FNG 數據的 DataFrame
                - cols: ['fng','timestamp']
                - timestamp: Asia/Taipei datetime
                - fng: numeric

        """
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        j = requests.get(url, timeout=30).json()
        df = pd.DataFrame(j["data"])

        # 轉 numeric + 判斷秒/毫秒（簡化）
        ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts_num.isna().all():
            raise ValueError("No valid numeric timestamps returned from FNG API")

        max_val = ts_num.max()
        unit_for_to_datetime = "ms" if max_val > 1e11 else "s"

        df = df.assign(timestamp_num=ts_num).dropna(subset=["timestamp_num"]).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp_num"].astype("int64"), unit=unit_for_to_datetime, utc=True).dt.tz_convert('Asia/Taipei')

        df = df.sort_values("timestamp")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # rename "value" column to "fng"
        df = df.rename(columns={"value": "fng"})
        if limit > 0 and "time_until_update" in df.columns and "value_classification" in df.columns:
            df = df.drop(columns=["time_until_update", "value_classification"]).reset_index(drop=True)

        df = df.set_index("timestamp",drop=False)
        df = df.dropna(subset=["timestamp_num"],inplace=False)
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
    

