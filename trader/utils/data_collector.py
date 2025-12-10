# ./app/data_collector.py
import ccxt
import pandas as pd
import os
import requests
import logging
class ExchangeDataCollector:
    def __init__(self, exchange_config: dict):
        """
        通用數據採集器，根據傳入的配置初始化交易所。
        :param exchange_config: 包含 id 和 default_type 的字典。
        """
        self.exchange_id = exchange_config.get('id', 'binance')
        self.default_type = exchange_config.get('default_type', 'spot')
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': os.getenv(f"{self.exchange_id.upper()}_API_KEY"),
                'secret': os.getenv(f"{self.exchange_id.upper()}_SECRET_KEY"),
                'options': {
                    'defaultType': self.default_type,
                },
                'enableRateLimit': True,
            })
            self.logger.info(f"Successfully connected to {self.exchange_id.upper()} ({self.default_type} market).")
        except AttributeError:
            self.logger.critical(f"Exchange with ID '{self.exchange_id}' not found in ccxt.")
            raise
        except Exception as e:
            self.logger.critical(f"Failed to initialize {self.exchange_id.upper()} exchange: {e}", exc_info=True)
            raise

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
        df["timestamp"] = pd.to_datetime(df["timestamp_num"].astype("int64"), unit=unit_for_to_datetime, utc=True)

        df = df.sort_values("timestamp")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        if limit > 0 and "time_until_update" in df.columns and "value_classification" in df.columns:
            df = df.drop(columns=["time_until_update", "value_classification"]).reset_index(drop=True)

        df = df.set_index("timestamp").sort_index()

        # 重採樣 15min
        fng_15m = df[["value"]].resample("15min").ffill().rename(columns={"value": "sent_fng"})
        fng_15m["sent_fng_diff1"] = fng_15m["sent_fng"].diff()

        roll = 96 * 7  # 15min bars per day = 96 -> 96*7 = 672
        mean = fng_15m["sent_fng"].rolling(roll, min_periods=24).mean()
        std = fng_15m["sent_fng"].rolling(roll, min_periods=24).std()
        fng_15m["sent_fng_z7d"] = (fng_15m["sent_fng"] - mean) / (std + 1e-6)

        out = fng_15m.copy()

        # --- 這裡是關鍵：把 index 轉成跟 OHLCV 相同的時區（Asia/Taipei） ---
        out.index = out.index.tz_convert("Asia/Taipei")

        # 建立 datetime 欄位（直接用 index，且為 tz-aware）
        out["datetime"] = out.index  # 已經是 timezone-aware datetimes

        # 建立 unix timestamp（秒為單位）
        out["timestamp"] = (out.index.astype("int64") // 10**9).astype("int64")

        out = out[["datetime", "timestamp", "sent_fng", "sent_fng_diff1", "sent_fng_z7d"]]
        return out
    def merge_ohlcv_fng(self, ohlcv_df: pd.DataFrame, fng_df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
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
        def _ensure_tz(df):
            if df.index.tz is None:
                return df.tz_localize("Asia/Taipei")
            else:
                return df.tz_convert("Asia/Taipei")

        ohlcv = _ensure_tz(ohlcv)
        fng = _ensure_tz(fng)

        # 3) 移除 fng 中會與 ohlcv 衝突的欄位（通常不需要 'timestamp' / 'datetime'）
        #    我們保留 fng 的指標欄位 (fng, fng_diff1, fng_z7d)
        conflict_cols = [c for c in fng.columns if c in ohlcv.columns]
        if conflict_cols:
            # 只要把衝突欄位從 fng 移除（因為它們跟 index 重複或是語意相同）
            fng = fng.drop(columns=conflict_cols, errors="ignore")
        for col in ["datetime", "timestamp"]:
            if col in fng.columns:
                fng = fng.drop(columns=[col])

        # 4) 以 ohlcv 的 index 為基準 join（left join），把 fng 貼上
        merged = ohlcv.join(fng, how="left")

        # 5) 若需要，對 fng 欄位做 forward-fill
        if fill_method == "ffill":
            for col in ["fng", "fng_diff1", "fng_z7d"]:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(method="ffill")

        return merged


def main():
    exchange_config = {
        'id': 'binance',
        'default_type': 'swap'
    }
    collector = ExchangeDataCollector(exchange_config)
    ohlcv_df = collector.fetch_ohlcv("BTCUSDT", "15m", 100)
    ohlcv_df.to_csv("btc_usdt_15m.csv")

    fng = collector.fetch_FNG(10)
   
    fng.to_csv("fng_sample.csv", index=False)

    merged_df = collector.merge_ohlcv_fng(ohlcv_df, fng)
    print(merged_df.head(5))
    
    merged_df.to_csv("merged_ohlcv_fng.csv",index=False)
if __name__ == "__main__":
    main()