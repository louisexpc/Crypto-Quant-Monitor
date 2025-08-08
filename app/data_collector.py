# ./app/data_collector.py
import ccxt
import pandas as pd
import os
from .logger import log

class ExchangeDataCollector:
    def __init__(self, exchange_config: dict):
        """
        通用數據採集器，根據傳入的配置初始化交易所。
        :param exchange_config: 包含 id 和 default_type 的字典。
        """
        self.exchange_id = exchange_config.get('id', 'binance')
        self.default_type = exchange_config.get('default_type', 'spot')
        
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
            log.info(f"Successfully connected to {self.exchange_id.upper()} ({self.default_type} market).")
        except AttributeError:
            log.critical(f"Exchange with ID '{self.exchange_id}' not found in ccxt.")
            raise
        except Exception as e:
            log.critical(f"Failed to initialize {self.exchange_id.upper()} exchange: {e}", exc_info=True)
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """獲取指定交易對和時間週期的 OHLCV 數據"""
        try:
            log.info(f"Fetching {limit} klines for {symbol} on {timeframe} timeframe.")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                log.warning(f"No data returned for {symbol} on {timeframe}.")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Taipei')
            df.set_index('datetime', inplace=True)
            
            log.info(f"Successfully fetched {len(df)} records for {symbol} on {timeframe} in UTC+8.")
            return df
        except ccxt.NetworkError as e:
            log.error(f"Network error while fetching {symbol}: {e}")
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error while fetching {symbol}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred while fetching {symbol}: {e}", exc_info=True)
        
        return None
