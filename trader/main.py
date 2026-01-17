import asyncio
import logging
import os
import signal
from pathlib import Path
import dotenv
import yaml
import argparse
# Module imports
from utils.data_collector import ExchangeDataCollector, ExchangeConfig
from utils.trader import Trader
from indicators.feature_computer import FeatureComputer
from predictor.predictor import Predictor
from strategy.strategy import SNRLiveStrategy, SNRCfg
dotenv.load_dotenv()

class TradingBot:
    """
    所有模組 Entry Point
    - data manager
    - strategy manager
    - model predictor
    - order manager
    """

    def __init__(self, api_key:str, api_secret:str, config:dict):
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret

        self.logger = logging.getLogger(self.__class__.__name__)

        self.pidfile = "./trader.pid"

        self._init_all_modules()
        self.logger.info("All modules initialized successfully.")
    # -----------------------------
    # Initialization Methods
    # -----------------------------
    def _init_all_modules(self):
        """初始化所有模組"""
        self.data_collector = self._init_exchange_data_collector()
        self.feature_computer = self._init_feature_computer()
        self.long_predictor, self.short_predictor = self._init_long_short_predictor()
        self.strategy_manager = self._init_strategy_manager()
        self.trader = self._init_trader()

        """Strategy Manager Bootstrap"""
        self.lookback_bars = self.feature_cfg.get("feat_normalization", {}).get("rolling_window", 144) + self.predictor_cfg.get("seq_len", 144)
        self.barTimeframe = self.feature_cfg.get("time",{}).get("freq", "15m")
        self.symbol = self.config.get("symbol","BTCUSDT")
        self.logger.info(f"Initilized with symbol={self.symbol}, barTimeframe={self.barTimeframe}, lookback_bars={self.lookback_bars}")

        history_df = self.data_collector.fetch_ohlcv_fng(self.symbol, self.barTimeframe, self.lookback_bars)
        self.logger.debug(f"Fetched history_df with shape={history_df.shape}:\n{history_df.tail(5)}")
        self.strategy_manager.bootstrap_from_df(history_df=history_df)

    def _init_exchange_data_collector(self) -> ExchangeDataCollector:
        self.exchange_cfg = self.config.get('exchange', None)
        if not self.exchange_cfg:
            raise ValueError("Exchange configuration not found in config file.")
        
        self.exchange_cfg = ExchangeConfig(**self.exchange_cfg)
        return ExchangeDataCollector(self.api_key, self.api_secret, self.exchange_cfg)
    
    def _init_feature_computer(self) -> FeatureComputer:
        feature_cfg_path = Path(self.config['feature'].get('config_path', None))
        if not feature_cfg_path.exists():
            raise FileNotFoundError("Feature configuration path not found in config file.")
        self.feature_cfg = feature_cfg = load_config(feature_cfg_path)
        return FeatureComputer(feature_cfg)
    def _init_strategy_manager(self):
        strategy_cfg_path = Path(self.config['strategy'].get('config_path', None))
        if not strategy_cfg_path.exists():
            raise FileNotFoundError("Strategy configuration path not found in config file.")
        strategy_cfg = load_config(strategy_cfg_path)['SNRStrategy']
        self.strategy_cfg = SNRCfg(**strategy_cfg)
        return SNRLiveStrategy(self.strategy_cfg)

    def _init_long_short_predictor(self):
        predictor_cfg_path = Path(self.config['predictor'].get('config_path', None))
        if not predictor_cfg_path.exists():
            raise FileNotFoundError("Predictor configuration path not found in config file.")
        self.predictor_cfg = load_config(predictor_cfg_path)
        return Predictor(self.predictor_cfg, 'long'), Predictor(self.predictor_cfg, 'short')

    def _init_trader(self):
        trader_cfg = self.config.get('trade', {})
        return Trader(
            tradeConfig=trader_cfg,
            exchangeConfig=self.exchange_cfg,
            apiKey=self.api_key,
            apiSecret=self.api_secret,
        )

    def run(self):
        """
        假設執行時間(interval 1hr (in case)) 確保 Klines 已經更新完畢，則按照以下步驟執行：
        1. 從 exchange 抓取最新 Klines + FNG 數據
        2. 提供最新 candle 給 staregy manager，來確認是否產生進場訊號
        3. 若有進場訊號，根據 long/short 計算 feature
        4. 使用 long/short predictor 進行預測
        5. 根據預測結果，透過 trader module 下單
        6. 結束後將結果回傳至 discord/ mongoDB/log file
        """
        # TODO: implement the run logic


    def test_pipeline(self):
        df_15m  = self.data_collector.fetch_ohlcv("BTCUSDT", "15m", 400)
        # df_1hr  = self.data_collector.fetch_ohlcv("BTCUSDT", "1h", 200)
        fng = self.data_collector.fetch_FNG(5)
        merged_df = self.data_collector.merge_ohlcv_fng(df_15m, fng)

        if merged_df is not None:
            X_short = self.short_feat_engine.compute_features(merged_df)
            self.logger.info(f"Short-term feature matrix shape: {X_short.shape}, columns: {X_short.columns.tolist()}\n{X_short}")

            X_long = self.long_feat_engine.compute_features(merged_df)
            self.logger.info(f"Long-term feature matrix shape: {X_long.shape}, columns: {X_long.columns.tolist()}\n{X_long}")

    async def run_until_signal(self):
        """
        建立 PID 檔案，並等待終止訊號
        1. 建立 PID 檔案
        2. 等待 SIGINT / SIGTERM 訊號
        3. 收到訊號後，停止服務並刪除 PID 檔案
        4. 結束程式
        """
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        # create pidfile
        try:
            Path(self.pidfile).write_text(str(os.getpid()))
        except Exception:
            self.logger.warning("Cannot write pidfile %s (permissions?)", self.pidfile)

        def _on_signal():
            self.logger.info("Signal received, setting stop_event")
            stop_event.set()

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, _on_signal)
            except NotImplementedError:
                self.logger.warning("loop.add_signal_handler not supported on this platform.")

        await self.data_manager.start()
        self.logger.info("Trader started (pid=%d). Waiting for SIGINT/SIGTERM...", os.getpid())

        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("Stopping trader...")
            try:
                await self.data_manager.stop()
            except Exception:
                self.logger.exception("Error while stopping data manager")
            try:
                Path(self.pidfile).unlink()
            except Exception:
                self.logger.warning("Could not remove pidfile %s", self.pidfile)
            self.logger.info("Trader stopped")




def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_dir / "trader.log",
        filemode="w"
    )
def main():
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("API_KEY and API_SECRET must be set in environment variables")
    
    parser = argparse.ArgumentParser(description="Trader Application")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store log files")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    setup_logging(log_dir)

    config_path = args.config
    config = load_config(config_path)

    trader = TradingBot(api_key, api_secret, config)

if __name__ == "__main__":
    main()




