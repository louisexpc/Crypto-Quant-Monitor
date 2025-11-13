import asyncio
import logging
import os
import signal
from pathlib import Path
from data_manager import DataManager
import dotenv
import yaml
import argparse
dotenv.load_dotenv()

class Trader:
    """
    所有模組 Entry Point
    - data manager
    - strategy manager
    - model predictor
    - order manager
    """

    def __init__(self, api_key:str, api_secret:str, config:dict):
        self.data_manager = DataManager(api_key, api_secret, config['data'])
        self.logger = logging.getLogger(self.__class__.__name__)

        self.pidfile = "./trader.pid"

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

def main():
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("API_KEY and API_SECRET must be set in environment variables")
    
    parser = argparse.ArgumentParser(description="Trader Application")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store log files")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config
    config = load_config(config_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_dir / "trader.log",
        filemode="w"
    )

    trader = Trader(api_key, api_secret, config)
    try:
        asyncio.run(trader.run_until_signal())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception("Unhandled exception in main: %s", e)





