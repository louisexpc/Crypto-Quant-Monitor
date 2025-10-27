import asyncio
import logging
import os
import signal
from pathlib import Path
from listener import AsyncBinanceStoragePipeline
# 請 import 你的 AsyncBinanceStoragePipeline 類別
# from bsm_storage_pipeline import AsyncBinanceStoragePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PIDFILE = "./binance_storage.pid"  # 你可以改成適合權限的路徑，例如 /tmp/... for dev

async def run_until_signal(svc, pidfile: str = PIDFILE):
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    # create pidfile
    try:
        Path(pidfile).write_text(str(os.getpid()))
    except Exception:
        logging.warning("Cannot write pidfile %s (permissions?)", pidfile)

    def _on_signal():
        logging.info("Signal received, setting stop_event")
        stop_event.set()

    # register handlers (may raise NotImplementedError on Windows)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _on_signal)
        except NotImplementedError:
            logging.warning("loop.add_signal_handler not supported on this platform.")

    await svc.start()
    logging.info("Service started (pid=%d). Waiting for SIGINT/SIGTERM...", os.getpid())

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logging.info("Stopping service...")
        try:
            await svc.stop()
        except Exception:
            logging.exception("Error while stopping service")
        # remove pidfile
        try:
            Path(pidfile).unlink()
        except Exception:
            pass
        logging.info("Service stopped")

def main():
    import dotenv
    import os
    dotenv.load_dotenv()  # load from .env if exists
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    symbols = ["BTCUSDT"]
    symbols = ["btcusdt"]  # start small
    svc = AsyncBinanceStoragePipeline(
        api_key, api_secret, symbols,
        diff_batch_size=1000, diff_max_interval=1.0,
        trade_batch_size=500, trade_max_interval=2.0,
        snapshot_interval_sec=60,
        snapshot_top_k=None,  # None -> write full depth from DepthCache
        diff_log_path="diff_log.jsonl",
        snapshot_path_template="snapshot_{ts}.jsonl",
        snapshot_latest_path="snapshot_latest.jsonl",
        trades_path="trades.jsonl"
    )
    try:
        asyncio.run(run_until_signal(svc))
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt caught in main()")
    except Exception:
        logging.exception("Unhandled exception in main()")

if __name__ == "__main__":
    main()
