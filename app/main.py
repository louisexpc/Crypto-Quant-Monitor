# ./app/main.py
import asyncio
import yaml
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pymongo import MongoClient

from .logger import log, add_discord_handler
from .data_collector import ExchangeDataCollector
from .analysis import StrategyAnalyzer
from .discord_bot import DiscordNotifier, DiscordLogHandler

# RF Model
from predictor import RFModelPredictor
RF_MODEL = None
# --- Globals and TZ setup are correct ---
config = {}
db_client = None
db = None
data_collector = None
discord_notifier = None
TZ = pytz.timezone('Asia/Taipei')

def load_config():
    """加載 YAML 配置文件"""
    global config
    with open('app/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    log.info("Configuration loaded.")

def connect_to_db():
    """連接到 MongoDB Atlas"""
    global db_client, db
    try:
        mongo_url = os.getenv("MONGO_DB_URL")
        if not mongo_url:
            raise ValueError("MONGO_DB_URL not found in .env file.")
        
        db_client = MongoClient(mongo_url)
        db_name = config['database']['db_name']
        db = db_client[db_name]
        db_client.admin.command('ping')
        log.info(f"Successfully connected to MongoDB Atlas. Database: '{db_name}'.")
    except Exception as e:
        log.critical(f"Failed to connect to MongoDB: {e}", exc_info=True)
        raise

async def check_and_process():
    """核心檢查任務，由排程器定時觸發"""
    log.info("Scheduler triggered. Checking for new candles...")
    
    # --- 關鍵修正：初始化一個列表來收集本輪所有的訊號 ---
    all_signals_in_run = []
    
    exchange_config = config.get('exchange', {})
    monitoring_list = config.get('monitoring', [])
    analysis_params = config.get('analysis_params', {})

    for item in monitoring_list:
        symbol = item['symbol']
        for timeframe in item['timeframes']:
            try:
                now_tz = datetime.now(TZ)
                cron_second = config.get('schedule', {}).get('cron_second', 5)
                if not is_candle_closed(now_tz, timeframe, cron_second):
                    continue

                log.info(f"New candle closed for {symbol} on {timeframe}. Starting process.")
                
                kline_limit = analysis_params.get(timeframe, analysis_params['default'])['klines_to_fetch']
                df = data_collector.fetch_ohlcv(symbol, timeframe, limit=kline_limit)
                
                if df is None or df.empty:
                    log.warning(f"No data fetched for {symbol}/{timeframe}. Skipping analysis.")
                    continue

                analyzer = StrategyAnalyzer(df, symbol, timeframe)
                signals = analyzer.analyze()

                # Extend RF Model Prediction
                if RF_MODEL and signals:
                    # Model Prediction Logic Here
                    # Batch Prediction
                    pass

                # --- 關鍵修正：將找到的訊號添加到總列表中，而不是立即處理 ---
                if signals:
                    all_signals_in_run.extend(signals)

            except Exception as e:
                log.error(f"Error processing {symbol}/{timeframe}: {e}", exc_info=True)

    # --- 關鍵修正：在所有分析都完成後，統一處理收集到的訊號 ---
    if all_signals_in_run:
        log.warning(f"Found a total of {len(all_signals_in_run)} signals in this run. Processing...")
        
        # 1. 一次性發送所有訊號到 Discord
        if discord_notifier:
            # 假設目前不對批次訊號進行單獨的 AI 預測
            await discord_notifier.send_signals_in_batch(all_signals_in_run, exchange_config.get('id', 'default'))
        
        # 2. 遍歷列表，將所有訊號存入資料庫
        if db is not None:
            collection_name = config['database']['collections']['signals']
            docs_to_insert = []
            for signal in all_signals_in_run:
                log.info(f"Saving signal for {signal['symbol']} to database.")
                docs_to_insert.append(
                    {**signal, "ai_prediction": None, "processed_at": datetime.now(TZ)}
                )
            
            if docs_to_insert:
                db[collection_name].insert_many(docs_to_insert)
                log.info(f"Successfully saved {len(docs_to_insert)} signals to database.")
    else:
        log.info("No new signals found in this run.")

def is_candle_closed(current_time: datetime, timeframe: str, cron_second: int) -> bool:
    """檢查指定時間週期的 K 線是否剛結束"""
    # 由於使用 cron, 我們檢查當前時間是否在 cron 觸發後的短時間內
    if not (cron_second <= current_time.second < cron_second + 15):
        return False

    timeframe_minutes = {'15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_min = timeframe_minutes.get(timeframe)
    
    # K線結束的判斷：當前分鐘數是該時間週期的整數倍
    if tf_min:
        return current_time.minute % tf_min == 0
    return False

async def main():
    """主函數：初始化並啟動服務"""
    load_dotenv()
    load_config()
    
    global data_collector, discord_notifier, RF_MODEL
    
    try:
        # Initialize RF Model Predictor
        rf_config = config.get('rf_model', {})
        RF_MODEL = RFModelPredictor(config=rf_config)
        log.info("RF Model Predictor initialized.")
        # Initialize Discord Notifier
        discord_notifier = DiscordNotifier()
        await discord_notifier.connect()
        log.info("Discord Notifier connected.")
        discord_log_handler = DiscordLogHandler(discord_notifier)
        add_discord_handler(discord_log_handler)
        # Connect to Database
        connect_to_db()
        # Initialize Data Collector
        exchange_config = config.get('exchange', {})
        if not exchange_config:
            raise ValueError("Exchange configuration is missing in config.yaml")
        data_collector = ExchangeDataCollector(exchange_config)
        log.info("Data Collector initialized.")
        # Setup Scheduler
        scheduler = AsyncIOScheduler(timezone='Asia/Taipei')
        cron_second = config.get('schedule', {}).get('cron_second', 5)
        # 使用 misfire_grace_time 確保即使系統短暫繁忙，錯過的任務也能被執行
        scheduler.add_job(check_and_process, 'cron', second=cron_second, misfire_grace_time=15)
        scheduler.start()
        log.info(f"Scheduler started in UTC+8. Will run checks every minute at second {cron_second}.")
        
        while True:
            await asyncio.sleep(3600)

    except Exception as e:
        log.critical(f"Application failed to start: {e}", exc_info=True)
        if discord_notifier:
            await discord_notifier.send_error(f"🚨 **Application Startup Failed!**\n**Error:** {e}")
    finally:
        if discord_notifier:
            await discord_notifier.close()
        if db_client:
            db_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Application shutting down...")