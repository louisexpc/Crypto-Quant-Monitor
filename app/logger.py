# ./app/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime
import pytz

# --- 新增 TimezoneFormatter ---
class TimezoneFormatter(logging.Formatter):
    """自定義 Formatter，將日誌時間轉換為指定時區"""
    def __init__(self, fmt=None, datefmt=None, tz=pytz.timezone('Asia/Taipei')):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            t = dt.strftime('%Y-%m-%d %H:%M:%S')
            return f"{t},{int(dt.microsecond / 1000):03d}"
        
# 確保 logs 目錄存在
if not os.path.exists('logs'):
    os.makedirs('logs')

def setup_logger():
    """配置全局日誌記錄器"""
    logger = logging.getLogger("CQM")
    logger.setLevel(logging.DEBUG)

    # 避免重複添加 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(module)s.%(funcName)s: %(message)s'
    )

    # 控制台輸出 Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    # 文件輸出 Handler (帶輪轉)
    file_handler = RotatingFileHandler(
        'logs/app.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger

# 創建一個全局 logger 實例
log = setup_logger()

# 稍後我們會將 Discord handler 添加到這裡
discord_handler_instance = None

def add_discord_handler(handler):
    global discord_handler_instance
    if handler and not discord_handler_instance:
       log.info("Adding Discord handler to logger.")
       discord_handler_instance = handler
       discord_handler_instance.setLevel(logging.ERROR) # 只發送 ERROR 和 CRITICAL
       log.addHandler(discord_handler_instance)