# Project Structure
```
crypto_quant_monitor/
├── app/
│   ├── __init__.py
│   ├── analysis.py           # 核心交易邏輯分析模組
│   ├── config.yaml           # 應用程式核心設定檔
│   ├── data_collector.py     # 數據採集模組
│   ├── discord_bot.py        # Discord 通知模組
│   ├── inference.py          # AI 模型推論服務
│   ├── logger.py             # 日誌設定模組
│   ├── main.py               # 應用程式主進入點與排程器
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py     # PyTorch LSTM 模型定義
│       └── (此處放置預訓練模型, e.g., btc_usdt_1h_model.pt)
│
├── train/
│   ├── __init__.py
│   ├── train.py              # 獨立的模型訓練腳本
│   └── utils.py              # 訓練時所需的輔助工具 (數據預處理等)
│
├── logs/                     # 日誌文件存放目錄 (由 logger.py 自動創建)
│   └── app.log
│
├── .env                      # 環境變數檔案 (敏感資訊)
├── .gitignore                # Git 忽略檔案清單
├── Dockerfile                # Docker 鏡像建構檔案
├── docker-compose.yml        # Docker Compose 設定檔 (方便本地開發與部署)
├── README.md                 # 專案說明文件
└── requirements.txt          # Python 依賴套件
```