#!/bin/bash

# 啟動
nohup python3 main.py > out.log 2>&1 &
echo $! > ./binance_storage.pid

# 停止
# kill $(cat ./binance_storage.pid)        # send SIGTERM -> your process should handle and stop gracefully
# # 若沒反應，強制
# kill -9 $(cat ./binance_storage.pid)
