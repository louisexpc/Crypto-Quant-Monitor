# ./Dockerfile

# --- Stage 1: Builder ---
# 使用一個完整的 Python 鏡像來編譯和安裝依賴
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

# 安裝構建工具
RUN apt-get update && apt-get install -y build-essential

# 複製依賴文件
COPY requirements.txt ./

# 創建一個虛擬環境並安裝依賴
# 這樣可以將依賴與系統 Python 隔離開來，並方便複製
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# --- Stage 2: Final Image ---
# 使用一個輕量級的基礎鏡像
FROM python:3.11-slim

WORKDIR /usr/src/app

# 從 builder 階段複製虛擬環境
COPY --from=builder /opt/venv /opt/venv

# 複製應用程式程式碼
COPY app/ ./app/
COPY .env ./
COPY train/ ./train/

# 將虛擬環境的路徑添加到 PATH，這樣可以直接執行
ENV PATH="/opt/venv/bin:$PATH"

# 設置時區為 UTC，對交易應用很重要
ENV TZ=UTC

# 執行主程式
CMD ["python", "-m", "app.main"]