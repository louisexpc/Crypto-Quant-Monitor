1. 嘗試 CNN(短期特徵: 15m) => Transformers(長期特徵: 1H)
    CNN: Adaptive Kernel Size: 近期波動度 決定有效感受野 ex: {3,7,15} softmax 模擬； 加殘差
    Transformers: 加入Multi-Scale Attention: 在自注意力前插入多尺度可學位置編碼
    融合: Late Fusion/Bilinear Fusion(分路logit相加) / Cross-Attention(以 1H 表徵為 query、15m 表徵為 key/value)

2. 嘗試minmax標準化 
3. Thresholded Accuracy (只保留模型輸出機率高於第 75 百分位)
4. EMA 作為學習目標 (最小化 MSE)
    MCC 做參考
    忽略模糊區（exclude ambiguity zone）? (-0.5% ~ +0.55不取)
    改Binary Cross Entropy?

5. finBert / BERTweet蒐集新聞
    [36 根 1H 價格 + 技術指標]
                    |
        ┌──────────▼────────────┐
        |     CNN/TCN          |   → 快通道
        └──────────┬────────────┘
                    |
    [每小時情緒向量]  ←  DeepSeek 摘要 + FinBERT encoding
                    |
        ┌──────────▼────────────┐
        |     LSTM / Transformer |  → 慢通道
        └──────────┬────────────┘
                    |
            Bilinear Fusion
                    ↓
            Multi-task Head
    → [三分類：漲 / 平 / 跌] + [幅度回歸]

6. Neutrosophic Sentiment Analysis
    Text → VADER 分數 → Neutrosophic 推理規則 → (T, I, F) 向量

    Price → 歷史價格序列 (OHLC)

    → [T, I, F, OHLC] → LSTM → 股價上漲/下跌預測
    I = 1 - |pos - neg| (e.g."It might go up" pos: 0.45, neg: 0.40 T = 0.45, I = 0.85, F = 0.40)


7. RSI / MACD / %K / %D / CCI / ATR / BB / CMF / OBV / ADL：

    RSI：相對強弱，衡量漲與跌的相對力度。

    MACD：兩條 EMA 的差（快慢線）及其信號線，抓趨勢變化。

    Stochastic（%K、%D）：收盤相對區間高低的位置（慣用超買超賣）。

    CCI：價格相對「典型價的均值」的偏離度。

    ATR：真實波幅的均值（衡量波動）。

    BB：移動均值 ± 2σ 的通道（波動帶）。

    CMF/OBV/ADL：把量價結合，近似衡量資金淨流入/流出。


8. trade資訊:
    (1) 價格與成交量統計（核心特徵）
    price_mean	均價（直接對價格平均）
    price_vwap	成交量加權平均價 sum(price × qty) / sum(qty)
    price_max,  price_min	價格區間最高/最低
    price_std	價格波動度
    price_range	最高 - 最低

    (2) 成交量與頻率特徵
    trade_count	            該分鐘有幾筆交易
    qty_sum, quote_qty_sum	該分鐘的總成交量、總成交額
    qty_mean	            平均單筆成交量
    qty_max, qty_std	    最大單筆、波動度
    qty_skew, qty_kurt	    偏度、峰度（可選進階特徵）

    (3) 買賣方向特徵（根據 is_buyer_maker）
    buy_qty, sell_qty	    買單/賣單總成交量
    buy_count, sell_count	買單/賣單筆數
    buy_ratio	            買單量佔比：buy_qty / (buy_qty + sell_qty)
    buy_count_ratio	        買單筆數佔比：buy_count / trade_count
    imbalance	            買賣量不均衡度 (buy_qty - sell_qty) / total_qty

    (4) 高級統計與推導特徵（可選
    volatility_proxy	例如 price_std / vwap，代表波動程度
    volume_spike	    突然放量（和過去幾分鐘平均比）
    price_jump_flag 	價格跳動是否超過門檻
    trend_slope	        該分鐘內的價格線性斜率
    aggressiveness	    平均每筆的 quote_qty（作為掛單壓力 proxy）

9. rule base 進場:
    獵取 第二根高(低)超過上一根，但body在中間，第三根上漲


