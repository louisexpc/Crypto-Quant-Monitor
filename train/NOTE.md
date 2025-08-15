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
