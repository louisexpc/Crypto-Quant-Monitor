import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas_ta import Strategy

# ====== 1) 產生特徵（pandas_ta）＋ 標籤（上/平/下） ======
def build_features_and_label(df: pd.DataFrame,
                             horizon=1,               # 預測 t+1 小時
                             flat_band_bps=10,        # 平盤區間寬度（bps）
                             ):
    """
    df: index=Datetime, columns=[open, high, low, close, volume]
    flat_band_bps: 例如 10 表示 ±0.10% 內視為平
    """

    df = df.copy()
    assert set(['open','high','low','close','volume']).issubset(df.columns)
    df = df.sort_index()

    # -- 指標 --
    df.ta.strategy("all")   


    # -- 建標籤：上(2) / 平(1) / 下(0)
    # 用對數報酬避免比例偏差
    ret = np.log(df['close'].shift(-horizon)/df['close'])
    y = pd.Series((ret > 0).astype(int), index=df.index, name="label")    
    # band = flat_band_bps / 10000.0    # bps → ratio
    # """
    # ret >= band  → label = 2  （例如：看漲／上漲）
    # ret <= -band → label = 0  （例如：看跌／下跌）
    # else         → label = 1  （例如：盤整／持平）
    # """
    # y = pd.Series(np.where(ret >= band, 2,
    #                        np.where(ret<= - band, 0, 1)),
    #                        index=df.index, name='label')

    nan_ratio = df.isna().mean().sort_values(ascending=False)
    nan_ratio = nan_ratio[nan_ratio>0].head(10)
    df = df.drop(columns=nan_ratio.index.to_list()) # 把 top10 超過 30% nan的 drop
    df = df.fillna(0)   # 剩下的nan不到 1% 補0
    
    X = df    
    Xy = pd.concat([X, y], axis=1)
    feature_cols = [c for c in Xy.columns if c != 'label']

    return Xy[feature_cols], Xy['label']    # 相當於回傳X, y


if __name__ == "__main__":

    csv_path = r"data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv"
    df = pd.read_csv(csv_path,
                     parse_dates=['datetime'],
                     index_col="datetime")
    
    X, y = build_features_and_label(df)
    print(X.head())
    print(y.value_counts(normalize=True))

    X["label"] = y
    X.to_csv("indicators.csv")



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
    
#     df = pd.read_csv(r"indicators_drop_top10_nan.csv")
#     nan_ratio = df.isna().mean().sort_values(ascending=False)
#     nan_ratio = nan_ratio[nan_ratio>0].head(30)

#     plt.figure(figsize=(12, 6))
#     nan_ratio.plot(kind='bar', color='orange')
#     plt.title("Top 30 NaN ratio")
#     plt.ylabel("NaN ratio(0~1)")
#     plt.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig("output_nan_ratio.png", dpi=300)