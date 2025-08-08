import numpy as np, pandas as pd

def make_price_label(df:pd.DataFrame, horizon:int=1, thr = 0.0, reg_type:str='logret'):
    """
    

    同時產生：
      - y_cls: 二分類標籤（Up=1/Down=0；若 thr>0，中間區間設 NaN 之後會被丟掉）
        (在df加一個col: r1「下一根收盤相對這一根是漲還是跌」=> log(下一根close - 當前close))
      
      - y_reg: 回歸目標（logret / pct / abs 三選一）
        (回傳close價差)

      - delta_abs_h{h}: 絕對價差
      - ret_pct_h{h}: 未來 1 小時的百分比回報
      - ret_log_h{h}: 對數報酬
      
    會自動去掉最後 horizon 根（避免偷看未來）。

    """
    fut_close = df['close'].shift(-horizon)
    ret_log  = np.log(fut_close / df['close'])
    ret_pct  = (fut_close / df['close']) - 1.0
    delta_abs = (fut_close - df['close'])

    # 1. 二分類: 漲跌
    if thr <= 0:
        y_cls = (ret_log > 0).astype('int8')
    else:
        y_cls = pd.Series(np.nan, index=df.index)
        y_cls[ret_log >  thr] = 1
        y_cls[ret_log < -thr] = 0

    # 2. 數值價差 (回歸標籤)
    if reg_type == 'abs':
        y_reg = delta_abs.astype('float32')
    elif reg_type == 'pct':
        y_reg = ret_pct.astype('float32')
    else:  # 'logret' (default)
        y_reg = ret_log.astype('float32')

    out = df.copy()
    out[f'delta_abs_h{horizon}'] = delta_abs.astype('float32')
    out[f'ret_pct_h{horizon}']   = ret_pct.astype('float32')
    out[f'ret_log_h{horizon}']   = ret_log.astype('float32')
    out['y_cls'] = y_cls
    out['y_reg'] = y_reg

    # 移除看未來的尾巴
    out = out.iloc[:-horizon]
    # 若有門檻，會留下中立區間為 NaN 的列，這裡丟掉
    if thr > 0:
        out = out.dropna(subset=['y_cls'])

    return out

# if __name__ == "__main__":
#     df = pd.read_csv("data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv",
#                      parse_dates=["datetime"],
#                      index_col="datetime")
#     if df.index.tz is None:
#         df.index = df.index.tz_localize("Asia/Taipei")
#     else:
#         df.index = df.index.tz_convert("Asia/Taipei")
#     df = df.sort_index()
#     df = df[~df.index.duplicated(keep="last")]  # 去重

#     df_lab = make_binary_label(df, thr=0.0)
#     print(df_lab.head(3)[["timestamp","open","high","low","close","volume","label"]])
#     print(df_lab.tail(3)[["close","label"]])

def monthly_pair_id(ts):  
    """
    兩個月為一組：Jan-Feb=0, Mar-Apr=1, ...
    e.g. 2025/01 → (2025*12+1-1)//2 → 同一組ID（Jan–Feb）
         2025/02 → 同一組ID（Jan–Feb）
    """
    m = ts.month
    y = ts.year
    return ((y * 12 + m - 1) // 2).astype(int)

def odd_month(ts):  # 奇數月
    return (ts.month % 2 == 1)

def build_folds(df, seq_len:int = 128, val_ratio:float = 0.2):
    """
    傳回 folds: 一組訓練月 => [(train_idx, val_idx, test_idx), ...]
    - train/val: 奇數月；同月內按時間 80/20 切
    - test: 下一個偶數月
    (內部再用seq_len當作當下資料的回看數量)
    """
    idx = df.index
    pair = monthly_pair_id(idx)
    is_odd = odd_month(idx)

    folds = []
    for pid in np.unique(pair):
        # 本組奇數月（train/val） & 下一個偶數月（test）
        tr_mask = (pair == pid) & is_odd
        test_mask = (pair == pid+1) & (~is_odd)

        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(test_mask)[0]

        if len(tr_idx) < seq_len or len(te_idx) < seq_len:
            continue

        cut = tr_idx[int(len(tr_idx)*(1-val_ratio))]
        train_idx = np.arange(tr_idx[0], cut)
        val_idx   = np.arange(cut, tr_idx[-1]+1)
        # 為了避免視窗越界，保證索引 >= seq_len-1
        valid = lambda a: a[a >= (seq_len-1)]
        folds.append((valid(train_idx), valid(val_idx), valid(te_idx)))
    return folds




def main(src:str, out:str, seq_len:int=36, thr:float =0.0, horizon:int=1, reg_type:str='logret'):
    from pathlib import Path
    df = pd.read_csv(src, parse_dates=["datetime"], index_col="datetime")

    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Taipei")
    else:
        df.index = df.index.tz_convert("Asia/Taipei")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]  # 去重

    df_lab = make_price_label(df, horizon=horizon, thr=thr, reg_type=reg_type)
    df_lab["month"] = df_lab.index.month
    df_lab["pair_id"] = ((df_lab.index.year * 12 + df_lab.index.month - 1) // 2)

    cols = ["timestamp","open","high","low","close","volume",
        "y_cls","y_reg", f"ret_log_h{horizon}", f"ret_pct_h{horizon}", f"delta_abs_h{horizon}"]
    cols = [c for c in cols if c in df_lab.columns]  # 避免某些欄位不存在

    print(df_lab.head(3)[cols])
    print(df_lab.tail(3)[["close","y_cls","y_reg"]])

    folds = build_folds(df_lab, seq_len=seq_len)
    print(f"#folds = {len(folds)}")


    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_lab.to_csv(out)
    pos_rate = float(df_lab["y_cls"].mean()) if df_lab["y_cls"].notna().any() else float('nan')
    print(f"saved: {out}  rows: {len(df_lab)}  pos_rate: {pos_rate:.3f}")


if __name__ == "__main__":
    src = r"data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv"
    out = r"data/ohlcv_labeled/btcusdt_1h_thr0_labeled.csv"   # 先用 csv

    main(src, out, seq_len=36, thr=0.0, horizon=1, reg_type='logret')

    


