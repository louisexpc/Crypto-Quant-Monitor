import os, math, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.build_features import Indicators
from utils.data_preprocess import make_price_label, monthly_pair_id

def set_seed(seed: int = 1337):
    import torch, random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_ohlcv_csv(path: str, time_col = 'datatime', tz = 'Asia/Taipei'):
    """
    讀取ohlcv，以date作為index
    """
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], utc = True).dt.tz_convert(tz)
    df = df.sort_values(time_col).set_index(time_col)
    required = {'open','high','low','close','volume'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV 需含欄位: {required}")
    return df

def build_features(df_ohlcv: pd.DataFrame, preset = 'fast36', prefix = 'f_'):
    ind = Indicators(df_ohlcv[['open','high','low','close','volume']])
    feat = ind.build(preset=preset, prefix=prefix)
    return feat 


class WindowIndexDataset(Dataset):
    """
    以「結尾索引 end_idx」取窗：X = X_all[end-L+1 : end+1]
    e.g. 取窗的方式: X_all[100-36+1 : 100+1] → 也就是「向前數 35 根 + 當前這根」共 36 根資料。
    """
    def __init__(self, X_all: np.ndarray, y_all: np.ndarray, end_indices: np.ndarray,
                 seq_len: int, mean: np.ndarray, std: np.ndarray, task: str):
        self.X_all = X_all      # [T, F]
        self.y_all = y_all      # [T]
        self.end_indices = end_indices.astype(int)
        self.seq_len = seq_len  # 一次看幾根
        self.mean = mean        
        self.std = std          # 訓練集的 Z-score 
        self.task = task        # classification 或 regression

    def __len__(self):
        """
        返回可取樣的數量：等於 end_indices 的長度，也就是這個 Dataset 有多少個結尾索引可用。
        """
        return len(self.end_indices)
    
    def __getitem__(self, i):
        end = self.end_indices[i]                   # 取出第 i 個樣本的結尾索引。
        sl = slice(end - self.seq_len+1, end +1)    # slice（切片物件），範圍是 end - L + 1 到 end
        X = self.X_all[sl]
        X = self (X - self.mean)/self.std           # 標準化
        y = self.X_all[end]                         # 對應的標籤是序列結尾那一根的標籤值
    
        if self.task == 'regression':
            y = np.float32(y)
        else:
            y = np.int64(y)
        return torch.from_numpy(X.astype(np.float32)), torch.tensor(y)


