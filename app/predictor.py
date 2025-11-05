import os, json, traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import talib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import torch
from torch.utils.data import Dataset, DataLoader
from joblib import dump, load

from train import data
class MultiTimeframeOHLCVDataset(Dataset):
    """
    多時間框架OHLCV數據集（與你原本一致的介面）
    """
    def __init__(self,
                 data_dict: Dict[str, np.ndarray],
                 labels: np.ndarray,
                 timeframe_windows: Dict[str, int],
                 normalize: bool = True):
        self.data_dict = data_dict
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.timeframe_windows = timeframe_windows
        self.timeframes = list(data_dict.keys())
        self.num_samples = len(labels)
        if normalize:
            self.normalize_data()

    def normalize_data(self):
        for timeframe in self.timeframes:
            data = self.data_dict[timeframe]
            mean = np.mean(data, axis=(0, 1), keepdims=True)
            std = np.std(data, axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1, std)
            self.data_dict[timeframe] = (data - mean) / std

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        batch_data = {}
        for timeframe in self.timeframes:
            seq_len = self.timeframe_windows[timeframe]
            data_slice = self.data_dict[timeframe][idx, -seq_len:, :]
            batch_data[timeframe] = torch.tensor(data_slice, dtype=torch.float32)
        label = self.labels[idx]
        return batch_data, label


def create_dataloader(dataset: MultiTimeframeOHLCVDataset,
                      batch_size: int = 32,
                      shuffle: bool = False,
                      num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )


# ========== DataLoader → sklearn 特徵 ==========
def dataloader_to_arrays(dataloader: DataLoader,
                         feature_mode: str = "stat"
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    將 MultiTimeframeOHLCVDataset 的批次資料轉成 sklearn 需要的 2D 特徵。
    - "flatten": 將 [B, T, F] 攤平為 [B, T*F]，多個 timeframe 串接
    - "stat":    沿時間軸做 mean/std/min/max/last，降維且較易解讀
    """
    X_list, y_list = [], []
    for batch_data, batch_labels in dataloader:
        feats_per_tf = []
        for tf, x in batch_data.items():
            x_np = x.numpy()  # [B, T, F]
            if feature_mode == "flatten":
                B, T, F = x_np.shape
                feats = x_np.reshape(B, T * F)
            elif feature_mode == "stat":
                mean = x_np.mean(axis=1)
                std = x_np.std(axis=1)
                mn = x_np.min(axis=1)
                mx = x_np.max(axis=1)
                last = x_np[:, -1, :]
                feats = np.concatenate([mean, std, mn, mx, last], axis=1)
            else:
                raise ValueError("feature_mode must be 'flatten' or 'stat'")
            feats_per_tf.append(feats)
        X_batch = np.concatenate(feats_per_tf, axis=1)
        y_batch = batch_labels.numpy()
        X_list.append(X_batch)
        y_list.append(y_batch)
    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0,))
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=int)
    return X, y

class RFModelPredictor:
    def __init__(self, 
                 cfg: Optional[Dict] = None):

        if cfg is None:
            raise ValueError("Configuration dictionary is required for RFModelPredictor.")
        
        for key in ['path','lookback']:
            if key not in cfg:
                raise ValueError(f"Missing required config key: '{key}'")
        self.cfg = cfg
        self.model_path = cfg['path']
        self.lookback = cfg['lookback']
        self.model = self.load_model()
        self.features = cfg.get('features', ['open', 'high', 'low', 'close', 'volume'])

    def load_model(self)-> RandomForestClassifier:
        """Load the RF model from disk."""
        try:
            model = load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
        return model
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features based on config feature list"""
        candidates = ['open', 'high', 'low', 'close', 'volume']
        candidates_with_range = ['ema','rsi','atr','cci','mfi', 'adx','willr'] # indicators needing price range: 以 format: {name}_{period} 呈現
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', and 'volume' columns.")
        for feature_name in self.features:
            if feature_name in candidates:
                continue  # 基本欄位不需計算
            elif feature_name.startswith('ema_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.EMA(df['close'], timeperiod=period)
            elif feature_name.startswith('rsi_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.RSI(df['close'], timeperiod=period)
            elif feature_name.startswith('atr_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            elif feature_name.startswith('cci_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
            elif feature_name.startswith('mfi_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)
            elif feature_name.startswith('adx_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
            elif feature_name.startswith('willr_'):
                period = int(feature_name.split('_')[1])
                df[feature_name] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
            else:
                raise ValueError(f"Unsupported feature: {feature_name}")
            
        df = df.drop(labels=['timestamp','open', 'high', 'low', 'close', 'volume'], axis=1)
        df.fillna(0, inplace=True)

        return df
    def data_pipeline(self,df_1h:pd.DataFrame, df_4h:pd.DataFrame, signals:List[Dict]) -> np.ndarray:
        """Perform batch predictions for all signals using the RF model."""
        signals_df = pd.DataFrame(signals)
        data_dict = {}

        df_1h = self.compute_features(df_1h)
        df_4h = self.compute_features(df_4h)


        # 將 index 改回時間索引便於切片
        df_1h = df_1h.copy()
        df_4h = df_4h.copy()
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            raise ValueError("df_1h index must be DatetimeIndex")
        if not isinstance(df_4h.index, pd.DatetimeIndex):
            raise ValueError("df_4h index must be DatetimeIndex")
        x_list = []
        for _, row in signals_df.iterrows():
            timestamp = pd.to_datetime(row['t0'])

            # make sure timestamp
            # 確保 df_4h.index 有時區
            if df_4h.index.tz is None:
                df_4h.index = df_4h.index.tz_localize("Asia/Taipei")

            if df_1h.index.tz is None:
                df_1h.index = df_1h.index.tz_localize("Asia/Taipei")

            # 確保 timestamp 有時區（若來源沒加 tz）
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("Asia/Taipei")


            # 1h
            bars_1h = self.lookback  # 100
            # 取 last bars_1h 根 K 線
            data_1h = df_1h.loc[df_1h.index < timestamp].tail(bars_1h)
            if len(data_1h) < 100:

                continue

            # 4h
            # 假設 df_4h 是連續 4h K 線
            bars_4h = int(self.lookback // 4)   # 25
            # 取 last bars_4h 根 K 線
            data_4h = df_4h[df_4h.index < timestamp].tail(bars_4h)
            if len(data_4h) < int(self.lookback / 4):
                continue
            

            if '1h' not in data_dict:
                data_dict['1h'] = []
            if '4h' not in data_dict:
                data_dict['4h'] = []

            # data_1h.shape = (lookback, features) => flatten (lookback * features)
            # data_4h.shape = (lookback/4, features) => flatten (lookback/4 * features)
            # combined_features = np.concatenate([data_1h.values.flatten(), data_4h.values.flatten()]) => shape = (features * (lookback + lookback/4),)
            flatten_1h = data_1h.values.flatten()
            flatten_4h = data_4h.values.flatten()
            combined_features = np.concatenate([flatten_1h, flatten_4h])
            x_list.append(combined_features)

        # Turn X_list into numpy array
        X = np.array(x_list) #shape = (num_samples, features * (lookback + lookback/4))
        print(f"[INFO] 特徵矩陣形狀: {X.shape}")
        
        
        return X
    def predict(self, df_1h:pd.DataFrame, df_4h:pd.DataFrame, signals:List[Dict]) -> List[int]:
        """Predict classes for the given signals."""
        if not signals:
            return []
        
        X = self.data_pipeline(df_1h, df_4h, signals)
        if X.size == 0:
            return []
        
        predictions = self.model.predict(X)
        return predictions.tolist()

if __name__ == "__main__":
    # Example usage
    cfg = {
        'path': 'app/models/default_model.pt',
        'lookback': 100,
        'features': ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'atr_14']
    }
    predictor = RFModelPredictor(cfg=cfg)
    # df_1h and df_4h should be provided as pd.DataFrame
    # signals should be provided as List[Dict]
    # predictions = predictor.predict(df_1h, df_4h, signals)