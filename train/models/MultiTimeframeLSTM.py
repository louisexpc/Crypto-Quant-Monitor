import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
import talib
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
import json
import os
from datetime import datetime
import seaborn as sns
from train_utils.feature_selectors import feature_selector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTimeframeEncoder(nn.Module):
    """
    單一時間框架的編碼器
    將OHLCV序列資料編碼成embedding vector
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, embed_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1):
        super(MultiTimeframeEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal encoding with LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Final embedding layer
        self.embedding_layer = nn.Linear(hidden_dim * 2, embed_dim)  # *2 for bidirectional
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] - OHLCV sequence
        Returns:
            embedding: [batch_size, embed_dim] - encoded representation
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction
        features = self.feature_net(x)  # [batch_size, seq_len, hidden_dim]
        
        # Temporal encoding
        lstm_out, _ = self.temporal_lstm(features)  # [batch_size, seq_len, hidden_dim*2]
        
        # Global pooling (mean over sequence)
        pooled = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # Final embedding
        embedding = self.embedding_layer(pooled)  # [batch_size, embed_dim]
        
        return embedding

class MultiHeadCrossAttention(nn.Module):
    """
    多頭注意力機制，用於聚合不同時間框架的embeddings
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of [batch_size, embed_dim] tensors from different timeframes
        Returns:
            aggregated: [batch_size, embed_dim] - aggregated representation
        """
        # Stack embeddings to create sequence
        # [batch_size, num_timeframes, embed_dim]
        stacked_embeddings = torch.stack(embeddings, dim=1)
        
        # Self-attention across timeframes
        attn_output, _ = self.multihead_attn(
            query=stacked_embeddings,
            key=stacked_embeddings,
            value=stacked_embeddings
        )
        
        # Residual connection and normalization
        attended = self.norm(stacked_embeddings + self.dropout(attn_output))
        
        # Global pooling across timeframes
        aggregated = torch.mean(attended, dim=1)  # [batch_size, embed_dim]
        
        return aggregated

class MultiTimeframeLSTMClassifier(nn.Module):
    """
    主要模型：Multi-Encoder + Multi-Head Attention + LSTM
    """
    def __init__(self, 
                 timeframe_configs: Dict[str, dict],
                 lstm_hidden_dim: int = 256,
                 lstm_num_layers: int = 2,
                 num_classes: int = 3,  # 買入/持有/賣出
                 dropout: float = 0.1,
                 attention_heads: int = 8):
        super(MultiTimeframeLSTMClassifier, self).__init__()
        
        self.timeframes = list(timeframe_configs.keys())
        self.embed_dim = timeframe_configs[self.timeframes[0]]['embed_dim']
        
        # 為每個時間框架創建encoder
        self.encoders = nn.ModuleDict()
        for timeframe, config in timeframe_configs.items():
            self.encoders[timeframe] = MultiTimeframeEncoder(**config)
        
        # Multi-head attention for aggregation
        self.attention = MultiHeadCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, num_classes)
        )
        
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch_data: Dict with keys as timeframes, values as [batch_size, seq_len, 5] tensors
        Returns:
            logits: [batch_size, num_classes] - classification logits
        """
        batch_size = batch_data[self.timeframes[0]].shape[0]
        
        # Encode each timeframe
        embeddings = []
        for timeframe in self.timeframes:
            if timeframe in batch_data:
                encoded = self.encoders[timeframe](batch_data[timeframe])
                embeddings.append(encoded)
        
        # Aggregate embeddings using attention
        aggregated = self.attention(embeddings)  # [batch_size, embed_dim]
        
        # Expand for LSTM (single time step)
        lstm_input = aggregated.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)  # [batch_size, 1, lstm_hidden_dim]
        final_hidden = lstm_out.squeeze(1)  # [batch_size, lstm_hidden_dim]
        
        # Classification
        logits = self.classifier(final_hidden)  # [batch_size, num_classes]
        
        return logits

class MultiTimeframeOHLCVDataset(Dataset):
    """
    多時間框架OHLCV數據集
    """
    def __init__(self, 
                 data_dict: Dict[str, np.ndarray], 
                 labels: np.ndarray,
                 timeframe_windows: Dict[str, int],
                 normalize: bool = True):
        """
        Args:
            data_dict: {'1h': [num_samples, max_seq_len, 5], '4h': [num_samples, max_seq_len, 5]}
            labels: [num_samples] - classification labels (0: sell, 1: hold, 2: buy)
            timeframe_windows: {'1h': 100, '4h': 25} - sequence lengths for each timeframe
            normalize: whether to normalize OHLCV data
        """
        self.data_dict = data_dict
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.timeframe_windows = timeframe_windows
        self.timeframes = list(data_dict.keys())
        self.num_samples = len(labels)
        
        # Normalize data if required
        if normalize:
            self.normalize_data()
    
    def normalize_data(self):
        """Z-score normalization for each timeframe"""
        for timeframe in self.timeframes:
            data = self.data_dict[timeframe]
            # Calculate mean and std across all samples and time steps
            mean = np.mean(data, axis=(0, 1), keepdims=True)
            std = np.std(data, axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            self.data_dict[timeframe] = (data - mean) / std
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            batch_data: Dict with timeframe keys and tensor values [seq_len, 5]
            > example: {'1h': tensor([[...]]), '4h': tensor([[...]])}
            label: scalar tensor
        """
        batch_data = {}
        
        for timeframe in self.timeframes:
            seq_len = self.timeframe_windows[timeframe]
            # Extract the required sequence length
            data_slice = self.data_dict[timeframe][idx, -seq_len:, :]
            batch_data[timeframe] = torch.tensor(data_slice, dtype=torch.float32)
        
        label = self.labels[idx]
        
        return batch_data, label

def create_dataloader(dataset: MultiTimeframeOHLCVDataset, 
                     batch_size: int = 32,
                     shuffle: bool = False,
                     num_workers: int = 4) -> DataLoader:
    """
    創建DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

def _range_ewm_vol(df: pd.DataFrame, halflife: int = 20) -> pd.Series:
    """
    OHL-range 的指數加權移動平均（EWM）波動率。
    Formula: range = (high - low) / open

    Args:
        - df:pd.DataFrame, 必須包含 'open','high','low' 欄位
        - halflife:int, 指數加權平滑的半衰期

    Returns: 
        - 相對波動率:pd.Series, 經過指數加權平滑處理，波動率序列，與 df 同索引，向前移動一格
    """
    if not all(col in df.columns for col in ['open', 'high', 'low']):
        raise ValueError("DataFrame must contain 'open', 'high', and 'low' columns.")
    rng = (df['high'] - df['low']) / df['open'].replace(0.0, np.nan)
    return rng.ewm(halflife=halflife, adjust=False).mean().shift(1)


def compute_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    計算技術指標特徵
    Args:
        data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    Returns:
        data: DataFrame with additional feature columns
    """
    data = df.copy()
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', and 'volume' columns.")
    data['return'] = data['close'].pct_change().shift(-1)

    #Add Features
    data[f"obv"] = talib.OBV(data['close'], data['volume'])
    data[f"volatility"] = _range_ewm_vol(data, halflife=20)
    
    data[f"momentum"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)

    for n in [5,10,20]:
        data[f"ema_{n}"] = talib.EMA(data['close'], timeperiod=n)
        data[f"rsi_{n}"] = talib.RSI(data['close'], timeperiod=n)
        data[f"atr_{n}"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"cci_{n}"] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"mfi_{n}"] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=n)
        data[f"adx_{n}"] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"willr_{n}"] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=n)
    data.fillna(0, inplace=True)
    return data

def compute_features(df:pd.DataFrame,alpha_dir:str = None) -> pd.DataFrame:
    """
    計算技術指標特徵
    Args:
        data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    Returns:
        data: DataFrame with additional feature columns
    """
    data = df.copy()
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', and 'volume' columns.")
    data['returns'] = data['close'].pct_change().shift(-1)

    #Add Features
    data[f"obv"] = talib.OBV(data['close'], data['volume'])
    data[f"volatility"] = _range_ewm_vol(data, halflife=20)
    
    data[f"momentum"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)

    for n in [5, 10, 20]:
        data[f"ema_{n}"] = talib.EMA(data['close'], timeperiod=n)
        data[f"rsi_{n}"] = talib.RSI(data['close'], timeperiod=n)
        data[f"atr_{n}"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"cci_{n}"] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"mfi_{n}"] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=n)
        data[f"adx_{n}"] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"willr_{n}"] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=n)
    if alpha_dir is not None:
        # Load and compute custom alphas
        from train_utils.random_alpha_generator.random_alpha_generator import load_alpha

        alpha_files = [f for f in os.listdir(alpha_dir) if f.endswith('.json')]
        print(f"[INFO] Loading {len(alpha_files)} custom alphas from {alpha_dir}")
        for file in alpha_files:
            try:
                alpha = load_alpha(os.path.join(alpha_dir, file))
                feature_name = f"alpha_{os.path.splitext(file)[0]}"
                data[feature_name] = alpha.tree.eval(data)
            except Exception as e:
                print(f"[ERROR] Failed to load/compute alpha from {file}: {e}")
                traceback.print_exc()
    data = data.drop(labels=['timestamp','open', 'high', 'low', 'close', 'volume'], axis=1)
    data.fillna(0, inplace=True)
    print(f"[INFO] 計算特徵完成，總共 {data.shape[1]} 個特徵欄位。\n包含特徵: {data.columns.tolist()}")
    print(f"[INFO] 開始特徵選擇...")
    fs = feature_selector(data, target='returns')
    fs.train_model(p_threshold=0.035)
    selected_features = fs.get_selected_features()
    print(f"[INFO] 選擇後的特徵數量: {len(selected_features)}")
    data = fs.transform(data)

    return data
def purged_split(labels:pd.DataFrame, max_timeframe_lookback_window:int, test_date: str, embargo_hour:int = 4):
    labels = labels.set_index(pd.to_datetime(labels['t0']))

    limit_date = pd.to_datetime(test_date).tz_localize("Asia/Taipei")  - pd.Timedelta(hours=max_timeframe_lookback_window*4+embargo_hour)

    labels_train_val = labels[labels.index < limit_date]
    labels_test = labels[labels.index >= limit_date]

    val_start_date = limit_date - pd.Timedelta(days = 60) 
    val_limit_date = val_start_date - pd.Timedelta(hours=max_timeframe_lookback_window*4+embargo_hour)
    labels_train = labels_train_val[labels_train_val.index < val_limit_date]
    labels_val = labels_train_val[labels_train_val.index >= val_start_date]
    print(f"Train: {labels_train.shape}, Val: {labels_val.shape}, Test: {labels_test.shape}")
    assert labels_train.index.max() < labels_val.index.min()-pd.Timedelta(hours=max_timeframe_lookback_window*4+embargo_hour), "Training and validation sets overlap!"
    assert labels_val.index.max() < labels_test.index.min()-pd.Timedelta(hours=max_timeframe_lookback_window*4+embargo_hour), "Validation and test sets overlap!"
    return labels_train, labels_val, labels_test

def data_pipeline(label:pd.DataFrame,df_1h:pd.DataFrame, df_4h:pd.DataFrame,use_padding : bool = False) -> DataLoader:
    """
    數據處理管線: 由於label 是事件，需要從事件時間點往前取K線數據
    1. 讀取1小時和4小時K線數據
    2. 計算技術指標特徵
    3. 根據label的timestamp往前取K線數據
    4. 返回多時間框架的OHLCV數據字典
    Args:
        label: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    Returns:
        data_dict: {'1h': np.ndarray, '4h': np.ndarray}, 每個key對應的值是形狀為 (num_samples, num_timesteps, num_features) 的numpy陣列
        labels: np.ndarray, 形狀為 (num_samples,) 的標籤陣列
    """
    # try:
    #     df_1h = pd.read_csv('../data/1h_klines.csv',index_col=['datetime'],parse_dates=['datetime'])
    #     df_4h = pd.read_csv('../data/4h_klines.csv',index_col=['datetime'],parse_dates=['datetime'])

    # except FileNotFoundError:
    #     raise FileNotFoundError("Kline data files not found. Please ensure 'data/1h_klines.csv' and 'data/4h_klines.csv' exist.")
    
    # try:
    #     df_1h = compute_features(df_1h)
    #     df_4h = compute_features(df_4h)
    # except Exception as e:
    #     raise ValueError(f"Error computing features: {e}")
    print(f"[INFO] 開始數據處理管線, padding : {use_padding}")
    data_dict = {}
    valid_labels = []
    if 'label' not in label.columns:
        raise ValueError("Label DataFrame must contain a 'label' column.")
    for _,row in label.iterrows():
        timestamp = pd.to_datetime(row['t0'])
        # 取1小時K線
        start_time_1h = timestamp - pd.Timedelta(hours=100)  # 假設取100個1小時K線
        mask_1h = (df_1h.index >= start_time_1h) & (df_1h.index < timestamp)
        data_1h = df_1h.loc[mask_1h]
        # print(f"Sampling start_time_1h: {data_1h.index[0]}, end_time_1h: {data_1h.index[-1]}, 1h data shape: {data_1h.shape}, timestamp: {timestamp}")
        if data_1h.index[-1] >= timestamp:
            raise ValueError(f"Data sampling error: end_time_1h {data_1h.index[-1]} should be before timestamp {timestamp}")

        # print(f"Timestamp: {timestamp}, 1h data shape: {data_1h.shape}")
        if len(data_1h) < 100:
            if use_padding :
                # padding
                padding = pd.DataFrame(np.zeros((100 - len(data_1h), data_1h.shape[1])), columns=data_1h.columns)
                data_1h = pd.concat([padding, data_1h], ignore_index=True)
                # print(f"Warning: Not enough 1h data before {timestamp}. Expected at least 100, got {len(data_1h)}")
            else:
                continue
        # 取4小時K線
        start_time_4h = timestamp - pd.Timedelta(hours=25*4)  # 假設取25個4小時K線
        mask_4h = (df_4h.index >= start_time_4h) & (df_4h.index < timestamp)
        data_4h = df_4h.loc[mask_4h]
        # print(f"Sampling start_time_4h: {data_4h.index[0]}, end_time_4h: {data_4h.index[-1]}, 4h data shape: {data_4h.shape}, timestamp: {timestamp}")
        if data_4h.index[-1] >= timestamp:
            raise ValueError(f"Data sampling error: end_time_4h {data_4h.index[-1]} should be before timestamp {timestamp}")
        
        # print(f"Timestamp: {timestamp}, 4h data shape: {data_4h.shape}")
        if  len(data_4h) < 25:
            if use_padding :
                # padding
                padding = pd.DataFrame(np.zeros((25 - len(data_4h), data_4h.shape[1])), columns=data_4h.columns)
                data_4h = pd.concat([padding, data_4h], ignore_index=True)
                # print(f"Warning: Not enough 4h data before {timestamp}. Expected at least 25, got {len(data_4h)}")
            else:
                continue
        if '1h' not in data_dict:
            data_dict['1h'] = []
        if '4h' not in data_dict:
            data_dict['4h'] = []
        
        data_dict['1h'].append(data_1h.values)
        data_dict['4h'].append(data_4h.values)

        valid_labels.append(row['label'])  # 假設label欄位名稱為'label' 
    # 將list轉為np.ndarray
    data_dict['1h'] = np.array(data_dict['1h'])
    data_dict['4h'] = np.array(data_dict['4h'])


    valid_labels = np.array(valid_labels)

    dataset = MultiTimeframeOHLCVDataset(
        data_dict=data_dict, 
        labels=valid_labels,
        timeframe_windows={'1h': 100, '4h': 25},
        normalize=True
    )
    dataloader = create_dataloader(dataset, batch_size=32, shuffle=False)
    return dataloader, len(dataset)


class TrainingManager:
    """訓練管理器，處理完整的訓練、驗證、測試和可視化流程"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', save_dir='./results'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        # 創建保存目錄
        # 檢查並創建必要的導入
        try:
            import numpy as np
            self.np = np
        except ImportError:
            raise ImportError("NumPy is required for TrainingManager")
        
        try:
            from sklearn.metrics import precision_recall_fscore_support
            self.precision_recall_fscore_support = precision_recall_fscore_support
        except ImportError:
            print("Warning: scikit-learn not found, some metrics may be unavailable")
        os.makedirs(save_dir, exist_ok=True)
        
        # 設置matplotlib樣式
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8')
        
    def plot_training_curves(self, save_path=None):
        """繪製訓練和驗證的損失/準確率曲線"""
        if not self.history['train_loss']:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 損失曲線
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 準確率曲線
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'training_curves_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['sp','tp'], save_path=None):
        """繪製混淆矩陣（數量和百分比版本）"""
        if class_names is None:
            class_names = ['Sell', 'Hold', 'Buy']
        
        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # 計算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始數量混淆矩陣
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar_kws={'shrink': .8})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # 百分比混淆矩陣
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax2, cbar_kws={'shrink': .8})
        ax2.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'confusion_matrix_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        
        return fig, cm
    
    def plot_classification_report(self, y_true, y_pred, class_names=['sp','tp'], save_path=None):
        """繪製分類報告的可視化版本 - 修復版本"""
        if class_names is None:
            class_names = ['Sell', 'Hold', 'Buy']
        
        # 生成分類報告 - 添加zero_division參數
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱狀圖：精確率、召回率、F1分數
        x = np.arange(len(class_names))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
        ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('Classes', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # 餅圖：支持樣本數
        ax2.pie(support, labels=class_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存圖表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'classification_report_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification report saved to: {save_path}")
        
        return fig
    
    def plot_loss_comparison(self, save_path=None):
        """繪製訓練損失與驗證損失的詳細比較"""
        if not self.history['train_loss']:
            print("No training history available!")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 主要曲線
        train_line = ax.plot(epochs, self.history['train_loss'], 'b-', 
                           label='Training Loss', linewidth=2.5, alpha=0.8)
        val_line = ax.plot(epochs, self.history['val_loss'], 'r-', 
                         label='Validation Loss', linewidth=2.5, alpha=0.8)
        
        # 填充區域顯示差異
        ax.fill_between(epochs, self.history['train_loss'], self.history['val_loss'], 
                       alpha=0.2, color='gray', label='Gap')
        
        # 標記最佳點
        min_val_loss_idx = np.argmin(self.history['val_loss'])
        best_epoch = min_val_loss_idx + 1
        best_val_loss = self.history['val_loss'][min_val_loss_idx]
        
        ax.plot(best_epoch, best_val_loss, 'ro', markersize=8, 
               label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
        
        ax.set_title('Training vs Validation Loss Detailed Comparison', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加統計信息
        final_gap = abs(self.history['train_loss'][-1] - self.history['val_loss'][-1])
        ax.text(0.02, 0.98, f'Final Gap: {final_gap:.4f}', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存圖表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'loss_comparison_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss comparison saved to: {save_path}")
        
        return fig
    
    def save_training_results(self, y_true=None, y_pred=None, test_acc=None, save_prefix=""):
        """保存所有訓練結果和可視化 - 完全修復版本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.save_dir, f'training_results_{save_prefix}{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Saving all results to: {results_dir}")
        
        # 輔助函數：確保數據為JSON可序列化類型 - 修復版本
        def make_serializable(obj):
            """將numpy類型轉換為Python原生類型 - 完全修復版本"""
            if obj is None:
                return None
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, (float, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                if obj.size == 0:
                    return []
                elif obj.size == 1:
                    # 單元素數組，提取標量值
                    return make_serializable(obj.flat[0])
                else:
                    # 多元素數組，轉換為列表
                    return obj.tolist()
            elif hasattr(obj, 'item') and not hasattr(obj, '__len__'):
                # numpy標量
                try:
                    return obj.item()
                except (ValueError, TypeError):
                    return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            else:
                # 嘗試其他轉換方法
                if hasattr(obj, 'tolist'):
                    try:
                        return obj.tolist()
                    except (ValueError, TypeError):
                        pass
                
                # 最後嘗試直接返回或轉為字符串
                try:
                    # 檢查是否是基本類型
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # 1. 保存訓練歷史數據
        try:
            history_path = os.path.join(results_dir, 'training_history.json')
            history_clean = make_serializable(self.history)
            
            with open(history_path, 'w') as f:
                json.dump(history_clean, f, indent=2)
            print(f"Training history saved to: {history_path}")
        except Exception as e:
            print(f"Error saving training history: {e}")
            traceback.print_exc()
        
        # 2. 保存訓練曲線
        try:
            curves_path = os.path.join(results_dir, 'training_curves.png')
            self.plot_training_curves(save_path=curves_path)
        except Exception as e:
            print(f"Error saving training curves: {e}")
        
        # 3. 保存損失比較圖
        try:
            loss_comp_path = os.path.join(results_dir, 'loss_comparison.png')
            self.plot_loss_comparison(save_path=loss_comp_path)
        except Exception as e:
            print(f"Error saving loss comparison: {e}")
        
        # 4. 如果有測試結果，保存相關圖表
        if y_true is not None and y_pred is not None:
            try:
                # 確保輸入是numpy數組或列表
                y_true_array = np.asarray(y_true)
                y_pred_array = np.asarray(y_pred)
                
                # 混淆矩陣
                cm_path = os.path.join(results_dir, 'confusion_matrix.png')
                _, cm = self.plot_confusion_matrix(y_true_array, y_pred_array, save_path=cm_path)
                
                # 分類報告
                report_path = os.path.join(results_dir, 'classification_report.png')
                self.plot_classification_report(y_true_array, y_pred_array, save_path=report_path)
                
                # 計算額外統計指標 - 修復precision警告
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true_array, y_pred_array, average=None, zero_division=0
                )
                
                # 計算宏平均和加權平均（也添加zero_division參數）
                macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                    y_true_array, y_pred_array, average='macro', zero_division=0
                )
                
                weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                    y_true_array, y_pred_array, average='weighted', zero_division=0
                )
                
                # 保存數值結果
                results_summary = {
                    # 基本測試結果
                    'test_accuracy': make_serializable(test_acc),
                    'confusion_matrix': make_serializable(cm),
                    
                    # 訓練過程結果
                    'final_train_loss': make_serializable(
                        self.history['train_loss'][-1] if self.history['train_loss'] else None
                    ),
                    'final_val_loss': make_serializable(
                        self.history['val_loss'][-1] if self.history['val_loss'] else None
                    ),
                    'final_train_acc': make_serializable(
                        self.history['train_acc'][-1] if self.history['train_acc'] else None
                    ),
                    'final_val_acc': make_serializable(
                        self.history['val_acc'][-1] if self.history['val_acc'] else None
                    ),
                    
                    # 最佳結果
                    'best_val_acc': make_serializable(
                        max(self.history['val_acc']) if self.history['val_acc'] else None
                    ),
                    'best_epoch': make_serializable(
                        np.argmax(self.history['val_acc']) + 1 if self.history['val_acc'] else None
                    ),
                    'total_epochs': len(self.history['train_loss']) if self.history['train_loss'] else 0,
                    
                    # 詳細分類指標
                    'class_metrics': {
                        'precision': make_serializable(precision),
                        'recall': make_serializable(recall),
                        'f1_score': make_serializable(f1),
                        'support': make_serializable(support)
                    },
                    
                    # 平均指標
                    'macro_avg': {
                        'precision': make_serializable(macro_precision),
                        'recall': make_serializable(macro_recall),
                        'f1_score': make_serializable(macro_f1)
                    },
                    
                    # 加權平均指標
                    'weighted_avg': {
                        'precision': make_serializable(weighted_precision),
                        'recall': make_serializable(weighted_recall),
                        'f1_score': make_serializable(weighted_f1)
                    },
                    
                    # 類別統計
                    'class_statistics': {
                        'total_samples': int(len(y_true_array)),
                        'class_distribution': make_serializable(
                            np.bincount(y_true_array, minlength=3)
                        ),
                        'prediction_distribution': make_serializable(
                            np.bincount(y_pred_array, minlength=3)
                        )
                    },
                    
                    # 元數據
                    'timestamp': timestamp,
                    'class_names': ['sp','tp'],
                    'model_info': {
                        'device': str(self.device),
                        'total_params': sum(p.numel() for p in self.model.parameters()),
                        'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    }
                }
                
                # 測試JSON序列化
                try:
                    json.dumps(results_summary)
                    
                except Exception as json_err:
                    print(f"JSON序列化測試失敗: {json_err}")
                    # 嘗試逐項檢查
                    for key, value in results_summary.items():
                        try:
                            json.dumps({key: value})
                        except Exception as item_err:
                            print(f"項目 '{key}' 序列化失敗: {item_err}")
                            results_summary[key] = str(value)
                
                summary_path = os.path.join(results_dir, 'results_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(results_summary, f, indent=2)
                print(f"Results summary saved to: {summary_path}")
                
            except Exception as e:
                print(f"Error saving test results: {e}")
                traceback.print_exc()
        
        print(f"All results saved successfully!")
        return results_dir
    def load_training_results(self, results_dir):
        """載入之前保存的訓練結果"""
        try:
            # 載入訓練歷史
            history_path = os.path.join(results_dir, 'training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.history = json.load(f)
                print(f"Training history loaded from: {history_path}")
            
            # 載入結果摘要
            summary_path = os.path.join(results_dir, 'results_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                print(f"Results summary loaded from: {summary_path}")
                return summary
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    def train_epoch(self, dataloader, optimizer, criterion):
        """單個epoch的訓練"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in tqdm(dataloader, desc="Training"):
            # 移動數據到設備
            batch_labels = batch_labels.to(self.device)
            for timeframe in batch_data:
                batch_data[timeframe] = batch_data[timeframe].to(self.device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 統計
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader, criterion):
        """單個epoch的驗證"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader, desc="Validation"):
                batch_labels = batch_labels.to(self.device)
                for timeframe in batch_data:
                    batch_data[timeframe] = batch_data[timeframe].to(self.device)
                
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=1e-5,weight=None, early_stop_patience:int = 20):
        """完整訓練流程"""
        if weight is not None:
            print(f"[INFO] Using class weights: {weight}")
            weight = torch.tensor(weight, dtype=torch.float).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_acc = 0
        early_stop_counter = 0
        early_stop_patience = early_stop_patience
        
        print(f"[INFO] 開始訓練，共 {num_epochs} epochs...")
        print(f"[INFO] 模型參數總數: {sum(p.numel() for p in self.model.parameters())}")
        print(f"[INFO] 可訓練參數數量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"[INFO] 設備: {self.device}")
        print(f"[INFO] 初始學習率: {lr}, 權重衰減: {weight_decay}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 訓練階段
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # 驗證階段
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # 學習率調整
            scheduler.step(val_loss)
            
            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 打印結果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停機制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"新的最佳驗證準確率: {best_val_acc:.2f}%")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= early_stop_patience:
                print(f"早停觸發，在第 {epoch+1} epoch 停止訓練")
                break
        
        print(f"\n訓練完成！最佳驗證準確率: {best_val_acc:.2f}%")
        
    def test(self, test_loader):
        """測試階段，生成混淆矩陣"""
        # 載入最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
                batch_labels = batch_labels.to(self.device)
                for timeframe in batch_data:
                    batch_data[timeframe] = batch_data[timeframe].to(self.device)
                
                outputs = self.model(batch_data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # 計算測試準確率
        test_acc = accuracy_score(all_labels, all_predictions)
        print(f"測試準確率: {test_acc:.4f}")
        
        # 生成混淆矩陣
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 分類報告
        class_names = ['fail', 'success']
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, digits=4)
        
        return cm, all_labels, all_predictions, test_acc, report
def immediate_diagnosis(trainer):
    history = trainer.history
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    print(f"最新Train Loss: {train_loss[-1]:.4f}")
    print(f"最新Val Loss: {val_loss[-1]:.4f}")
    print(f"Loss差距: {val_loss[-1] - train_loss[-1]:.4f}")
    
    if len(train_loss) >= 5:
        recent_train_trend = train_loss[-1] - train_loss[-5]
        recent_val_trend = val_loss[-1] - val_loss[-5]
        
        print(f"近5epoch Train趨勢: {recent_train_trend:.4f}")
        print(f"近5epoch Val趨勢: {recent_val_trend:.4f}")
        
        if recent_train_trend < -0.1 and recent_val_trend > 0.05:
            return "SEVERE_OVERFITTING"
        elif recent_train_trend > 0 and recent_val_trend > 0:
            return "LEARNING_RATE_TOO_HIGH"
    
    return "UNCLEAR"

# 檢查預測偏向
def check_predictions(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_data, labels in val_loader:
            labels = labels.to("cuda")
            for timeframe in batch_data:
                batch_data[timeframe] = batch_data[timeframe].to("cuda")
            outputs = model(batch_data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.to("cpu").numpy())
    
    pred_dist = np.bincount(all_preds, minlength=3)
    true_dist = np.bincount(all_labels, minlength=3)
    
    print(f"真實分布: {true_dist}")
    print(f"預測分布: {pred_dist}")
    
    zero_classes = np.sum(pred_dist == 0)
    if zero_classes >= 1:
        return "EXTREME_BIAS"
    return "NORMAL"
def main(label:pd.DataFrame,side:str):
    # train_size = int(len(label) * 0.6)
    # val_size = int(len(label) * 0.2)
    # test_size = len(label) - train_size - val_size

    # train_label = label.iloc[:train_size]
    # val_label = label.iloc[train_size:train_size+val_size]
    # test_label = label.iloc[train_size+val_size:]
    label_train, label_val, label_test = purged_split(label, max_timeframe_lookback_window=25, test_date='2025-04-30 23:00:00')
    print(f"Train end time : {label_train['t0'].max()}, Val end time: {label_val['t0'].max()}, Test end time: {label_test['t0'].max()}")
    try:
        df_1h = pd.read_csv('./data/1h_klines.csv',index_col=['datetime'],parse_dates=['datetime'])
        df_4h = pd.read_csv('./data/4h_klines.csv',index_col=['datetime'],parse_dates=['datetime'])

    except FileNotFoundError:
        raise FileNotFoundError("Kline data files not found. Please ensure 'data/1h_klines.csv' and 'data/4h_klines.csv' exist.")

    try:
        df_1h = compute_features(df_1h, alpha_dir="./random_alpha_generator")
        df_4h = compute_features(df_4h, alpha_dir="./random_alpha_generator")

        df_1h.to_csv('./data/1h_klines_features.csv')
        df_4h.to_csv('./data/4h_klines_features.csv')
        print("Features computed and saved successfully.")
    except Exception as e:
        raise ValueError(f"Error computing features: {e}")
    
    train_dataloader, train_size = data_pipeline(label_train, df_1h, df_4h)
    val_dataloader, val_size = data_pipeline(label_val, df_1h, df_4h)
    test_dataloader, test_size = data_pipeline(label_test, df_1h, df_4h)
    print(f"訓練集大小: {train_size}, 驗證集大小: {val_size}, 測試集大小: {test_size}")
    # 模型配置
    timeframe_configs = {
        '1h': {'input_dim': df_1h.shape[1], 'hidden_dim': 96, 'embed_dim': 128, 'num_layers': 1, 'dropout': 0.3},
        '4h': {'input_dim': df_4h.shape[1], 'hidden_dim': 96, 'embed_dim': 128, 'num_layers': 1, 'dropout': 0.3}
    }
    
    # 建立模型
    model = MultiTimeframeLSTMClassifier(
        timeframe_configs=timeframe_configs,
        lstm_hidden_dim=128,
        lstm_num_layers=1,
        num_classes=2,
        dropout=0.3,
        attention_heads=4
    )
    

    
    # 訓練範例
    train_labels = []
    for _, labels in train_dataloader:
        train_labels.extend(labels.numpy())
    classes = np.unique(train_labels)
    weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    print(f"類別權重: {dict(zip(classes, weights))}")

    trainer = TrainingManager(model)
    trainer.train(train_dataloader, val_dataloader, num_epochs=100, lr=0.00015, weight_decay=5e-4, weight=weights)
    cm, y_true, y_pred, test_acc, report = trainer.test(test_dataloader)
    label_test['pred_vote'] = y_pred
    
    # 打印結果
    print("\n" + "="*60)
    print("最終測試結果")
    print("="*60)
    print(f"測試準確率: {test_acc:.4f}")
    print("\n混淆矩陣:")
    print(cm)
    print("\n詳細分類報告:")
    print(report)
    # 單獨調用可視化方法
    trainer.plot_training_curves()
    trainer.plot_confusion_matrix(y_true, y_pred)
    trainer.plot_classification_report(y_true, y_pred)
   


    # 一鍵保存所有結果
    results_dir = trainer.save_training_results(y_true, y_pred, test_acc, save_prefix=f"{side}_final_")

    problem_type = immediate_diagnosis(trainer)
    bias_type = check_predictions(model, val_dataloader)
    print(f"問題診斷: {problem_type}, 預測偏向: {bias_type}")
    return label_test
# ===== 使用範例 =====
if __name__ == "__main__":
    labels = [
        "data/BTC-USDT_1h_ewma_up3_dn3_lookback36_label.csv",
        "data/BTC-USDT_1h_atr_up4_dn3_lookback72_label.csv",
        "data/BTC-USDT_1h_atr_up4_dn2.5_lookback108_label.csv"
    ]
    for file,lookback in zip(labels,[36,72,108]):
        label = pd.read_csv(file)
        long_label = label[label['side']==1].reset_index(drop=True)
        short_label = label[label['side']==-1].reset_index(drop=True)
        print(f"Long labels: {len(long_label)}, Short labels: {len(short_label)}")


        # main(long_label,'long')
        label_test= main(short_label,'short')
        label_test.to_csv(f'./data/short_{lookback}.csv',index=False)

