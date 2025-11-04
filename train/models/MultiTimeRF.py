# rf_pipeline_integrated.py
# -*- coding: utf-8 -*-
"""
完整整合：RandomForest 版端到端訓練／驗證／測試（含 DataLoader→sklearn 特徵轉換與新版 Trainer）

需求：
- pandas, numpy, scikit-learn, matplotlib, seaborn, talib
- 你自家的 feature_selector(data, target='returns') 介面（train_model、get_selected_features、transform）
- 檔案：
    ./data/1h_klines.csv
    ./data/4h_klines.csv
    label 檔（見 __main__ 範例）
    ./random_alpha_generator/*.json (選配)

使用：
    python rf_pipeline_integrated.py
或在其他程式中：
    from rf_pipeline_integrated import main
    main(your_label_df, side='short', use_rf=True, feature_mode='stat')
"""

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

# ========== 你的外部模組：特徵選擇器 ==========
from train_utils.feature_selectors import BiserialRankEvaluator


# ========== Dataset & DataLoader（版本精簡，沿用你的介面） ==========
import torch
from torch.utils.data import Dataset, DataLoader


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


# ========== RF 版 Trainer ==========
class SklearnRFTrainer:
    """
    Sklearn RandomForest 的 Trainer（與你的 TrainingManager 風格對齊）：
    - fit(train, val) + refit(train+val)
    - test(test)
    - 繪製混淆矩陣（數量/百分比）、分類報告（柱狀＋餅圖）
    - Top-k 特徵重要度
    - JSON 摘要存檔
    """
    def __init__(self,
                 feature_mode: str = "stat",
                 n_estimators: int = 800,
                 max_depth: Optional[int] = None,
                 max_features: str = "sqrt",
                 min_samples_leaf: int = 1,
                 class_weight: str = "balanced_subsample",
                 random_state: int = 42,
                 save_dir: str = "./rf_results",
                 topk_importance: int = 30):
        self.feature_mode = feature_mode
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
            oob_score=False,
        )
        self.model = RandomForestClassifier(**self.params)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.topk_importance = topk_importance
        self.history = {}  # 讓介面與 Torch 版本一致地有個 history 掛勾

        # 畫圖風格
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8')

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _plot_confusion_matrices(self, cm: np.ndarray, class_names: List[str], ts: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Percent)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')

        path = os.path.join(self.save_dir, f'confusion_matrix_{ts}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[RF] Confusion matrix saved to {path}")
        plt.close(fig)
        return path

    def _plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    class_names: List[str], ts: str):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        x = np.arange(len(class_names))
        width = 0.25
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.85)
        ax1.bar(x, recall, width, label='Recall', alpha=0.85)
        ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names)
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Classification Metrics by Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.pie(support, labels=class_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution in Test Set')

        path = os.path.join(self.save_dir, f'classification_report_{ts}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[RF] Classification report saved to {path}")
        plt.close(fig)
        return path

    def _plot_feature_importances(self, importances: np.ndarray, ts: str):
        idx = np.argsort(importances)[::-1][:self.topk_importance]
        fig, ax = plt.subplots(figsize=(9, 10))
        ax.barh(range(len(idx)), importances[idx][::-1])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([f"f{j}" for j in idx[::-1]])
        ax.set_title(f"Top-{len(idx)} Feature Importances")
        path = os.path.join(self.save_dir, f'feature_importances_{ts}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[RF] Feature importances saved to {path}")
        plt.close(fig)
        return path

    def fit_validate(self, train_loader: DataLoader, val_loader: DataLoader):
        X_tr, y_tr = dataloader_to_arrays(train_loader, feature_mode=self.feature_mode)
        X_va, y_va = dataloader_to_arrays(val_loader, feature_mode=self.feature_mode)
        print(f"[RF] train: {X_tr.shape}, val: {X_va.shape}")

        self.model.fit(X_tr, y_tr)
        va_pred = self.model.predict(X_va)
        va_acc = accuracy_score(y_va, va_pred)
        self.history['val_acc'] = float(va_acc)
        print(f"[RF] Validation acc = {va_acc:.4f}")

        # 簡易策略：train+val 合併重訓
        X_trva = np.concatenate([X_tr, X_va], axis=0)
        y_trva = np.concatenate([y_tr, y_va], axis=0)
        self.model.fit(X_trva, y_trva)
        return va_acc

    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        X_te, y_te = dataloader_to_arrays(test_loader, feature_mode=self.feature_mode)
        te_pred = self.model.predict(X_te)
        return y_te, te_pred

    def evaluate_and_save(self, y_true: np.ndarray, y_pred: np.ndarray):
        ts = self._timestamp()
        test_acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report_str = classification_report(y_true, y_pred, digits=4, zero_division=0)

        print(f"[RF] Test acc = {test_acc:.4f}")
        print("[RF] Classification report:\n", report_str)

        # 視覺化輸出
        class_names = [str(c) for c in sorted(np.unique(y_true))]
        cm_path = self._plot_confusion_matrices(cm, class_names, ts)
        cr_path = self._plot_classification_report(y_true, y_pred, class_names, ts)
        fi_path = self._plot_feature_importances(self.model.feature_importances_, ts)

        # JSON 摘要
        summary = dict(
            feature_mode=self.feature_mode,
            params=self.params,
            val_acc=float(self.history.get('val_acc', np.nan)),
            test_acc=float(test_acc),
            confusion_matrix=cm.tolist(),
            classification_report=report_str,
            artifacts=dict(
                confusion_matrix_png=cm_path,
                classification_report_png=cr_path,
                feature_importances_png=fi_path
            ),
            timestamp=ts
        )
        jpath = os.path.join(self.save_dir, f"rf_summary_{ts}.json")
        with open(jpath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[RF] Summary saved to {jpath}")

        return test_acc, cm, report_str, summary


# ========== 你原本資料處理：特徵工程、切割、管線 ==========
def _range_ewm_vol(df: pd.DataFrame, halflife: int = 20) -> pd.Series:
    rng = (df['high'] - df['low']) / df['open'].replace(0.0, np.nan)
    return rng.ewm(halflife=halflife, adjust=False).mean().shift(1)


def compute_features(df: pd.DataFrame, alpha_dir: str = None) -> pd.DataFrame:
    data = df.copy()
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', and 'volume' columns.")
    data['returns'] = data['close'].pct_change().shift(-1)

    data["obv"] = talib.OBV(data['close'], data['volume'])
    data["volatility"] = _range_ewm_vol(data, halflife=20)
    data["momentum"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)

    for n in [5, 10, 20]:
        data[f"ema_{n}"] = talib.EMA(data['close'], timeperiod=n)
        data[f"rsi_{n}"] = talib.RSI(data['close'], timeperiod=n)
        data[f"atr_{n}"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"cci_{n}"] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"mfi_{n}"] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=n)
        data[f"adx_{n}"] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"willr_{n}"] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=n)

    if alpha_dir is not None and os.path.isdir(alpha_dir):
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

    # 丟掉原 K 線欄位，保留特徵
    data = data.drop(labels=['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
    data.fillna(0, inplace=True)

    # 特徵選擇（沿你原設計）
    # print(f"[INFO] 計算特徵完成，總共 {data.shape[1]} 個欄位，開始特徵選擇...")
    # fs = feature_selector(data, target='returns')
    # fs.train_model(p_threshold=0.035)
    # selected_features = fs.get_selected_features()
    # print(f"[INFO] 選擇後特徵數: {len(selected_features)}")
    # data = fs.transform(data)
    return data


def purged_split(labels: pd.DataFrame,
                 max_timeframe_lookback_window: int,
                 test_date: str,
                 embargo_hour: int = 4):
    labels = labels.set_index(pd.to_datetime(labels['t0']))
    labels = labels.drop(columns=['t0'])
    limit_date = pd.to_datetime(test_date).tz_localize("Asia/Taipei") - pd.Timedelta(
        hours=max_timeframe_lookback_window * 4 + embargo_hour
    )
    labels_train_val = labels[labels.index < limit_date]
    labels_test = labels[labels.index >= limit_date]
    val_start_date = limit_date - pd.Timedelta(days=60)
    val_limit_date = val_start_date - pd.Timedelta(hours=max_timeframe_lookback_window * 4 + embargo_hour)
    labels_train = labels_train_val[labels_train_val.index < val_limit_date]
    labels_val = labels_train_val[labels_train_val.index >= val_start_date]
    print(f"Train: {labels_train.shape}, Val: {labels_val.shape}, Test: {labels_test.shape}")
    assert labels_train.index.max() < labels_val.index.min() - pd.Timedelta(
        hours=max_timeframe_lookback_window * 4 + embargo_hour), "Training and validation sets overlap!"
    assert labels_val.index.max() < labels_test.index.min() - pd.Timedelta(
        hours=max_timeframe_lookback_window * 4 + embargo_hour), "Validation and test sets overlap!"
    return labels_train.reset_index(drop=False), labels_val.reset_index(drop=False), labels_test.reset_index(drop=False)


def data_pipeline(label: pd.DataFrame,
                  df_1h: pd.DataFrame,
                  df_4h: pd.DataFrame,
                  use_padding: bool = False
                  ) -> Tuple[DataLoader, int]:
    """
    與你原本的一致：依事件時間往前取 1h*100、4h*25 的片段
    輸出：DataLoader（batch_size=32, shuffle=False）
    """
    print(f"[INFO] 開始數據處理管線, padding: {use_padding}")
    data_dict = {}
    valid_labels = []

    if 'label' not in label.columns:
        raise ValueError("Label DataFrame must contain a 'label' column.")

    # 將 index 改回時間索引便於切片
    df_1h = df_1h.copy()
    df_4h = df_4h.copy()
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        raise ValueError("df_1h index must be DatetimeIndex")
    if not isinstance(df_4h.index, pd.DatetimeIndex):
        raise ValueError("df_4h index must be DatetimeIndex")

    for _, row in label.iterrows():
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
        start_time_1h = timestamp - pd.Timedelta(hours=100)
        mask_1h = (df_1h.index >= start_time_1h) & (df_1h.index < timestamp)
        data_1h = df_1h.loc[mask_1h]
        if len(data_1h) < 100:
            if use_padding:
                padding = pd.DataFrame(np.zeros((100 - len(data_1h), data_1h.shape[1])), columns=data_1h.columns)
                data_1h = pd.concat([padding, data_1h], ignore_index=True)
            else:
                continue

        # 4h
        start_time_4h = timestamp - pd.Timedelta(hours=25 * 4)
        mask_4h = (df_4h.index >= start_time_4h) & (df_4h.index < timestamp)
        data_4h = df_4h.loc[mask_4h]
        if len(data_4h) < 25:
            if use_padding:
                padding = pd.DataFrame(np.zeros((25 - len(data_4h), data_4h.shape[1])), columns=data_4h.columns)
                data_4h = pd.concat([padding, data_4h], ignore_index=True)
            else:
                continue

        if '1h' not in data_dict:
            data_dict['1h'] = []
        if '4h' not in data_dict:
            data_dict['4h'] = []

        data_dict['1h'].append(data_1h.values)
        data_dict['4h'].append(data_4h.values)
        valid_labels.append(row['label'])

    # 堆疊
    data_dict['1h'] = np.array(data_dict['1h'])
    data_dict['4h'] = np.array(data_dict['4h'])
    valid_labels = np.array(valid_labels)

    print(f"[Debug] Unique labels in data: {pd.Series(valid_labels).value_counts().to_dict()}")

    dataset = MultiTimeframeOHLCVDataset(
        data_dict=data_dict,
        labels=valid_labels,
        timeframe_windows={'1h': 100, '4h': 25},
        normalize=True
    )
    dataloader = create_dataloader(dataset, batch_size=32, shuffle=False)
    return dataloader, len(dataset)


# ========== End-to-end main（切到 RF） ==========
def main(label: pd.DataFrame,
         side: str,
         lookback: int,
         feature_mode: str = "stat",
         use_rf: bool = True):
    """
    - purged_split
    - 載入K線→compute_features（含alpha+特徵選擇）
    - 建立 train/val/test dataloader
    - 用 RF 訓練、驗證、重訓、測試並輸出圖與JSON摘要
    - 回傳：帶 preds 的 test label
    """
    # 1) 時間切分
    label_train, label_val, label_test = purged_split(
        label, max_timeframe_lookback_window=25, test_date='2025-04-30 23:00:00'
    )
    print(f"Train end time : {label_train['t0'].max()}, Val end time: {label_val['t0'].max()}, Test end time: {label_test['t0'].max()}")

    # 2) 載入 K 線並算特徵
    df_1h = pd.read_csv('/home/louisexpc/Crypto-Quant-Monitor/train/data/binanceusdm_swap_BTC-USDT-USDT_1h.csv', index_col=['datetime'], parse_dates=['datetime'])
    df_4h = pd.read_csv('/home/louisexpc/Crypto-Quant-Monitor/train/data/binanceusdm_swap_BTC-USDT-USDT_4h.csv', index_col=['datetime'], parse_dates=['datetime'])
    
    df_1h = compute_features(df_1h, alpha_dir="/home/louisexpc/Crypto-Quant-Monitor/train/data/alpha")
    df_4h = compute_features(df_4h, alpha_dir="/home/louisexpc/Crypto-Quant-Monitor/train/data/alpha")

    # 2.1) 獨立出 feature selector（如有需要）
    _, features_1h = BiserialRankEvaluator(event_df=label_train, df=df_1h, lookback=lookback).evaluate(threshold=0.01)
    print(f"[INFO] 1h selected features: {features_1h}")
    df_1h_fs = df_1h[features_1h]
    # df_4h_fs, features_4h = BiserialRankEvaluator(event_df=label_train, df=df_4h, lookback=lookback).evaluate(threshold=0.1)
    df_4h_fs = df_4h[features_1h]  # 簡單起見，4h 用同一組特徵
    print(f"[INFO] 1h selected features: {len(features_1h)}, 4h selected features: {len(features_1h)}")
    df_1h_fs.to_csv('./data/1h_klines_features.csv')
    df_4h_fs.to_csv('./data/4h_klines_features.csv')
    print("[INFO] Features computed and saved.")
    # 3) dataloader
    
    train_loader, train_size = data_pipeline(label_train, df_1h_fs, df_4h_fs)
    val_loader, val_size = data_pipeline(label_val, df_1h_fs, df_4h_fs)
    test_loader, test_size = data_pipeline(label_test, df_1h_fs, df_4h_fs)
    print(f"[INFO] 訓練集: {train_size}, 驗證集: {val_size}, 測試集: {test_size}")

    if not use_rf:
        raise NotImplementedError("本檔專注 RF；若要切回 Torch 模型，請沿用你原 TrainingManager。")
    
    # 4) RF Trainer
    trainer = SklearnRFTrainer(
        feature_mode=feature_mode,
        n_estimators=800,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        save_dir=f"./rf_results_{side}",
        topk_importance=30
    )
    trainer.fit_validate(train_loader, val_loader)
    y_true, y_pred = trainer.predict(test_loader)
    test_acc, cm, report_str, summary = trainer.evaluate_and_save(y_true, y_pred)

    out = label_test.copy()
    out['pred_vote'] = y_pred
    print("\n" + "=" * 60)
    print("[RF] 最終測試結果")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report_str)
    return out


# ========== CLI 例子 ==========
if __name__ == "__main__":
    # 這裡示範你的 label 三檔的流程（與你原程式一致）
    labels_files = [
        "data/BTC-USDT_1h_atr_up4_dn2_lookback36_label.csv",
        "data/BTC-USDT_1h_ewma_up8_dn6_lookback72_label.csv",
        "data/BTC-USDT_1h_ewma_up8_dn10_lookback108_label.csv"
    ]
    lookbacks = [36, 72, 108]
    for file, lookback in zip(labels_files, lookbacks):
        print(f"\n=== Processing label file: {file} with lookback {lookback} ===")
        label_df = pd.read_csv(file)
        print(f"[Debug] Unique labels in {file}: {label_df['label'].value_counts().to_dict()}")
        long_label = label_df[label_df['side'] == 1].reset_index(drop=True)
        short_label = label_df[label_df['side'] == -1].reset_index(drop=True)
        print(f"[Debug] Long labels: {len(long_label)}, Short labels: {len(short_label)}")
        print(f"[INFO] =========  Short Side  =========")
        # 以 short 當例子；feature_mode 可切 "stat" 或 "flatten"
        short_label_test_out = main(short_label, side='short', lookback=lookback, feature_mode='stat', use_rf=True)
        short_label_test_out.to_csv(f'./data/short_{lookback}_rf.csv', index=False)
        print(f"[INFO] =========  Long Side  =========")
        long_label_test_out = main(long_label, side='long', lookback=lookback, feature_mode='stat', use_rf=True)
        long_label_test_out.to_csv(f'./data/long_{lookback}_rf.csv', index=False)
