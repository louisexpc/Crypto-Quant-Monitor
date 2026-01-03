import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
import warnings
from tqdm import tqdm

# 忽略運算中的一些除零警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================
OHLCV_PATH = "data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_15m.csv"
BASE_DIR = "feature_selection/results/ts_kmeans_msm"
OUTPUT_CSV = "feature_selection/results/feature_selection_summary.csv"

# 參數組合
H_LIST = [100, 80, 60, 40, 20]
K_LIST = [40, 30, 20, 15, 10]

# 動態生成檔案清單
FEAT_LIST_PATHS = [
    "data/precomputed/btcusdt_15m_features_VBT_z_norm.csv", # baseline
    "feature_selection/results/pca_60/pca_output.csv", # standard baseline
    "feature_selection/results/hierarchical_corr/hcorr_pearson_avg_k60/hcorr_pearson_avg_k60_selected_feat.csv", # h_corr
]

# 加入 MSM KMeans 組合
for h in H_LIST:
    for k in K_LIST:
        if h >= k:
            path = f"{BASE_DIR}/msm_kmeans_h{h}_k{k}/msm_kmeans_h{h}_k{k}_selected_feat.csv"
            FEAT_LIST_PATHS.append(path)

# 回測參數
N_FOLDS = 5
EMBARGO_PCT = 0.01  # 在 Train 和 Test 之間清除 1% 的數據以防止洩漏

# ==========================================
# 2. 核心函數：Purged K-Fold 與 Metrics
# ==========================================

class PurgedKFold:
    """
    實現簡單版的 Purged K-Fold。
    在時間序列中，將數據分為 N 塊。
    測試集為第 k 塊，訓練集為其餘塊，但在訓練集和測試集接壤處進行 Purge/Embargo。
    """
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        embargo = int(n_samples * self.embargo_pct)
        
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        
        for train_idx, test_idx in kf.split(X):
            # 簡單處理：如果是時序數據，通常我們希望測試集在未來的某一段
            # 這裡為了標準 K-Fold，我們會遍歷每一段作為測試集
            
            # 實施 Purging: 從訓練集中移除緊鄰測試集之前的數據 (如果是預測未來)
            # 實施 Embargo: 從訓練集中移除緊鄰測試集之後的數據
            
            # 找出測試集的邊界
            test_start = test_idx.min()
            test_end = test_idx.max()
            
            # 過濾訓練集
            # 1. 移除測試集區間
            # 2. 移除測試集前後的緩衝區 (Embargo)
            
            valid_train_mask = np.ones(len(train_idx), dtype=bool)
            
            # 這裡簡化邏輯：只要訓練樣本在 (test_start - embargo) 到 (test_end + embargo) 之間就剔除
            # 這是為了確保沒有重疊標籤或序列相關性洩漏
            train_indices_val = indices[train_idx]
            
            purge_start = test_start - embargo
            purge_end = test_end + embargo
            
            clean_train_idx = train_indices_val[
                (train_indices_val < purge_start) | (train_indices_val > purge_end)
            ]
            
            yield clean_train_idx, test_idx

def calculate_vif(df_features):
    """計算平均與最大 VIF，處理潛在的無限值"""
    # 如果特徵數太多(>100)，VIF計算會極慢，這裡做一個安全限制
    if df_features.shape[1] > 100:
        return -1, -1 # Skip VIF for high dim
        
    vif_data = []
    # 添加常數項
    X = df_features.assign(const=1)
    
    # 處理可能的 Singular Matrix
    try:
        for i in range(len(X.columns)-1):
            vif = variance_inflation_factor(X.values, i)
            vif_data.append(vif)
        return np.mean(vif_data), np.max(vif_data)
    except:
        return 999, 999 # Singular matrix or error

def get_performance_metrics(returns):
    """計算年化 Sharpe (假設 15m 頻率)"""
    if len(returns) < 2: return 0
    # 15m kline, 24h trading => 4 * 24 = 96 periods per day
    # Crypto 365 days
    ann_factor = np.sqrt(96 * 365)
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * ann_factor
    return sharpe

# ==========================================
# 3. 主程序 (Main Loop)
# ==========================================

def main():
    print(f"Loading OHLCV from {OHLCV_PATH}...")
    try:
        df_ohlcv = pd.read_csv(OHLCV_PATH)
        # 關鍵修正 1: 強制轉為 UTC，然後移除時區資訊 (變為 Naive UTC)
        df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['datetime'], utc=True)
        df_ohlcv = df_ohlcv.set_index('datetime').sort_index()
        df_ohlcv.index = df_ohlcv.index.tz_convert(None) 
        
        # 建構 Target: Log Returns (Next Period)
        # 預測下一根 K 線的收益率
        df_ohlcv['target'] = np.log(df_ohlcv['close'] / df_ohlcv['close'].shift(1)).shift(-1)
        target_series = df_ohlcv['target'].dropna()
        
    except Exception as e:
        print(f"Error loading OHLCV: {e}")
        return

    results = []

    print(f"Starting evaluation of {len(FEAT_LIST_PATHS)} feature sets...")
    
    for feat_path in tqdm(FEAT_LIST_PATHS):
        if not os.path.exists(feat_path):
            # 某些組合可能不存在，跳過不報錯
            continue
            
        try:
            # 解析檔名以獲取 metadata
            path_parts = feat_path.split('/')
            filename = path_parts[-1]
            if "msm_kmeans" in feat_path:
                method = "TS_KMeans_MSM"
            elif "pca" in feat_path:
                method = "PCA"
            elif "hcorr" in feat_path:
                method = "Hierarchical_Corr"
            else:
                method = "Baseline"
                
            # 讀取特徵
            df_feats = pd.read_csv(feat_path)
            if 'datetime' in df_feats.columns:
                # 關鍵修正 2: 對特徵檔做同樣的時區移除處理
                df_feats['datetime'] = pd.to_datetime(df_feats['datetime'], utc=True)
                df_feats = df_feats.set_index('datetime').sort_index()
                df_feats.index = df_feats.index.tz_convert(None)
            
            # 資料對齊 (Align Data)
            # 現在兩者都是 Naive UTC，可以安全地取交集
            common_idx = df_feats.index.intersection(target_series.index)
            
            if len(common_idx) == 0:
                print(f"Warning: No overlapping data found for {filename}. Check timestamps.")
                continue

            X = df_feats.loc[common_idx]
            y = target_series.loc[common_idx]
            
            # 移除包含 NaN 的行
            valid_mask = ~X.isna().any(axis=1) & ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 1000:
                print(f"Not enough data for {filename} (n={len(X)}), skipping.")
                continue

            # --- Metric 1: VIF ---
            avg_vif, max_vif = calculate_vif(X)
            
            # --- Metric 2: Global IC ---
            feature_ics = []
            for col in X.columns:
                ic, _ = spearmanr(X[col], y)
                feature_ics.append(ic)
            avg_abs_ic = np.mean(np.abs(feature_ics))
            # 防止 std 為 0
            ic_std = np.std(feature_ics)
            ic_ir = np.mean(feature_ics) / (ic_std + 1e-9)

            # --- Metric 3: Purged K-Fold Backtest ---
            pkf = PurgedKFold(n_splits=N_FOLDS, embargo_pct=EMBARGO_PCT)
            model = Ridge(alpha=10.0) 
            
            oos_preds = []
            oos_true = []
            
            for train_idx, test_idx in pkf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                
                oos_preds.extend(pred)
                oos_true.extend(y_test)
            
            oos_preds = np.array(oos_preds)
            oos_true = np.array(oos_true)
            
            # 策略回報 = 訊號方向 * 真實回報
            strategy_rets = np.sign(oos_preds) * oos_true
            sharpe = get_performance_metrics(strategy_rets)
            ic_oos, _ = spearmanr(oos_preds, oos_true)
            
            results.append({
                'Path': feat_path,
                'Method': method,
                'Filename': filename,
                'Num_Features': X.shape[1],
                'Avg_VIF': avg_vif,
                'Max_VIF': max_vif,
                'Feature_Avg_IC_Abs': avg_abs_ic,
                'Backtest_Sharpe': sharpe,
                'Backtest_IC': ic_oos,
            })
            
        except Exception as e:
            print(f"Error processing {feat_path}: {e}")
            import traceback
            traceback.print_exc()

    # 輸出結果
    if not results:
        print("No results generated.")
        return

    df_results = pd.DataFrame(results)
    df_results.sort_values(by='Backtest_Sharpe', ascending=False, inplace=True)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCompleted! Results saved to {OUTPUT_CSV}")
    print(df_results[['Filename', 'Avg_VIF', 'Backtest_Sharpe', 'Backtest_IC']].head(10))
    
if __name__ == "__main__":
    main()