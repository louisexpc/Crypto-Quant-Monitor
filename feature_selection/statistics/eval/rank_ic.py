import pandas as pd
import numpy as np
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# 忽略常數輸入導致的相關係數警告
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= 設定區 =================
# 建議設定為 CPU 核心數 - 2，保留資源給系統
MAX_WORKERS = 10

OHLCV_PATH = "data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_15m.csv"
H_LIST = [100, 80, 60, 40, 20]
K_LIST = [40, 30, 20, 15, 10]
BASE_DIR = "feature_selection/results/ts_kmeans_msm"

# 構建路徑清單
FEAT_LIST_PATHS = [
    "data/precomputed/btcusdt_15m_features_VBT_z_norm.csv", # baseline
    "feature_selection/results/pca_60/pca_output.csv", # standard baseline
    "feature_selection/results/hierarchical_corr/hcorr_pearson_avg_k60/hcorr_pearson_avg_k60_selected_feat.csv", # h_corr
    *[f"{BASE_DIR}/msm_kmeans_h{h}_k{k}/msm_kmeans_h{h}_k{k}_selected_feat.csv"
      for h in H_LIST for k in K_LIST if h >= k]
]

OUTPUT_CSV = "feature_selection/results/feature_selection_summary_parallel.csv"


# ================= 輔助函式 =================

def prepare_target(ohlcv_path):
    """讀取 OHLCV 並製作預測目標 (Log Return)"""
    df = pd.read_csv(ohlcv_path)
    # 時間處理
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
        
    # Log Return
    df['target'] = np.log(df['close'].shift(-1) / df['close'])
    return df[['target']].dropna()


def calc_daily_ic(df_day):
    """計算單日的 Rank IC (修正版：保證回傳 Series)"""
    # 取得所有特徵名稱，用於構建回傳格式
    # 注意：這裡假設傳入時 'target' 以外的都是特徵
    feature_cols = [c for c in df_day.columns if c != 'target']
    
    # 定義失敗時的回傳值 (全 NaN 的 Series)
    nan_result = pd.Series(np.nan, index=feature_cols)

    # 1. 樣本太少不計
    if len(df_day) < 10: 
        return nan_result
    
    # 2. 確保是 DataFrame
    if isinstance(df_day, pd.Series):
        df_day = df_day.to_frame()

    # 3. 檢查 Target 是否為常數
    if df_day['target'].std() == 0:
        return nan_result

    features_df = df_day[feature_cols]
    
    # 4. 檢查特徵標準差
    feat_std = features_df.std()
    
    # 處理單一特徵的情況 (std 回傳 scalar)
    if not isinstance(feat_std, pd.Series):
        feat_std = pd.Series([feat_std], index=feature_cols)
        
    valid_feats = feat_std[feat_std > 1e-12].index
    
    if len(valid_feats) == 0:
        return nan_result

    # 5. 計算 Correlation
    # corrwith 回傳的是 Series (Index=Feature Names)，這正是我們要的
    ic_series = features_df[valid_feats].corrwith(df_day['target'], method='spearman')
    
    # 6. 對齊 Index (如果有特徵被過濾掉，要補 NaN，確保 Series 形狀一致)
    return ic_series.reindex(feature_cols)


def calculate_long_short_return_rolling(features_df, target_series, min_periods=200):
    """
    嚴謹版 Long-Short Backtest (Rolling Window)
    避免 Look-ahead Bias，僅使用「當下時間點之前」的資訊來決定分位數。
    """
    # 1. 合成訊號 (Composite Signal)
    # 注意：嚴謹來說，連「特徵方向調整」都不該用 Global IC。
    # 但為了效能，這裡假設特徵的方向性是穩定的 (或已在特徵工程階段處理好方向)。
    # 我們這裡簡單做 Z-Score 平均
    
    # 對齊 index
    common_idx = features_df.index.intersection(target_series.index)
    features_df = features_df.loc[common_idx]
    target_series = target_series.loc[common_idx]
    
    if len(features_df) < min_periods:
        return 0.0

    # 簡單等權重合成 (假設你輸入的特徵都已經調整過方向，例如都是正相關)
    # 如果特徵未調整方向，建議在 feature engineering 階段處理，或使用 expanding correlation (計算極慢)
    composite_signal = features_df.mean(axis=1)

    # 2. 計算 Rolling Percentile (關鍵修改)
    # 我們不看未來的數據，只看「目前這個值在過去歷史中算不算高」
    # expanding(): 從開始累積到現在
    # rank(pct=True): 計算百分位 (0.0 ~ 1.0)
    rolling_rank = composite_signal.expanding(min_periods=min_periods).rank(pct=True)

    # 3. 根據 Rolling Rank 生成多空訊號
    # 歷史前 80% 強 -> 做多
    # 歷史後 20% 弱 -> 做空
    signal_long = (rolling_rank >= 0.8).astype(int)
    signal_short = (rolling_rank <= 0.2).astype(int)

    # 4. 計算收益 (Shift 1, 因為訊號是用來預測下一根)
    # target 已經是 next log return，所以直接相乘
    # Long 收益: signal=1 * return
    # Short 收益: signal=1 * return * -1
    
    ret_long = signal_long * target_series
    ret_short = signal_short * target_series * -1
    
    # 合併多空收益 (假設資金一半做多，一半做空)
    strategy_ret = (ret_long + ret_short) / 2
    
    # 移除前面的暖身期 (min_periods)
    strategy_ret = strategy_ret.iloc[min_periods:]
    
    # 5. 年化計算 (平均每根 K 的收益 * 一年 K 棒數)
    # 這裡用 mean() 代表期望值
    avg_ret = strategy_ret.mean()
    
    # 96根(15m) * 365天
    annualized_ret = avg_ret * 96 * 365
    
    return annualized_ret

def evaluate_feature_set(feat_path, target_df):
    """評估單一特徵集 (核心邏輯)"""
    try:
        # 讀取特徵
        if not os.path.exists(feat_path):
            return {'id': os.path.basename(feat_path), 'error': 'File not found'}

        feat_df = pd.read_csv(feat_path)
        
        # 時間索引處理
        if 'datetime' in feat_df.columns:
            feat_df['datetime'] = pd.to_datetime(feat_df['datetime'], utc=True)
            feat_df = feat_df.set_index('datetime')
        elif 'timestamp' in feat_df.columns:
            feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp'], utc=True)
            feat_df = feat_df.set_index('timestamp')

        # 合併標籤
        merged = feat_df.join(target_df, how='inner')
        if merged.empty:
            return {'id': os.path.basename(feat_path), 'error': 'Empty intersection'}

        target = merged['target']
        features = merged.drop(columns=['target'])
        
        # 確保 features 是 DataFrame (即使只有一欄)
        if isinstance(features, pd.Series):
            features = features.to_frame()

        stats = {}
        stats['id'] = os.path.basename(feat_path).replace('_selected_feat.csv', '')
        stats['n_features'] = features.shape[1]

        # 1. Turnover Proxy (Autocorrelation)
        # 用 np.mean 處理 scalar 結果
        autocorr = features.apply(lambda col: col.autocorr(lag=1)).mean()
        stats['autocorr_turnover_proxy'] = float(autocorr)

        # 2. Daily IC / ICIR / t-stat
        daily_ics = merged.groupby(merged.index.date).apply(calc_daily_ic)
        
        # [關鍵修正] 如果 daily_ics 是空的或全 NaN
        if daily_ics.empty or daily_ics.isnull().all().all():
             return {'id': stats['id'], 'error': 'IC calculation failed (all NaNs)'}

        # [關鍵修正] 使用 np.nanmean 和 np.nanstd 避免 scalar 報錯
        feat_ic_mean = daily_ics.mean(axis=0) # Series
        feat_ic_std = daily_ics.std(axis=0)   # Series
        
        # 計算 ICIR (Per Feature)
        feat_icir = feat_ic_mean / (feat_ic_std + 1e-9)
        
        # 計算 t-stat
        n_days = daily_ics.shape[0]
        feat_t_stat = feat_icir * np.sqrt(n_days)

        # 彙總 (使用 np.abs 和 np.mean 確保穩健性)
        stats['ic_mean_abs'] = np.mean(np.abs(feat_ic_mean))
        stats['icir_mean_abs'] = np.mean(np.abs(feat_icir))
        stats['t_stat_mean_abs'] = np.mean(np.abs(feat_t_stat))
        
        # 顯著特徵比例
        if isinstance(feat_t_stat, pd.Series):
             stats['significant_feat_ratio'] = (feat_t_stat.abs() > 2.0).mean()
        else:
             stats['significant_feat_ratio'] = 1.0 if abs(feat_t_stat) > 2.0 else 0.0

        # 3. Long-Short Return
        stats['ls_annual_ret'] = calculate_long_short_return_rolling(features, target)

        # 4. Redundancy
        # 取樣避免記憶體爆炸
        if len(features) > 2000:
            sample_feat = features.sample(n=2000)
        else:
            sample_feat = features
            
        # 如果只有一個特徵，冗餘度為 0
        if features.shape[1] > 1:
            corr_matrix = sample_feat.corr(method='spearman').abs()
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            # 使用 np.nanmean 處理可能的 NaN
            redundancy = corr_matrix.where(mask).stack().mean()
            stats['feature_redundancy'] = float(redundancy) if not np.isnan(redundancy) else 0.0
        else:
            stats['feature_redundancy'] = 0.0

        return stats

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'id': os.path.basename(feat_path), 'error': str(e)}


# ================= 主程式 =================
def main():
    print(f"1. 準備目標標籤... (Log Return)")
    # 為了避免在每個 Process 重複讀取大檔案，我們在主進程讀好，但 DataFrame 在 MP 中傳遞會有序列化開銷
    # 由於 target 很小 (只有 datetime 和 float)，傳遞還好。
    # 更好的方式是傳遞 path，每個 process 自己讀 (IO bound)，或使用 memory map。
    # 這裡為了簡單，我們直接傳遞 DataFrame (Copy-on-Write 機制下還算快)。
    target_df = prepare_target(OHLCV_PATH)
    
    print(f"2. 開始評估 {len(FEAT_LIST_PATHS)} 組特徵集 (Parallel Workers: {MAX_WORKERS})...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任務
        future_to_path = {
            executor.submit(evaluate_feature_set, path, target_df): path 
            for path in FEAT_LIST_PATHS
        }
        
        count = 0
        total = len(FEAT_LIST_PATHS)
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            count += 1
            filename = os.path.basename(path)
            
            try:
                res = future.result()
                if 'error' in res:
                    print(f"[{count}/{total}] [FAIL] {res['id']}: {res['error']}")
                else:
                    print(f"[{count}/{total}] [OK] {res['id']} | ICIR: {res['icir_mean_abs']:.4f} | LS: {res['ls_annual_ret']:.2%}")
                    results.append(res)
            except Exception as e:
                print(f"[{count}/{total}] [CRASH] {filename}: {e}")

    # 3. 輸出 summary
    if not results:
        print("沒有產生任何結果。")
        return

    summary_df = pd.DataFrame(results)
    
    # 欄位排序
    cols_order = [
        'id', 'n_features', 
        'ic_mean_abs', 'icir_mean_abs', 't_stat_mean_abs', 
        'ls_annual_ret', 'significant_feat_ratio', 
        'autocorr_turnover_proxy', 'feature_redundancy'
    ]
    # 只選存在的欄位
    cols_order = [c for c in cols_order if c in summary_df.columns]
    summary_df = summary_df[cols_order]
    
    # 依照 ICIR 排序
    summary_df = summary_df.sort_values(by='icir_mean_abs', ascending=False)
    
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n全部完成！報告已儲存至: {OUTPUT_CSV}")
    # 顯示前 10 名
    print(summary_df.head(10).to_string())

if __name__ == "__main__":
    main()