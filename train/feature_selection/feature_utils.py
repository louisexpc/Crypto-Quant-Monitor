# feature_utils.py
import pandas as pd, numpy as np

def create_labels_adaptive(df: pd.DataFrame,
                           horizon: int = 1,
                           mode: str = "vol",          # "bps" | "vol" | "quantile" | "cost" | "triple"
                           flat_band_bps: float = 5.0, # for mode="bps"
                           k_vol: float = 0.5,         # for mode="vol"
                           vol_window: int = 96,       # 以 1H 資料為例，過去 4 天
                           q_flat: float = 0.3,        # for mode="quantile": 中間 40% = 持平
                           roundtrip_cost_bps: float = 10.0, # for mode="cost"
                           triple_k: float = 0.5,      # for mode="triple"
                           triple_window: int = 96):
    """
    根據選擇定義flat的模式
    回傳含 y_reg, y_cls 的 DataFrame；末端 horizon 筆 y 會是 NaN/-1（請用 mask 排除）。
    """
    df = df.copy()
    close = df["close"].astype(float)
    logret_fwd = np.log(close.shift(-horizon)) - np.log(close)
    df["y_reg"] = logret_fwd

    # ----------- 計算 threshold_t -----------
    if mode == "bps":
        thr = flat_band_bps / 10000.0
        thr_t = pd.Series(thr, index=df.index)

    elif mode == "cost":
        # 交易回合成本（手續費+滑價）換成 log 門檻；近似用 bps/10000
        thr = roundtrip_cost_bps / 10000.0
        thr_t = pd.Series(thr, index=df.index)

    elif mode == "vol":
        # 過去 vol_window 的歷史波動（以 logret 計），不看未來 → 無洩漏
        lr_past = np.log(close).diff()
        sigma = lr_past.rolling(vol_window, min_periods=vol_window//2).std()
        thr_t = (k_vol * sigma * np.sqrt(horizon)).reindex(df.index)

    elif mode == "quantile":
        # 用過去資料的雙尾分位數，控制類別比例；expanding 不看未來
        lr_past = (np.log(close).shift(-horizon) - np.log(close)).shift(+horizon)  # 對齊到當下，不含未來
        q_low  = lr_past.expanding(min_periods=500).quantile(q_flat)
        q_high = lr_past.expanding(min_periods=500).quantile(1 - q_flat)
        # 對稱化一個閾值（也可做非對稱，見下方註解）
        thr_t = pd.concat([q_low.abs(), q_high.abs()], axis=1).max(axis=1)

    elif mode == "triple":
        # 簡化版 triple-barrier：若在 horizon 內最高/最低越過 ±kσ_t 則標 up/down，否則 flat
        lr_past = np.log(close).diff()
        sigma = lr_past.rolling(triple_window, min_periods=triple_window//2).std()
        up_bar = k_vol * sigma
        dn_bar = -k_vol * sigma

        y_cls = np.full(len(df), -1, dtype=int)
        for t in range(len(df) - horizon):
            # 期初價格與後續區間的相對 logret path
            rel = np.log(close.iloc[t+1:t+1+horizon].values) - np.log(close.iloc[t])
            hit_up = (rel >= up_bar.iloc[t]).any()
            hit_dn = (rel <= dn_bar.iloc[t]).any()
            if hit_up and not hit_dn: y_cls[t] = 2
            elif hit_dn and not hit_up: y_cls[t] = 0
            else: y_cls[t] = 1
        df["y_cls"] = y_cls
        return df

    else:
        raise ValueError("Unknown mode")

    # ----------- 依 threshold_t 標註 -----------
    y_cls = np.full(len(df), -1, dtype=int)
    cond_up   = logret_fwd >  thr_t
    cond_down = logret_fwd < -thr_t
    cond_flat = (~cond_up) & (~cond_down)

    y_cls[cond_up.fillna(False).values]   = 2
    y_cls[cond_down.fillna(False).values] = 0
    y_cls[cond_flat.fillna(False).values] = 1

    df["y_cls"] = y_cls
    return df

def load_labeled_data(cfg):
    # 1. 包含ohlcv+pandas_ta_all的指標
    csv_path = cfg["csv_path"]
    df = pd.read_csv(csv_path)

    # 2. 包含未來資訊的指標 => drop    
    leakage_blocklist = cfg["leakage_blocklist"]
    assert leakage_blocklist is not None
    df = df.drop(columns=[col for col in leakage_blocklist])

    # 3. 根據mode 回傳加上 [y_reg, y_cls]
    df = create_labels_adaptive(df,
                                horizon=1,
                                mode=cfg["labels"]["mode"],
                                k_vol=cfg["labels"]["k_vol"],
                                vol_window=cfg["labels"]["vol_window"])

    # 4. 依時間切資料
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(None)  # 去掉時區
    start_date, end_date = cfg["start_date"], cfg["end_date"]
    df = df[df["datetime"] >= pd.to_datetime(start_date)]
    df = df[df["datetime"] <= pd.to_datetime(end_date)]

    # 5. 取出feature cols
    df_feat  = df.drop(columns=["label", "datetime", "timestamp","y_cls", "y_reg"])
    df_feat = df_feat.fillna(0).astype(float)

    # 6. 取出label cols
    y_cls, y_reg = df["y_cls"], df["y_reg"]
    return df, df_feat, y_cls, y_reg



from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
def scalar_data(df_feat, scalar_type: str = "StandardScaler"):
    
    # 0. 挑選scalar:
    scalar = None
    if scalar_type == "StandardScaler":
        scalar = StandardScaler()
    elif scalar_type == "RobustScaler":
        scalar = RobustScaler(with_centering=False)
    elif scalar_type == "MinMaxScaler":
        scalar = MinMaxScaler()
    else:
        scalar = StandardScaler()

    # 1. 對每個特徵標準化（沿時間軸）
    X_scaled = scalar.fit_transform(df_feat)
    X_scaled = pd.DataFrame(X_scaled, columns=df_feat.columns)

    # 2. 再對不同 feature 標準化（避免均值/幅度影響距離）
    X_t = X_scaled.T    # Transpose：每個 row 是一個指標在所有樣本上的表現
    X_t_scaled = StandardScaler().fit_transform(X_t)
    
    return X_scaled, X_t





def topk_features_per_cluster_cls(
    X_scaled: pd.DataFrame,
    feature_clusters: pd.DataFrame,
    y_cls: pd.Series,
    best_k:int = 10,
    topk: int = 10,
    save_dir: str | None = None,
) -> pd.DataFrame:
    """
    依 ANOVA F-score（分類 y_cls）在每個 cluster 內挑 Top-k features。
    參數：
      - X_scaled: 標準化後的特徵矩陣 DataFrame，shape=[n_samples, n_features]
      - feature_clusters: DataFrame(columns=["feature","cluster"]) 對照表
      - y_cls: 類別標籤（含 -1 表未定義）
      - topk: 每群取前幾名
      - save_dir: 若提供路徑，會輸出 CSV 檔 top{topk}_by_cls_per_cluster.csv
    回傳：
      - top_cls_df: 各群排名後的 DataFrame（含 feature, cluster, f_score, rank_cls）
    """
    import numpy as np
    import pandas as pd
    import warnings, os
    from sklearn.feature_selection import f_classif

    # 對齊可用特徵集合，避免名稱不交集造成對不齊
    feat_ok = X_scaled.columns.intersection(feature_clusters["feature"])
    X = X_scaled[feat_ok]
    fc = feature_clusters[feature_clusters["feature"].isin(feat_ok)].copy()

    # 過濾掉未定義/NaN 的樣本
    y = y_cls.to_numpy()
    mask = (y != -1) & np.isfinite(y)
    X_for_cls = X.loc[mask].values
    y_for_cls = y[mask].astype(int)

    # 計算每個特徵的 ANOVA F-score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_vals, _ = f_classif(X_for_cls, y_for_cls)

    feat_f = pd.DataFrame({"feature": feat_ok, "f_score": f_vals}).fillna(0.0)

    # 併回群資訊並在群內排名
    feat_scores = fc.merge(feat_f, on="feature", how="left").fillna({"f_score": 0.0})
    feat_scores["rank_cls"] = (
        feat_scores.groupby("cluster")["f_score"]
        .rank(method="first", ascending=False)
    )

    top_cls_df = (
        feat_scores[feat_scores["rank_cls"] <= topk]
        # .sort_values(["cluster", "rank_cls"])
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )

    # 輸出 CSV（可選）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        top_cls_df.to_csv(
            os.path.join(save_dir, f"cluster_{best_k}_top{topk}_by_cls_per_cluster.csv"), index=False
        )

    return top_cls_df
