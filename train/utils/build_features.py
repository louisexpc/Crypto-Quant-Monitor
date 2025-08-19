import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas_ta import Strategy
from pathlib import Path

all_feat = [
    "datetime", "timestamp", "open", "high", "low", "close", "volume", "ABER_ZG_5_15", "ABER_SG_5_15", "ABER_XG_5_15",
    "ABER_ATR_5_15", "ACCBL_20", "ACCBM_20", "ACCBU_20", "AD", "ADOSC_3_10", "ADX_14", "DMP_14", "DMN_14", "ALMA_10_6.0_0.85",
    "AMATe_LR_8_21_2", "AMATe_SR_8_21_2", "AO_5_34", "OBV", "OBV_min_2", "OBV_max_2", "OBVe_4", "OBVe_12", "AOBV_LR_2",
    "AOBV_SR_2", "APO_12_26", "AROOND_14", "AROONU_14", "AROONOSC_14", "ATRr_14", "BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0",
    "BBB_5_2.0", "BBP_5_2.0", "BIAS_SMA_26", "BOP", "AR_26", "BR_26", "CCI_14_0.015", "CDL_2CROWS", "CDL_3BLACKCROWS",
    "CDL_3INSIDE", "CDL_3LINESTRIKE", "CDL_3OUTSIDE", "CDL_3STARSINSOUTH", "CDL_3WHITESOLDIERS", "CDL_ABANDONEDBABY",
    "CDL_ADVANCEBLOCK", "CDL_BELTHOLD", "CDL_BREAKAWAY", "CDL_CLOSINGMARUBOZU", "CDL_CONCEALBABYSWALL",
    "CDL_COUNTERATTACK", "CDL_DARKCLOUDCOVER", "CDL_DOJI_10_0.1", "CDL_DOJISTAR", "CDL_DRAGONFLYDOJI", "CDL_ENGULFING",
    "CDL_EVENINGDOJISTAR", "CDL_EVENINGSTAR", "CDL_GAPSIDESIDEWHITE", "CDL_GRAVESTONEDOJI", "CDL_HAMMER",
    "CDL_HANGINGMAN", "CDL_HARAMI", "CDL_HARAMICROSS", "CDL_HIGHWAVE", "CDL_HIKKAKE", "CDL_HIKKAKEMOD",
    "CDL_HOMINGPIGEON", "CDL_IDENTICAL3CROWS", "CDL_INNECK", "CDL_INSIDE", "CDL_INVERTEDHAMMER", "CDL_KICKING",
    "CDL_KICKINGBYLENGTH", "CDL_LADDERBOTTOM", "CDL_LONGLEGGEDDOJI", "CDL_LONGLINE", "CDL_MARUBOZU",
    "CDL_MATCHINGLOW", "CDL_MATHOLD", "CDL_MORNINGDOJISTAR", "CDL_MORNINGSTAR", "CDL_ONNECK", "CDL_PIERCING",
    "CDL_RICKSHAWMAN", "CDL_RISEFALL3METHODS", "CDL_SEPARATINGLINES", "CDL_SHOOTINGSTAR", "CDL_SHORTLINE",
    "CDL_SPINNINGTOP", "CDL_STALLEDPATTERN", "CDL_STICKSANDWICH", "CDL_TAKURI", "CDL_TASUKIGAP", "CDL_THRUSTING",
    "CDL_TRISTAR", "CDL_UNIQUE3RIVER", "CDL_UPSIDEGAP2CROWS", "CDL_XSIDEGAP3METHODS", "open_Z_30_1", "high_Z_30_1",
    "low_Z_30_1", "close_Z_30_1", "CFO_9", "CG_10", "CHOP_14_1_100", "CKSPl_10_3_20", "CKSPs_10_3_20", "CMF_20", "CMO_14",
    "COPC_11_14_10", "CTI_12", "LDECAY_5", "DEC_1", "DEMA_10", "DCL_20_20", "DCM_20_20", "DCU_20_20", "DPO_20",
    "EBSW_40_10", "EFI_13", "EMA_10", "ENTP_10", "ER_10", "BULLP_13", "BEARP_13", "FISHERT_9_1", "FISHERTs_9_1", "FWMA_10",
    "HA_open", "HA_high", "HA_low", "HA_close", "HILO_13_21", "HL2", "HLC3", "HMA_10", "HWM", "HWU", "HWL",
    "HWMA_0.2_0.1_0.1", "ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26", "INC_1", "INERTIA_20_14", "JMA_7_0",
    "KAMA_10_2_30", "KCLe_20_2", "KCBe_20_2", "KCUe_20_2", "K_9_3", "D_9_3", "J_9_3", "KST_10_15_20_30_10_10_10_15", "KSTs_9",
    "KURT_30", "KVO_34_55_13", "KVOs_34_55_13", "LR_14", "LOGRET_1", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "MAD_30", "MASSI_9_25", "MCGD_10", "MEDIAN_30", "MFI_14", "MIDPOINT_2", "MIDPRICE_2", "MOM_10", "NATR_14", "NVI_1",
    "OHLC4", "PDIST", "PCTRET_1", "PGO_14", "PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9", "PSARaf_0.02_0.2",
    "PSARr_0.02_0.2", "PSL_12", "PVI_1", "PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9", "PVOL", "PVR", "PVT", "PWMA_10",
    "QQE_14_5_4.236", "QQE_14_5_4.236_RSIMA", "QS_10", "QTL_30_0.5", "RMA_10", "ROC_10", "RSI_14", "RSX_14", "RVGI_14_4",
    "RVGIs_14_4", "RVI_14", "SINWMA_14", "SKEW_30", "SLOPE_1", "SMA_10", "SMI_5_20_5", "SMIs_5_20_5", "SMIo_5_20_5",
    "SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO", "SQZPRO_20_2.0_20_2_1.5_1", "SQZPRO_ON_WIDE",
    "SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO", "SSF_10_2", "STC_10_12_26_0.5",
    "STCmacd_10_12_26_0.5", "STCstoch_10_12_26_0.5", "STDEV_30", "STOCHk_14_3_3", "STOCHd_14_3_3",
    "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3", "SUPERT_7_3.0", "SUPERTd_7_3.0", "SWMA_10", "T3_10_0.7", "TEMA_10",
    "THERMO_20_2_0.5", "THERMOma_20_2_0.5", "THERMOl_20_2_0.5", "THERMOs_20_2_0.5", "TOS_STDEVALL_LR",
    "TOS_STDEVALL_L_1", "TOS_STDEVALL_U_1", "TOS_STDEVALL_L_2", "TOS_STDEVALL_U_2", "TOS_STDEVALL_L_3",
    "TOS_STDEVALL_U_3", "TRIMA_10", "TRIX_30_9", "TRUERANGE_1", "TSI_13_25_13", "TSIs_13_25_13", "TTM_TRND_6", "UI_14",
    "UO_7_14_28", "VAR_30", "VHF_28", "VIDYA_14", "VTXP_14", "VTXM_14", "VWAP_D", "VWMA_10", "WCP", "WILLR_14", "WMA_10",
    "ZL_EMA_10", "ZS_30"
]


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




# def _to_utc_index(idx) -> pd.DatetimeIndex:
#     """把任何 DatetimeIndex 統一成 tz-aware UTC。"""
#     idx = pd.DatetimeIndex(idx)
#     if idx.tz is None:
#         # 原始為 tz-naive，直接視為 UTC
#         idx = idx.tz_localize("UTC")
#     else:
#         # 已有時區，一律轉成 UTC
#         idx = idx.tz_convert("UTC")
#     return idx

# def build_features_and_label_runtime(
#     df_base: pd.DataFrame,
#     feat_parquet_path: str | None = None,
#     feat_df: pd.DataFrame | None = None,
#     horizon: int = 1,
#     ret_kind: str = "logret",
# ):
#     """
#     將 runtime 產生的特徵（parquet 或 DF）與原始 OHLCV（df_base）對齊，
#     產生二分類標籤：0=down / 1=up。
#     - 內部會統一 index 到 tz-aware UTC
#     - 用「交集」對齊，避免 KeyError
#     """

#     # 1) 讀特徵
#     if feat_df is None:
#         if not feat_parquet_path or not Path(feat_parquet_path).exists():
#             raise FileNotFoundError(f"特徵檔不存在：{feat_parquet_path}")
#         feat = pd.read_parquet(feat_parquet_path)
#     else:
#         feat = feat_df.copy()

#     # 2) 統一 index（tz-aware UTC）並排序
#     dfb = df_base.copy()
#     feat.index = _to_utc_index(feat.index)
#     dfb.index  = _to_utc_index(dfb.index)
#     feat = feat.sort_index()
#     dfb  = dfb.sort_index()

#     # 3) 取「交集」對齊（避免用 .loc[dfb.index] 觸發 KeyError）
#     common_idx = dfb.index.intersection(feat.index)
#     if len(common_idx) == 0:
#         raise ValueError("對齊後沒有任何共同時間戳（可能是時區或頻率不一致）。")

#     # 若你希望檢查覆蓋率（避免交集太小）
#     cover_ratio = len(common_idx) / min(len(dfb), len(feat))
#     if cover_ratio < 0.5:
#         print(f"[WARN] 對齊覆蓋率偏低：{cover_ratio:.2%}（df_base={len(dfb)}, feat={len(feat)}, common={len(common_idx)}）")

#     dfb  = dfb.loc[common_idx]
#     feat = feat.loc[common_idx]

#     # ===============================
#     # raw OHLCV 特徵（與指標一致，全部 shift(1)）
#     # ===============================
#     ohlcv_cols = ["open", "high", "low", "close", "volume"]
#     if not set(ohlcv_cols).issubset(dfb.columns):
#         raise KeyError(f"df_base 缺少 OHLCV 欄位：{ohlcv_cols}")

#     ohlcv = dfb[ohlcv_cols].copy().shift(1)  # ★ 關鍵：與指標一致回移一根
#     # （可選）避免名稱混淆，幫 OHLCV 加個前綴
#     # ohlcv = ohlcv.add_prefix("ohlcv_")

#     # feat = pd.concat([ohlcv, feat], axis=1)

#     # 4) 產生標籤（0/1）
#     if "close" not in dfb.columns:
#         raise KeyError("df_base 缺少 close 欄位，無法產生標籤。")

#     if ret_kind == "logret":
#         ret = np.log(dfb["close"].shift(-horizon) / dfb["close"])
#     else:
#         ret = (dfb["close"].shift(-horizon) - dfb["close"]) / dfb["close"]

#     y = (ret > 0).astype(int).rename("label")

#     # 5) 合併 + 清理
#     out = feat.join(y, how="inner")
#     out = out.replace([np.inf, -np.inf], np.nan).dropna()

#     # 6) 回傳 X, y（符合你 pipeline 介面）
#     return out.drop(columns=["label"]), out["label"]



def _to_utc_index(idx, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)  # ← 讓呼叫端決定
    return idx.tz_convert("UTC")

def build_features_and_label_runtime(
    df_base: pd.DataFrame,
    feat_parquet_path: str | None = None,
    feat_df: pd.DataFrame | None = None,
    horizon: int = 1,
    ret_kind: str = "logret",
    assume_tz: str = "UTC",
    enforce_grid: bool = True,
    lag_features_by: int = 0,
    # --- 新增參數 ---
    task_type: str = "classification",   # "classification" | "regression"
    cls_threshold: float = 0.0,          # 分類門檻；回歸無效
):
    """
    對齊你的原版行為（UTC/網格/shift），但：
    - regression：y 為連續報酬（未來 horizon 的 log 或 simple）
    - classification：y 為二值 (ret > cls_threshold)
    """

    # 1) 讀特徵
    if feat_df is None:
        if not feat_parquet_path or not Path(feat_parquet_path).exists():
            raise FileNotFoundError(f"特徵檔不存在：{feat_parquet_path}")
        feat = pd.read_parquet(feat_parquet_path)
    else:
        feat = feat_df.copy()

    # 2) 時區與排序（統一 UTC）
    dfb  = df_base.copy()
    feat.index = _to_utc_index(feat.index, assume_tz=assume_tz)
    dfb.index  = _to_utc_index(dfb.index,  assume_tz=assume_tz)
    feat = feat.sort_index()
    dfb  = dfb.sort_index()

    # 去重
    dfb  = dfb[~dfb.index.duplicated(keep="last")]
    feat = feat[~feat.index.duplicated(keep="last")]

    # 3) 強制完整 1H 網格（推薦）
    if enforce_grid:
        full_idx = pd.date_range(dfb.index.min(), dfb.index.max(), freq="1H", tz="UTC")
        dfb  = dfb.reindex(full_idx)
        feat = feat.reindex(full_idx)

    # 4) 對齊並做「只用過去」策略
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    if not set(ohlcv_cols).issubset(dfb.columns):
        raise KeyError(f"df_base 缺少 OHLCV 欄位：{ohlcv_cols}")

    # 防洩漏：當期輸入只看得到 t-1 的 OHLCV 與 features
    ohlcv = dfb[ohlcv_cols].shift(1)
    if lag_features_by > 0:
        feat = feat.shift(lag_features_by)

    X = pd.concat([ohlcv, feat], axis=1)

    # 5) 產生未來 horizon 報酬（建立於完整網格上）
    if ret_kind.lower() == "logret":
        ret = np.log(dfb["close"].shift(-horizon) / dfb["close"])
    else:
        ret = (dfb["close"].shift(-horizon) - dfb["close"]) / dfb["close"]
    ret.name = "target_cont"

    # 6) 依任務建立 y
    tt = str(task_type).lower()
    if tt == "regression":
        y = ret.rename("target")  # 連續值
    elif tt == "classification":
        y = (ret > float(cls_threshold)).astype(int).rename("label")
    else:
        raise ValueError(f"Unknown task_type={task_type}; use 'classification' or 'regression'.")

    # 7) 僅保留「特徵完整 + 目標存在（未來 close 存在）」的時點
    valid_now = X.notna().all(axis=1)
    valid_fut = dfb["close"].shift(-horizon).notna()
    keep_mask = valid_now & valid_fut
    X, y = X[keep_mask], y[keep_mask]

    # 8) 清理數值（inf→nan、刪 nan）並對齊 index
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.loc[y.index]  # 再次對齊

    # （可選）若擔心全 0 方差，這裡可做保底處理，但通常不用
    return X, y