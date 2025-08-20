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

# build_feature.py
import numpy as np
import pandas as pd
from pathlib import Path


def _to_utc_index(idx, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)  # ← 讓呼叫端決定
    return idx.tz_convert("UTC")

def build_features_and_label(
    df_base: pd.DataFrame,
    feat_parquet_path: str | None = None,
    feat_df: pd.DataFrame | None = None,
    *,
    cfg
    ):
    """
    對齊你的原版行為（UTC/網格/shift），但：
    - regression：y 為連續報酬（未來 horizon 的 log 或 simple）
    - classification：y 為二值 (ret > cls_threshold)
    """

    # === 0) 時間設定 ===
    start_date = pd.Timestamp(cfg["cv"]["start_date"]).tz_localize("UTC")
    end_date   = pd.Timestamp(cfg["cv"]["end_date"]).tz_localize("UTC")
    freq = cfg["data"]["freq"]

    # === 1) 讀特徵 ===
    if feat_df is None:
        if not feat_parquet_path or not Path(feat_parquet_path).exists():
            raise FileNotFoundError(f"特徵檔不存在：{feat_parquet_path}")
        X = pd.read_parquet(feat_parquet_path)
    else:
        X = feat_df.copy()

    # === 2) 轉換時區與排序 ===
    dfb = df_base.copy()
    X.index   = _to_utc_index(X.index)
    dfb.index = _to_utc_index(dfb.index)
    X   = X.sort_index()
    dfb = dfb.sort_index()

    X   = X[~X.index.duplicated(keep="last")]
    dfb = dfb[~dfb.index.duplicated(keep="last")]

    # 3) 強制完整 1H 網格
    full_idx = pd.date_range(dfb.index.min(), dfb.index.max(), freq=str(freq), tz="UTC")
    dfb = dfb.reindex(full_idx)
    X   = X.reindex(full_idx)

    # === 4) 補充 horizon 與 shift(1) 防洩漏
    horizon = int(cfg["label"]["horizon"])
    if not {"open", "high", "low", "close", "volume"}.issubset(dfb.columns):
        raise KeyError("df_base 缺少 OHLCV 欄位")    
        
    # 5) 產生未來 horizon 報酬（建立於完整網格上）
    ret_kind = cfg["label"]["ret_kind"]

    close = dfb["close"]
    if ret_kind.lower() == "logret":
        ret = np.log(close.shift(-horizon) / close)
    else:
        ret = (close.shift(-horizon) - close) / close
    ret.name = "target"

    # 6) 依任務建立 y
    task_type = str(cfg["task"]["type"]).lower()
    if task_type == "regression":
        y = ret.rename("target")
    else:
        y = (ret > float(cfg["label"]["cls_threshold"])).astype(int).rename("label")

    # === 6) 去掉未來 close 為 nan 或特徵不完整的時點 ===
    valid_now = X.notna().all(axis=1)
    valid_fut = dfb["close"].shift(-horizon).notna()
    keep = valid_now & valid_fut
    X, y = X[keep], y[keep]

    # === 7) 篩選時間區間（最重要）===
    mask_range = (X.index >= start_date) & (X.index <= end_date)
    X = X.loc[mask_range]
    y = y.loc[mask_range]

    # === 8) 再次清理數值（NaN / inf） ===
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.loc[y.index]
    
    X, y = X.align(y, join="inner", axis=0)
    return X, y

# def build_features_and_label(
#     df_base: pd.DataFrame,
#     feat_parquet_path: str | None = None,
#     feat_df: pd.DataFrame | None = None,
#     *,
#     cfg
# ):
#     start_date = pd.Timestamp(cfg["cv"]["start_date"]).tz_localize("UTC")
#     end_date   = pd.Timestamp(cfg["cv"]["end_date"]).tz_localize("UTC")
#     freq = cfg["data"]["freq"]

#     # === 1) 讀特徵 ===
#     if feat_df is None:
#         if not feat_parquet_path or not Path(feat_parquet_path).exists():
#             raise FileNotFoundError(f"特徵檔不存在：{feat_parquet_path}")
#         X = pd.read_parquet(feat_parquet_path)
#     else:
#         X = feat_df.copy()
    
#     # === 2) 時間對齊 ===
#     dfb = df_base.copy()
#     X.index   = _to_utc_index(X.index)
#     dfb.index = _to_utc_index(dfb.index)
#     X   = X.sort_index().loc[~X.index.duplicated(keep="last")]
#     dfb = dfb.sort_index().loc[~dfb.index.duplicated(keep="last")]

#     # === 3) 強制網格化 ===
#     full_idx = pd.date_range(dfb.index.min(), dfb.index.max(), freq=str(freq), tz="UTC")
#     dfb = dfb.reindex(full_idx)
#     X   = X.reindex(full_idx)

#     # === 4) 防洩漏與報酬計算 ===
#     horizon = int(cfg["label"]["horizon"])
#     close = dfb["close"]

#     X = X.shift(1)  # 防止當期特徵洩漏未來報酬

#     ret = np.log(close.shift(-horizon) / close)

#     ret.name = "target"
#     # print("[debug] ret.describe():", ret.describe())
#     # print("[debug] ret.head(10)")

#     # === 5) 建立 y ===
#     task_type = str(cfg["task"]["type"]).lower()
#     if task_type == "regression":
#         y = ret.rename("target")
#     elif task_type == "classification":
#         y = (ret > float(cfg["label"]["cls_threshold"])).astype(int).rename("label")
#     else:
#         raise ValueError(f"Unknown task type: {task_type}")

#     # === 6) 清除無效時間點 ===
#     valid_now = X.notna().all(axis=1)
#     valid_fut = close.shift(-horizon).notna()
#     keep = valid_now & valid_fut
#     X, y = X[keep], y[keep]

#     # === 7) 篩選訓練區間 ===
#     mask_range = (X.index >= start_date) & (X.index <= end_date)
#     X, y = X.loc[mask_range], y.loc[mask_range]

#     # === 8) 清除 inf 與 NaN ===
#     X = X.replace([np.inf, -np.inf], np.nan).dropna()
#     y = y.loc[X.index].replace([np.inf, -np.inf], np.nan).dropna()
#     X = X.loc[y.index]

#     # === 9) 對齊 index 並輸出 ===
#     X, y = X.align(y, join="inner", axis=0)
#     return X, y




