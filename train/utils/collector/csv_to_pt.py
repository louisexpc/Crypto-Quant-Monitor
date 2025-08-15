# import pandas as pd
# from build_features import build_features_and_label

# df = pd.read_csv(
#     r"data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv",
#     parse_dates=['datetime'],
#     index_col="datetime"
# )

# X, y = build_features_and_label(df, horizon=1, flat_band_bps=10)
# import pyarrow as pa
# import pyarrow.parquet as pq

# df_out = X.copy()
# df_out['label'] = y
# table = pa.Table.from_pandas(df_out.reset_index(), preserve_index=False)
# pq.write_table(table, "btc_1h_features.parquet", compression="zstd", compression_level=3)
# import torch

# X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
# y_tensor = torch.tensor(y.to_numpy(), dtype=torch.int64)

# # 用 ns 級整數保存時間索引，先轉 Asia/Taipei 再去時區，避免之後 fold 錯位
# idx = pd.DatetimeIndex(df.index)
# if getattr(idx, "tz", None) is not None:
#     idx = idx.tz_convert("Asia/Taipei").tz_localize(None)
# ts_np = idx.view("int64")  # nanoseconds since epoch

# blob = {
#     "X": X_tensor,                    # [N, F] float32
#     "y": y_tensor,                    # [N]    int64
#     "ts": torch.from_numpy(ts_np),    # [N]    int64 (ns)
#     "cols": X.columns.to_list(),      # 特徵名稱
# }
# torch.save(blob, "btc_1h_features.pt")
# # torch.save({"X": X_tensor, "y": y_tensor}, "btc_1h_features.pt")

# # build_pt_from_csv.py
# import pandas as pd, numpy as np, torch
# from build_features import build_features_and_label

# csv_path = "data/ohlcv/mexc_swap_BTC-USDT-USDT_1h.csv"  # 你的原始CSV
# dt_col   = "datetime"  # 你給的欄位名
# tz = "Asia/Taipei"

# # 讀入：datetime 是 tz-aware（你原始例子就是 +08:00）
# df_raw = pd.read_csv(csv_path, parse_dates=[dt_col])
# # 若它已是字符串含 +08:00，下面這行會保留時區；若不是，可用 .dt.tz_localize(tz)
# if df_raw[dt_col].dt.tz is None:
#     df_raw[dt_col] = df_raw[dt_col].dt.tz_localize(tz)

# df_raw = df_raw.set_index(dt_col).sort_index()

# # 做特徵與標籤（你原本的流程）
# X, y = build_features_and_label(df_raw, horizon=1, flat_band_bps=10)

# # 整理：去 NaN（末端 horizon/shift 可能產生缺值）
# mask = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
# X = X.loc[mask]
# y = y.loc[mask]

# # 存成 .pt：ts 用 ISO 字串，feat_cols 用 list[str]
# blob = {
#     "X": torch.tensor(X.to_numpy(), dtype=torch.float32),
#     "y": torch.tensor(y.to_numpy(), dtype=torch.int64),
#     "ts": X.index.astype(str).tolist(),           # <-- 關鍵：用字串就不會 TYPEERROR
#     "feat_cols": X.columns.tolist(),              # 供 feature 過濾/抽樣用
# }
# torch.save(blob, "data/ohlcv_labeled/btc_1h_features.pt")
# print("saved to data/ohlcv_labeled/btc_1h_features.pt")


import pandas as pd

# df = pd.read_csv("data/ohlcv_labeled/indicators.csv")
# df.to_parquet("data/ohlcv/BTC_1h.parquet", index=False)
# df = pd.read_parquet(r"data/ohlcv_labeled/BTC_1h_features.parquet")
# print(df)

