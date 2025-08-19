# y_corr.py
import yaml, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from feature_utils import load_labeled_data, scalar_data, topk_features_per_cluster_cls

# 1) 讀data, hyp
with open(r"train/feature_selection/feature_selection.yaml",encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

df, df_feat, y_cls, y_reg = load_labeled_data(cfg=cfg)

# 2) 、標準化
scalar_type=str(cfg["scalar"])
X_scaled, X_t_scaled = scalar_data(df_feat=df_feat, scalar_type=scalar_type)


# 3) 根據 k_means 結果分群
best_k = cfg["best_k"]
kmeans = KMeans(n_clusters=best_k, random_state=0, n_init="auto")
cluster_labels = kmeans.fit_predict(X_t_scaled)

# 4) 建立特徵群對照表
feature_clusters = pd.DataFrame({
    "feature": X_scaled.columns,
    "cluster": cluster_labels
})

# 以原始 df 取 y（避免上方 y_cls 變成 ndarray 之後不好用）
top_cls_df = topk_features_per_cluster_cls(
    X_scaled=X_scaled,
    feature_clusters=feature_clusters,
    best_k=best_k,
    y_cls=df["y_cls"],
    topk=cfg["top_n"],
    save_dir=r"train/feature_selection/outputs",  # 你上面已經設定好的目錄；不想存檔可改成 None
)
print(top_cls_df.head(20))




# # print(feature_clusters)

# # 5) 建立每群的平均特徵值（作為代表）
# cluster_features = {}
# for c in sorted(feature_clusters["cluster"].unique()):
#     members = feature_clusters[feature_clusters["cluster"] == c]["feature"].tolist()
#     cluster_features[f"cluster_{c}"] = X_scaled[members].mean(axis=1)
# df_cluster = pd.DataFrame(cluster_features)



# valid_mask = np.isfinite(y_reg.values)        # 去掉 y_reg 的 NaN（最後一筆）
# # 如果你也想把分類未定義(-1)的樣本拿掉，打開下一行：
# # valid_mask &= (y_cls.values != -1)

# # 取出對齊後的資料
# Xc = df_cluster.loc[valid_mask].copy()
# Xc.index = y_reg[valid_mask].index  # 強制同步 index

# yr = y_reg[valid_mask].astype(float)
# yc = y_cls[valid_mask].astype(int)
# assert (Xc.index == yr.index).all()
# assert (Xc.index == yc.index).all()

# # ---- (可選) 先檢查一下是否還有 NaN ----
# assert np.isfinite(Xc.values).all(), "Xc still has NaN/inf"
# assert np.isfinite(yr.values).all(), "yr still has NaN/inf"

# # Step 7-1: 對y_cls標籤做 ANOVA
# from sklearn.feature_selection import f_classif
# X_cls = df_cluster.values
# y_cls = y_cls.values.astype(int)
# f_vals, p_vals = f_classif(X_cls, y_cls)
# f_score_df = pd.DataFrame({
#     "cluster": df_cluster.columns.tolist(),
#     "f_score": f_vals,
#     "p_value": p_vals
# }).sort_values("f_score", ascending=False)
# print(f"Class:\n{f_score_df}\n")


# # # ---- Step 7-2: 對 y_reg 做 Pearson correlation（逐欄）----
# # from scipy.stats import pearsonr  # 比 np.corrcoef 更穩健，會忽略常數向量與回傳 p-value
# # print(f"Regression:")
# # for c in Xc.columns:
# #     r, p = pearsonr(Xc[c].values, yr.values)
# #     print(f"{c}: Pearson r = {r:.6f}, p = {p:.3e}")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os 
# save_dir = fr"train/feature_selection/plots/{best_k}_group_y_corr_analysis"
# os.makedirs(save_dir, exist_ok=True)

# # 1. ANOVA F-score 排序圖
# plt.figure(figsize=(6,4))
# sns.barplot(
#     data=f_score_df,
#     x="f_score",
#     y="cluster",
#     hue="cluster",       # 加這行避免未來警告
#     palette="viridis",
#     legend=False         # 不顯示圖例
# )
# plt.title(f"{str(scalar_type)} ANOVA F-score by Cluster")
# plt.xlabel("F-score")
# plt.ylabel("Cluster")
# plt.tight_layout()
# save_path = os.path.join(save_dir,f"{str(scalar_type)}ANOVA F-score by Cluster.png")
# plt.savefig(save_path, dpi = 300)

# import math
# # 2. 各 cluster 的 boxplot（對應 y_cls）
# cols = list(df_cluster.columns)     # K 個 cluster 欄位
# K = len(cols)
# ncols = min(3, K)                   # 每列最多放 3 張
# nrows = math.ceil(K / ncols)

# # 每個子圖大約 5x4 inches，依列欄數動態放大
# fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
# fig.suptitle(f"{scalar_type} Cluster_boxplot (K={K})", fontsize=16)
# axes = axes.flatten()

# df_plot = df_cluster.copy()
# df_plot["y_cls"] = y_cls

# for i, col in enumerate(cols):
#     ax = axes[i]
#     sns.boxplot(data=df_plot, x="y_cls", y=col, order=[0,1,2], ax=ax)
#     ax.set_title(f"{col} vs y_cls")
#     ax.set_xlabel("y_cls"); ax.set_ylabel(col)

# # 把多餘子圖關掉
# for j in range(K, len(axes)):
#     axes[j].axis("off")

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# save_path = os.path.join(save_dir, f"{scalar_type}Cluster_boxplot_K{K}.png")
# plt.savefig(save_path, dpi=300)
# plt.close(fig)

# # # 3. 散佈圖 + 回歸線
# # top_corr = "cluster_1"  # 你覺得最強的
# # plt.figure(figsize=(6,4))
# # sns.regplot(x=Xc[top_corr], y=yr, scatter_kws={"s":5}, line_kws={"color":"red"})
# # plt.title(f"{top_corr} vs y_reg")
# # plt.xlabel(top_corr); plt.ylabel("y_reg")
# # plt.tight_layout()
# # plt.show()


