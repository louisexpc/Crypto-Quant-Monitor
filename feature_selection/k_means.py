# === 必放在最前面：修正 pyclustering 對 numpy.warnings 的誤用 ===
import warnings
import numpy as np

# 有些 pyclustering 版本會錯用 numpy.warnings；我們把它指到標準 warnings
if not hasattr(np, "warnings"):
    np.warnings = warnings

# # （可選）為了避免 C-Core 不同環境造成的額外問題，先關掉 C-Core
# import os
# os.environ["PYCLUSTERING_PACKAGE"] = "pure"   # 強制使用純 Python 實作

import collections
import collections.abc
# 修正舊版 pyclustering 引用錯誤
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable


import pandas as pd, numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os, random, yaml
from feature_utils import load_labeled_data, scalar_data
from sklearn.cluster import KMeans


# 1) 讀data, hyp
with open(r"train/feature_selection/feature_selection.yaml",encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


df, df_feat, y_cls, y_reg = load_labeled_data(cfg=cfg)
start_date, end_date = cfg["start_date"], cfg["end_date"]
save_dir = r"train/feature_selection/plots"
os.makedirs(save_dir, exist_ok=True)

# 2) 、標準化
X_scaled, X_t_scaled = scalar_data(df_feat=df_feat)

# 3) PCA 降到 2D (Principal Component Analysis = 主成分分析)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_t_scaled)


# 用於找出k群
# Step 4: KMeans 分群（你可照舊嘗試不同 k）

k_min, k_max = cfg["k_min"], cfg["k_max"]
range_k = range(k_min, k_max + 1)   # 包含 group_h




def run_kmeans_clustering(X_t_scaled: np.ndarray,
                          X_pca: np.ndarray,
                          save_dir: str,
                          range_k: range,
                          start_date=None,
                          end_date=None):
    """
    對每個 k 執行 KMeans 分群，儲存每張分群圖與 Silhouette 分數，並找出最佳 k。

    參數：
        - X_t_scaled: 每個 feature 的標準化向量（shape: n_features x n_samples）
        - X_pca: PCA 降到 2 維的特徵表示（shape: n_features x 2）
        - save_dir: 儲存圖片的資料夾
        - range_k: 嘗試的 k 值範圍（例如 range(2, 40+1)）
        - start_date, end_date: 若提供會自動加進儲存資料夾名稱
    回傳：
        - best_k: 最佳的 k 值（對應最大 Silhouette 分數）
        - sil_scores: 每個 k 對應的 Silhouette 分數 list
    """
    
    if start_date and end_date:
        save_dir = fr"{save_dir}/k_group_v2_{str(start_date)}_to_{str(end_date)}"
    os.makedirs(save_dir, exist_ok=True)

    sil_scores = []

    for k in tqdm(range_k):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X_t_scaled)
        score = silhouette_score(X_t_scaled, labels)
        sil_scores.append(score)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="nipy_spectral", s=50)

        # ===== 只標 centroid =====
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            center = X_pca[idx].mean(axis=0)  # 該群在 2D PCA 的中心
            ax.text(center[0], center[1], f"Cluster {c}", fontsize=9, weight="bold",
                    ha="center", va="center", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_title(f"K={k} | Silhouette={score:.4f}")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True)

        save_path = os.path.join(save_dir, f"group_{k}_{score:.4f}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # 儲存 silhouette 分數圖
    plt.plot(range_k, sil_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("KMeans Optimal Cluster Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"silhouette_scores.png"), dpi=200)
    plt.close()

    for idx, k in enumerate(range_k):
        print(f"group={k}_{sil_scores[idx]:.4f}")

    return sil_scores

# run_kmeans_clustering(X_t_scaled = X_t_scaled,
#                           X_pca = X_pca,
#                           save_dir = save_dir,
#                           range_k=range_k,
#                           start_date = start_date, 
#                           end_date=end_date)




from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# X_means
def run_xmeans_once(data_2d, k_min=2, k_max=30, tol=0.01):
    """
    data_2d: shape = [n_features, n_samples]，每列是一個「指標的整段向量」
    回傳: (labels, centers)；labels 是長度為 n_features 的群標籤
    """
    data_list = np.asarray(data_2d, dtype=float).tolist()
    init = kmeans_plusplus_initializer(data_list, k_min).initialize()
    xm = xmeans(data_list, init, kmax=k_max,
                criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
                tolerance=tol)
    xm.process()
    clusters = xm.get_clusters()
    # 轉成 label 向量
    labels = np.empty(len(data_list), dtype=int)
    for cid, idxs in enumerate(clusters):
        labels[idxs] = cid
    return labels, xm.get_centers()

def run_many_xmeans(data_2d, n_runs=100, k_min=2, k_max=30, tol=0.01):
    ks, all_labels = [], []
    for _ in trange(n_runs, desc="Running X-Means"):
        labels, _ = run_xmeans_once(data_2d, k_min, k_max, tol)
        ks.append(int(labels.max()+1))
        all_labels.append(labels)
    ks = np.array(ks)
    return ks, all_labels

def clt_confidence_interval(values, alpha=0.05):
    mu = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    n  = len(values)
    se = sd / np.sqrt(n)
    # 近似常態：mu ± 1.96*SE
    lo, hi = mu - 1.96*se, mu + 1.96*se
    return mu, sd, (lo, hi)

# random.seed(42); np.random.seed(42)
# n_runs = 100
# ks, all_labels = run_many_xmeans(X_t_scaled, n_runs=n_runs, k_min=k_min, k_max=k_max, tol=0.01)
# mu, sd, (lo, hi) = clt_confidence_interval(ks)

# print(f"[K distribution] mean={mu:.2f}, sd={sd:.2f}, 95% CI≈[{lo:.2f}, {hi:.2f}]")

# # --- 畫群數直方圖
# plt.figure(figsize=(6,4))
# plt.hist(ks, bins=np.arange(ks.min()-0.5, ks.max()+1.5, 1), edgecolor='k')
# plt.axvline(mu, linestyle='--', label=f"mean={mu:.2f}")
# plt.axvspan(lo, hi, alpha=0.2, label=f"95% CI")
# plt.xlabel("Number of clusters (K)"); plt.ylabel("Frequency")
# plt.title("X-Means K distribution over runs"); plt.legend(); plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"{n_runs}-k_distribution_mean={mu:.2f}.png"), dpi=150); plt.close()





