from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    prange = None
    _NUMBA_AVAILABLE = False


def _msm_cost(a: float, b: float, c_val: float, cost: float) -> float:
    """
    1. 說明: 計算 MSM 中的拆分/合併成本。
    2. inputs:
       - a: 新點（被插入或移除的點值）
       - b: 其前一個時間點的值
       - c_val: 另一個鄰近點（來自另一條序列）
       - cost: 固定成本參數 c
    3. return:
       - MSM 中的拆分/合併成本
    """
    if (b <= a <= c_val) or (c_val <= a <= b):
        return cost
    return cost + min(abs(a - b), abs(a - c_val))


def _msm_distance_py(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """
    MSM 距離的純 Python 版本（供 numba 不可用時 fallback）。
    """
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        raise ValueError("MSM distance 需要非空的時間序列")

    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        prev_x = x[i - 2] if i > 1 else x[i - 1]
        dp[i, 0] = dp[i - 1, 0] + _msm_cost(x[i - 1], prev_x, y[0], c)
    for j in range(1, m + 1):
        prev_y = y[j - 2] if j > 1 else y[j - 1]
        dp[0, j] = dp[0, j - 1] + _msm_cost(y[j - 1], x[0], prev_y, c)

    for i in range(1, n + 1):
        prev_x = x[i - 2] if i > 1 else x[i - 1]
        for j in range(1, m + 1):
            prev_y = y[j - 2] if j > 1 else y[j - 1]
            cost_move = dp[i - 1, j - 1] + abs(x[i - 1] - y[j - 1])
            cost_split = dp[i - 1, j] + _msm_cost(x[i - 1], prev_x, y[j - 1], c)
            cost_merge = dp[i, j - 1] + _msm_cost(y[j - 1], x[i - 1], prev_y, c)
            dp[i, j] = min(cost_move, cost_split, cost_merge)

    return float(dp[n, m])


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _msm_cost_nb(a: float, b: float, c_val: float, cost: float) -> float:
        if (b <= a <= c_val) or (c_val <= a <= b):
            return cost
        return cost + min(abs(a - b), abs(a - c_val))

    @njit(cache=True)
    def _msm_distance_nb(x: np.ndarray, y: np.ndarray, c: float) -> float:
        n = x.shape[0]
        m = y.shape[0]
        if n == 0 or m == 0:
            raise ValueError("MSM distance 需要非空的時間序列")

        dp = np.zeros((n + 1, m + 1), dtype=np.float32)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            prev_x = x[i - 2] if i > 1 else x[i - 1]
            dp[i, 0] = dp[i - 1, 0] + _msm_cost_nb(x[i - 1], prev_x, y[0], c)
        for j in range(1, m + 1):
            prev_y = y[j - 2] if j > 1 else y[j - 1]
            dp[0, j] = dp[0, j - 1] + _msm_cost_nb(y[j - 1], x[0], prev_y, c)

        for i in range(1, n + 1):
            prev_x = x[i - 2] if i > 1 else x[i - 1]
            for j in range(1, m + 1):
                prev_y = y[j - 2] if j > 1 else y[j - 1]
                cost_move = dp[i - 1, j - 1] + abs(x[i - 1] - y[j - 1])
                cost_split = dp[i - 1, j] + _msm_cost_nb(x[i - 1], prev_x, y[j - 1], c)
                cost_merge = dp[i, j - 1] + _msm_cost_nb(y[j - 1], x[i - 1], prev_y, c)
                dp[i, j] = min(cost_move, cost_split, cost_merge)

        return float(dp[n, m])

    @njit(parallel=True, fastmath=True, cache=True)
    def _assign_clusters_parallel_nb(X: np.ndarray, centroids: np.ndarray, cost: float):
        """
        使用 numba 在多核上並行計算樣本到各 centroid 的 MSM 距離。
        傳回每筆樣本的最佳群標籤與距離。
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)
        distances = np.zeros(n_samples, dtype=np.float32)
        for i in prange(n_samples):
            min_dist = np.inf
            best_label = -1
            for k in range(n_clusters):
                d = _msm_distance_nb(X[i], centroids[k], cost)
                if d < min_dist:
                    min_dist = d
                    best_label = k
            labels[i] = best_label
            distances[i] = min_dist
        return labels, distances
else:
    _msm_cost_nb = None  # type: ignore
    _msm_distance_nb = None  # type: ignore
    _assign_clusters_parallel_nb = None  # type: ignore


def msm_distance(x: np.ndarray, y: np.ndarray, c: float = 0.1, use_numba: bool = False) -> float:
    """
    1. 說明: 計算兩條時間序列之間的 MSM (Move-Split-Merge) 距離。
    2. inputs:
       - x: shape = (T,), 第一條時間序列
       - y: shape = (T,), 第二條時間序列
       - c: Split/Merge 的固定成本參數
       - use_numba: 若為 True 且已安裝 numba，會使用編譯版以加速
    3. return:
       - MSM 距離 (float)
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if use_numba and _msm_distance_nb is not None:
        return float(_msm_distance_nb(x_arr, y_arr, c))
    return _msm_distance_py(x_arr, y_arr, c)


def _progress(it: Iterable, enable: bool, desc: str):
    """
    1. 說明: 視需要包裝 tqdm 進度條。
    2. inputs:
       - it: 可迭代物件
       - enable: 是否啟用 tqdm
       - desc: tqdm 的說明文字
    3. return:
       - 原始或 tqdm 包裝的 iterable
    """
    if enable and tqdm is not None:
        return tqdm(it, desc=desc)
    return it


@dataclass
class TimeSeriesKMeansMSM:
    """
    1. 說明: 使用 MSM 距離對等長時間序列進行 k-means 分群。
    2. inputs:
       - n_clusters: 群數 k
       - msm_cost: MSM 中 Split/Merge 的成本 c
       - max_iter: 單次 k-means 的最大迭代次數
       - n_init: 不同隨機初始化的重複次數
       - tol: 收斂門檻 (centroid 變化量)
       - random_state: 隨機種子
       - use_tqdm: 是否顯示 tqdm 進度條（需安裝 tqdm）
       - use_numba: 是否啟用 numba (若安裝) 來加速 MSM 距離
    3. return:
       - 物件本身，包含訓練後的 centroids_、labels_ 等屬性
    """

    n_clusters: int
    msm_cost: float = 0.1
    max_iter: int = 100
    n_init: int = 3
    tol: float = 1e-3
    random_state: Optional[int] = None
    use_tqdm: bool = False
    use_numba: bool = True

    cluster_centers_: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None
    inertia_: Optional[float] = None
    _use_numba_backend: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._use_numba_backend = bool(self.use_numba and _msm_distance_nb is not None)
        if self.use_numba and not self._use_numba_backend:
            print("[WARN] numba 未安裝，MSM 距離將使用純 Python 版本")

    def fit(self, X: np.ndarray) -> "TimeSeriesKMeansMSM":
        """
        1. 說明: 對輸入時間序列執行 MSM k-means 分群。
        2. inputs:
           - X: shape = (n_samples, seq_len) 的時間序列矩陣
        3. return:
           - 回傳已訓練好的物件本身
        """
        X = self._validate_input(X)
        rng = np.random.default_rng(self.random_state)
        best_inertia = math.inf
        best_centroids = None
        best_labels = None

        for _ in _progress(range(self.n_init), self.use_tqdm, "init"):
            centroids = self._init_centroids(X, rng)
            for _ in _progress(range(self.max_iter), self.use_tqdm, "iter"):
                labels, distances = self._assign_clusters(X, centroids)
                inertia = float(np.sum(distances ** 2))
                new_centroids = self._compute_centroids(X, labels, centroids, rng)
                shift = self._max_centroid_shift(centroids, new_centroids)
                centroids = new_centroids
                if shift <= self.tol:
                    break
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        1. 說明: 對新的時間序列資料指派群集標籤。
        2. inputs:
           - X: shape = (n_samples, seq_len)
        3. return:
           - 每筆樣本的群集標籤 (shape = (n_samples,))
        """
        if self.cluster_centers_ is None:
            raise ValueError("模型尚未訓練，請先呼叫 fit()")
        X = self._validate_input(X)
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        1. 說明: 先訓練再回傳訓練資料的群集標籤。
        2. inputs:
           - X: shape = (n_samples, seq_len)
        3. return:
           - 每筆樣本的群集標籤 (shape = (n_samples,))
        """
        return self.fit(X).labels_

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        1. 說明: 驗證輸入矩陣並轉成 float numpy array。
        2. inputs:
           - X: 任意可轉成 (n_samples, seq_len) 的陣列
        3. return:
           - 轉換後的 numpy ndarray
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X 必須是 shape = (n_samples, seq_len) 的 2D 陣列")
        n_samples, seq_len = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("樣本數需大於等於群數")
        if seq_len == 0:
            raise ValueError("序列長度需大於 0")
        return X

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        1. 說明: 隨機初始化群中心。
        2. inputs:
           - X: 訓練資料
           - rng: numpy 隨機數產生器
        3. return:
           - 初始 centroids 陣列
        """
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        1. 說明: 以 MSM 距離將樣本指派至最近的群中心。
        2. inputs:
           - X: 樣本矩陣
           - centroids: 目前群中心
        3. return:
           - 樣本標籤與對應距離
        """
        if self._use_numba_backend and _assign_clusters_parallel_nb is not None:
            # 全部移入 numba 並行執行，讓每個樣本距離計算在多核上並行
            labels, distances = _assign_clusters_parallel_nb(
                np.asarray(X, dtype=np.float32), np.asarray(centroids, dtype=np.float32), float(self.msm_cost)
            )
            return labels.astype(int), distances.astype(float)

        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        distances = np.zeros(n_samples, dtype=float)
        iterable = _progress(range(n_samples), self.use_tqdm, "assign")
        for i in iterable:
            dists = np.array(
                [
                    msm_distance(X[i], centroid, self.msm_cost, use_numba=self._use_numba_backend)
                    for centroid in centroids
                ]
            )
            labels[i] = int(np.argmin(dists))
            distances[i] = float(dists[labels[i]])
        return labels, distances

    def _compute_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        old_centroids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        1. 說明: 依據指派結果更新群中心。
        2. inputs:
           - X: 樣本矩陣
           - labels: 群集標籤
           - old_centroids: 舊的群中心
           - rng: 隨機數產生器，用來處理空群
        3. return:
           - 更新後的群中心
        """
        new_centroids = np.zeros_like(old_centroids)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if members.size == 0:
                new_centroids[k] = X[rng.integers(0, X.shape[0])]
            else:
                new_centroids[k] = np.mean(members, axis=0, dtype=np.float32)
        return new_centroids

    def _max_centroid_shift(self, centroids: np.ndarray, new_centroids: np.ndarray) -> float:
        """
        1. 說明: 計算群中心更新後的最大 L2 變化量。
        2. inputs:
           - centroids: 舊群中心
           - new_centroids: 新群中心
        3. return:
           - 最大 L2 shift
        """
        diffs = new_centroids - centroids
        norms = np.linalg.norm(diffs, axis=1)
        return float(np.max(norms))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 60)
    waves = []
    for phase in [0.0, 0.8, 1.6]:
        for _ in range(15):
            waves.append(np.sin(t + phase) + 0.05 * rng.normal(size=t.shape))
    X_demo = np.vstack(waves)
    model = TimeSeriesKMeansMSM(n_clusters=3, msm_cost=0.1, max_iter=30, n_init=2, random_state=0)
    labels = model.fit_predict(X_demo)
    unique, counts = np.unique(labels, return_counts=True)
    print("Inertia:", model.inertia_)
    print("Cluster sizes:", dict(zip(unique.tolist(), counts.tolist())))
