# train/evaluation/reporters/regression_reporter.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np, matplotlib.pyplot as plt
from ._utils import _ensure_dir, _ensure_2d_prob
from sklearn.metrics import fbeta_score

class RegressionReporter:
    """
    1. 說明: 回歸任務可視化與摘要（散點/殘差/相關係數）與回歸→分類 Fβ 門檻掃描
    2. inputs: 於 __init__ 與各方法傳入
    3. return: plot_eval 回傳指標 dict；threshold_sweep 回傳最佳門檻/分數
    """
    def __init__(self, save_dir, prefix: str = ""):
        """
        1. 說明: 初始化 Reporter（設定輸出路徑、檔名前綴）
        2. inputs:
           - save_dir: 圖表輸出資料夾
           - prefix: 檔名前綴
        3. return: None
        """
        self.save_dir = _ensure_dir(save_dir)
        self.prefix = prefix or ""

    @staticmethod
    def _corr_rmse_mae_spearman(y_true, y_pred):
        """
        1. 說明: 計算 Pearson/Spearman 相關、RMSE、MAE；回傳清理後的 y_true/y_pred
        2. inputs:
           - y_true: ndarray-like
           - y_pred: ndarray-like
        3. return:
           - (pearson: float, rmse: float, mae: float, spearman: float, y_true: np.ndarray, y_pred: np.ndarray)
        """
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if y_true.size == 0:
            return 0.0, float("nan"), float("nan"), float("nan"), np.array([]), np.array([])

        err = y_pred - y_true
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))

        yt = y_true - y_true.mean()
        yp = y_pred - y_pred.mean()
        denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()))
        pearson = float((yt * yp).sum() / denom) if denom > 1e-12 else 0.0

        # Spearman：用平均名次的 Pearson
        def _rank_avg(a):
            a = np.asarray(a)
            n = a.size
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty(n, dtype=float)
            i = 0
            while i < n:
                j = i
                ai = a[order[i]]
                while j + 1 < n and a[order[j + 1]] == ai:
                    j += 1
                avg_rank = 0.5 * (i + j) + 1.0
                ranks[order[i:j + 1]] = avg_rank
                i = j + 1
            return ranks

        r1 = _rank_avg(y_true)
        r2 = _rank_avg(y_pred)
        r1c = r1 - r1.mean()
        r2c = r2 - r2.mean()
        denom_s = (np.sqrt((r1c**2).sum()) * np.sqrt((r2c**2).sum()))
        spearman = float((r1c * r2c).sum() / denom_s) if denom_s > 1e-12 else float("nan")

        return pearson, rmse, mae, spearman, y_true, y_pred

    def plot_eval(self, y_true, y_pred) -> Dict[str, float]:
        """
        1. 說明: 產生三張圖（散點、殘差直方、殘差 vs y_pred），並回傳摘要指標
        2. inputs:
           - y_true: [N] 真實數值
           - y_pred: [N] 預測數值
        3. return:
           - dict: {"pearson": r, "spearman": ρ, "rmse": rmse, "mae": mae}
        """
        r, rmse, mae, spr, y_true, y_pred = self._corr_rmse_mae_spearman(y_true, y_pred)
        if y_true.size == 0:
            print("[RegressionReporter] empty inputs after masking; skip.")
            return {"pearson": 0.0, "spearman": float("nan"), "rmse": float("nan"), "mae": float("nan")}

        err = y_pred - y_true

        # (1) y_true vs y_pred
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, s=6, alpha=0.6)
        lim_min = float(min(y_true.min(), y_pred.min()))
        lim_max = float(max(y_true.max(), y_pred.max()))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle='--', color='gray', linewidth=1.0)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.axis("equal")
        plt.xlabel("y_true"); plt.ylabel("y_pred")
        plt.title(f"y_true vs y_pred\nPearson r={r:.3f} | Spearman ρ={spr:.3f} | RMSE={rmse:.4g} | MAE={mae:.4g}")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_scatter.png", dpi=200)
        plt.close()

        # (2) 殘差直方圖
        plt.figure(figsize=(6, 4))
        abs_max = float(np.abs(err).max())
        plt.hist(err, bins=50)
        plt.xlim(-abs_max, abs_max)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.xlabel("Residual (y_pred - y_true)"); plt.ylabel("Count")
        plt.title("Residual Histogram")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_residual_hist.png", dpi=200)
        plt.close()

        # (3) 殘差 vs y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, err, s=6, alpha=0.6)
        plt.axhline(0.0, linewidth=1.0)
        plt.axvline(0.0, linestyle='--', color='gray', linewidth=1.0)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.xlabel("y_pred"); plt.ylabel("Residual")
        plt.title("Residual vs y_pred")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_residual_vs_pred.png", dpi=200)
        plt.close()

        return {"pearson": r, "spearman": spr, "rmse": rmse, "mae": mae}

    def threshold_sweep(
        self, y_true_reg, y_pred_reg, *, true_threshold: float = 0.0, beta: float = 0.5, grid_points: int = 101
    ) -> Dict[str, float]:
        """
        1. 說明: 把回歸輸出用門檻轉成二分類，掃描 Fβ 最佳門檻
        2. inputs:
           - y_true_reg: [N] 真實回歸標籤
           - y_pred_reg: [N] 預測回歸數值
           - true_threshold: 將 y_true_reg 二值化時的門檻
           - beta: Fβ 的 β
           - grid_points: 門檻掃描點數
        3. return:
           - dict: {"best_threshold": t, "best_fbeta": f}
        """
        y_true_reg = np.asarray(y_true_reg).reshape(-1)
        y_pred_reg = np.asarray(y_pred_reg).reshape(-1)
        mask = np.isfinite(y_true_reg) & np.isfinite(y_pred_reg)
        y_true_reg, y_pred_reg = y_true_reg[mask], y_pred_reg[mask]

        y_true_bin = (y_true_reg >= float(true_threshold)).astype(int)
        qs = np.linspace(0, 1, int(grid_points))
        cand = np.quantile(y_pred_reg, qs)

        fvals = []
        for t in cand:
            yhat = (y_pred_reg >= t).astype(int)
            f = fbeta_score(y_true_bin, yhat, beta=beta, zero_division=0)
            fvals.append(f)

        fvals = np.asarray(fvals)
        best_idx = int(np.argmax(fvals))
        best_t, best_f = float(cand[best_idx]), float(fvals[best_idx])

        plt.figure(figsize=(6, 4))
        plt.plot(cand, fvals)
        plt.axvline(best_t, linestyle="--")
        plt.xlabel("prediction threshold on y_pred")
        plt.ylabel(f"F_{beta}")
        plt.title(f"F_{beta} vs threshold | best_t={best_t:.6g}, best_F={best_f:.3f}")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_threshold_sweep_f{beta}.png", dpi=200)
        plt.close()

        return {"best_threshold": best_t, "best_fbeta": best_f}
