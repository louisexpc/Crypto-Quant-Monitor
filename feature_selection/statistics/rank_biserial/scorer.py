# feature_select/scorer.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, norm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
from train.data.folds import split_fold_to_indices

class VetScorer:
    """
    1. 說明:
        在給定的時間序列折疊上，計算每個特徵的 OOS AUC 與 rank-biserial (r_rb=2*AUC-1)，
        彙整 mean/std 與 ICIR=mean/std；同時以全體樣本做 Mann–Whitney U 檢定拿 p 值，
        之後用 BH-FDR 控制多重假設。
    2. inputs:
        cfg (dict): 需含 cv.train_val_split、scoring.min_non_nan、scoring.fdr_q
        evt_df (DataFrame): index=t0；cols=['y','side','entry_price','t1']
        feat_at_t0 (DataFrame): index=t0；每欄一特徵的事件時刻分數
        bars (DataFrame): bar×特徵（僅用其 index 做時間對齊）
    3. return:
        由 run(folds) 回傳結果 DataFrame
    """
    def __init__(self, cfg: Dict, evt_df: pd.DataFrame,
                 feat_at_t0: pd.DataFrame, bars: pd.DataFrame):
        self.cfg = cfg
        self.evt = evt_df
        self.X = feat_at_t0
        self.bars = bars

    def _test_event_index_by_timerange(self, fold: Dict) -> pd.DatetimeIndex:
        """
        1. 說明:
            以 fold['test_times'] 的最小/最大時間作為區間，挑出位於該區間的事件 t0。
            用於 event-driven 場景中，事件時戳未必等於 bar 時戳，避免 'isin' 對不上。
        2. inputs:
            fold (dict): FoldGenerator 產生的折資訊，需含 'test_times'
        3. return:
            DatetimeIndex: 位於測試時間範圍內的事件索引（可能為空）
        """
        te_times = fold.get("test_times")
        if te_times is None or len(te_times) == 0:
            return self.evt.index[:0]
        start, end = pd.DatetimeIndex(te_times).min(), pd.DatetimeIndex(te_times).max()
        idx = pd.DatetimeIndex(self.evt.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        mask = (idx >= start) & (idx <= end)
        return idx[mask]

    def _fold_auc(self, col: str, folds: List[Dict]) -> Tuple[List[float], List[int]]:
        """
        1. 說明:
            於各折的「測試區間」計算該特徵的 OOS AUC；同時回傳每折有效樣本數供除錯。
        2. inputs:
            col (str): 特徵名
            folds (list[dict]): 折疊資訊
        3. return:
            (aucs, ns): aucs=list[float]、ns=list[int]（各折有效樣本數）
        """
        aucs, ns = [], []
        for fold in folds:
            # 先嘗試用 split_fold_to_indices（時間集合對齊事件）
            try:
                _, _, te_idx = split_fold_to_indices(
                    self.evt, fold, {"cv": {"train_val_split": self.cfg["cv"]["train_val_split"]}}
                )
            except Exception:
                te_idx = self.evt.index[:0]

            # 若沒對到任何事件，改用「時間範圍包含」做 fallback
            if len(te_idx) == 0:
                te_idx = self._test_event_index_by_timerange(fold)

            if len(te_idx) == 0:
                ns.append(0)
                continue

            y = self.evt.loc[te_idx, "y"].astype(int)
            s = self.X.loc[te_idx, col]
            mask = ~(s.isna() | y.isna())
            n_eff = int(mask.sum())
            ns.append(n_eff)
            if n_eff < int(self.cfg["scoring"]["min_non_nan"]):
                continue

            auc = float(roc_auc_score(y[mask], s[mask]))  # 可加 sample_weight
            aucs.append(auc)

        return aucs, ns

    @staticmethod
    def _u_test(y: pd.Series, s: pd.Series) -> float:
        """
        1. 說明: Mann–Whitney U 檢定（等價於檢定 AUC=0.5）
        2. inputs: y (0/1), s (連續分數)
        3. return: p 值（雙尾）
        """
        yb = y.astype(int).values
        sb = s.values
        return float(mannwhitneyu(sb[yb == 1], sb[yb == 0], alternative="two-sided").pvalue)

    def _fold_metrics(self, col: str, folds: list[dict]) -> list[dict]:
        """
        1. 說明:
            逐折收集單變量表現：AUC、該折 U 檢定 p 值、有效樣本數。
            - 測試集以 fold 的 test_times 對齊事件索引（同你既有邏輯）。
            - 若該折有效樣本數 < min_non_nan 或 y 單一類別，略過該折。
        2. inputs:
            col (str): 目標特徵欄名
            folds (list[dict]): FoldGenerator 產生的折疊清單
        3. return:
            list[dict]: 每個元素含 {auc, pval, n, sign}，sign=sign(auc-0.5)
        """
        out = []
        min_non_nan = int(self.cfg["scoring"]["min_non_nan"])
        for fold in folds:
            te_start = pd.DatetimeIndex(fold["test_times"]).min()
            te_end   = pd.DatetimeIndex(fold["test_times"]).max()

            # 用「時間範圍包含」對齊該折的事件（避免事件時間≠bar時間）
            mask_te = (self.evt.index >= te_start) & (self.evt.index <= te_end)
            if mask_te.sum() == 0:
                continue

            y = self.evt.loc[mask_te, "y"].astype(int)
            s = self.X.loc[mask_te, col]
            mask = ~(s.isna() | y.isna())
            if mask.sum() < min_non_nan or y[mask].nunique() < 2:
                continue

            auc = float(roc_auc_score(y[mask], s[mask]))
            # U 檢定（雙尾）
            yb = y[mask].values
            sb = s[mask].values
            try:
                p = float(mannwhitneyu(sb[yb==1], sb[yb==0], alternative="two-sided").pvalue)
            except Exception:
                p = np.nan

            out.append(dict(
                auc=auc,
                pval=p,
                n=int(mask.sum()),
                sign=np.sign(auc - 0.5) if not np.isclose(auc, 0.5) else 0.0
            ))
        return out

    @staticmethod
    def _combine_pvals_stouffer(pvals: np.ndarray, signs: np.ndarray, weights: np.ndarray | None = None) -> float:
        """
        1. 說明:
            以（加權）Stouffer 法合併多個 p 值（雙尾），方向用 sign（+1/-1）帶入。
            Z = sum(w_i * z_i) / sqrt(sum w_i^2)，其中 z_i = sign_i * Phi^{-1}(1 - p_i/2)
        2. inputs:
            pvals (np.ndarray): 各折 p 值（雙尾）
            signs (np.ndarray): 各折方向（+1/-1/0），建議取 sign(AUC-0.5)
            weights (np.ndarray|None): 權重；None 則等權。通常取 sqrt(n_i)
        3. return:
            float: 合併後雙尾 p 值
        """
        p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0)  # 避免極端 0
        z = norm.isf(p / 2.0) * np.sign(signs)
        if weights is None:
            Z = z.mean() * np.sqrt(len(z))
        else:
            w = np.asarray(weights, dtype=float)
            Z = np.sum(w * z) / np.sqrt(np.sum(w**2))
        return float(2.0 * norm.sf(abs(Z)))

    def _pval_cv_aware(self, fold_stats: list[dict]) -> float:
        """
        1. 說明:
            將逐折 U 檢定 p 值用 Stouffer（加權）合併為單一 CV-aware p 值。
            權重採 sqrt(n_k)，方向採 sign(AUC_k - 0.5)。
            若無可用折，回傳 NaN（上層可選擇 fallback）。
        2. inputs:
            fold_stats (list[dict]): 由 _fold_metrics 回傳的逐折統計
        3. return:
            float: 合併後 p 值（NaN 表示無法計算）
        """
        if not fold_stats:
            return float("nan")
        pvals  = np.array([d["pval"] for d in fold_stats if np.isfinite(d["pval"])])
        signs  = np.array([d["sign"] for d in fold_stats if np.isfinite(d["pval"])])
        weights= np.array([np.sqrt(d["n"]) for d in fold_stats if np.isfinite(d["pval"])])
        if len(pvals) == 0:
            return float("nan")
        return self._combine_pvals_stouffer(pvals, signs, weights)
    
    def run(self, folds: List[Dict]) -> pd.DataFrame:
        """
        1. 說明:
            主程式：對每個特徵計算跨折 AUC、r_rb、ICIR 與「CV-aware」合併 p 值，最後做 BH-FDR。
            p 值以逐折 Mann–Whitney U 檢定後使用（加權）Stouffer 合併，與 OOS 計分對齊。
        2. inputs:
            folds (list[dict]): 時間序列折疊
        3. return:
            DataFrame: index=feature；cols=auc_mean, auc_std, rrb_mean, rrb_std, icir, pval, fdr_reject
        """
        rows = []
        for col in self.X.columns:
            # 逐折統計（包含 auc、該折 p 值、樣本數、方向）
            fold_stats = self._fold_metrics(col, folds)
            if not fold_stats:
                continue

            aucs = [d["auc"] for d in fold_stats]
            rrb = 2 * np.array(aucs) - 1.0

            # 以 Stouffer（權重=√n）合併逐折 p 值 → CV-aware p 值
            pval_cv = self._pval_cv_aware(fold_stats)

            rows.append(dict(
                feature=col,
                auc_mean=float(np.mean(aucs)),
                auc_std=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                rrb_mean=float(np.mean(rrb)),
                rrb_std=float(np.std(rrb, ddof=1)) if len(rrb) > 1 else 0.0,
                icir=float(np.mean(rrb) / (np.std(rrb, ddof=1) + 1e-12)),
                pval=pval_cv,   # ← 現在是 CV-aware 的合併 p 值
            ))

        # 無任何特徵通過有效樣本數門檻 → 回傳空表
        if not rows:
            cols = ["auc_mean","auc_std","rrb_mean","rrb_std","icir","pval","fdr_reject"]
            return pd.DataFrame(columns=cols)

        out = pd.DataFrame(rows).set_index("feature").sort_values("rrb_mean", ascending=False)

        # multipletests 不吃 NaN；保守起見把 NaN 當 1.0（完全不顯著）
        p_in = out["pval"].fillna(1.0).values
        out["fdr_reject"] = multipletests(p_in, alpha=self.cfg["scoring"]["fdr_q"], method="fdr_bh")[0]
        return out
