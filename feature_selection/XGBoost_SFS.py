# XGBoost_SFS.py

import yaml, pandas as pd, numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans
from feature_utils import load_labeled_data, scalar_data, topk_features_per_cluster_cls
import sys, os
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.dataloader import FoldGenerator

# 1) 讀data, hyp
with open(r"train/feature_selection/feature_selection.yaml",encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


def k_means_candidates(cfg):
    # 1) 讀資料
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
        y_cls=df["y_cls"],
        topk=10,
        save_dir=None,  # 你上面已經設定好的目錄；不想存檔可改成 None
    )
    # print(top_cls_df.head(100))

    return df, X_scaled, y_cls, top_cls_df

def convert_folds_to_cv_splits(fold_dicts: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    把 FoldGenerator 輸出的 fold dicts 轉成 SFS 相容的 (train_idx, val_idx) list
    """
    cv_splits = []
    for fold in fold_dicts:
        tr_idx = np.where(fold["train_val_mask"])[0]
        va_idx = np.where(fold["test_mask"])[0]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            cv_splits.append((tr_idx, va_idx))
    return cv_splits



def prepare_sfs_data(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
):
    """
    將 X, y 依 mask 濾掉 y==-1，轉 float32 numpy，並把 CV folds 映射到濾後索引。
    回傳：
      X_all_np: (n, d) float32
      y_all_np: (n,) int
      feat_to_col: dict[str,int]  # feature 名稱 -> 欄位位置
      mapped_cv: list[(train_idx_np, val_idx_np)]  # 濾後索引
      mask: 原始樣本有效遮罩（可選）
    """
    mask = (y.values != -1) & np.isfinite(y.values)
    X_all_df = X.loc[mask].copy()
    y_all_sr = y.loc[mask].astype(int).copy()

    # 重要：float32，減少 GPU/CPU/PCIe 負擔
    X_all_np = X_all_df.to_numpy(dtype=np.float32, copy=False)
    y_all_np = y_all_sr.to_numpy()

    feat_to_col = {f: i for i, f in enumerate(X_all_df.columns)}

    # 原始位置 -> 濾後位置
    orig_pos = np.where(mask)[0]
    pos_map = np.full(len(mask), -1, dtype=np.int32)
    pos_map[orig_pos] = np.arange(len(orig_pos), dtype=np.int32)

    mapped_cv = []
    for tr_raw, va_raw in cv_splits:
        tr = pos_map[tr_raw]; tr = tr[tr >= 0]
        va = pos_map[va_raw]; va = va[va >= 0]
        mapped_cv.append((tr, va))

    return X_all_np, y_all_np, feat_to_col, mapped_cv, mask

# ====== XGBoost SFS：分類 y_cls，時間序列CV安全 ======
from typing import List, Tuple
import time
import numpy as np, pandas as pd
from sklearn.metrics import fbeta_score

def _class_weights(y: np.ndarray) -> dict[int, float]:
    """balanced class weights: n / (k * count_c)"""
    classes, counts = np.unique(y, return_counts=True)
    k = len(classes); n = len(y)
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}

# === CHANGED: SFS 主函式（GPU 最佳化 + 剪枝 + 快篩） ===
import time
import numpy as np, pandas as pd
from sklearn.metrics import fbeta_score

def sfs_select_features_cls_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: List[str],
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    max_features: int = 30,
    min_improve: float = 0.002,
    patience: int = 2,
    beta: float = 0.5,
    xgb_params: dict | None = None,
    random_state: int = 42,
    # 快篩參數
    prefilter: bool = True,
    prefilter_folds: int = 5,
    prefilter_keep: int = 25,
    prefilter_rounds: int = 400,
) -> tuple[list[str], pd.DataFrame]:
    import xgboost as xgb

    # --- 一次性預處理 ---
    X_all_np, y_all_np, feat_to_col, mapped_cv, _ = prepare_sfs_data(X, y, cv_splits)
    n_folds = len(mapped_cv)

    # --- 預設參數 ---
    default_params = dict(
        learning_rate=0.08,
        max_depth=4,
        min_child_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_estimators=1500,        # 暖身足夠，反正有 early stop
        eval_metric="mlogloss",
        random_state=random_state,
        verbosity=0,
        tree_method="gpu_hist",   # 嘗試 GPU；不行再降
    )
    if xgb_params:
        default_params.update(xgb_params)

    # --- 嘗試 GPU，不行就回退 CPU ---
    device = "cuda" if default_params.get("tree_method", "hist") == "gpu_hist" else "cpu"
    try:
        tiny_X = np.zeros((8, 2), dtype=np.float32)
        tiny_y = np.zeros(8, dtype=np.int32)
        dtiny = xgb.QuantileDMatrix(tiny_X, label=tiny_y)
        xgb.train({"device": "cuda"}, dtrain=dtiny, num_boost_round=1, evals=[(dtiny, "e")], verbose_eval=False)
        device = "cuda"
        print("[gpu-check] OK: using CUDA")
    except Exception as e:
        device = "cpu"
        print("[gpu-check] FAIL -> fallback to CPU. Reason:\n", repr(e))

    train_params = {
        "objective": "multi:softprob",
        "num_class": int(np.unique(y_all_np).size),
        "eta": float(default_params.get("learning_rate", 0.08)),
        "max_depth": int(default_params.get("max_depth", 4)),
        "min_child_weight": float(default_params.get("min_child_weight", 2.0)),
        "subsample": float(default_params.get("subsample", 0.8)),
        "colsample_bytree": float(default_params.get("colsample_bytree", 0.8)),
        "lambda": float(default_params.get("reg_lambda", 1.0)),
        "alpha": float(default_params.get("reg_alpha", 0.0)),
        "device": device,
        "predictor": "gpu_predictor" if device == "cuda" else "cpu_predictor",
        "single_precision_histogram": True,
        "max_bin": 256,
        "eval_metric": default_params.get("eval_metric", "mlogloss"),
        "seed": int(default_params.get("random_state", 42)),
        "verbosity": int(default_params.get("verbosity", 0)),
    }
    num_boost_round = int(default_params.get("n_estimators", 1500))
    early_rounds = 50

    DMAT = xgb.QuantileDMatrix if device == "cuda" and hasattr(xgb, "QuantileDMatrix") else xgb.DMatrix

    def _class_weights_np(y_arr: np.ndarray) -> dict[int, float]:
        classes, counts = np.unique(y_arr, return_counts=True)
        k, n = len(classes), len(y_arr)
        return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}

    # --- 帶 cutoff 的 CV：跨折剪枝 ---
    def cv_score(feats: list[str], cutoff: float, folds_subset=None, rounds=None) -> float:
        if not feats:
            return -np.inf
        cols = np.fromiter((feat_to_col[f] for f in feats), dtype=np.int32)
        folds = folds_subset if folds_subset is not None else mapped_cv
        F = len(folds)
        rounds = int(rounds or num_boost_round)

        s_sum, done = 0.0, 0
        for tr_idx, va_idx in folds:
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            Xtr = X_all_np[tr_idx][:, cols]
            Xva = X_all_np[va_idx][:, cols]
            ytr = y_all_np[tr_idx]; yva = y_all_np[va_idx]

            cw = _class_weights_np(ytr)
            sw = np.array([cw[int(t)] for t in ytr], dtype=np.float32)

            dtrain = DMAT(Xtr, label=ytr, weight=sw)
            dval   = DMAT(Xva, label=yva)

            bst = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=rounds,
                evals=[(dval, "eval")],
                early_stopping_rounds=early_rounds,
                verbose_eval=False,
            )
            proba = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
            yhat = proba.argmax(axis=1)
            f = fbeta_score(yva, yhat, beta=beta, average="macro", zero_division=0)

            s_sum += f; done += 1

            # 上界（未來折全拿 1.0）都追不過 cutoff 就提前放棄
            ub = (s_sum + (F - done) * 1.0) / F
            if ub < cutoff:
                return -np.inf

        return s_sum / max(done, 1)

    # --- 快篩：先用少數 folds + 少量 rounds 對所有候選打分，保留前 L ---
    feat_ok = [f for f in candidates if f in X.columns]
    if prefilter and len(feat_ok) > prefilter_keep:
        quick_folds = mapped_cv[: min(prefilter_folds, n_folds)]
        print(f"[prefilter] scanning {len(feat_ok)} feats on {len(quick_folds)} folds, "
              f"rounds={prefilter_rounds}, keep={prefilter_keep}")
        quick_scores = []
        for f in feat_ok:
            s = cv_score([f], cutoff=-1e9, folds_subset=quick_folds, rounds=prefilter_rounds)
            quick_scores.append((f, s))
        quick_scores.sort(key=lambda x: x[1], reverse=True)
        feat_ok = [f for f, _ in quick_scores[:prefilter_keep]]
        print(f"[prefilter] kept {len(feat_ok)} features")
    else:
        feat_ok = [f for f in candidates if f in X.columns]

    print(f"[device] {'GPU' if device=='cuda' else 'CPU'}  | folds={n_folds}  | candidates={len(feat_ok)}")

    # --- SFS 主迴圈 ---
    selected: list[str] = []
    trace = []
    best_score = -np.inf
    no_improve = 0
    t0 = time.time()

    while len(selected) < min(max_features, len(feat_ok)):
        remaining = [f for f in feat_ok if f not in selected]
        gains = []
        base = selected.copy()
        cutoff = best_score + min_improve

        for f in remaining:
            s = cv_score(base + [f], cutoff=cutoff)
            gains.append((f, s))

        gains.sort(key=lambda x: x[1], reverse=True)
        best_feat, cand_score = gains[0]
        trace.append({
            "step": len(selected) + 1,
            "added": best_feat,
            "score": cand_score,
            "prev_score": best_score,
            "improve": cand_score - best_score
        })
        print(f"[step {len(selected)+1:02d}] +{best_feat:>20s}  score={cand_score:.4f}  "
              f"Δ={cand_score - best_score:+.4f}  (remain {len(remaining)-1})", flush=True)

        if cand_score > cutoff:
            selected.append(best_feat)
            best_score = cand_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[SFS] early stop: no improvement for {patience} steps.")
                break

    dt = time.time() - t0
    print(f"[SFS] done. selected={len(selected)}  best_score={best_score:.4f}  time={dt:.1f}s")
    trace_df = pd.DataFrame(trace)
    return selected, trace_df


def main():
    
    df, X_scaled, y_cls, top_cls_df = k_means_candidates(cfg)

    candidates = top_cls_df["feature"].unique().tolist()
    print(f"共 {len(candidates)} 個候選特徵")

    # 用 datetime 欄位（含 +08:00），轉成台北時區後去時區 → DatetimeIndex
    dt_index = (
        pd.to_datetime(df["datetime"], utc=True)  # 安全：若原本帶 +08:00 也能 parse 成 tz-aware
        .dt.tz_convert("Asia/Taipei")
        .dt.tz_localize(None)
    )

    # 這一步把「Series of datetime64[ns]」變成真正的 DatetimeIndex
    dt_index = pd.DatetimeIndex(dt_index)

    fold_gen = FoldGenerator(dt_index=dt_index, start_month=cfg["start_date"])

    folds = fold_gen.make_rolling_folds(train_window=cfg["train_window"], embargo_hours=cfg["embargo_hours"], test_freq=cfg["test_freq"])
    cv_splits = convert_folds_to_cv_splits(folds)
    print(f"共 {len(cv_splits)} 個 fold，可供 SFS 使用")


    out_dir = os.path.join("train", "feature_selection", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    selected_features, trace_df = sfs_select_features_cls_xgb(
        X=X_scaled,
        y=df["y_cls"],
        candidates=candidates,
        cv_splits=cv_splits,
        max_features=30,
        min_improve=0.002,
        patience=5,
        beta=0.5,
        xgb_params=dict(
            tree_method="gpu_hist",   # 嘗試 GPU；函式內會自動 fallback 到 CPU
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_estimators=1500,        # 先暖身；之後可調 2500~3000
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        ),
        # 快篩（強烈建議打開）
        prefilter=True,
        prefilter_folds=5,
        prefilter_keep=40,
        prefilter_rounds=400,
    )

    # 儲存結果
    sel_path = os.path.join(out_dir, "selected_features.txt")
    trace_path = os.path.join(out_dir, "sfs_trace.csv")
    with open(sel_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))
    trace_df.to_csv(trace_path, index=False)

    print("\n=== Selected features (top 20) ===")
    for i, f in enumerate(selected_features[:20], 1):
        print(f"{i:2d}. {f}")
    print(f"\nSaved: {sel_path}\nSaved: {trace_path}")


if __name__ == "__main__":
    import xgboost as xgb
    print("[xgboost]", xgb.__version__)
    try:
        import cupy as cp
        print("[cupy] devices =", cp.cuda.runtime.getDeviceCount())
    except Exception as e:
        print("[cupy] not available:", repr(e))
    main()