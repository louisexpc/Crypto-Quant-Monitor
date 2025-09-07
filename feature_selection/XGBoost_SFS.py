# XGBoost_SFS.py

import os, random, yaml, numpy as np, pandas as pd
from pathlib import Path

import sys, os 
curr_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir)) 
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
from train_core.build_feature_loader.dataloader import FoldGenerator



def setup_reproducible(cfg):
    os.environ["PYTHONHASHSEED"] = str(int(cfg["seed"]))
    import random as _random
    _random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

def load_precomputed_csv(cfg):
    csv_path = Path(cfg["data"]["precomputed_csv_path"])
    df = pd.read_csv(csv_path)

    tcol = cfg["data"]["time_col"]
    ycol = cfg["data"]["label_reg_col"]
    assert tcol in df.columns and ycol in df.columns, f"missing {tcol} or {ycol}"

    # 時間欄轉台北時區的 naive DatetimeIndex（FoldGenerator 需求）
    dt = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    df[tcol] = dt
    assert df[tcol].notna().all(), f"{tcol} parse failed."

    # 準備 X / y
    drop_cols = [tcol, ycol] + list(cfg["data"]["drop_cols"])
    X = df[[c for c in df.columns if c not in drop_cols]].astype(np.float32)
    y = df[ycol].astype(np.float32)
  

    # 有限值與零方差清理
    mask_finite = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
    if not mask_finite.all():
        bad = (~mask_finite).sum()
        print(f"[warn] drop {bad} rows (non-finite in X or y)")
        X, y, df = X.loc[mask_finite], y.loc[mask_finite], df.loc[mask_finite]

    nunique = X.nunique(dropna=False)
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        print(f"[warn] drop zero-variance cols: {len(zero_var_cols)}")
        X = X.drop(columns=zero_var_cols)

    return df.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)


# ============ candidates builder ============
def build_candidates(cfg, X: pd.DataFrame):
    mode = cfg["candidates_mode"]
    if mode == "all_columns":
        feats = X.columns.tolist()
    elif mode == "list_file":
        txt = Path(cfg["candidates_list_path"]).read_text(encoding="utf-8").splitlines()
        feats = [f for f in txt if f in X.columns]
    elif mode == "cluster_topk":
        # 可替換為你既有的分群+TopK流程；此處簡化為方差排序後每群前 topk（示意）
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X.values.astype(np.float32))
        k = int(cfg["best_k"])
        km = KMeans(n_clusters=k, random_state=int(cfg["seed"]), n_init="auto")
        cid = km.fit_predict(Xt)
        Xdf = pd.DataFrame(Xt, columns=X.columns)
        feats, topk = [], int(cfg["cluster_topk"])
        for g in range(k):
            idx = np.where(cid == g)[0]
            # 以群內的變異度為 proxy（你也可替換為與 y 的相關）
            cols_sorted = X.columns[np.argsort(-Xdf.iloc[idx].var().values)]
            feats.extend(cols_sorted[:topk].tolist())
        feats = list(dict.fromkeys(feats))
    else:
        raise ValueError(f"unknown candidates_mode={mode}")

    block = set(cfg["data"]["blocklist"])
    feats = [f for f in feats if f not in block]
    print(f"[candidates] {len(feats)} features")
    return feats

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



# ============ build folds ============
def build_cv_splits(cfg, df: pd.DataFrame):


    dt = pd.DatetimeIndex(df[cfg["data"]["time_col"]])
    fg = FoldGenerator(dt_index=dt, start_month=cfg["start_date"])
    folds = fg.make_rolling_folds(
        train_window=cfg["train_window"],
        embargo_hours=cfg["embargo_hours"],
        test_freq=cfg["test_freq"],
    )
    cv_splits = []
    for fd in folds:
        tr = np.where(fd["train_val_mask"])[0]
        te = np.where(fd["test_mask"])[0]
        if len(tr) and len(te):
            cv_splits.append((tr, te))
    print(f"[cv] folds={len(cv_splits)}")
    return cv_splits

# ============ metrics ============
def _safe_corrcoef(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def pearson_np(y_true, y_pred):
    return _safe_corrcoef(y_true, y_pred)

def spearman_np(y_true, y_pred):
    # 以 rank 轉換近似 Spearman（無需 scipy）
    ar = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64, copy=False)
    br = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64, copy=False)
    return _safe_corrcoef(ar, br)

def direction_acc(y_true, y_pred, zero_tol=0.0):
    st, sp = np.sign(y_true), np.sign(y_pred)
    if zero_tol > 0:
        mask = np.abs(y_true) > zero_tol
        if mask.sum() == 0:
            return 0.0
        st, sp = st[mask], sp[mask]
    return float((st == sp).mean())

def neg_rmse(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return -rmse  # 越大越好（負RMSE）

def mixed_metric(y_true, y_pred, w_corr=0.6, w_dir=0.3, w_rmse=0.1):
    # 無量綱化 RMSE：除以 y_true 的 std
    stdy = float(np.std(y_true)) + 1e-12
    m_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)) / stdy)  # 越小越好
    pc = max(0.0, pearson_np(y_true, y_pred))  # 負相關不加分（可依需求調整）
    da = direction_acc(y_true, y_pred, zero_tol=0.0)
    return float(w_corr * pc + w_dir * da - w_rmse * m_rmse)

def score_by_name(metric_name: str, y_true, y_pred, weights=None, mix_w=None):
    if metric_name == "pearson":
        return pearson_np(y_true, y_pred)
    if metric_name == "spearman":
        return spearman_np(y_true, y_pred)
    if metric_name == "direction":
        return direction_acc(y_true, y_pred, zero_tol=0.0)
    if metric_name == "neg_rmse":
        return neg_rmse(y_true, y_pred)
    if metric_name == "mixed":
        if mix_w is None:
            mix_w = dict(w_corr=0.6, w_dir=0.3, w_rmse=0.1)
        return mixed_metric(y_true, y_pred, **mix_w)
    raise ValueError(metric_name)

# ============ prefilter（corr / xgb） ============
def prefilter_corr(X: pd.DataFrame, y: np.ndarray, keep: int, method: str):
    # method in ["pearson", "spearman"]
    vals = []
    for f in X.columns:
        x = X[f].to_numpy(np.float32, copy=False)
        c = pearson_np(y, x) if method == "pearson" else spearman_np(y, x)
        vals.append((f, abs(c)))
    vals.sort(key=lambda z: z[1], reverse=True)
    kept = [f for f, _ in vals[:keep]]
    return kept, vals

# ============ SFS core (regression, XGB) ============
def sfs_select_features_reg_xgb(
    cfg,
    X: pd.DataFrame,
    y: pd.Series,
    candidates: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
):
    import xgboost as xgb

    # ---- 全域設定 ----
    metric_name = cfg["reg"]["primary_metric"]            # "pearson"|"spearman"|"direction"|"neg_rmse"|"mixed"
    early_rounds = int(cfg["xgb"]["early_stopping_rounds"])
    num_rounds   = int(cfg["xgb"]["n_estimators"])
    min_improve  = float(cfg["sfs"]["min_improve"])
    patience     = int(cfg["sfs"]["patience"])
    max_features = int(cfg["sfs"]["max_features"])

    # 權重（for mixed）
    mix_w = dict(
        w_corr=float(cfg["reg"]["mixed_w"]["w_corr"]),
        w_dir=float(cfg["reg"]["mixed_w"]["w_dir"]),
        w_rmse=float(cfg["reg"]["mixed_w"]["w_rmse"]),
    )

    # ---- 準備 NumPy 與欄位映射 ----
    X_all = X.to_numpy(dtype=np.float32, copy=False)
    y_all = y.to_numpy(dtype=np.float32, copy=False)
    feat_to_col = {f: i for i, f in enumerate(X.columns)}
    mapped_cv = [(tr, va) for tr, va in cv_splits]

    # ---- GPU 檢查 / 強制策略 ----
    require_gpu = bool(cfg["xgb"]["require_gpu"])
    device = "cuda"
    try:
        dtiny = xgb.QuantileDMatrix(np.zeros((16, 2), np.float32), label=np.zeros(16, np.float32))
        xgb.train({"device": "cuda", "objective": "reg:squarederror"}, dtiny, num_boost_round=1, verbose_eval=False)
        print("[gpu-check] OK: CUDA enabled")
    except Exception as e:
        if require_gpu:
            raise RuntimeError(f"require_gpu=True but CUDA not usable: {repr(e)}")
        device = "cpu"
        print("[gpu-check] FAIL -> fallback to CPU:", repr(e))

    DMAT = xgb.QuantileDMatrix if device == "cuda" and hasattr(xgb, "QuantileDMatrix") else xgb.DMatrix

    train_params = {
        "objective": "reg:squarederror",
        "device": device,
        "predictor": "gpu_predictor" if device == "cuda" else "cpu_predictor",
        "tree_method": "gpu_hist" if device == "cuda" else "hist",
        "max_bin": int(cfg["xgb"]["max_bin"]),
        "single_precision_histogram": True,
        "eta": float(cfg["xgb"]["learning_rate"]),
        "max_depth": int(cfg["xgb"]["max_depth"]),
        "min_child_weight": float(cfg["xgb"]["min_child_weight"]),
        "subsample": float(cfg["xgb"]["subsample"]),
        "colsample_bytree": float(cfg["xgb"]["colsample_bytree"]),
        "lambda": float(cfg["xgb"]["reg_lambda"]),
        "alpha": float(cfg["xgb"]["reg_alpha"]),
        "eval_metric": "rmse",  # 用於早停；真正的選擇分數由 metric_name 決定
        "seed": int(cfg["seed"]),
        "verbosity": int(cfg["xgb"]["verbosity"]),
    }

    # --- 針對不同指標設定「理論上界」：用於跨折剪枝 ---
    if cfg["reg"]["primary_metric"] == "mixed":
        ub_max = float(cfg["reg"]["mixed_w"]["w_corr"]) + float(cfg["reg"]["mixed_w"]["w_dir"])
    elif cfg["reg"]["primary_metric"] in ("pearson", "spearman", "direction"):
        ub_max = 1.0
    elif cfg["reg"]["primary_metric"] == "neg_rmse":
        # 負RMSE的理論上界約為0（越接近0越好）
        ub_max = 0.0
    else:
        ub_max = 1.0

    # ---- 評分 with cutoff（跨折剪枝）----
    def cv_score(cols_idx: np.ndarray, cutoff: float, rounds: int | None = None) -> float:
        if cols_idx.size == 0:
            return -np.inf
        rounds = int(rounds or num_rounds)

        s_sum, done = 0.0, 0
        for tr, va in mapped_cv:
            Xtr, Xva = X_all[tr][:, cols_idx], X_all[va][:, cols_idx]
            ytr, yva = y_all[tr], y_all[va]

            dtr = DMAT(Xtr, label=ytr)
            dva = DMAT(Xva, label=yva)

            bst = xgb.train(
                params=train_params,
                dtrain=dtr,
                num_boost_round=rounds,
                evals=[(dva, "eval")],
                early_stopping_rounds=early_rounds,
                verbose_eval=False,
            )
            yhat = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
            s = score_by_name(metric_name, yva, yhat, mix_w=mix_w)
            s_sum += float(s); done += 1

            # 上界（假設剩餘折都拿到 1.0）也過不了 cutoff → 提前放棄
            ub = (s_sum + (len(mapped_cv) - done) * ub_max) / len(mapped_cv)
            if ub < cutoff:
                return -np.inf  # 標記為剪枝

        return s_sum / max(done, 1)

    # ---- Prefilter ----
    feat_ok = [f for f in candidates if f in feat_to_col]
    if bool(cfg["prefilter"]["enable"]) and len(feat_ok) > int(cfg["prefilter"]["keep"]):
        mode = cfg["prefilter"]["mode"]   # "corr" or "xgb"
        keep = int(cfg["prefilter"]["keep"])
        print(f"[prefilter] mode={mode} scan {len(feat_ok)} features")

        if mode == "corr":
            kept, stat = prefilter_corr(X[feat_ok], y_all, keep=keep, method=cfg["prefilter"]["corr_method"])
            feat_ok = kept
            print(f"[prefilter] kept={len(feat_ok)} by {cfg['prefilter']['corr_method']}")
        elif mode == "xgb":
            quick_rounds = int(cfg["prefilter"]["rounds"])
            quick_folds  = min(int(cfg["prefilter"]["folds"]), len(mapped_cv))
            base_cutoff  = -1e9
            scores = []
            for f in feat_ok:
                idx = np.array([feat_to_col[f]], dtype=np.int32)
                s = cv_score(idx, cutoff=base_cutoff, rounds=quick_rounds)
                scores.append((f, s))
            scores.sort(key=lambda z: z[1], reverse=True)
            feat_ok = [f for f, _ in scores[:keep]]
            print(f"[prefilter] kept={len(feat_ok)} by quick xgb")
        else:
            raise ValueError("prefilter.mode must be 'corr' or 'xgb'")

    print(f"[device] {device.upper()} | candidates={len(feat_ok)}")

    # ---- SFS 主流程 ----
    selected, trace = [], []
    best_score, no_improve = -np.inf, 0
    attempt = 0  # 同一步的第幾次嘗試

    while len(selected) < min(max_features, len(feat_ok)):
        remaining = [f for f in feat_ok if f not in selected]
        if not remaining:
            print("[SFS] no remaining candidates.")
            break

        cutoff = best_score + min_improve
        attempt += 1

        gains = []
        base_cols = np.array([feat_to_col[f] for f in selected], dtype=np.int32) if selected else np.empty((0,), np.int32)
        for f in remaining:
            cols = np.append(base_cols, feat_to_col[f]).astype(np.int32)
            s = cv_score(cols, cutoff=cutoff)
            gains.append((f, s))

        # 只保留有限值的候選；若全被剪枝，直接停止，避免「看起來重複選同一個」
        gains_valid = [(f, s) for (f, s) in gains if np.isfinite(s)]
        if not gains_valid:
            print(f"[SFS] all candidates pruned at step {len(selected)+1}. stop.")
            break

        gains_valid.sort(key=lambda z: z[1], reverse=True)
        best_feat, cand_score = gains_valid[0]
        improve = cand_score - best_score

        step_num = len(selected) + 1
        print(f"[step {step_num:02d} | try {attempt}] +{best_feat:>20s}  score={cand_score:.5f}  "
              f"Δ={improve:+.5f}  (remain {len(remaining)-1})", flush=True)

        trace.append(dict(step=step_num, added=best_feat, score=cand_score,
                          prev_score=best_score, improve=improve))

        if cand_score > cutoff:
            selected.append(best_feat)
            best_score = cand_score
            no_improve = 0
            attempt = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[SFS] early stop (no improve {patience} tries at step {step_num}).")
                break

    return selected, pd.DataFrame(trace)

# ============ main ============
def main():
    with open("feature_selection/feature_selection.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_reproducible(cfg)
    df, X, y = load_precomputed_csv(cfg)
    assert cfg["task_type"] == "reg", "請在 cfg 設定 task_type: 'reg'"

    candidates = build_candidates(cfg, X)
    cv_splits  = build_cv_splits(cfg, df)

    selected, trace_df = sfs_select_features_reg_xgb(cfg, X, y, candidates, cv_splits)

    out_dir = Path("train/feature_selection/outputs_reg"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_features.txt").write_text("\n".join(selected), encoding="utf-8")
    trace_df.drop_duplicates().to_csv(out_dir / "sfs_trace.csv", index=False)

    print("\n=== Selected (top 20) ===")
    for i, f in enumerate(selected[:20], 1):
        print(f"{i:2d}. {f}")
    print(f"\nSaved: {out_dir/'selected_features.txt'}")
    print(f"Saved: {out_dir/'sfs_trace.csv'}")

if __name__ == "__main__":
    main()