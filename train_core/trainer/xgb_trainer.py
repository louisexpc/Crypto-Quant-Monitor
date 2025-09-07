# xgb_trainer.py
import cupy as cp
import xgboost as xgb

def _train_one_fold_xgb(model_wrapper, cfg, fold_id, export_dir):
    import numpy as np
    from pathlib import Path
    from utils.regression_utils import compute_regression_metrics
    from utils.compute_export_metrices import (
        compute_mixed_objective_np,  # ← 與 trainer_reg.py 對齊
        plot_regression_eval, plot_regression_threshold_sweep, save_fold_metrics
    )

    export_dir = Path(export_dir); export_dir.mkdir(parents=True, exist_ok=True)

    # ===== 讀早停與損失權重（對齊 trainer_reg：直接 index，不用 .get）=====
    mcfg = cfg["model"]
    total_estimators = int(mcfg["n_estimators"])
    es_step   = int(mcfg["es_step"])      if "es_step"      in mcfg else 50
    patience  = int(mcfg["es_patience"])  if "es_patience"  in mcfg else 5
    min_delta = float(mcfg["es_min_delta"]) if "es_min_delta" in mcfg else 0.0

    lcfg  = cfg["loss"]
    alpha = float(lcfg["alpha"]) if "alpha" in lcfg else 0.7
    beta  = float(lcfg["beta"])  if "beta"  in lcfg else 0.3
    decay = float(lcfg["ema_decay"]) if "ema_decay" in lcfg else 0.9

    # ===== 取資料包 =====
    xgbp = cfg["_xgb_pack"]  # dict: X_tr/y_tr/X_va/y_va/X_te/y_te
    Xtr, ytr = xgbp["X_tr"], xgbp["y_tr"]
    Xva, yva = xgbp["X_va"], xgbp["y_va"]
    Xte, yte = xgbp["X_te"], xgbp["y_te"]

    # ===== 建模 & GPU handoff =====
    model = model_wrapper.build()
    dev = str(model.get_params().get("device", "cpu")).lower()
    use_cuda = (dev == "cuda")
    if use_cuda:
        Xtr, ytr = cp.asarray(Xtr), cp.asarray(ytr)
        Xva, yva = cp.asarray(Xva), cp.asarray(yva)
        Xte, yte = cp.asarray(Xte), cp.asarray(yte)

    # 讓 evals_result 可用（RMSE 只做參考曲線）
    try:
        model.set_params(eval_metric="rmse")
    except Exception:
        pass

    # ===== 小工具 =====
    def to_np(a):  # cp.ndarray -> np.ndarray
        return cp.asnumpy(a) if use_cuda else a

    def mixed_from_util(y_true_np: np.ndarray, y_pred_np: np.ndarray):
        """對齊 DL 版：用你的 compute_mixed_objective_np 算 mixed + pearson + ema_mse"""
        vmix, comps = compute_mixed_objective_np(
            y_true=y_true_np, y_pred=y_pred_np, alpha=alpha, beta=beta, ema_decay=decay
        )
        r = float(comps.get("pearson", np.corrcoef(y_true_np, y_pred_np)[0, 1]))
        ema_mse = float(comps.get("ema_mse", np.mean((y_true_np - y_pred_np) ** 2)))
        rmse = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))
        return float(vmix), rmse, r, ema_mse

    # ===== Warm-start 早停：以 val_mixed 作為判準 =====
    best_mixed = float("inf")
    best_t = 0
    no_improve = 0
    curr = 0

    while curr < total_estimators:
        prev = curr
        curr = min(curr + es_step, total_estimators)

        fit_kwargs = dict(
            X=Xtr, y=ytr,
            eval_set=[(Xtr, ytr), (Xva, yva)],
            verbose=False
        )
        if prev > 0:  # 第一次不能帶 booster；之後用 booster warm-start
            fit_kwargs["xgb_model"] = model.get_booster()

        model.set_params(n_estimators=curr)
        model.fit(**fit_kwargs)

        # 驗證集 mixed（用前 curr 棵）
        yva_t = model.predict(Xva, iteration_range=(0, curr))
        vmix, vrmse, vr, v_ema = mixed_from_util(to_np(yva), to_np(yva_t))
        print(f"[XGB][fold {fold_id}] trees={curr:04d} | "
              f"val_mixed={vmix:.8e} (rmse={vrmse:.6f}, r={vr:.4f}) | "
              f"best_mixed={best_mixed:.8e} (best_t={best_t})")

        if vmix < (best_mixed - min_delta):
            best_mixed = vmix
            best_t = curr
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[XGB][fold {fold_id}] Early stop at {curr} trees "
                  f"(best_t={best_t}, best_mixed={best_mixed:.8e}\n)")
            break

    if best_t == 0:
        best_t = curr

    # ===== 學習曲線 history（1..best_t）=====
    ytr_np, yva_np = to_np(ytr), to_np(yva)
    history = []
    for t in range(1, best_t + 1):
        ytr_t = model.predict(Xtr, iteration_range=(0, t))
        yva_t = model.predict(Xva, iteration_range=(0, t))
        tr_mix, tr_rmse, tr_r, _ = mixed_from_util(ytr_np, to_np(ytr_t))
        va_mix, va_rmse, va_r, _ = mixed_from_util(yva_np, to_np(yva_t))

        print(f"[XGB][fold {fold_id}] round={t:04d} | "
              f"tr_mix={tr_mix:.8e} va_mix={va_mix:.8e} | "
              f"tr_rmse={tr_rmse:.6f} va_rmse={va_rmse:.6f} | "
              f"tr_r={tr_r:.4f} va_r={va_r:.4f}")

        history.append({
            "epoch": t,
            "train_rmse": tr_rmse,
            "val_rmse": va_rmse,
            "train_pearson": tr_r,
            "val_pearson": va_r,
            "train_loss": tr_mix,   # ← 對齊 DL：loss 用 mixed
            "val_loss": va_mix,     # ← 對齊 DL：loss 用 mixed
        })

    # ===== 最終驗證/測試推論（用 best_t 截斷）=====
    yva_pred = model.predict(Xva, iteration_range=(0, best_t))
    yte_pred = model.predict(Xte, iteration_range=(0, best_t))
    yva_pred, yte_pred = to_np(yva_pred), to_np(yte_pred)
    yva_np, yte_np = to_np(yva), to_np(yte)

    best_epoch = best_t

    # ===== valid 主目標（沿用你的 util）=====
    val_mixed, comps = compute_mixed_objective_np(
        y_true=yva_np, y_pred=yva_pred, alpha=alpha, beta=beta, ema_decay=decay
    )

    # ===== 測試指標 =====
    m_te = compute_regression_metrics(yte_np, yte_pred)

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": {},
        "test_metrics_reg": {"test_loss": float(np.mean((yte_np - yte_pred) ** 2)), **m_te},
        "temperature": 1.0,
        "best_val_mixed": float(val_mixed),
        "best_val_pearson": float(comps.get("pearson", 0.0)),
        "best_val_arrays": {"y_va": yva_np.tolist(), "y_pred_va": yva_pred.tolist()},
    }

    # ===== 視覺化 / 匯出 =====
    plot_regression_eval(y_true=yte_np, y_pred=yte_pred, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    reg2cls = cfg["regression_to_class"] if "regression_to_class" in cfg else {}
    if ("enabled" in reg2cls) and bool(reg2cls["enabled"]):
        plot_regression_threshold_sweep(
            y_true_reg=yte_np, y_pred_reg=yte_pred,
            true_threshold=float(reg2cls["true_threshold"]) if "true_threshold" in reg2cls else 0.0,
            beta=float(reg2cls["fbeta"]) if "fbeta" in reg2cls else 0.5,
            grid_points=int(reg2cls["grid_points"]) if "grid_points" in reg2cls else 101,
            save_dir=export_dir, prefix=f"fold_{fold_id}_"
        )
    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    # save_result(fold_id=fold_id, export_dir=export_dir, result=result)
    return model, result

