# train/training/trainers/xgb.py
import cupy as cp
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from sklearn.metrics import log_loss

from train.training.metrics.metrics_reg import compute_regression_metrics
from train.training.metrics.metrics_cls import compute_cls_metrics
from train.training.trainers.utils import find_best_threshold_by_fbeta, get_task_type
from train.models.xgb_model import XGBClassifierModel, XGBRegressorModel
from train.evaluation.utils import (
    compute_mixed_objective_np,
    plot_regression_eval,
    plot_regression_threshold_sweep,
    save_fold_metrics,
)


class _XGBClassifierInferenceModule(torch.nn.Module):
    """
    Lightweight torch.Module wrapper that adapts sklearn's XGBClassifier
    for downstream evaluation/inference routines (expects logits output).
    """

    def __init__(self, sklearn_model, best_iteration: int):
        super().__init__()
        self.model = sklearn_model
        self.best_iteration = int(best_iteration) if int(best_iteration or 0) > 0 else int(
            sklearn_model.get_params().get("n_estimators", 1)
        )

    def forward(self, X):  # type: ignore[override]
        if isinstance(X, torch.Tensor):
            device = X.device
            X_np = X.detach().to("cpu").contiguous()
            X_np = X_np.view(X_np.shape[0], -1).numpy()
        else:
            device = torch.device("cpu")
            X_np = np.asarray(X, dtype=np.float32)
            if X_np.ndim > 2:
                X_np = X_np.reshape(X_np.shape[0], -1)

        probs = self.model.predict_proba(X_np, iteration_range=(0, self.best_iteration))
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if probs.shape[1] == 2:
            p1 = probs[:, 1]
            logits = np.log(p1 / np.clip(1.0 - p1, 1e-12, None))
            logits = logits.reshape(-1, 1)
        else:
            logits = np.log(probs)

        return torch.from_numpy(logits.astype(np.float32)).to(device)


def _train_one_fold_xgb_reg(model_wrapper, cfg, fold_id, export_dir):
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


def _train_one_fold_xgb_cls(model_wrapper, cfg, fold_id, export_dir):
    export_dir = Path(export_dir); export_dir.mkdir(parents=True, exist_ok=True)
    if "_xgb_pack" not in cfg:
        raise KeyError("[XGB][classification] cfg 缺少 '_xgb_pack'，請確認 also_XGB 為 True 且 loader 有回傳資料。")

    xgbp = cfg["_xgb_pack"]
    Xtr, ytr = xgbp["X_tr"], xgbp["y_tr"]
    Xva, yva = xgbp["X_va"], xgbp["y_va"]
    Xte, yte = xgbp["X_te"], xgbp["y_te"]

    num_classes = int(getattr(model_wrapper, "num_classes", cfg["model"].get("num_classes", 2)))
    model = model_wrapper.build()

    dev = str(model.get_params().get("device", "cpu")).lower()
    use_cuda = (dev == "cuda")

    if use_cuda:
        Xtr = cp.asarray(Xtr, dtype=cp.float32); ytr = cp.asarray(ytr, dtype=cp.int64)
        Xva = cp.asarray(Xva, dtype=cp.float32); yva = cp.asarray(yva, dtype=cp.int64)
        Xte = cp.asarray(Xte, dtype=cp.float32); yte = cp.asarray(yte, dtype=cp.int64)
    else:
        Xtr = np.asarray(Xtr, dtype=np.float32)
        Xva = np.asarray(Xva, dtype=np.float32)
        Xte = np.asarray(Xte, dtype=np.float32)
        ytr = np.asarray(ytr, dtype=np.int64)
        yva = np.asarray(yva, dtype=np.int64)
        yte = np.asarray(yte, dtype=np.int64)

    try:
        model.set_params(eval_metric="logloss" if num_classes == 2 else "mlogloss")
    except Exception:
        pass

    labels = list(range(num_classes))

    def to_np(a):
        return cp.asnumpy(a) if use_cuda else np.asarray(a)

    def normalize_prob(prob):
        prob_np = to_np(prob)
        prob_np = np.clip(prob_np, 1e-12, None)
        prob_np = prob_np / prob_np.sum(axis=1, keepdims=True)
        return prob_np

    def safe_logloss(y_true, prob):
        prob_norm = normalize_prob(prob)
        return float(log_loss(y_true, prob_norm, labels=labels))

    train_cfg = (cfg.get("train") or {})
    thr_mode = str(train_cfg.get("threshold_mode", "auto_auc")).lower()
    beta = float(train_cfg.get("threshold_fbeta", 0.5))
    grid_points = int(train_cfg.get("threshold_grid_points", 201))
    default_thr = float(train_cfg["threshold"]) if train_cfg.get("threshold") is not None else 0.5

    def compute_threshold(y_true_np, y_score_np):
        if num_classes != 2:
            return None
        if thr_mode == "auto_fbeta":
            thr, _ = find_best_threshold_by_fbeta(
                y_true_np, y_score_np, beta=beta, grid_points=grid_points, task="cls"
            )
            return float(thr)
        return float(default_thr)

    primary_metric = str(cfg.get("objective", {}).get("primary_metric", "val_loss")).lower()

    def primary_value(metrics: dict, val_loss: float):
        name = primary_metric
        if name in {"macro_f05", "f_05_macro", "threshold_macro_f05"}:
            return metrics.get("macro_f05", metrics.get("f_05_macro", -np.inf)), True
        if name in {"macro_f1"}:
            return metrics.get("macro_f1", -np.inf), True
        if name in {"acc", "accuracy"}:
            return metrics.get("acc", -np.inf), True
        if name in {"mcc"}:
            return metrics.get("mcc", -np.inf), True
        if name in {"macro_precision"}:
            return metrics.get("macro_precision", -np.inf), True
        if name in {"macro_recall"}:
            return metrics.get("macro_recall", -np.inf), True
        if name in {"logloss", "loss", "val_loss"}:
            return val_loss, False
        return val_loss, False

    mcfg = cfg["model"]
    total_estimators = int(mcfg["n_estimators"])
    es_step = int(mcfg.get("es_step", 25))
    patience = int(mcfg.get("es_patience", 5))
    min_delta = float(mcfg.get("es_min_delta", 0.0))

    ytr_np = to_np(ytr)
    yva_np = to_np(yva)
    yte_np = to_np(yte)

    best_score = None
    best_info = {"iteration": 0, "threshold": None, "val_loss": float("inf"), "metrics": {}}
    curr = 0
    no_improve = 0

    while curr < total_estimators:
        prev = curr
        curr = min(curr + es_step, total_estimators)

        fit_kwargs = dict(X=Xtr, y=ytr, eval_set=[(Xva, yva)], verbose=False)
        if prev > 0:
            fit_kwargs["xgb_model"] = model.get_booster()

        model.set_params(n_estimators=curr)
        model.fit(**fit_kwargs)

        yva_prob = model.predict_proba(Xva, iteration_range=(0, curr))
        yva_prob_np = normalize_prob(yva_prob)
        val_loss = safe_logloss(yva_np, yva_prob_np)

        if num_classes == 2:
            yva_score = yva_prob_np[:, 1]
            curr_thr = compute_threshold(yva_np, yva_score)
            yva_pred = (yva_score >= curr_thr).astype(int)
        else:
            curr_thr = None
            yva_pred = yva_prob_np.argmax(axis=1)

        m_va = compute_cls_metrics(yva_np, yva_pred)
        m_va["macro_f05"] = m_va.get("macro_f05", m_va.get("f_05_macro", 0.0))
        score, maximize = primary_value(m_va, val_loss)

        improved = False if best_score is not None else True
        if best_score is None:
            improved = True
        else:
            if maximize:
                improved = score > (best_score + min_delta)
            else:
                improved = score < (best_score - min_delta)

        print(f"[XGB-CLS][fold {fold_id}] trees={curr:04d} | "
              f"val_loss={val_loss:.6f} | val_acc={m_va.get('acc', float('nan')):.4f} | "
              f"val_f05={m_va.get('macro_f05', float('nan')):.4f} | thresh={curr_thr if curr_thr is not None else -1:.4f}")

        if improved:
            best_score = score
            best_info = {
                "iteration": curr,
                "threshold": curr_thr,
                "val_loss": val_loss,
                "metrics": m_va,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[XGB-CLS][fold {fold_id}] Early stop at {curr} trees (best_t={best_info['iteration']})")
                break

    if int(best_info["iteration"]) <= 0:
        best_info["iteration"] = curr

    best_t = int(best_info["iteration"])
    history = []

    for t in range(1, best_t + 1):
        ytr_prob_np = normalize_prob(model.predict_proba(Xtr, iteration_range=(0, t)))
        yva_prob_np = normalize_prob(model.predict_proba(Xva, iteration_range=(0, t)))

        train_loss = safe_logloss(ytr_np, ytr_prob_np)
        val_loss = safe_logloss(yva_np, yva_prob_np)

        if num_classes == 2:
            yva_score = yva_prob_np[:, 1]
            ytr_score = ytr_prob_np[:, 1]
            thr_t = compute_threshold(yva_np, yva_score)
            yva_pred = (yva_score >= thr_t).astype(int)
            ytr_pred = (ytr_score >= thr_t).astype(int)
        else:
            thr_t = None
            yva_pred = yva_prob_np.argmax(axis=1)
            ytr_pred = ytr_prob_np.argmax(axis=1)

        m_va = compute_cls_metrics(yva_np, yva_pred)
        m_tr = compute_cls_metrics(ytr_np, ytr_pred)

        history.append({
            "epoch": t,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": m_tr.get("acc", np.nan),
            "val_acc": m_va.get("acc", np.nan),
            "train_macro_f1": m_tr.get("macro_f1", np.nan),
            "val_macro_f1": m_va.get("macro_f1", np.nan),
            "train_macro_precision": m_tr.get("macro_precision", np.nan),
            "val_macro_precision": m_va.get("macro_precision", np.nan),
            "train_macro_recall": m_tr.get("macro_recall", np.nan),
            "val_macro_recall": m_va.get("macro_recall", np.nan),
            "val_f_05_macro": m_va.get("macro_f05", m_va.get("f_05_macro", np.nan)),
        })

    yva_prob_best = normalize_prob(model.predict_proba(Xva, iteration_range=(0, best_t)))
    yte_prob_best = normalize_prob(model.predict_proba(Xte, iteration_range=(0, best_t)))

    if num_classes == 2:
        yva_score_best = yva_prob_best[:, 1]
        best_thresh = best_info.get("threshold")
        if best_thresh is None:
            best_thresh = compute_threshold(yva_np, yva_score_best)
        yva_pred_best = (yva_score_best >= best_thresh).astype(int)
        yte_pred = (yte_prob_best[:, 1] >= best_thresh).astype(int)
    else:
        best_thresh = None
        yva_pred_best = yva_prob_best.argmax(axis=1)
        yte_pred = yte_prob_best.argmax(axis=1)

    val_loss_best = safe_logloss(yva_np, yva_prob_best)
    m_va_best = compute_cls_metrics(yva_np, yva_pred_best)
    m_te = compute_cls_metrics(yte_np, yte_pred)

    class_names = cfg["model"].get("class_names", None)
    val_loss_history = [float(h.get("val_loss", np.nan)) for h in history if "val_loss" in h]

    label_counts = {
        "train": {int(k): int(v) for k, v in Counter(ytr_np).items()},
        "val": {int(k): int(v) for k, v in Counter(yva_np).items()},
    }

    test_metrics = {
        "test_acc": m_te.get("acc", 0.0),
        "test_macro_f1": m_te.get("macro_f1", 0.0),
        "test_weighted_f1": m_te.get("weighted_f1", m_te.get("macro_f1", 0.0)),
        "test_macro_f05": m_te.get("macro_f05", m_te.get("f_05_macro", 0.0)),
        "test_mcc": m_te.get("mcc", 0.0),
        "test_macro_precision": m_te.get("macro_precision", 0.0),
        "test_macro_recall": m_te.get("macro_recall", 0.0),
    }

    result = {
        "history": history,
        "val_loss_history": val_loss_history,
        "label_counts": label_counts,
        "best_epoch": best_t,
        "state_dict": {},
        "val_metrics": {
            "val_loss": float(val_loss_best),
            "macro_f1": float(m_va_best.get("macro_f1", 0.0)),
            "f_05_macro": float(m_va_best.get("macro_f05", m_va_best.get("f_05_macro", 0.0))),
            "macro_precision": float(m_va_best.get("macro_precision", 0.0)),
            "macro_recall": float(m_va_best.get("macro_recall", 0.0)),
        },
        "test_metrics": test_metrics,
        "best_val_thresh": float(best_thresh) if (best_thresh is not None) else None,
        "temperature": 1.0,
        "threshold_metrics": {},
    }

    if num_classes == 2 and best_thresh is not None:
        result["threshold_metrics"] = {
            "best_threshold": float(best_thresh),
            "acc": test_metrics["test_acc"],
            "macro_f1": test_metrics["test_macro_f1"],
            "macro_precision": test_metrics["test_macro_precision"],
            "macro_recall": test_metrics["test_macro_recall"],
            "macro_f05": test_metrics["test_macro_f05"],
        }
    else:
        result["threshold_metrics"] = {
            "best_threshold": None,
            "acc": test_metrics["test_acc"],
            "macro_f1": test_metrics["test_macro_f1"],
            "macro_precision": test_metrics["test_macro_precision"],
            "macro_recall": test_metrics["test_macro_recall"],
            "macro_f05": test_metrics["test_macro_f05"],
        }

    result["eval_payload"] = {
        "y_true": yte_np,
        "y_pred": yte_pred,
        "y_prob": yte_prob_best,
        "class_names": class_names,
        "best_threshold": float(best_thresh) if (num_classes == 2 and best_thresh is not None) else None,
    }

    save_fold_metrics(
        history,
        save_dir=export_dir,
        prefix=f"fold_{fold_id}_",
        y_true=yte_np,
        y_pred=yte_pred,
        class_names=class_names,
    )

    inference_model = _XGBClassifierInferenceModule(model, best_t)
    return inference_model, result


def _train_one_fold_xgb(model_wrapper, cfg, fold_id, export_dir):
    if isinstance(model_wrapper, XGBClassifierModel):
        return _train_one_fold_xgb_cls(model_wrapper, cfg, fold_id, export_dir)
    if isinstance(model_wrapper, XGBRegressorModel):
        return _train_one_fold_xgb_reg(model_wrapper, cfg, fold_id, export_dir)

    task = get_task_type(cfg)
    if task == "classification":
        return _train_one_fold_xgb_cls(model_wrapper, cfg, fold_id, export_dir)
    return _train_one_fold_xgb_reg(model_wrapper, cfg, fold_id, export_dir)
