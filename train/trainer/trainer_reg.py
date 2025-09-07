# trainer_reg.py
import numpy as np
import torch
import torch.nn as nn
from torch import amp
from sklearn.metrics import fbeta_score
from .trainer_base import (
    amp_dtype, build_optimizer, build_warmup_scheduler, build_grad_scaler
)
from .xgb_trainer import _train_one_fold_xgb

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 回歸度量/損失/圖表
from train_utils.regression_utils import build_regression_loss

from train_utils.compute_export_metrices import (
    save_fold_metrics, plot_regression_eval, 
    plot_test_eval
)
from train_utils.metrics_reg import compute_regression_metrics, mixed_objective  # ★ 用這個
from train_utils.metrics_cls import compute_cls_metrics               # ★ 回歸→分類要用

from models.xgb_model import XGBRegressorModel

def train_one_fold(
    model,
    train_loader,
    val_loader,
    test_loader,
    cfg,
    device: str = None,
    fold_id: int | None = None,
    export_dir: str | None = None
):
    
    # === XGB 分支：直接改走 numpy 訓練，略過整個 torch 流程 ===
    if XGBRegressorModel is not None and isinstance(model, XGBRegressorModel):
        return _train_one_fold_xgb(model, cfg, fold_id, export_dir)
    

    # === DL: PyTorch 訓練迴圈 ===
    if len(train_loader) == 0:
        print("[ERROR][trainer_reg] empty train loader")
        return None, None

    assert torch.cuda.is_available(), "[trainer_reg] 需要 CUDA GPU 環境。"
    device = device or "cuda"

    # 讀設定
    lr        = float(cfg["train"]["lr"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    primary   = str(cfg["objective"]["primary_metric"]).lower()

    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_scheduler(optimizer, steps_per_epoch, cfg)
    dtype = amp_dtype()
    scaler = build_grad_scaler(dtype)

    # 回歸損失（由 cfg 決定：mse/huber 或 混合 α·EMA-MSE + β·(1-Pearson) 等）
    loss_fn = build_regression_loss(cfg)

    # Early-stop 狀態
    best_epoch = 0
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    wait = 0
    prefix = f"[fold {fold_id}] " if fold_id is not None else ""
    history = []

    best_reg_val_loss = float("inf")
    best_reg_pearson  = -1.0
    best_reg_spearman = -1.0   
    best_reg_mixed    = float("inf")
    best_va_arrays    = None  # (y_va, y_pred_va)

    printed_shape = False

    # ----------------- Epoch Loop -----------------
    for epoch in range(1, epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            if not printed_shape:
                print("[trainer_reg] DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
                pred = model(Xb).squeeze(-1)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            if clip and clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            bs = Xb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            tr_preds.append(pred.detach().float().cpu())
            tr_tgts.append(yb.detach().cpu())

        avg_tr_loss = train_loss_sum / max(1, train_n)
        y_tr = torch.cat(tr_tgts).numpy()
        yhat_tr = torch.cat(tr_preds).numpy()
        m_tr = compute_regression_metrics(y_tr, yhat_tr)

        # -------- VAL --------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_tgts = []
        val_preds = []

        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                pred = model(Xb).squeeze(-1)
                loss = loss_fn(pred, yb)

                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb.detach().cpu())
                val_preds.append(pred.detach().float().cpu())

        avg_va_loss = val_loss_sum / max(1, val_n)
        y_va = torch.cat(val_tgts, dim=0).to(torch.float32).cpu().numpy()
        y_pred_va = torch.cat(val_preds, dim=0).to(torch.float32).cpu().numpy()

        m_va = compute_regression_metrics(y_va, y_pred_va)

        # 組合指標（Optuna 主目標 / Early-stop 也可用）
        val_mixed, comps = mixed_objective(
            y_true=y_va,
            y_pred=y_pred_va,
            alpha=float(cfg["loss"]["alpha"]),
            beta=float(cfg["loss"]["beta"]),
            ema_decay=float(cfg["loss"]["ema_decay"]),
        )
        m_va["pearson_global"] = comps["pearson"]
        m_va["spearman_global"]  = comps.get("spearman", np.nan)   # NEW
        m_va["ema_mse_global"] = comps["ema_mse"]
        m_va["mixed"]          = val_mixed

        if epoch == 1:
            std_pred = float(np.std(y_pred_va))
            r_sanity = float(np.corrcoef(y_va, y_pred_va)[0,1]) if len(y_va) > 2 else 0.0
            print(f"[trainer_reg][DEBUG] val std(y_pred)={std_pred:.6g}, corr={r_sanity:.4f}")

        # Early-Stopping 規則
        if primary == "mixed":
            improved = (epoch == 1) or (val_mixed < (best_reg_mixed - 1e-6))
        elif primary == "pearson":
            improved = (epoch == 1) or (m_va["pearson"] > (best_reg_pearson + 1e-6))
        elif primary == "spearman":                                        
            improved = (epoch == 1) or (m_va["spearman"] > (best_reg_spearman + 1e-6))
        else:
            improved = (epoch == 1) or (avg_va_loss < (best_reg_val_loss - 1e-6))

        if improved:
            best_reg_val_loss = avg_va_loss
            best_reg_pearson  = m_va["pearson"]
            best_reg_spearman = m_va.get("spearman", best_reg_spearman)  # NEW
            best_reg_mixed    = val_mixed
            best_epoch        = epoch
            best_state        = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_va_arrays    = (y_va.copy(), y_pred_va.copy())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_mixed={best_reg_mixed:.6g} | val_pearson={best_reg_pearson:.4f}")
                break

        history.append({
            "epoch": epoch,
            "train_loss": avg_tr_loss, "val_loss": avg_va_loss,
            "train_mse": m_tr.get("mse", np.nan), "val_mse": m_va.get("mse", np.nan),
            "train_mae": m_tr.get("mae", np.nan), "val_mae": m_va.get("mae", np.nan),
            "train_rmse": m_tr.get("rmse", np.nan), "val_rmse": m_va.get("rmse", np.nan),
            "train_pearson": m_tr.get("pearson", np.nan), "val_pearson": m_va.get("pearson", np.nan),

            # ★ 這兩行是關鍵：讓 save_fold_metrics 找得到 spearman 並畫曲線
            "train_spearman": m_tr.get("spearman", np.nan),
            "val_spearman":   m_va.get("spearman", np.nan),

            "val_pearson_global": m_va.get("pearson_global", np.nan),
            "val_spearman_global": m_va.get("spearman_global", np.nan),
            "val_ema_mse_global": m_va.get("ema_mse_global", np.nan),
            "val_mixed": val_mixed,
        })
        print(f"{prefix}[Epoch {epoch:03d}] tr_loss={avg_tr_loss:.6f} | val_loss={avg_va_loss:.6f} "
                f"| val_pearson={m_va['pearson']:.4f} | val_spearman={m_va.get('spearman', np.nan):.4f} "  
                f"| val_rmse={m_va['rmse']:.6f}")
        
    # 載回最佳
    model.load_state_dict(best_state)
    model.eval()

    # -------- TEST --------
    te_tgts, te_preds = [], []
    test_loss_sum, test_n = 0.0, 0

    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            pred = model(Xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs

            te_preds.append(pred.detach().float().cpu())
            te_tgts.append(yb.detach().cpu())

    avg_te_loss = test_loss_sum / max(1, test_n)
    y_te = torch.cat(te_tgts,  dim=0).numpy().astype(np.float64)
    y_pred_te = torch.cat(te_preds, dim=0).numpy().astype(np.float64)
    m_te = compute_regression_metrics(y_te, y_pred_te)
    print(f"{prefix}Test_reg: pearson={m_te['pearson']:.4f} | spearman={m_te['spearman']:.4f} "
      f"| rmse={m_te['rmse']:.6f} | mae={m_te['mae']:.6f}\n")

    val_metrics_reg = {
        "val_pearson": float(best_reg_pearson),
        "val_spearman": float(best_reg_spearman),  # 新增，方便 cv_summary 彙總
        "val_mixed": float(best_reg_mixed),
        "val_loss": float(best_reg_val_loss)
    }
    
    if best_va_arrays is not None and best_va_arrays[0] is not None:        
        _m = compute_regression_metrics(best_va_arrays[0], best_va_arrays[1])
        val_metrics_reg.update({
            "val_rmse": float(_m["rmse"]),
            "val_mae":  float(_m["mae"]),
            "val_mse":  float(_m["mse"]),
        })



    result = {
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        
        "test_metrics_reg": {
            "test_loss": avg_te_loss, **m_te
        },
        "temperature": 1.0,
        "best_val_mixed": float(best_reg_mixed),
        "best_val_pearson": float(best_reg_pearson),
        "best_val_spearman": float(best_reg_spearman),
        "best_val_arrays": {
            "y_va": best_va_arrays[0].tolist() if best_va_arrays else None,
            "y_pred_va": best_va_arrays[1].tolist() if best_va_arrays else None
        }
    }
    result["val_metrics_reg"] = val_metrics_reg  # ★★ 新增：VAL 摘要

    # ====== Reg -> Cls：用「val 找到的最佳門檻」轉為 test 的分類並保存 ======
    # reg2cls = cfg["regression_to_class"]
    # if reg2cls["enabled"]:
    #     true_thr   = float(reg2cls.get("true_threshold", 0.0))
    #     beta       = float(reg2cls.get("fbeta", 0.5))
    #     grid_points = int(reg2cls.get("grid_points", 101))

    #     # 1) 用「最佳 epoch 的 val 預測」來挑門檻（避免 test 洩漏）
    #     if best_va_arrays is not None and best_va_arrays[0] is not None:
    #         y_va_best, y_pred_va_best = best_va_arrays
    #         best_t_val, best_f_val = find_best_threshold_for_regression(
    #             y_true_reg=y_va_best, y_pred_reg=y_pred_va_best,
    #             fbeta=beta, grid_points=grid_points, true_threshold=true_thr
    #         )
    #     else:
    #         # 理論上不會發生；保底（用 test 掃，僅供報表，不建議用於正式評估）
    #         sweep_info = plot_regression_threshold_sweep(
    #             y_true_reg=y_te, y_pred_reg=y_pred_te,
    #             true_threshold=true_thr, beta=beta,
    #             grid_points=grid_points, save_dir=export_dir, prefix=f"fold_{fold_id}_"
    #         )
    #         best_t_val = float(sweep_info["best_threshold"])
    #         best_f_val = float(sweep_info["best_fbeta"])


    #     # 2) 在 test 上用「val 的最佳門檻」做二值化並計分
    #     y_true_te_bin = binarize_regression(y_te, threshold=true_thr)          # 以真值閾值把 y_true 轉成 0/1
    #     y_hat_te_bin  = (y_pred_te >= best_t_val).astype(int)                  # 用「val 的最佳門檻」切 y_pred

    #     m_te_cls = compute_cls_metrics(y_true_te_bin, y_hat_te_bin)                # acc / macro-F1 / F0.5 / MCC...


    #     # （可選）為了診斷，也可以在 test 上「自己找一個最佳門檻」但**僅作報表**，不要拿來主張泛化表現
    #     best_t_te = None
    #     if bool(reg2cls.get("also_sweep_on_test_for_report", False)):
    #         best_t_te, best_f_te = find_best_threshold_for_regression(
    #             y_true_reg=y_te, y_pred_reg=y_pred_te,
    #             fbeta=beta, grid_points=grid_points, true_threshold=true_thr
    #         )

    #     # 3) 併入 result，並可選保存 test 的原始 y / y_pred（之後你要重算別的 β 或門檻會很方便）
    #     result_reg2cls = {
    #         "true_threshold": true_thr,        # 把連續 y 轉 0/1 的真值門檻
    #         "beta": beta,                      # F_beta 的 beta
    #         "best_threshold_val": float(best_t_val),
    #         "val_best_fbeta": float(best_f_val),
    #         "test_metrics_cls_at_val_threshold": {
    #             "acc": m_te_cls["acc"],
    #             "macro_f1": m_te_cls["macro_f1"],
    #             "macro_precision": m_te_cls["macro_precision"],
    #             "macro_recall": m_te_cls["macro_recall"],
    #             "f_05_macro": m_te_cls["f_05_macro"],
    #             "MCC": m_te_cls["mcc"],
    #         }
    #     }
    
    # ====== Reg -> 3-class Cls（對稱雙閾值 ±t）======
    reg2cls = cfg.get("regression_to_class", {})
    if reg2cls.get("enabled", False):
        # 取最佳 epoch 的 val (y, y_pred)；若沒有就退回 test（僅診斷用）
        if best_va_arrays is not None and best_va_arrays[0] is not None:
            y_va_best, y_pred_va_best = best_va_arrays
        else:
            y_va_best, y_pred_va_best = y_te, y_pred_te  # fallback only

        method       = str(reg2cls.get("method", "opt_fbeta_on_pred")).lower()
        beta         = float(reg2cls.get("fbeta", 0.5))
        grid_points  = int(reg2cls.get("grid_points", 201))
        q_flat       = float(reg2cls.get("q_flat", 0.3))
        # 讓 flat 比例 ≈ 1 - 2*q_flat；加上安全界限
        flat_ratio   = float(reg2cls.get("flat_ratio", 1.0 - 2.0 * q_flat))
        flat_ratio   = float(np.clip(flat_ratio, 0.05, 0.9))

        def tri_from(a: np.ndarray, thr: float) -> np.ndarray:
            """對稱雙閾值：<-thr=0, |<=thr|=1, >thr=2"""
            lab = np.ones_like(a, dtype=int)
            lab[a >  +thr] = 2
            lab[a <  -thr] = 0
            return lab

        # --- 定義真值的平坦帶閾值 thr_true（把 y_true 切 3 類） ---
        if method == "fixed_bps":
            thr_true = float(reg2cls.get("flat_band_bps", 5.0)) / 10000.0
        else:
            # 用 val 的 |y_true| 分位數，讓 flat ≈ flat_ratio
            thr_true = float(np.quantile(np.abs(y_va_best), flat_ratio))

        # --- 定義預測的平坦帶閾值 thr_pred（把 y_pred 切 3 類） ---
        if method == "abs_quantile_true":
            thr_pred = float(thr_true)
        elif method == "abs_quantile_pred":
            thr_pred = float(np.quantile(np.abs(y_pred_va_best), flat_ratio))
        else:
            # opt_fbeta_on_pred：在 |y_pred(val)| 的分位數網格上找 t，使 macro Fβ 最大
            qs   = np.linspace(0.05, 0.95, grid_points)
            cand = np.quantile(np.abs(y_pred_va_best), qs)
            y_true_va_cls = tri_from(y_va_best, thr_true)
            best_t, best_score = None, -1.0
            for t in cand:
                y_pred_va_cls = tri_from(y_pred_va_best, float(t))
                sc = fbeta_score(y_true_va_cls, y_pred_va_cls, beta=beta,
                                average="macro", zero_division=0)
                if sc > best_score + 1e-12:
                    best_score, best_t = sc, float(t)
            thr_pred = float(best_t if best_t is not None else thr_true)

        # --- 用 val 定好的門檻，在 test 上評估 ---
        y_true_te_cls = tri_from(y_te,      thr_true)
        y_pred_te_cls = tri_from(y_pred_te, thr_pred)
        m_te_cls      = compute_cls_metrics(y_true_te_cls, y_pred_te_cls)

        # --- 畫圖（暫以 one-hot 機率；要 soft-binning 再加強）---
        y_prob_te_3 = np.eye(3, dtype=float)[y_pred_te_cls]
        plot_test_eval(
            y_true=y_true_te_cls, y_pred=y_pred_te_cls, y_prob=y_prob_te_3,
            class_names=cfg["model"].get("class_names", ["down","flat","up"]),
            save_dir=os.path.join(export_dir, "cls_results_3c"),
            prefix=f"fold_{fold_id}_",
            threshold=None
        )

        # --- 回寫結果 ---
        result["regression_to_class_3c"] = {
            "method": method,
            "flat_ratio": float(flat_ratio),
            "thr_true": float(thr_true),
            "thr_pred": float(thr_pred),
            "test_metrics_3c": {
                "acc": m_te_cls["acc"],
                "macro_f1": m_te_cls["macro_f1"],
                "macro_precision": m_te_cls["macro_precision"],
                "macro_recall": m_te_cls["macro_recall"],
                "macro_f05": m_te_cls.get("macro_f05", m_te_cls.get("f_05_macro", 0.0)),
                "mcc": m_te_cls["mcc"],
            }
        }
        if bool(reg2cls.get("save_test_arrays", False)):
            result["regression_to_class_3c"]["arrays"] = {
                "y_te": y_te.tolist(),
                "y_pred_te": y_pred_te.tolist(),
            }



        # if best_t_te is not None:
        #     result_reg2cls["diagnostic_test_best_threshold"] = float(best_t_te)




        # # === 以 val 做機率校準（避免 test 洩漏） ===
        # from sklearn.isotonic import IsotonicRegression

        # # 1) 先把 val 的連續 y / ẑ 轉二分類標籤當作校準目標
        # y_va_best, y_pred_va_best = best_va_arrays  # 這是你早就存好的 (val_y, val_pred) of best epoch
        # y_va_bin = (y_va_best >= true_thr).astype(int)

        # # 2) 對回歸分數做 Isotonic calibration 得到 p(pos)
        # iso = IsotonicRegression(out_of_bounds="clip")
        # iso.fit(y_pred_va_best, y_va_bin)

        # p_te = iso.transform(y_pred_te)                         # p(pos) on test
        # p_te = np.clip(p_te, 1e-6, 1.0 - 1e-6)                 # 數值安全
        # y_prob_te = np.stack([1.0 - p_te, p_te], axis=1)       # (N,2) → [p(neg), p(pos)]

        # # 3) 畫圖到 export_dir/cls_results
        # cls_dir = os.path.join(export_dir, "cls_results")
        # plot_test_eval(
        #     y_true=y_true_te_bin,               # 0/1
        #     y_pred=y_hat_te_bin,                # 0/1（用 val 門檻切出來）
        #     y_prob=y_prob_te,                   # (N,2)
        #     save_dir=cls_dir,
        #     prefix=f"fold_{fold_id}_",
        #     class_names=["neg", "pos"],         # ★ 強制二分類名稱，避免 3 類設定造成維度不符
        #     threshold=float(best_t_val)         # 讓圖上標記門檻
        # )
        # # result["regression_to_class"] = result_reg2cls  # ← 關鍵

    # 繪圖 / 匯出
    plot_regression_eval(
        y_true=y_te, y_pred=y_pred_te,
        save_dir=export_dir, prefix=f"fold_{fold_id}_"
    )
    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    # save_result(fold_id=fold_id, export_dir=export_dir, result=result)
    return model, result
