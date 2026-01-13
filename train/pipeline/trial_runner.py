# train/pipeline/trial_runner.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import numpy as np
import pandas as pd
import torch
import optuna

# -- 1) 來自 search.space（新位置）：避免拿舊 objective_utils --
from train.pipeline.search.space import (
    get_task_type, _format_score_tag, _safe_rename_trial_dir, compute_trial_score
)

# -- single-fold trainer factory --
from train.training.trainers.utils import get_trainer
from train.evaluation.exporters.cv_summary import save_cv_summary
from train.evaluation.exporters.tbm_exporter import TBMExporter
from train.inference.predictor import Predictor
from train.evaluation.utils import save_fold_metrics
from train.evaluation.reporters.classification_reporter import ClassificationReporter
from train.evaluation.reporters.regression_reporter import RegressionReporter

# -- 3) Loader：優先新 -> 退回舊 --
from train.data.dataloaders.time_loader import make_time_loaders_for_fold  # type: ignore
from train.data.dataloaders.event_loader import make_event_loaders_for_fold  # type: ignore
from train.data.dataloaders.base import load_precomputed_features, flatten_micro_features
from train.models.model_factory import build_model


@dataclass
class TrialOutputs:
    """
    1. 說明: 封裝一次 Trial 的主要輸出，方便 pipeline 與上層使用
    2. inputs: （建構子自動賦值）
    3. return: 無（資料屬性）
    """
    mean_score: float
    trial_dir: Path
    val_avg: Dict[str, float]
    test_avg: Dict[str, float]
    fold_results: List[Dict[str, Any]]
    fold_models_for_infer: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]]


class TrialRunner:
    """
    1. 說明: 單次 Optuna Trial 的執行器；負責：
       - 依 cfg/task_type 產生每個 fold 的資料載入器
       - 建模、訓練、收集指標、PRUNE 回報
       - 彙整 CV 平均、命名試驗資料夾、寫入報表與（可選）TBM 併回
       注意：不做「超參建議」與「fold 生成」，那屬於 objective_utils。
    2. inputs: 於 run() 傳入
    3. return: TrialOutputs（含 mean_score 與各彙整資訊）
    """

    def run(
        self,
        *,
        trial: optuna.Trial,
        cfg: Dict[str, Any],
        df,  # 時間索引骨架（或完整 DF；維持與你現行 objective 相容）
        trial_dir: Path,
        folds: List[Dict[str, Any]],
        device: Optional[str] = None,
        effective_seed: Optional[int] = None,
    ) -> TrialOutputs:
        """
        1. 說明: 執行一次 Trial 的完整折訓練流程與彙整
        2. inputs:
           - trial: Optuna Trial 物件（用於回報/PRUNE）
           - cfg: 已套用建議超參的設定 dict
           - df: 你原先傳入 make_*_loaders 的 df（保相容）
           - trial_dir: 此次 trial 的輸出資料夾
           - folds: 由 make_folds() 產生的 Fold 規格清單
           - device: "cuda" 或 "cpu"（未提供時使用 cfg['device']）
           - effective_seed: 這次真正使用的 seed（用於寫回可重現 config）
        3. return:
           - TrialOutputs: 包含 mean_score、平均指標、每折結果與模型清單等
        """
        task_type = get_task_type(cfg)
        device = device or str(cfg.get("device", "cuda")).lower()
        train_one_fold = get_trainer(cfg)

        # === 每 fold 訓練 ===
        fold_scores: List[float] = []
        fold_results: List[Dict[str, Any]] = []
        fold_models_for_infer: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]] = []

        label_mode = str(cfg.get("label", {}).get("mode", "")).lower()
        also_xgb = bool(cfg.get("also_XGB", cfg.get("also_XGB", False)))

        for i, fold in enumerate(folds):
            fold_dir = trial_dir / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # 依任務選擇資料載入器
            if label_mode == "event_tbm":
                tr_loader, va_loader, te_loader, info = make_event_loaders_for_fold(
                    df, [], fold, cfg, also_XGB=also_xgb
                )
            else:
                tr_loader, va_loader, te_loader, info = make_time_loaders_for_fold(
                    df, None, None, fold, cfg, also_XGB=also_xgb
                )

            # XGB 選配（不再硬性相依）
            cfg_fold = dict(cfg)  # 淺拷貝足夠：下層 trainer 會 copy/record
            if also_xgb and (("XGB" not in info) or (info["XGB"] is None)):
                # 不中斷訓練，僅提示；若你仍想嚴格，改成 raise ValueError
                print("[Runner][WARN] also_XGB=True 但資料載入器未回傳 XGB pack -> 跳過 XGB 訓練")
                cfg_fold.pop("_xgb_pack", None)
            elif also_xgb:
                cfg_fold["_xgb_pack"] = info["XGB"]
            else:
                cfg_fold.pop("_xgb_pack", None)

            feature_columns = info.get("feat_cols")
            if feature_columns is not None:
                feature_columns = list(feature_columns)
            n_features = len(feature_columns) if feature_columns is not None else None

            model = build_model(cfg, n_features, feature_columns)

            # 單 fold 訓練
            model_trained, result = train_one_fold(
                model, tr_loader, va_loader, te_loader,
                cfg_fold, device=device, fold_id=i, export_dir=fold_dir
            )
            if result is not None and feature_columns is not None:
                result["_feature_columns"] = feature_columns
            fold_results.append(result)

            self._export_fold_artifacts(
                task_type=task_type,
                fold_result=result,
                export_dir=fold_dir,
                fold_id=i,
            )

            # 釋放顯存，保留 CPU 權重供 TBM 併回
            model_ref: Any = None
            try:
                checkpoint_payload = None
                if result is not None:
                    checkpoint_payload = result.get("state_dict")
                if checkpoint_payload:
                    checkpoint_path = fold_dir / f"model_state_{i}.pt"
                    best_thr = result.get("best_val_thresh", None)
                    temperature = result.get("temperature", None)
                    meta = {
                        "state_dict": checkpoint_payload,
                        "feature_columns": result.get("_feature_columns"),
                        "model_cfg": cfg.get("model"),
                        "temperature": float(temperature) if temperature is not None else None,
                        "best_val_thresh": float(best_thr) if best_thr is not None else None,
                        "amp": (cfg.get("train", {}) or {}).get("amp"),
                        "amp_dtype": (cfg.get("train", {}) or {}).get("amp_dtype"),
                    }
                    torch.save(meta, checkpoint_path)
                    model_ref = checkpoint_path
                    # 釋放記憶體：state_dict 已寫入檔案
                    result.pop("state_dict", None)
                else:
                    model_ref = model_trained
                if model_trained is not None:
                    try:
                        model_trained = model_trained.to("cpu")
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Runner][WARN] failed to persist fold model: {e}")
                model_ref = model_trained

            fold_models_for_infer.append((model_ref, fold, result))

            # 試驗分數 + PRUNE 回報
            score = compute_trial_score(result, cfg)
            fold_scores.append(float(score))
            trial.report(float(score), step=i)
            if cfg.get("objective", {}).get("enable_prune", False) and trial.should_prune():
                raise optuna.TrialPruned()

        # === CV 平均 ===
        val_avg, test_avg = self._compute_cv_avgs(task_type, fold_results)
        self._print_cv(task_type, val_avg, test_avg, len(fold_results))

        # === Optional: TBM 併回 ===
        self._maybe_export_tbm(cfg, folds, fold_models_for_infer, trial_dir, task_type)

        # === 保存特徵欄位 ===
        try:
            feat_sets = [
                set(res.get("_feature_columns", []))
                for res in fold_results
                if isinstance(res, dict) and res.get("_feature_columns")
            ]
            if feat_sets:
                # 取第一個的順序
                ordered_cols = fold_results[0].get("_feature_columns", [])
                common = set.intersection(*feat_sets)
                final_cols = [c for c in ordered_cols if c in common]
                feat_path = trial_dir / "feat.txt"
                with open(feat_path, "w", encoding="utf-8") as fh:
                    for col in final_cols:
                        fh.write(f"{col}\n")
        except Exception as e:
            print(f"[Runner][WARN] unable to write feat.txt: {e}")

        # === 寫 CV 摘要 ===
        save_cv_summary(fold_results, export_dir=trial_dir, task_type=task_type)
        self._export_holdout_metrics(
            task_type=task_type,
            fold_results=fold_results,
            export_dir=trial_dir,
        )

        # === 依型別為 trial 資料夾加上分數 tag ===
        trial_dir = self._tag_trial_dir(trial_dir, task_type, test_avg, trial=trial)

        # === 存可重現 config（用實際 effective_seed）===
        mean_score = float(np.mean(fold_scores)) if fold_scores else float(-1e9)
        self._dump_reproducible_cfg(cfg, trial_dir, mean_score, effective_seed)

        return TrialOutputs(
            mean_score=mean_score,
            trial_dir=trial_dir,
            val_avg=val_avg,
            test_avg=test_avg,
            fold_results=fold_results,
            fold_models_for_infer=fold_models_for_infer,
        )

    # ----------------- helpers -----------------

    def _export_fold_artifacts(
        self,
        *,
        task_type: str,
        fold_result: Dict[str, Any],
        export_dir: Path,
        fold_id: int,
    ) -> None:
        """Generate per-fold plots and metrics outside the trainers."""

        history = fold_result.get("history") or []
        if history:
            save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")

        prefix = f"fold_{fold_id}_"

        if task_type == "classification":
            payload = fold_result.get("eval_payload") or {}
            y_true = payload.get("y_true")
            y_pred = payload.get("y_pred")
            y_prob = payload.get("y_prob")
            if y_true is not None and y_pred is not None and y_prob is not None:
                reporter = ClassificationReporter(
                    save_dir=export_dir,
                    prefix=prefix,
                    class_names=payload.get("class_names"),
                )
                reporter.plot_eval(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    threshold=payload.get("best_threshold"),
                )

        else:  # regression
            payload = fold_result.get("eval_payload") or {}
            y_true = payload.get("y_true")
            y_pred = payload.get("y_pred")
            if y_true is not None and y_pred is not None:
                reporter = RegressionReporter(save_dir=export_dir, prefix=prefix)
                reporter.plot_eval(y_true=y_true, y_pred=y_pred)

            reg2cls = (fold_result.get("regression_to_class_3c") or {}).get("eval_payload")
            if reg2cls:
                cls_dir = export_dir / "cls_results_3c"
                reporter = ClassificationReporter(
                    save_dir=cls_dir,
                    prefix=prefix,
                    class_names=reg2cls.get("class_names"),
                )
                reporter.plot_eval(
                    y_true=reg2cls.get("y_true"),
                    y_pred=reg2cls.get("y_pred"),
                    y_prob=reg2cls.get("y_prob"),
                    threshold=None,
                )

    def _export_holdout_metrics(
        self,
        *,
        task_type: str,
        fold_results: List[Dict[str, Any]],
        export_dir: Path,
    ) -> None:
        """Aggregate all test folds once and persist metrics/confusion matrix."""
        if task_type != "classification":
            return

        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )

        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []
        y_prob_list: List[np.ndarray] = []
        thresholds: List[float] = []
        class_names: Optional[List[str]] = None

        for res in fold_results:
            payload = (res or {}).get("eval_payload") or {}
            y_true = payload.get("y_true")
            y_pred = payload.get("y_pred")
            if y_true is None or y_pred is None:
                continue
            y_true_list.append(np.asarray(y_true))
            y_pred_list.append(np.asarray(y_pred))
            y_prob = payload.get("y_prob")
            if y_prob is not None:
                y_prob_list.append(np.asarray(y_prob))
            if class_names is None and payload.get("class_names") is not None:
                class_names = list(payload["class_names"])
            if payload.get("best_threshold") is not None:
                thresholds.append(float(payload["best_threshold"]))

        if not y_true_list:
            return

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else None

        if class_names is not None:
            labels_for_metrics = list(range(len(class_names)))
        else:
            labels_for_metrics = sorted({int(v) for v in np.unique(np.concatenate([y_true, y_pred]))})

        acc = float(accuracy_score(y_true, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0, labels=labels_for_metrics
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels_for_metrics
        )
        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=labels_for_metrics
        )
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels_for_metrics).tolist()
        if class_names is not None:
            labels = [class_names[i] for i in labels_for_metrics]
        else:
            labels = [str(i) for i in labels_for_metrics]
        per_class = [
            {
                "class": labels[i],
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        ]

        metrics_dir = export_dir / "holdout_eval"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "samples": int(len(y_true)),
            "accuracy": acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
            "confusion_matrix": conf_mat,
        }
        if thresholds:
            metrics["threshold_mean"] = float(np.mean(thresholds))
            metrics["thresholds"] = thresholds

        with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)

        if y_prob is not None:
            reporter = ClassificationReporter(
                save_dir=metrics_dir,
                prefix="holdout_",
                class_names=class_names,
            )
            reporter.plot_eval(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                threshold=(float(np.mean(thresholds)) if thresholds else None),
            )

    def _numeric_only(self, d: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        1. 說明: 過濾非數值鍵值，避免平均時出錯
        2. inputs: d: 字典或 None
        3. return: 只含數值型的字典
        """
        out: Dict[str, float] = {}
        for k, v in (d or {}).items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                out[k] = float(v)
        return out

    def _avg_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        1. 說明: 對齊鍵名後逐欄平均
        2. inputs: rows: 字典清單
        3. return: 每個指標鍵的平均數
        """
        pool: Dict[str, List[float]] = {}
        for d in rows:
            for k, v in self._numeric_only(d).items():
                pool.setdefault(k, []).append(v)
        return {k: float(np.mean(vs)) for k, vs in pool.items()}

    def _compute_cv_avgs(
        self, task_type: str, fold_results: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        1. 說明: 依任務型別聚合 VAL/TEST 平均
        2. inputs:
           - task_type: "classification" | "regression"
           - fold_results: 每折結果清單
        3. return: (val_avg, test_avg)
        """
        if task_type == "classification":
            val_rows  = [r.get("val_metrics",  {}) for r in fold_results]
            test_rows = [r.get("test_metrics", {}) for r in fold_results]
        else:
            val_rows  = [r.get("val_metrics_reg",  {}) for r in fold_results]
            test_rows = [r.get("test_metrics_reg", {}) for r in fold_results]
        return self._avg_rows(val_rows), self._avg_rows(test_rows)

    def _print_cv(self, task_type: str, val_avg: Dict[str, float], test_avg: Dict[str, float], k: int) -> None:
        """
        1. 說明: 以人類可讀的方式列印 CV 結果
        2. inputs: task_type, val_avg, test_avg, k 折數
        3. return: None
        """
        print(f"\n[CV] {task_type.upper()} | folds={k}")
        if val_avg:
            print("[CV] VAL  avg:")
            for k in sorted(val_avg):
                print(f"  {k}: {val_avg[k]:.6g}")
        if test_avg:
            print("[CV] TEST avg:")
            for k in sorted(test_avg):
                print(f"  {k}: {test_avg[k]:.6g}")

    def _tag_trial_dir(
        self, trial_dir: Path, task_type: str, test_avg: Dict[str, float], trial: optuna.Trial
    ) -> Path:
        """
        1. 說明: 依任務指標為資料夾加上分數 tag，並寫入 trial user_attr
        2. inputs: trial_dir, task_type, test_avg, trial
        3. return: 可能被改名後的新路徑
        """
        if task_type == "classification":
            mcc_cv = test_avg.get("test_mcc", test_avg.get("mcc", np.nan))
            trial.set_user_attr("test_mcc_avg", float(mcc_cv) if np.isfinite(mcc_cv) else None)
            if np.isfinite(mcc_cv):
                tag = _format_score_tag("mcc", mcc_cv, digits=3, signed=True)
                return _safe_rename_trial_dir(trial_dir, [tag])
        else:
            pearson_cv = test_avg.get("pearson", np.nan)
            trial.set_user_attr("test_pearson_avg", float(pearson_cv) if np.isfinite(pearson_cv) else None)
            if np.isfinite(pearson_cv):
                tag = _format_score_tag("pearson", pearson_cv, digits=4, signed=True)
                return _safe_rename_trial_dir(trial_dir, [tag])
        return trial_dir

    def _dump_reproducible_cfg(self, cfg: Dict[str, Any], trial_dir: Path, mean_score: float, effective_seed: Optional[int]) -> None:
        """
        1. 說明: 將「實際使用的 seed」覆寫後的 cfg 存成可重現 YAML
        2. inputs: cfg, trial_dir, mean_score, effective_seed
        3. return: None
        """
        import yaml
        cfg = dict(cfg)
        if effective_seed is not None:
            cfg["seed"] = int(effective_seed)
        path = trial_dir / f"trial_config_{cfg['objective']['primary_metric']}={mean_score:.6g}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        print(f"[runner] saved reproducible config -> {path}")

    def _maybe_export_tbm(
        self,
        cfg: Dict[str, Any],
        folds: List[Dict[str, Any]],
        fold_models_for_infer: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]],
        trial_dir: Path,
        task_type: str,
    ) -> None:
        """
        1. 說明: 依設定執行 TBM 併回匯出（僅分類任務）
        2. inputs: cfg, folds, fold_models_for_infer, trial_dir, task_type
        3. return: None
        """
        try:
            post_infer = cfg.get("post_infer", {}) or {}
            debug_lines = [
                f"task_type={task_type}",
                f"post_infer_enabled={bool(post_infer.get('enabled', False))}",
                f"trial_dir={trial_dir}",
                f"n_fold_models={len(fold_models_for_infer)}",
            ]
            if bool(post_infer.get("enabled", False)) and task_type == "classification":
                if not folds or not fold_models_for_infer:
                    debug_lines.append("skip_export=empty_folds_or_models")
                    with open(trial_dir / "post_infer_debug.txt", "w", encoding="utf-8") as dfp:
                        dfp.write("\n".join(debug_lines))
                    return
                ds = str(post_infer["date_start"])
                de = str(post_infer["date_end"])
                out_csv = str(post_infer.get("csv_path_override") or cfg.get("label", {}).get("tbm_csv_path"))
                trial_name = trial_dir.name
                match = re.match(r"^(trial_\d+)", trial_name)
                trial_token = match.group(1) if match else trial_name
                keep_sides = str(cfg["label"]["keep_sides"]).lower()
                lookback = cfg["label"]["lookback"]
                lookback_token = f"{int(lookback)}"
                save_csv = trial_dir / f"{trial_token}_{keep_sides}_{lookback_token}.csv"
                debug_lines += [f"date_range=[{ds},{de}]", f"tbm_src={out_csv}", f"save_to={save_csv}"]

                # 準備特徵（與 loader/tbm_exporter 同步展平邏輯）
                tbm_df = pd.read_csv(out_csv, parse_dates=["t0"])
                feat_df = load_precomputed_features(path=cfg["data"]["path"])
                micro_cfg = (cfg.get("data", {}) or {}).get("micro", {}) or {}
                micro_df = None
                micro_end = None
                if micro_cfg.get("enabled") and micro_cfg.get("path"):
                    micro_df = load_precomputed_features(path=micro_cfg["path"])
                    if len(micro_df.index):
                        micro_end = pd.DatetimeIndex(micro_df.index).max()

                def _to_utc(ts_like):
                    ts = pd.Timestamp(ts_like)
                    if ts.tzinfo is None:
                        return ts.tz_localize("UTC")
                    return ts.tz_convert("UTC")

                cv_start = _to_utc(cfg["cv"]["start_date"])
                ts_end_param = _to_utc(de)
                ts_end_candidates = [ts_end_param, pd.DatetimeIndex(feat_df.index).max()]
                if micro_end is not None:
                    ts_end_candidates.append(micro_end)
                ts_end = min(ts_end_candidates)

                if micro_df is not None:
                    window_len = int(micro_cfg.get("window_len", 15))
                    feat_df = flatten_micro_features(
                        feat_df=feat_df,
                        micro_df=micro_df,
                        cv_start=cv_start,
                        ts_end=ts_end,
                        window_len=window_len,
                    )

                feat_df = feat_df.loc[(feat_df.index >= cv_start) & (feat_df.index <= ts_end)]

                # 收集 checkpoint 路徑（僅保留有路徑的）
                model_paths = []
                for model_ref, _fold, _res in fold_models_for_infer:
                    if isinstance(model_ref, (str, Path)):
                        model_paths.append(Path(model_ref))
                if not model_paths:
                    debug_lines.append("skip_export=no_checkpoints")
                else:
                    predictor = Predictor(cfg=cfg)
                    pred_df = predictor.predict_vote(
                        feat_df,
                        tbm_df=tbm_df,
                        model_paths_or_dir=model_paths,
                        date_start=ds,
                        date_end=de,
                    )
                    exporter = TBMExporter(cfg)
                    exporter.export_csv(pred_df, save_to_path=str(save_csv))
                    print(f"[PostInfer] Saved TBM with predictions: {save_csv}")
                    debug_lines.append("status=ok")

            # 記錄 debug
            with open(trial_dir / "post_infer_debug.txt", "w", encoding="utf-8") as dfp:
                dfp.write("\n".join(debug_lines))

        except Exception as e:
            print(f"[PostInfer][WARN] skip due to error: {e}")
            try:
                with open(trial_dir / "post_infer_error.txt", "w", encoding="utf-8") as ef:
                    ef.write(str(e))
            except Exception:
                pass


def run_trial(
    *,
    optuna_trial,
    cfg: Dict[str, Any],
    df,
    trial_dir: Path,
    folds: List[Dict[str, Any]],
    device: Optional[str] = None,
    effective_seed: Optional[int] = None,
) -> TrialOutputs:
    """Convenience wrapper used by the Optuna objective layer."""
    runner = TrialRunner()
    return runner.run(
        trial=optuna_trial,
        cfg=cfg,
        df=df,
        trial_dir=trial_dir,
        folds=folds,
        device=device,
        effective_seed=effective_seed,
    )
