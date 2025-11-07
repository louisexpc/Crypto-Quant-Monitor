# models/xgb_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from xgboost.sklearn import (
    XGBRegressor as _XGBRegressor,
    XGBClassifier as _XGBClassifier,
)

from train.models.registry import register

@dataclass
class XGBRegressorModel:
    """
    輕量 wrapper：
    - from_cfg(cfg) 讀取 YAML 中的 model 參數
    - build() 產生 xgboost.XGBRegressor
    - early_stopping_rounds 留在這裡讓 trainer 可讀
    """
    params: Dict[str, Any]
    early_stopping_rounds: int = 100

    @classmethod
    def from_cfg(cls, cfg: dict) -> "XGBRegressorModel":
        # 必須存在的頂層區塊
        if "model" not in cfg:
            raise KeyError("config 缺少 'model' 區塊")
        m = cfg["model"]

        # ---- 讀取 model 參數 ----
        params = dict(        
            n_estimators      = int(m["n_estimators"]),     
            max_depth         = int(m["max_depth"]),      
            learning_rate     = float(m["learning_rate"]),  
            subsample         = float(m["subsample"]),     
            colsample_bytree  = float(m["colsample_bytree"]),
            reg_alpha         = float(m["reg_alpha"]),       
            reg_lambda        = float(m["reg_lambda"]),     
            min_child_weight  = float(m["min_child_weight"]),
            tree_method       = str(m["tree_method"]),           # "hist" | "gpu_hist"
            n_jobs            = int(m["n_jobs"]),           
            device            = cfg["device"],
            objective         = "reg:squarederror",
            random_state      = cfg["seed"],
            )

        esr = int(m["early_stopping_rounds"])
        return cls(params=params, early_stopping_rounds=esr)

    def build(self):
        return _XGBRegressor(**self.params)


@dataclass
class XGBClassifierModel:
    """
    XGBoost classifier wrapper used by the training pipeline.
    Responsible for translating cfg -> sklearn constructor args.
    """
    params: Dict[str, Any]
    num_classes: int
    early_stopping_rounds: int = 100

    @classmethod
    def from_cfg(cls, cfg: dict) -> "XGBClassifierModel":
        if "model" not in cfg:
            raise KeyError("config 缺少 'model' 區塊")
        m = cfg["model"]

        num_classes = int(m.get("num_classes", cfg.get("model", {}).get("num_classes", 2)))
        device = cfg.get("device", "cpu")

        params: Dict[str, Any] = dict(
            n_estimators=int(m["n_estimators"]),
            max_depth=int(m["max_depth"]),
            learning_rate=float(m["learning_rate"]),
            subsample=float(m["subsample"]),
            colsample_bytree=float(m["colsample_bytree"]),
            reg_alpha=float(m.get("reg_alpha", 0.0)),
            reg_lambda=float(m.get("reg_lambda", 1.0)),
            min_child_weight=float(m.get("min_child_weight", 1.0)),
            tree_method=str(m.get("tree_method", "hist")),
            n_jobs=int(m.get("n_jobs", -1)),
            random_state=int(cfg.get("seed", 42)),
            device=device,
        )

        # Optional knobs commonly surfaced in YAML configs
        for key in ("gamma", "max_delta_step", "max_leaves", "max_bin", "colsample_bylevel", "colsample_bynode", "scale_pos_weight"):
            if key in m:
                params[key] = m[key]

        if num_classes <= 2:
            params["objective"] = str(m.get("objective", "binary:logistic"))
            params.setdefault("eval_metric", "logloss")
        else:
            params["objective"] = str(m.get("objective", "multi:softprob"))
            params["num_class"] = num_classes
            params.setdefault("eval_metric", "mlogloss")

        esr = int(m.get("early_stopping_rounds", 100))
        return cls(params=params, num_classes=num_classes, early_stopping_rounds=esr)

    def build(self):
        return _XGBClassifier(**self.params)


@register("xgb")
def build_xgb_model(cfg, n_features, columns):
    """
    Shared entry that dispatches to classifier or regressor wrapper
    based on task type / num_classes.
    """
    num_classes = int(cfg.get("model", {}).get("num_classes", 1))
    task = str(cfg.get("task", {}).get("type", "regression")).lower()
    if task == "classification" or num_classes >= 2:
        return XGBClassifierModel.from_cfg(cfg)
    return XGBRegressorModel.from_cfg(cfg)


@register("xgb_cls")
def build_xgb_classifier(cfg, n_features, columns):
    return XGBClassifierModel.from_cfg(cfg)


@register("xgb_reg")
def build_xgb_regressor(cfg, n_features, columns):
    return XGBRegressorModel.from_cfg(cfg)
