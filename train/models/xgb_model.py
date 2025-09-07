# models/xgb_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from xgboost.sklearn import XGBRegressor as _XGBRegressor

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