# model_factory.py
from __future__ import annotations
from importlib import import_module
from train.models.registry import get as _get_builder, list_models as _list_models

def _ensure_registry_loaded(name: str) -> None:
    """
    1. 說明: 依模型名懶載入目標模組，讓 @register(...) 生效
    2. inputs:
       - name: str, cfg["model"]["name"]
    3. return:
       - None
    """
    mapping = {
        "LSTM_SE": "train.models.LSTM",
        "LSTM_Temporal": "train.models.LSTM",
        "TemporalTransformer": "train.models.transformer_model",
        "TwoStreamHybrid": "train.models.two_stream_model",
        "xgb": "train.models.xgb_model",
    }
    mod = mapping.get(name)
    if mod:
        try:
            import_module(mod)
        except Exception:
            # 保守：即便 import 失敗，仍讓後續報錯顯示可用名單
            pass

def build_model(cfg, n_features, columns):
    """
    1. 說明: 根據 cfg.model.name 建立模型。先查註冊表；找不到就清楚報錯。
    2. inputs:
       - cfg: dict, 設定（含 cfg["model"]）
       - n_features: int, 特徵數
       - columns: list[str], 欄位名稱
    3. return:
       - object: 已建好的模型（nn.Module 或 XGBRegressor）
    """
    name = cfg["model"]["name"]

    # 先嘗試懶載入對應子模組，觸發 @register
    _ensure_registry_loaded(name)

    # 優先用 registry
    try:
        builder = _get_builder(name)
        return builder(cfg, n_features, columns)
    except KeyError:
        # 清楚提示：目前有效的名稱有哪些
        available = sorted(_list_models().keys())
        raise ValueError(
            f"Model '{name}' not registered. "
            f"Available: {available}. "
            f"可能原因：1) 名稱拼錯或大小寫不同；2) 未 import 對應模組（請檢查 mapping）。"
        )
