# train/models/registry.py
from __future__ import annotations
from typing import Callable, Dict
'''
這段程式是在做一個「模型註冊表」（registry）
——把「模型名稱 → 建構器函式（builder）」配對起來，
之後用名稱就能拿到對應的 builder 來產生模型物件。
builder 在各個model裡
'''

# 全域字典，存放已註冊的 builder。key 是字串名稱（要對應 cfg.model.name），value 模型 class）。
_REGISTRY: Dict[str, Callable[..., object]] = {}

def register(name: str):
    """
    1. 說明: 註冊模型建構器（builder），名稱對應 cfg.model.name
    2. inputs:
       - name: str, 模型名稱（例如 "LSTM_SE"、"TemporalTransformer"）
    3. return:
       - Callable: 用於裝飾 builder 的裝飾器
    """
    def deco(fn: Callable[..., object]):
        _REGISTRY[name] = fn
        return fn
    return deco

def get(name: str) -> Callable[..., object]:
    """
    1. 說明: 依名稱取得已註冊的模型建構器
    2. inputs:
       - name: str, 模型名稱
    3. return:
       - Callable: builder(cfg, n_features, columns) -> model 物件
    """
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return _REGISTRY[name]

def list_models() -> Dict[str, Callable[..., object]]:
    """
    1. 說明: 列出目前註冊的模型
    2. inputs: 無
    3. return:
       - dict: {name: builder}
    """
    return dict(_REGISTRY)
