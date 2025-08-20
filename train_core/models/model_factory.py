# model_factory.py
from typing import Dict
from .LSTM import LSTM_SE
from .Transformer import TemporalTransformer

def build_model(cfg: Dict, n_features: int):
    """
    根據 cfg["model"]["name"] 建立模型。
    支援的模型：
      - "LSTMHead"
      - "LSTM_SE"
      - "TemporalTransformer"
    """
    
    mcfg = cfg["model"]
    name = mcfg['name']
    num_classes = cfg["model"]["num_classes"]

    
    if name == "LSTM_SE":
        return LSTM_SE(
            input_dim     = n_features,
            hidden_size   = mcfg.get("hidden_size", 256),
            n_layers      = mcfg.get("n_layers", 2),
            dropout       = mcfg.get("dropout", 0.1),
            bidirectional = mcfg.get("bidirectional", False),
            num_classes   = num_classes,
        )
    

    elif name.lower() == "temporaltransformer":
        return TemporalTransformer(
            input_dim      = n_features,
            num_classes    = num_classes,
            d_model        = int(mcfg.get("d_model", mcfg.get("hidden_size", 128))),  # 後備到 hidden_size
            n_heads        = int(mcfg.get("n_heads", 4)),
            num_layers     = int(mcfg.get("n_layers", 3)),
            mlp_ratio      = float(mcfg.get("mlp_ratio", 4.0)),
            dropout        = float(mcfg.get("dropout", 0.1)),
            attn_dropout   = float(mcfg.get("attn_dropout", 0.0)),
            pooling        = str(mcfg.get("pooling", "attn")),       # ← 預設 attn，比 mean 更穩
        )
    
    if name in {"xgb", "xgboost", "xgb_reg", "xgbregressor"}:
        from models.xgb_model import XGBRegressorModel
        return XGBRegressorModel.from_cfg(cfg)
    

    else:
        raise ValueError(f"Unknown model name: {name}")


