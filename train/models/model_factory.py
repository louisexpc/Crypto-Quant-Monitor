# model_factory.py
from typing import Dict
from .LSTM import LSTM_SE, LSTM_Temporal
from .Transformer import TemporalTransformer
from .two_stream_model import TwoStreamHybrid

def build_model(cfg: Dict, n_features: int, columns:list[str]):
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
            hidden_size   = mcfg["hidden_size"],
            n_layers      = mcfg.get("n_layers", 2),
            dropout       = mcfg.get("dropout", 0.1),
            bidirectional = mcfg.get("bidirectional", False),
            num_classes   = num_classes,
        )
    
    elif name in {"LSTM_Temporal", "LSTM_Attn"}:
        return LSTM_Temporal(
            input_dim     = n_features,
            hidden_size   = mcfg["hidden_size"],
            n_layers      = mcfg["n_layers"],
            dropout       = mcfg["dropout"],
            bidirectional = mcfg["bidirectional"],
            num_classes   = num_classes,
            use_se        = mcfg["use_se"],
            se_ratio      = mcfg["se_ratio"],
            pooling       = mcfg["pooling"],
            return_alpha  = mcfg["return_alpha"],
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
    
    elif name.lower() in {"twostreamhybrid", "two_stream_hybrid"}:
        return TwoStreamHybrid(
            num_classes     = num_classes,
            # minute stream
            minute_steps    = int(mcfg.get("minute_steps", 15)),
            minute_hidden   = int(mcfg.get("minute_hidden", 64)),
            minute_layers   = int(mcfg.get("minute_layers", 1)),
            minute_dropout  = float(mcfg.get("minute_dropout", 0.0)),

            # backbone (沿用你 TemporalTransformer 的參數命名)
            d_model       = int(mcfg.get("d_model", 128)),
            n_heads       = int(mcfg.get("n_heads", 4)),
            num_layers    = int(mcfg.get("n_layers", 2)),
            mlp_ratio     = float(mcfg.get("mlp_ratio", 4.0)),
            dropout       = float(mcfg.get("dropout", 0.1)),
            attn_dropout  = float(mcfg.get("attn_dropout", 0.0)),
            pooling       = str(mcfg.get("pooling", "attn")),
            use_causal    = bool(mcfg.get("use_causal", True)),

            columns      = columns,
            minute_prefixes = "m_"
        )
    
    if name in {"xgb", "xgboost", "xgb_reg", "xgbregressor"}:
        from models.xgb_model import XGBRegressorModel
        return XGBRegressorModel.from_cfg(cfg)
    

    else:
        raise ValueError(f"Unknown model name: {name}")


