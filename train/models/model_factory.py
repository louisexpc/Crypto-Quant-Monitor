from typing import Dict
from .LSTM import LSTMHead, LSTM_SE
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
    common_kwargs = dict(
        input_dim   = n_features,
        hidden_size = mcfg.get("hidden_size", 256),
        n_layers    = mcfg.get("n_layers", 2),
        dropout     = mcfg.get("dropout", 0.1),
        bidirectional = mcfg.get("bidirectional", False),
        num_classes = mcfg.get("num_classes", 3),
    )

    if name == "LSTMHead":
        return LSTMHead(**common_kwargs)
    
    elif name == "LSTM_SE":
        return LSTM_SE(**common_kwargs)
    

    elif name.lower() == "temporaltransformer":
        return TemporalTransformer(
            input_dim=n_features,
            num_classes=mcfg.get("num_classes", 3),
            d_model=mcfg.get("d_model", 128),
            n_heads=mcfg.get("n_heads", 4),
            num_layers=mcfg.get("n_layers", 2),
            mlp_ratio=mcfg.get("mlp_ratio", 4.0),
            dropout=mcfg.get("dropout", 0.1),
            attn_dropout=mcfg.get("attn_dropout", 0.1),
            pooling=mcfg.get("pooling", "mean")
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


