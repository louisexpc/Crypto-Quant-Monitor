from typing import Dict
from .LSTM import LSTMHead, LSTM_SE

def build_model(cfg: Dict, n_features: int):
    """
    依 cfg["model"]["name"] 建立模型。
    需要的鍵：
      cfg["model"]["name"] in {"LSTMHead", "LSTM_SE"}
      cfg["model"]["hidden_size"], ["n_layers"], ["dropout"], ["bidirectional"], ["num_classes"]
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
    else:
        raise ValueError(f"Unknown model name: {name}")


