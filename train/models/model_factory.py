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
    num_classes = int(mcfg.get("num_classes", 2))

    
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
            # use_learned_pos= bool(mcfg.get("use_learned_pos", False)),
            causal     = bool(mcfg.get("use_causal", True)),     # ← 預設打開因果注意力
            # use_alibi      = bool(mcfg.get("use_alibi", True)),      # ← 相對距離偏置
            # alibi_slope    = float(mcfg.get("alibi_slope", 0.05)),
            # use_conv_stem  = bool(mcfg.get("use_conv_stem", True)),  # ← 局部形狀
            # droppath       = float(mcfg.get("droppath", 0.05)),      # ← Stochastic Depth
            # use_input_norm = bool(mcfg.get("use_input_norm", True)), # ← 對輸入做 LN
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


