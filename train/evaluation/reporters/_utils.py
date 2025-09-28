import numpy as np
from pathlib import Path 
def _ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ensure_2d_prob(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    return y_prob
