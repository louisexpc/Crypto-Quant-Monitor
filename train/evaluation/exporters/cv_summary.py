"""Export cross-validation metric summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import numpy as np

__all__ = ["save_cv_summary"]


def _ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _numeric_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            out[k] = float(v)
    return out


def _avg_std_dict(rows: List[Dict[str, float]]) -> tuple[Dict[str, float], Dict[str, float]]:
    pool: Dict[str, List[float]] = {}
    for d in rows:
        for k, v in (d or {}).items():
            pool.setdefault(k, []).append(float(v))
    avg = {k: float(np.mean(vs)) for k, vs in pool.items()} if pool else {}
    std = {k: float(np.std(vs, ddof=0)) for k, vs in pool.items()} if pool else {}
    return avg, std


def save_cv_summary(fold_results: List[Dict[str, Any]], export_dir: str | Path, task_type: str) -> Path:
    export_dir = _ensure_dir(export_dir)
    folds_out = []

    for i, res in enumerate(fold_results):
        if task_type == "classification":
            val = _numeric_dict(res.get("val_metrics", {}))
            test = _numeric_dict(res.get("test_metrics", {}))
            extra: Dict[str, Any] = {}
        else:
            val = _numeric_dict(res.get("val_metrics_reg", {}))
            test = _numeric_dict(res.get("test_metrics_reg", {}))
            extra = {}
            if "regression_to_class" in res:
                extra["regression_to_class"] = res["regression_to_class"]
        folds_out.append({"fold_id": i, "val": val, "test": test, **extra})

    test_avg, test_std = _avg_std_dict([f["test"] for f in folds_out])
    val_avg, val_std = _avg_std_dict([f["val"] for f in folds_out])

    out = {
        "test_avg": test_avg,
        "test_std": test_std,
        "val_avg": val_avg,
        "val_std": val_std,
        "folds": folds_out,
    }

    out_path = Path(export_dir) / "cv_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        import json

        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out_path
