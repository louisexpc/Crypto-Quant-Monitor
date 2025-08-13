# config_export.py

import yaml
import copy
from pathlib import Path
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def dump_best_yaml(study: optuna.Study, cfg: dict, run_dir: Path):
    """將最佳 trial 的參數與設定儲存成 YAML 檔與 txt 檔。"""
    best = study.best_trial
    params = best.params
    feats = best.user_attrs.get("selected_features", [])

    outdir = run_dir / "best"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 純參數 YAML
    with open(outdir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    # 2. 完整 config（將搜尋空間替換為實際值）
    frozen = copy.deepcopy(cfg)

    def _set_num(node, key, default):
        val = params.get(key, None)
        if val is None:
            return default
        try:
            v = float(val)
            if v.is_integer():
                v = int(v)
        except Exception:
            v = val
        node[key] = v
        return v

    _set_num(frozen["train"], "lr", frozen["train"]["lr"])
    _set_num(frozen["train"], "weight_decay", frozen["train"]["weight_decay"])
    _set_num(frozen["train"], "epochs", frozen["train"]["epochs"])
    _set_num(frozen["train"], "grad_clip", frozen["train"]["grad_clip"])

    if "hidden_size" in params: frozen["model"]["hidden_size"] = params["hidden_size"]
    if "n_layers" in params:    frozen["model"]["n_layers"]    = params["n_layers"]
    if "dropout" in params:     frozen["model"]["dropout"]     = float(params["dropout"])
    if "seq_len" in params:     frozen["sequence"]["seq_len"]  = params["seq_len"]
    if "flat_band_bps" in params: frozen["label"]["flat_band_bps"] = params["flat_band_bps"]

    sel = frozen.setdefault("features", {}).setdefault("selection", {})
    if "k_features" in params: sel["k_range"] = [params["k_features"], params["k_features"]]
    if "feat_seed" in params:  sel["feat_seed"] = params["feat_seed"]

    # 3. 寫出 selected_features.txt
    with open(outdir / "selected_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feats))

    # 4. 寫出 best_config.yaml
    with open(outdir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(frozen, f, sort_keys=False, allow_unicode=True)

def save_fold_metrics(metrics: list[dict], save_dir: Path, prefix: str = ""):
    """
    儲存單個 fold 的訓練過程：
    - metrics: list of dicts，每個 epoch 的訓練記錄
    - save_dir: 儲存路徑
    - prefix: 檔名前綴（如 fold_0）

    會儲存：
    - metrics_epoch.csv
    - loss_curve.png
    - accuracy_curve.png
    - precision_recall_curve.png
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(save_dir / f"{prefix}metrics_epoch.csv", index=False)

    # ----- Loss curve -----
    plt.figure()
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}loss_curve.png")
    plt.close()

    # ----- Accuracy curve -----
    plt.figure()
    plt.plot(df["train_acc"], label="Train Acc")
    plt.plot(df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}accuracy_curve.png")
    plt.close()

    # ----- Precision / Recall curve -----
    # plt.figure()
    # plt.plot(df["train_macro_precision"], label="Train Precision")
    # plt.plot(df["val_macro_precision"], label="Val Precision")
    # plt.plot(df["train_macro_recall"], label="Train Recall")
    # plt.plot(df["val_macro_recall"], label="Val Recall")
    # plt.xlabel("Epoch")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_dir / f"{prefix}precision_recall_curve.png")
    # plt.close()

    # ----- f1 curve -----
    plt.figure()
    plt.plot(df["train_macro_f1"], label="Train f1")
    plt.plot(df["val_macro_f1"], label="Val f1")
    plt.xlabel("Epoch")
    plt.ylabel("f1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}f1_curve.png")
    plt.close()