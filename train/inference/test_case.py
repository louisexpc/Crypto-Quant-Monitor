from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from train.core.config_loader import load_cfg
from train.data.dataloaders.base import flatten_micro_features, load_precomputed_features
from train.evaluation.exporters.tbm_exporter import TBMExporter
from train.inference.predictor import Predictor


def _parse_args() -> argparse.Namespace:
    """
    1. 說明:
        解析指令列參數，允許覆寫設定檔路徑、輸出路徑與模型清單。
    2. inputs:
        - 無（直接讀取 sys.argv）
    3. return:
        - argparse.Namespace: 已解析的參數物件。
    """
    default_models = [
        "runs/BTC_test_long_108/trial_021_mcc=-0.086/fold_0/model_state.pt",
        "runs/BTC_test_long_108/trial_021_mcc=-0.086/fold_1/model_state.pt",
        "runs/BTC_test_long_108/trial_021_mcc=-0.086/fold_2/model_state.pt",
    ]

    parser = argparse.ArgumentParser(description="BTC event inference test case")
    parser.add_argument(
        "--config",
        type=str,
        default="train/inference/test_predictor.yaml",
        help="Path to the YAML config file used for training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train/inference/pred.csv",
        help="Where to save the prediction CSV.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=default_models,
        help="List of checkpoint paths. Defaults to the three provided BTC folds.",
    )
    return parser.parse_args()


def _to_utc(ts_like: Any) -> pd.Timestamp:
    """
    1. 說明:
        將任意時間格式轉為 UTC 時區的 Timestamp（tz-aware）。
    2. inputs:
        - ts_like: 可轉為 pandas Timestamp 的物件。
    3. return:
        - pd.Timestamp: UTC 時區的時間戳。
    """
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class EventInferenceRunner:
    """
    1. 說明:
        單次事件推論流程：讀取 config、預算特徵（含 1m micro 展平）、以多個 checkpoint 做投票並輸出 CSV。
    2. inputs:
        - cfg: 設定檔 dict。
        - model_paths: 模型 checkpoint 路徑列表。
        - output_path: 預測結果輸出路徑。
    3. return:
        - EventInferenceRunner instance
    """

    def __init__(self, cfg: Dict[str, Any], model_paths: Sequence[Path], output_path: Path):
        """
        1. 說明:
            建立推論 runner 並檢查輸入檔案是否存在。
        2. inputs:
            - cfg: 設定檔 dict。
            - model_paths: 模型 checkpoint 路徑列表。
            - output_path: 輸出 CSV 路徑。
        3. return:
            - None
        """
        self.cfg = cfg
        self.model_paths = [Path(p) for p in model_paths]
        self.output_path = Path(output_path)
        self._validate_model_paths()

    def _validate_model_paths(self) -> None:
        """
        1. 說明:
            確保提供的 checkpoint 路徑皆存在，否則拋出錯誤。
        2. inputs:
            - None
        3. return:
            - None
        """
        missing = [str(p) for p in self.model_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    def _resolve_time_bounds(
        self, feat_df: pd.DataFrame, micro_df: pd.DataFrame | None
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        1. 說明:
            決定特徵裁剪範圍：起點用 cv.start_date，終點取 post_infer.date_end 與特徵/micro 可用範圍的最小值。
        2. inputs:
            - feat_df: 15m 預算特徵 DataFrame。
            - micro_df: 1m micro 特徵 DataFrame 或 None。
        3. return:
            - (cv_start, ts_end): UTC Timestamp 範圍。
        """
        cv_start = _to_utc(self.cfg["cv"]["start_date"])

        post_root = (self.cfg.get("post_infer", {}) or {})
        post_cfg = post_root if isinstance(post_root, dict) else {}
        ts_end_param = post_cfg.get("date_end") or (self.cfg.get("cv", {}) or {}).get("end_date")
        ts_end_param = _to_utc(ts_end_param) if ts_end_param else _to_utc(pd.DatetimeIndex(feat_df.index).max())

        feat_end = _to_utc(pd.DatetimeIndex(feat_df.index).max())
        candidates: List[pd.Timestamp] = [feat_end, ts_end_param]

        if micro_df is not None and len(micro_df.index):
            micro_end = _to_utc(pd.DatetimeIndex(micro_df.index).max())
            candidates.append(micro_end)

        ts_end = min(candidates)
        return cv_start, ts_end

    def _prepare_features(self) -> pd.DataFrame:
        """
        1. 說明:
            讀取 15m 預算特徵並視需要展平 1m micro，裁剪到有效推論範圍。
        2. inputs:
            - None（使用 runner 內部的 cfg）
        3. return:
            - pd.DataFrame: 已排序且展平的特徵表。
        """
        feat_df = load_precomputed_features(path=self.cfg["data"]["path"])

        micro_cfg = (self.cfg.get("data", {}) or {}).get("micro", {}) or {}
        micro_df = None
        if micro_cfg.get("enabled") and micro_cfg.get("path"):
            micro_df = load_precomputed_features(path=micro_cfg["path"])

        cv_start, ts_end = self._resolve_time_bounds(feat_df, micro_df)

        if micro_df is not None:
            window_len = int(micro_cfg.get("window_len", 15))
            feat_df = flatten_micro_features(
                feat_df=feat_df,
                micro_df=micro_df,
                cv_start=cv_start,
                ts_end=ts_end,
                window_len=window_len,
            )

        feat_df = feat_df.loc[(feat_df.index >= cv_start) & (feat_df.index <= ts_end)]
        if feat_df.empty:
            raise ValueError("Feature dataframe is empty after applying time bounds.")
        return feat_df

    def run(self) -> str:
        """
        1. 說明:
            完整推論流程：建特徵 → 多模型投票 → 將 TBM 事件與預測輸出為 CSV。
        2. inputs:
            - None
        3. return:
            - str: 寫出的 CSV 路徑。
        """
        feat_df = self._prepare_features()
        predictor = Predictor(cfg=self.cfg)
        pred_df = predictor.predict_vote(feat_df, model_paths_or_dir=self.model_paths)

        exporter = TBMExporter(self.cfg)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = exporter.export_csv(pred_df, save_to_path=str(self.output_path))
        return save_path


def main() -> None:
    """
    1. 說明:
        腳本入口：解析參數、載入設定並執行事件推論。
    2. inputs:
        - None
    3. return:
        - None
    """
    args = _parse_args()
    cfg = load_cfg(args.config)

    model_paths = [Path(p) for p in args.models]
    runner = EventInferenceRunner(cfg=cfg, model_paths=model_paths, output_path=Path(args.output))
    save_path = runner.run()
    print(f"[Inference] Saved predictions to: {save_path}")


if __name__ == "__main__":
    main()
