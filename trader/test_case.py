import argparse
import os
from pathlib import Path

import dotenv
import yaml

# Module imports (do not modify other modules)
from utils.data_collector import ExchangeDataCollector, ExchangeConfig
from indicators.feature_computer import FeatureComputer
from predictor.predictor import Predictor
from typing import Tuple
dotenv.load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _init_exchange_data_collector(config: dict, api_key: str, api_secret: str) -> ExchangeDataCollector:
    exchange_cfg = config.get("exchange", None)
    if not exchange_cfg:
        raise ValueError("Exchange configuration not found in config file.")
    return ExchangeDataCollector(api_key, api_secret, ExchangeConfig(**exchange_cfg))


def _init_feature_computer(config: dict) -> Tuple[FeatureComputer, int]:
    feature_cfg_path = Path(config["feature"].get("config_path", ""))
    if not feature_cfg_path.exists():
        raise FileNotFoundError("Feature configuration path not found in config file.")
    feature_cfg = load_config(feature_cfg_path)
    feat_normalization_bars = feature_cfg.get("feat_normalization", {}).get("rolling_window", 144)
    return FeatureComputer(feature_cfg), feat_normalization_bars


def _init_long_short_predictor(config: dict) -> tuple[Predictor, Predictor, int]:
    predictor_cfg_path = Path(config["predictor"].get("config_path", ""))
    if not predictor_cfg_path.exists():
        raise FileNotFoundError("Predictor configuration path not found in config file.")
    predictor_cfg = load_config(predictor_cfg_path)
    predictor_bars = predictor_cfg.get("seq_len", 144)
    return Predictor(predictor_cfg, "long"), Predictor(predictor_cfg, "short"), predictor_bars


def main() -> None:
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("API_KEY and API_SECRET must be set in environment variables")

    parser = argparse.ArgumentParser(description="Test data_collector/feature_computer/predictor pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="15m", help="Kline timeframe")
    parser.add_argument("--lookback", type=int, default=120, help="Number of bars to fetch")
    args = parser.parse_args()

    config = load_config(args.config)

    

    data_collector = _init_exchange_data_collector(config, api_key, api_secret)
    feature_computer, feat_normalization_bars = _init_feature_computer(config)
    long_predictor, short_predictor, predictor_bars = _init_long_short_predictor(config)

    total_lookbacks = feat_normalization_bars + predictor_bars

    model_df = data_collector.fetch_ohlcv_fng(args.symbol, args.timeframe, total_lookbacks)
    print(f"[Debug] Fetched {len(model_df)} rows of data with cols: {len(model_df.columns)} for symbol {args.symbol} with timeframe {args.timeframe}")
    long_feature = feature_computer.compute(model_df, "long")
    print(f"[Debug] Computed long features with shape: {long_feature.shape},chunk shape: {long_feature.iloc[-predictor_bars:].shape}")
    long_consume_time, can_long_entry = long_predictor.predict(long_feature.iloc[-predictor_bars:])
    print(f"long consume time: {long_consume_time:.4f} seconds, can long entry: {can_long_entry}")
    short_feature = feature_computer.compute(model_df, "short")
    print(f"[Debug] Computed short features with shape: {short_feature.shape},chunk shape: {short_feature.iloc[-predictor_bars:].shape}")
    # 從 latest bar 往前數 predictor_bars 
    short_consume_time, can_short_entry = short_predictor.predict(short_feature.iloc[-predictor_bars:])



    print(f"short consume time: {short_consume_time:.4f} seconds, can short entry: {can_short_entry}")


if __name__ == "__main__":
    main()
