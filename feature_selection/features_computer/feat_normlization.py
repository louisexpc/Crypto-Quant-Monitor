"""
Utility for applying per-feature z-score normalization to precomputed feature CSVs.

Example
-------
python -m train.data.features.feat_normlization \
    --input data/precomputed/btcusdt_15m_features_all.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

_DEFAULT_SKIP_COLS = ("datetime", "timestamp")
_DEFAULT_SUFFIX = "_z_norm"


def z_normalize_dataframe(
    df: pd.DataFrame,
    *,
    skip_columns: Sequence[str] = _DEFAULT_SKIP_COLS,
) -> pd.DataFrame:
    """
    Apply z-score normalization to each column except those listed in skip_columns.

    All operations are column-wise across the full time series; statistics ignore NaNs.
    Columns with zero variance are returned as zeros to avoid NaNs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    skip_set = {c for c in skip_columns}
    target_cols = [c for c in df.columns if c not in skip_set]
    if not target_cols:
        return df.copy()

    normalized = df.copy()
    to_numeric_cols: list[str] = []

    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(normalized[col]):
            to_numeric_cols.append(col)

    if to_numeric_cols:
        normalized[to_numeric_cols] = normalized[to_numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    stats = normalized[target_cols]
    means = stats.mean(skipna=True)
    stds = stats.std(skipna=True, ddof=0)

    zero_std_cols = stds[stds == 0].index.tolist()
    non_zero = [c for c in target_cols if c not in zero_std_cols]

    if non_zero:
        normalized[non_zero] = (stats[non_zero] - means[non_zero]) / stds[non_zero]

    for col in zero_std_cols:
        normalized[col] = 0.0

    return normalized


def rolling_zscore_dataframe(
    df: pd.DataFrame,
    *,
    window: int,
    warmup: int | None = None,
    skip_columns: Sequence[str] = _DEFAULT_SKIP_COLS,
) -> pd.DataFrame:
    """
    Apply a past-only rolling z-score to each numeric column.

    Parameters
    ----------
    window
        Number of prior observations to include in the rolling statistics.
    warmup
        Minimum count of historical observations required before emitting a value.
        Defaults to ``window`` if not provided.
    skip_columns
        Columns to exclude from normalization (copied as-is).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if window <= 0:
        raise ValueError("window must be a positive integer")

    warmup = window if warmup is None else warmup
    if warmup <= 0:
        raise ValueError("warmup must be a positive integer")
    if warmup > window:
        raise ValueError("warmup cannot exceed window")

    skip_set = {c for c in skip_columns}
    target_cols = [c for c in df.columns if c not in skip_set]
    if not target_cols:
        return df.copy()

    normalized = df.copy()
    to_numeric_cols: list[str] = []

    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(normalized[col]):
            to_numeric_cols.append(col)

    if to_numeric_cols:
        normalized[to_numeric_cols] = normalized[to_numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    stats = normalized[target_cols]
    history = stats.shift(1)
    rolling_mean = history.rolling(window=window, min_periods=warmup).mean()
    rolling_std = history.rolling(window=window, min_periods=warmup).std(ddof=0)

    z_scores = (stats - rolling_mean) / rolling_std
    z_scores = z_scores.where(rolling_std != 0, 0.0)
    normalized[target_cols] = z_scores

    return normalized


def resolve_output_path(input_path: Path, suffix: str) -> Path:
    """Return the normalized file path by inserting suffix before the extension."""
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def z_normalize_csv(
    input_path: Path,
    *,
    output_path: Path | None = None,
    skip_columns: Iterable[str] = _DEFAULT_SKIP_COLS,
    suffix: str = _DEFAULT_SUFFIX,
    rolling_window: int | None = None,
    rolling_warmup: int | None = None,
) -> Path:
    """
    Read input CSV, z-normalize numeric feature columns, and write the result.

    Returns the Path to the written CSV.
    """
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if rolling_window is None:
        normalized = z_normalize_dataframe(df, skip_columns=tuple(skip_columns))
    else:
        normalized = rolling_zscore_dataframe(
            df,
            window=int(rolling_window),
            warmup=None if rolling_warmup is None else int(rolling_warmup),
            skip_columns=tuple(skip_columns),
        )

    output_path = (
        resolve_output_path(input_path, suffix) if output_path is None else output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply per-column z-score normalization to a feature CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path for the normalized CSV. Defaults to appending '_z_norm'.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=list(_DEFAULT_SKIP_COLS),
        help="Column names to exclude from normalization.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=_DEFAULT_SUFFIX,
        help="Suffix to append to the input filename when deriving the output path.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=None,
        help="Enable past-only rolling z-score using this window length.",
    )
    parser.add_argument(
        "--rolling-warmup",
        type=int,
        default=None,
        help=(
            "Minimum historical samples required before emitting a rolling z-score. "
            "Defaults to the rolling window if omitted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = z_normalize_csv(
        args.input,
        output_path=args.output,
        skip_columns=args.skip,
        suffix=args.suffix,
        rolling_window=args.rolling_window,
        rolling_warmup=args.rolling_warmup,
    )
    print(f"Normalized CSV written to: {output_path}")


if __name__ == "__main__":
    main()

"""
python3 -m feature_selection.features_computer.feat_normlization \
    --input data/precomputed/btcusdt_15m_features_VBT_1min.csv
    --rolling-window 144
"""
