from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd


def _side_label(x) -> str | None:
    s = str(x).strip().lower()
    if s in {"long", "buy", "l", "+1", "1"}:
        return "long"
    if s in {"short", "sell", "s", "-1", "-1.0"}:
        return "short"
    try:
        v = float(s)
    except Exception:
        return None
    if v > 0:
        return "long"
    if v < 0:
        return "short"
    return None


def _to_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")
    return dt


def load_train_csv(path: Path, side_hint: Literal["long", "short"]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t0" not in df.columns:
        raise ValueError(f"{path} missing column 't0'")
    side_norm = df["side"].map(_side_label) if "side" in df.columns else side_hint

    pred_col = "pred_vote"
    if pred_col is None:
        raise ValueError(f"{path} missing pred/pred_vote column")

    out = pd.DataFrame(
        {
            "t0": _to_utc(df["t0"]),
            "side": side_norm,
            "pred_train": pd.to_numeric(df[pred_col], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=["t0", "side"])
    # 去重時優先保留有 pred 的列
    out["_has_pred"] = out["pred_train"].notna().astype(int)
    out = out.sort_values(["t0", "side", "_has_pred"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["t0", "side"], keep="first")
    out = out.drop(columns=["_has_pred"])
    return out


def load_runtime_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["t0", "side", "pred"])
    if "t0" not in df.columns or "pred" not in df.columns:
        raise ValueError(f"{path} needs columns t0 and pred")
    side_norm = df["side"].map(_side_label) if "side" in df.columns else pd.NA
    out = pd.DataFrame(
        {
            "t0": _to_utc(df["t0"]),
            "side": side_norm,
            "pred_runtime": pd.to_numeric(df["pred"], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=["t0", "side"])
    out = out.drop_duplicates(subset=["t0", "side"], keep="last")
    out = out.sort_values("t0")
    return out


def compare(train_long: Path, train_short: Path, runtime_path: Path, output: Path | None) -> None:
    # 1) train 兩份合併並依 t0 排序
    df_long = load_train_csv(train_long, side_hint="long")
    df_short = load_train_csv(train_short, side_hint="short")
    train_merged = pd.concat([df_long, df_short], ignore_index=True)
    train_merged["_has_pred"] = train_merged["pred_train"].notna().astype(int)
    train_merged = (
        train_merged.sort_values(["t0", "side", "_has_pred"], ascending=[True, True, False])
        .drop_duplicates(subset=["t0", "side"], keep="first")
        .drop(columns=["_has_pred"])
        .sort_values(["t0", "side"])
    )

    # 2) runtime 載入
    runtime_df = load_runtime_csv(runtime_path)

    # 3) 取聯集並比較
    merged = runtime_df.merge(train_merged, on=["t0", "side"], how="outer", suffixes=("_runtime", "_train"))
    both_mask = merged["pred_runtime"].notna() & merged["pred_train"].notna()
    match_mask = both_mask & (merged["pred_runtime"] == merged["pred_train"])
    mismatch_mask = both_mask & (merged["pred_runtime"] != merged["pred_train"])

    print(f"Train merged rows: {len(train_merged)}")
    print(f"Runtime rows: {len(runtime_df)}")
    print(f"Union rows (t0,side): {len(merged)}")
    print(f"Both present: {both_mask.sum()} | match: {match_mask.sum()} | mismatch: {mismatch_mask.sum()}")
    print(f"Only runtime (no train pred): {(merged['pred_train'].isna() & merged['pred_runtime'].notna()).sum()}")
    print(f"Only train (no runtime pred): {(merged['pred_runtime'].isna() & merged['pred_train'].notna()).sum()}")

    if output:
        merged = merged.sort_values(["t0", "side"])
        merged["match"] = match_mask
        merged["mismatch"] = mismatch_mask
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
        print(f"Saved merged comparison -> {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare train inference CSVs with runtime predictions (union on t0,side).")
    p.add_argument("--train-long", required=True, type=Path)
    p.add_argument("--train-short", required=True, type=Path)
    p.add_argument("--runtime", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    compare(args.train_long, args.train_short, args.runtime, args.output)


if __name__ == "__main__":
    main()

"""
python trader/testcase/compare_result.py \
  --train-long runs/BTC_tbm_long_v3_ft32/trial_009_mcc=+0.231/trial_009_long_108.csv \
  --train-short runs/BTC_tbm_short_v3_ft32/trial_006_mcc=+0.172/trial_006_short_108.csv \
  --runtime trader/testcase/test_case_pred.csv \
  --output trader/testcase/compare_diff.csv
"""
