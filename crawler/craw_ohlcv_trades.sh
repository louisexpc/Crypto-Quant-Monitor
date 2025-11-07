#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: crawler/craw_ohlcv_trades.sh [options]

Runs the full data pipeline: OHLCV -> FNG -> Binance trades -> merge FNG ->
trade-minute features -> combine.

Options (defaults in brackets):
  --start DATE              Global start date for crawlers [2023-01-01]
  --end DATE                Global end date (exclusive for live, default: today)
  --ohlcv-start DATE        Override OHLCV start
  --ohlcv-end DATE          Override OHLCV end
  --trades-start DATE       Override trades start
  --trades-end DATE         Override trades end
  --fng-start DATE          Override FNG start
  --fng-end DATE            Override FNG end
  --timeframes "15m"        Timeframe for OHLCV (single value required downstream)
  --config PATH             OHLCV YAML config [utils/collector/collector_config.yaml]
  --exchange-id ID          Exchange id for OHLCV [binanceusdm]
  --default-type TYPE       Default market type [swap]
  --symbol SYMBOL           Symbol like BTC/USDT:USDT
  --ohlcv-outdir DIR        OHLCV output directory [data/ohlcv_2023_new]
  --ohlcv-fmt FMT           csv or parquet [csv]
  --fng-outdir DIR          FNG output directory [data/FNG]
  --trades-symbol SYM       Binance symbol [BTCUSDT]
  --trades-outdir DIR       Trades zip directory [data/binance_trades/SYM]
  --merged-csv PATH         Output of 15m+FNG merge
  --trades-features PATH    Output CSV for 1m features
  --final-out PATH          Final combined CSV path
  --minute-steps N          Minutes to flatten for each 15m bar [15]
  --fill-policy POLICY      zero|ffill|drop for 1m gaps [zero]
  --bar-is-end              Set if OHLCV timestamps already represent bar end
  --tz TIMEZONE             Time zone for OHLCV datetime when no timestamp [Asia/Taipei]
  -h, --help                Show this help
EOF
}

OHLCV_CONFIG="utils/collector/collector_config.yaml"
EXCHANGE_ID="binanceusdm"
DEFAULT_TYPE="swap"
SYMBOL="BTC/USDT:USDT"
OHLCV_OUTDIR="data/ohlcv_2023_new"
OHLCV_FMT="csv"
TIMEFRAMES="15m"
GLOBAL_START="2023-01-01"
GLOBAL_END=""

OHLCV_START="$GLOBAL_START"
OHLCV_END="$GLOBAL_END"
TRADES_START="$GLOBAL_START"
TRADES_END="$GLOBAL_END"
FNG_START="$GLOBAL_START"
FNG_END="$GLOBAL_END"

FNG_OUTDIR="data/FNG"
FNG_FILENAME="fng_15m_utc.csv"
TRADES_SYMBOL="BTCUSDT"
TRADES_OUTDIR="data/binance_trades/${TRADES_SYMBOL}"

MERGED_CSV=""
TRADES_FEATURES=""
FINAL_OUT=""
MINUTE_STEPS=15
FILL_POLICY="zero"
BAR_IS_END=0
TZ_15M="Asia/Taipei"
PYTHON_BIN=${PYTHON_BIN:-python}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) GLOBAL_START="$2"; OHLCV_START="$2"; TRADES_START="$2"; FNG_START="$2"; shift 2 ;;
    --end) GLOBAL_END="$2"; OHLCV_END="$2"; TRADES_END="$2"; FNG_END="$2"; shift 2 ;;
    --ohlcv-start) OHLCV_START="$2"; shift 2 ;;
    --ohlcv-end) OHLCV_END="$2"; shift 2 ;;
    --trades-start) TRADES_START="$2"; shift 2 ;;
    --trades-end) TRADES_END="$2"; shift 2 ;;
    --fng-start) FNG_START="$2"; shift 2 ;;
    --fng-end) FNG_END="$2"; shift 2 ;;
    --timeframes) TIMEFRAMES="$2"; shift 2 ;;
    --config) OHLCV_CONFIG="$2"; shift 2 ;;
    --exchange-id) EXCHANGE_ID="$2"; shift 2 ;;
    --default-type) DEFAULT_TYPE="$2"; shift 2 ;;
    --symbol) SYMBOL="$2"; shift 2 ;;
    --ohlcv-outdir) OHLCV_OUTDIR="$2"; shift 2 ;;
    --ohlcv-fmt) OHLCV_FMT="$2"; shift 2 ;;
    --fng-outdir) FNG_OUTDIR="$2"; shift 2 ;;
    --trades-symbol) TRADES_SYMBOL="$2"; shift 2 ;;
    --trades-outdir) TRADES_OUTDIR="$2"; shift 2 ;;
    --merged-csv) MERGED_CSV="$2"; shift 2 ;;
    --trades-features) TRADES_FEATURES="$2"; shift 2 ;;
    --final-out) FINAL_OUT="$2"; shift 2 ;;
    --minute-steps) MINUTE_STEPS="$2"; shift 2 ;;
    --fill-policy) FILL_POLICY="$2"; shift 2 ;;
    --bar-is-end) BAR_IS_END=1; shift 1 ;;
    --tz|--time-zone) TZ_15M="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

SYM_TAG=$(echo "$SYMBOL" | sed 's#[/:]#-#g')
TRADES_OUTDIR=${TRADES_OUTDIR%/}
if [[ -z "$TRADES_OUTDIR" ]]; then
  TRADES_OUTDIR="data/binance_trades/${TRADES_SYMBOL}"
fi

IFS=' ' read -r -a TF_ARRAY <<< "$TIMEFRAMES"
if [[ ${#TF_ARRAY[@]} -ne 1 ]]; then
  echo "Please provide exactly one timeframe via --timeframes (current: $TIMEFRAMES)" >&2
  exit 1
fi
PRIMARY_TF=${TF_ARRAY[0]}

OHLCV_FILE="$OHLCV_OUTDIR/${EXCHANGE_ID}_${DEFAULT_TYPE}_${SYM_TAG}_${PRIMARY_TF}.${OHLCV_FMT}"
FNG_CSV="$FNG_OUTDIR/$FNG_FILENAME"
SYMBOL_LOWER=$(echo "$TRADES_SYMBOL" | tr '[:upper:]' '[:lower:]')
MERGED_TARGET=${MERGED_CSV:-"data/derived/${SYMBOL_LOWER}_${PRIMARY_TF}_with_fng.csv"}
TRADES_FEAT_PATH=${TRADES_FEATURES:-"data/derived/${SYMBOL_LOWER}_trades_1min_features.csv"}
FINAL_DATA_PATH=${FINAL_OUT:-"data/derived/${SYMBOL_LOWER}_${PRIMARY_TF}_with_flat_1m.csv"}

echo "[1/6] Fetch OHLCV"
ohlcv_cmd=($PYTHON_BIN crawler/ohlcv/ohlcv.py --config "$OHLCV_CONFIG" --start "$OHLCV_START" --outdir "$OHLCV_OUTDIR")
if [[ -n "$OHLCV_END" ]]; then
  ohlcv_cmd+=(--end "$OHLCV_END")
fi
ohlcv_cmd+=(--timeframes)
ohlcv_cmd+=("${TF_ARRAY[@]}")
"${ohlcv_cmd[@]}"

echo "[2/6] Fetch FNG"
fng_cmd=($PYTHON_BIN crawler/fng/fng_index.py --start "$FNG_START" --outdir "$FNG_OUTDIR" --outfile "$FNG_FILENAME")
if [[ -n "$FNG_END" ]]; then
  fng_cmd+=(--end "$FNG_END")
fi
"${fng_cmd[@]}"

echo "[3/6] Fetch trades"
mkdir -p "$TRADES_OUTDIR"
trades_cmd=($PYTHON_BIN crawler/trades/binance_trades.py --symbol "$TRADES_SYMBOL" --start "$TRADES_START" --outdir "$TRADES_OUTDIR")
if [[ -n "$TRADES_END" ]]; then
  trades_cmd+=(--end "$TRADES_END")
fi
"${trades_cmd[@]}"

if [[ ! -f "$OHLCV_FILE" ]]; then
  echo "[error] Expected OHLCV file not found: $OHLCV_FILE" >&2
  exit 1
fi
if [[ ! -f "$FNG_CSV" ]]; then
  echo "[error] Expected FNG CSV not found: $FNG_CSV" >&2
  exit 1
fi

echo "[4/6] Merge FNG into 15m"
"$PYTHON_BIN" crawler/utils/merge_fng_into_15m.py --base_csv "$OHLCV_FILE" --fng_csv "$FNG_CSV" --out_csv "$MERGED_TARGET"

echo "[5/6] Build 1m trade features"
t2m_cmd=($PYTHON_BIN crawler/utils/trades_to_1min_features.py --input_dir "$TRADES_OUTDIR" --symbol "$TRADES_SYMBOL" --start "$TRADES_START" --output_csv "$TRADES_FEAT_PATH")
if [[ -n "$TRADES_END" ]]; then
  t2m_cmd+=(--end "$TRADES_END")
fi
"${t2m_cmd[@]}"

echo "[6/6] Combine 15m + flattened 1m"
combine_cmd=($PYTHON_BIN crawler/utils/combine_15m_1m.py --ohlcv_csv "$MERGED_TARGET" --trades_1m_csv "$TRADES_FEAT_PATH" --out_csv "$FINAL_DATA_PATH" --minute_steps "$MINUTE_STEPS" --fill_policy "$FILL_POLICY" --time_zone "$TZ_15M")
if [[ $BAR_IS_END -eq 1 ]]; then
  combine_cmd+=(--bar_is_end)
fi
"${combine_cmd[@]}"

echo "Done. Final dataset → $FINAL_DATA_PATH"
