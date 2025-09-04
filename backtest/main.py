# ./backtest.py
import math
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import (
    SizeType, StopEntryPrice, StopExitPrice, AccumulationMode,
    ConflictMode, DirectionConflictMode
)

from strategy.analysis import StrategyAnalyzer




def _parse_tf_to_timedelta(tf: str) -> Optional[pd.Timedelta]:
    """'15m'/'1h'/'1d' -> Timedelta"""
    if not isinstance(tf, str):
        return None
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    return None


class backtestStrategy:
    """
    U本位永續回測（多空雙開；TP/SL=10%/10%；槓桿5x；倉位=資產目標權重）
    - fees: 0.0005
    - slippage: 0.0002
    - position_size: 0.1           # 單邊基準倉位（以權重表示）
    - leverage: 5                  # 名目曝險 = position_size * leverage
    - target return ratio: 10% per order
    - stop loss: 10% per order
    - funding rate: 固定每8小時 0.0005（正值 → 多付空收）
    """
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        volume_ema_window: int = 10,
        *,
        fees: float = 0.0005,
        slippage: float = 0.0002,
        position_size: float = 0.10,
        leverage: float = 5.0,
        tp: float = 0.10,
        sl: float = 0.02,
        funding_rate_per_8h: float = 0.0005,
        init_cash: float = 100_000.0,
    ):
        if df is None or df.empty:
            raise ValueError("df 不可為空")
        req_cols = {"open", "high", "low", "close"}
        if not req_cols.issubset(df.columns):
            raise ValueError(f"df 需包含欄位：{sorted(req_cols)}")
        self.volume_ema_window = volume_ema_window
        self.df = df.copy()
        # 統一索引：DatetimeIndex、唯一且排序；保留時區亦可（vbt支援），但避免混用
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "datetime" in self.df.columns:
                self.df.index = pd.to_datetime(self.df["datetime"], utc=True)
            elif "timestamp" in self.df.columns:
                self.df.index = pd.to_datetime(self.df["timestamp"], unit="ms", utc=True)
            else:
                raise ValueError("找不到可轉換成 DatetimeIndex 的欄位（datetime/timestamp）")
        self.df = self.df[~self.df.index.duplicated()].sort_index()

        self.symbol = symbol
        self.timeframe = timeframe
        self.fees = fees
        self.slippage = slippage
        self.position_size = float(position_size)
        self.leverage = float(leverage)
        self.tp = float(tp)
        self.sl = float(sl)
        self.funding_rate_per_8h = float(funding_rate_per_8h)
        self.init_cash = float(init_cash)

        self.price = self.df["close"].astype(float)

        # entries/exits 會是 DataFrame（兩欄：LONG/SHORT）
        self.entries: Optional[pd.DataFrame] = None
        self.exits: Optional[pd.DataFrame] = None

        # for debug/檢視
        self.raw_signals_df: Optional[pd.DataFrame] = None

    # ------------------------- 訊號生成 ------------------------- #
    def _generate_entry_signals(self, lookback: int = 36) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        以滑動視窗觸發 StrategyAnalyzer：每次用 [i-lookback+1 : i] 的資料
        只要該視窗最後一根「完成K線」觸發，就在該根時間點標記 True。
        - 多個同根訊號：同側合併為 True（避免同根加倉）
        """
        idx = self.df.index
        long_s = pd.Series(False, index=idx, name="LONG")
        short_s = pd.Series(False, index=idx, name="SHORT")

        records: list[Dict[str, Any]] = []

        for i in range(lookback - 1, len(self.df)):
            window_df = self.df.iloc[i - lookback + 1 : i + 1]
            try:
                analyzer = StrategyAnalyzer(window_df, symbol=self.symbol, timeframe=self.timeframe, volume_ema_window=self.volume_ema_window)
                sigs = analyzer.analyze()  # ← 只判斷最後一根是否觸發
            except Exception:
                continue

            if not sigs:
                continue

            for sig in sigs:
                t = sig.get("test_trigger_time")
                side = sig.get("signal_type")  # 'Long' / 'Short'
                if t in idx:
                    if side == "Long":
                        long_s.loc[t] = True
                    elif side == "Short":
                        short_s.loc[t] = True
                records.append(sig)

        self.raw_signals_df = pd.DataFrame.from_records(records) if records else pd.DataFrame()

        entries = pd.concat([long_s, short_s], axis=1)
        exits = pd.DataFrame(False, index=idx, columns=["LONG", "SHORT"])  # 出場靠 TP/SL
        return entries, exits

    def _generate_exits_signals(self) -> pd.DataFrame:
        """這裡不用顯式 exits；TP/SL 交給 vectorbt 的 sl_stop/tp_stop。"""
        return pd.DataFrame(False, index=self.df.index, columns=["LONG", "SHORT"])

    # ------------------------- Funding 現金流 ------------------------- #
    def _compute_equity_with_funding(self, pf: vbt.Portfolio) -> pd.Series:
        """固定每 8 小時 funding：rate>0 → 多付空收"""
        tf_td = _parse_tf_to_timedelta(self.timeframe)
        steps = max(1, int(pd.Timedelta(hours=8) / tf_td)) if tf_td else 8
        fund_idx = self.price.index[::steps]

        # ---- 取得持倉單位數（單欄 Series，正=多、負=空）----
        pos_series = None

        # 1) 常見：position_size（不同版本可能是屬性或方法）
        cand = getattr(pf, "position_size", None)
        if cand is not None:
            cand = cand() if callable(cand) else cand
            if hasattr(cand, "to_series"):
                try:
                    pos_series = cand.to_series()
                except Exception:
                    pos_series = None
            elif isinstance(cand, pd.Series):
                pos_series = cand
            elif isinstance(cand, pd.DataFrame) and cand.shape[1] >= 1:
                pos_series = cand.iloc[:, 0]

        # 2) 次選：positions（有些版本是 accessor/包裝器）
        if pos_series is None:
            cand = getattr(pf, "positions", None)
            if cand is not None:
                if hasattr(cand, "to_series"):
                    try:
                        pos_series = cand.to_series()
                    except Exception:
                        pos_series = None
                elif isinstance(cand, pd.Series):
                    pos_series = cand
                elif isinstance(cand, pd.DataFrame) and cand.shape[1] >= 1:
                    pos_series = cand.iloc[:, 0]

        # 3) 保底：全 0（不中斷執行）
        if pos_series is None:
            pos_series = pd.Series(0.0, index=self.price.index, dtype=float)

        pos_series = pos_series.reindex(self.price.index, method="pad").fillna(0.0)

        # ---- 計算 funding 現金流 ----
        notional = (pos_series * self.price).reindex(fund_idx, method="pad").fillna(0.0)
        funding_rate = pd.Series(float(self.funding_rate_per_8h), index=fund_idx, dtype=float)

        # 現金流 = - rate * |notional| * sign(notional)（正率：多付空收）
        cashflow = -funding_rate * notional.abs() * np.sign(notional)

        eq = pf.value()
        adj = cashflow.cumsum().reindex(eq.index, method="pad").fillna(0.0)
        return eq + adj





    # ------------------------- 執行回測 ------------------------- #
    def run_backtest(self) -> Dict[str, Any]:
        # 1) 產生 entries/exits
        self.entries, self.exits = self._generate_entry_signals()

        long_entries  = self.entries["LONG"].astype(bool)
        long_exits    = self.exits["LONG"].astype(bool)
        short_entries = self.entries["SHORT"].astype(bool)
        short_exits   = self.exits["SHORT"].astype(bool)

        # 2) 每筆名目下單金額（例：0.1 * 5 * 100000 = 50000）
        order_pct = float(self.position_size) * float(self.leverage)
        order_value = order_pct * float(self.init_cash)

        # 3) 回測（注意：不要帶不存在的參數，如 conflict_mode / shorting）
        pf = vbt.Portfolio.from_signals(
            close=self.price,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            size=order_value,
            size_type="value",            # 你的 vbt 版本支援 value/percent/amount；value 可反手
            sl_stop=float(self.sl),
            tp_stop=float(self.tp),
            stop_entry_price="fillprice",
            stop_exit_price="stopmarket",
            accumulate=False,             # 若報參數錯誤就移除這行
            fees=float(self.fees),
            slippage=float(self.slippage),
            cash_sharing=True,
            init_cash=float(self.init_cash),
            freq=_parse_tf_to_timedelta(self.timeframe) or "1H",
        )

        equity_with_funding = self._compute_equity_with_funding(pf)

        out = {
            "portfolio": pf,
            "stats": pf.stats(),
            "value_raw": pf.value(),
            "value_with_funding": equity_with_funding,
            "orders": getattr(pf.orders, "records_readable", pf.orders),
            "trades": getattr(pf.trades, "records_readable", pf.trades),
            "raw_signals": self.raw_signals_df,
        }
        return out



if __name__ == "__main__":

    df = pd.read_csv("mexc_swap_BTC-USDT-USDT_1h.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    bt = backtestStrategy(
        df=df,
        symbol="BTCUSDT",
        timeframe="1h",            # 以你的資料頻率為準
        volume_ema_window=25,
        position_size=0.10,
        leverage=3,
        tp=0.02,
        sl=0.01,
        fees=0.0005,
        slippage=0.0002,
        funding_rate_per_8h=0.0005,
        init_cash=100_000,
    )
    result = bt.run_backtest()
    print(result["stats"].to_string())
    # 互動圖（在 Notebook 中使用）
    # result["portfolio"].plot().show()




