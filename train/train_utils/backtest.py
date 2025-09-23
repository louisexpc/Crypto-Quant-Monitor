import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# [Warning] vectorbt are dependent on NumPy version < 2.0
import vectorbt as vbt
from vectorbt.portfolio.enums import Direction

class BacktestExecutor:
    def __init__(self, 
                 olhcv_df: pd.DataFrame, 
                 predictions_df:pd.DataFrame,
                 init_cash: float = 10000.0, 
                 fees: float = 0.001, 
                 slippage: float = 0.001):
        """
        Initialize the BacktestExecutor with OHLCV data and predictions.
        Arguments:
            olhcv_df (pd.DataFrame): DataFrame containing OHLCV data with 'datetime' and 'close' columns.
            predictions_df (pd.DataFrame): DataFrame containing predictions with 'side' and 'pred_vote' columns.
                - 'side': 1 for long, -1 for short.
                - 'pred_vote': 1 for successful trade, and 0 for failed trade.
            init_cash (float): Initial cash for the backtest.
            fees (float): Trading fees as a decimal (e.g., 0.001 for 0.1%).
            slippage (float): Slippage as a decimal (e.g., 0.001 for 0.1%).
        """
        
        if 'datetime' not in olhcv_df.columns or 'close' not in olhcv_df.columns:
            raise ValueError("DataFrame must contain 'datetime' and 'close' columns.")
        
        if 'side' not in predictions_df.columns or 'pred_vote' not in predictions_df.columns:
            raise ValueError("Predictions DataFrame must contain 'side' and 'pred_vote' columns.")
        
        self.olhcv_df = olhcv_df
        self.olhcv_df.index = pd.to_datetime(self.olhcv_df['datetime'])

        self.price = self.olhcv_df['close']
        self.predictions_df = predictions_df

        self.long_entry, self.long_exit, self.short_entry, self.short_exit = self._init_long_short_signals()

        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
    
    def _init_long_short_signals(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Initialize long and short entry/exit signals based on predictions DataFrame.
        Returns:
            Tuple containing four pd.Series: long_entry, long_exit, short_entry, short_exit.
        """
        long_condition = (self.predictions_df['side'] == 1) & (self.predictions_df['pred_vote'] == 1)
        long_only = self.predictions_df[long_condition].dropna().reset_index(drop=True)

        short_condition = (self.predictions_df['side'] == -1) & (self.predictions_df['pred_vote'] == 1)
        short_only = self.predictions_df[short_condition].dropna().reset_index(drop=True)


        close = self.price.copy()
        long_entry = pd.Series([False] * len(close), index=close.index)
        long_exit = pd.Series([False] * len(close), index=close.index)
        short_entry = pd.Series([False] * len(close), index=close.index)
        short_exit = pd.Series([False] * len(close), index=close.index)

        for _,event in long_only.iterrows():
            t0 = event.loc['t0']
            t1 = event.loc['t1']
            long_entry.at[t0] = True
            long_exit.at[t1] = True

        for _,event in short_only.iterrows():
            t0 = event.loc['t0']
            t1 = event.loc['t1']

            short_entry.at[t0] = True
            short_exit.at[t1] = True
        
        long_same_bar = long_entry & long_exit
        short_same_bar = short_entry & short_exit

        if int(long_same_bar.sum()) > 0:
            print(f"[Warning] Found {long_same_bar.sum()} long entry/exit signals on the same bar. Adjusting to avoid conflicts.")
            long_exit[long_same_bar] = False

        if int(short_same_bar.sum()) > 0:
            print(f"[Warning] Found {short_same_bar.sum()} short entry/exit signals on the same bar. Adjusting to avoid conflicts.")
            short_exit[short_same_bar] = False

        long_entry_count, long_exit_count, short_entry_count, short_exit_count = long_entry.sum(), long_exit.sum(), short_entry.sum(), short_exit.sum()

        if long_entry_count == 0 :
            raise ValueError("No valid long entry signals found in the predictions DataFrame.")
        if short_entry_count == 0 :
            raise ValueError("No valid short entry signals found in the predictions DataFrame.")
        
        if long_exit_count == 0 :
            raise ValueError("No valid long exit signals found in the predictions DataFrame.")
        if short_exit_count == 0 :
            raise ValueError("No valid short exit signals found in the predictions DataFrame.")
        

        if long_entry_count != long_exit_count:
            print(f"[Warning] Mismatched long entry/exit signals: {long_entry_count} entries vs {long_exit_count} exits.")
        if short_entry_count != short_exit_count:
            print(f"[Warning] Mismatched short entry/exit signals: {short_entry_count} entries vs {short_exit_count} exits.")

        return long_entry, long_exit, short_entry, short_exit

    def run(self, save_results:bool = False) ->Tuple[float, float, float, float]:
        """
        Execute backtest for both long and short strategies using vectorbt.
        Arguments:
            save_results (bool): If True, saves trade and order details to CSV files.
        Returns:
            Tuple containing total return and Sharpe ratio for long and short strategies.

        """
        long_pf = vbt.Portfolio.from_signals(
            self.price,
            entries = self.long_entry,
            exits = self.long_exit,
            init_cash= self.init_cash,
            fees= self.fees,
            slippage= self.slippage,
            size_type= "value",
            size=1000,
        )

        self.long_status = long_pf.stats()

        short_pf = vbt.Portfolio.from_signals(
            self.price,
            entries = pd.Series(False, index=self.price.index),
            exits   = pd.Series(False, index=self.price.index),
            short_entries = self.short_entry.astype(bool),
            short_exits   = self.short_exit.astype(bool),
            direction = Direction.ShortOnly,             # # Key: allows short selling
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            size_type="value",
            size=1000,
        )

        self.short_status = short_pf.stats()


        if save_results:
            long_pf.trades.records_readable.to_csv("long_trade.csv", index=False)
            long_pf.orders.records_readable.to_csv("long_order.csv", index=False)
            short_pf.trades.records_readable.to_csv("short_trade.csv", index=False)
            short_pf.orders.records_readable.to_csv("short_order.csv", index=False)
            print(f"[Info] Long and Short trade/order details saved to current folder.")
        return self.long_status['Total Return [%]'], self.long_status['Sharpe Ratio'], self.short_status['Total Return [%]'], self.short_status['Sharpe Ratio']

    def get_short_status(self) -> pd.Series:
        """Get the statistics of the short strategy."""
        return self.short_status

    def get_long_status(self) -> pd.Series:
        """Get the statistics of the long strategy."""
        return self.long_status
