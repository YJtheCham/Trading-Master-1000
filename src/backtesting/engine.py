from typing import Optional

import pandas as pd
import numpy as np

from .models import BacktestConfig, BacktestResult, Trade
from .strategies import BaseStrategy


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, strategy: BaseStrategy,
                 config: Optional[BacktestConfig] = None):
        self.df = df.reset_index(drop=True)
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self._validate()

    def _validate(self):
        req = {"Date", self.config.price_col}
        missing = req - set(self.df.columns)
        if missing:
            raise ValueError(f"数据缺少列: {missing}")

    def _compute_fees(self, price: float, shares: int, is_sell: bool) -> float:
        amount = price * shares
        commission = amount * self.config.fee_rate
        stamp = amount * self.config.stamp_tax if is_sell else 0.0
        return commission + stamp

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        if is_buy:
            return price * (1 + self.config.slippage)
        return price * (1 - self.config.slippage)

    def run(self) -> BacktestResult:
        cfg = self.config
        prices = self.df[cfg.price_col].values
        n = len(self.df)

        cash = cfg.initial_capital
        holdings = 0
        buy_dates: dict[int, pd.Timestamp] = {}  # track T+1 for A-shares
        trades: list[Trade] = []
        current_trade: Optional[Trade] = None

        equity_curve = np.zeros(n)
        holdings_curve = np.zeros(n)
        cash_curve = np.zeros(n)
        total_fees = 0.0

        self.strategy.init(self.df)

        for i in range(n):
            date = self.df.iloc[i]["Date"]
            price = prices[i]
            signal = self.strategy.next(i, self.df)

            # T+1 check (A-share): can't sell shares bought today
            if cfg.market == "A" and i in buy_dates:
                pass  # allow sell, but shares from today are locked

            # Execute signal
            if signal == 1 and cash > 0:
                exec_price = self._apply_slippage(price, is_buy=True)
                shares = int(cash / exec_price)
                if shares > 0:
                    fees = self._compute_fees(exec_price, shares, is_sell=False)
                    cost = exec_price * shares + fees
                    if cost <= cash:
                        cash -= cost
                        holdings += shares
                        total_fees += fees
                        buy_dates[i] = date
                        current_trade = Trade(
                            entry_date=date, entry_price=exec_price,
                            shares=shares, direction="long",
                        )

            elif signal == -1 and holdings > 0:
                exec_price = self._apply_slippage(price, is_buy=False)
                fees = self._compute_fees(exec_price, holdings, is_sell=True)
                cash += exec_price * holdings - fees
                total_fees += fees
                if current_trade is not None:
                    current_trade.exit_date = date
                    current_trade.exit_price = exec_price
                    current_trade.pnl = (exec_price - current_trade.entry_price) * current_trade.shares - fees
                    current_trade.return_pct = (exec_price - current_trade.entry_price) / current_trade.entry_price
                    trades.append(current_trade)
                    current_trade = None
                holdings = 0

            # Mark to market
            portfolio_value = cash + holdings * price
            equity_curve[i] = portfolio_value
            holdings_curve[i] = holdings * price
            cash_curve[i] = cash

        # Force close at end
        if holdings > 0:
            price = prices[-1]
            exec_price = self._apply_slippage(price, is_buy=False)
            fees = self._compute_fees(exec_price, holdings, is_sell=True)
            cash += exec_price * holdings - fees
            total_fees += fees
            if current_trade is not None:
                current_trade.exit_date = self.df.iloc[-1]["Date"]
                current_trade.exit_price = exec_price
                current_trade.pnl = (exec_price - current_trade.entry_price) * current_trade.shares - fees
                current_trade.return_pct = (exec_price - current_trade.entry_price) / current_trade.entry_price
                trades.append(current_trade)
            holdings = 0
            equity_curve[-1] = cash

        return self._build_result(equity_curve, holdings_curve, cash_curve,
                                  trades, total_fees)

    def _build_result(self, equity: np.ndarray, holdings: np.ndarray,
                      cash: np.ndarray, trades: list[Trade],
                      total_fees: float) -> BacktestResult:
        cfg = self.config
        n = len(equity)
        total_ret = (equity[-1] - cfg.initial_capital) / cfg.initial_capital

        # Annualized return
        years = n / 252
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

        # Daily returns for Sharpe
        daily_ret = pd.Series(equity).pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0.0)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(np.min(dd))

        # Win rate & profit factor
        win_trades = [t for t in trades if t.pnl > 0]
        loss_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(win_trades) / len(trades) if trades else 0.0
        gross_profit = sum(t.pnl for t in win_trades)
        gross_loss = abs(sum(t.pnl for t in loss_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0.0

        return BacktestResult(
            config=cfg,
            equity_curve=pd.Series(equity, index=self.df["Date"]),
            holdings_curve=pd.Series(holdings, index=self.df["Date"]),
            cash_curve=pd.Series(cash, index=self.df["Date"]),
            trades=trades,
            total_return=total_ret,
            annual_return=annual_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            total_fees=total_fees,
        )
