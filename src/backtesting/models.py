from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0
    return_pct: float = 0.0
    direction: str = "long"

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    fee_rate: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.001
    market: str = "A"
    price_col: str = "Close"


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.Series
    trades: list[Trade]
    holdings_curve: pd.Series
    cash_curve: pd.Series
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0

    @property
    def metrics(self) -> dict:
        return {
            "总收益率": f"{self.total_return*100:.2f}%",
            "年化收益率": f"{self.annual_return*100:.2f}%",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "最大回撤": f"{self.max_drawdown*100:.2f}%",
            "卡玛比率": f"{self.calmar_ratio:.2f}",
            "总交易次数": str(self.total_trades),
            "胜率": f"{self.win_rate*100:.2f}%",
            "盈亏比": f"{self.profit_factor:.2f}",
            "总手续费": f"{self.total_fees:.2f}",
        }

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "equity": self.equity_curve,
            "holdings": self.holdings_curve,
            "cash": self.cash_curve,
        })

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "买入日": t.entry_date,
                "卖出日": t.exit_date or "",
                "买入价": t.entry_price,
                "卖出价": t.exit_price if t.exit_price else "",
                "股数": t.shares,
                "盈亏": t.pnl,
                "收益率": f"{t.return_pct*100:.2f}%",
            })
        return pd.DataFrame(rows)
