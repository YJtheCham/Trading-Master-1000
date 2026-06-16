"""
回测历史存储: JSON 文件
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.config import DATA_DIR

BT_HISTORY_FILE = DATA_DIR / "backtest_history.json"


@dataclass
class BacktestHistory:
    id: str
    symbol: str
    market: str
    stock_name: str
    strategy: str
    capital: float
    total_return: float
    annual_return: float
    sharpe: float
    max_dd: float
    win_rate: float
    profit_factor: float
    total_trades: int
    total_fees: float
    predicted_at: str


def load_bt_history() -> list[BacktestHistory]:
    if not BT_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(BT_HISTORY_FILE.read_text())
        return [BacktestHistory(**r) for r in data]
    except Exception:
        return []


def save_bt_history(records: list[BacktestHistory]):
    BT_HISTORY_FILE.write_text(
        json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))


def add_bt_record(symbol: str, market: str, stock_name: str,
                  strategy: str, capital: float,
                  total_return: float, annual_return: float,
                  sharpe: float, max_dd: float, win_rate: float,
                  profit_factor: float, total_trades: int, total_fees: float):
    record = BacktestHistory(
        id=f"bt_{market}_{symbol}_{strategy}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        symbol=symbol, market=market, stock_name=stock_name,
        strategy=strategy, capital=capital,
        total_return=total_return, annual_return=annual_return,
        sharpe=sharpe, max_dd=max_dd, win_rate=win_rate,
        profit_factor=profit_factor, total_trades=total_trades,
        total_fees=total_fees,
        predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    history = load_bt_history()
    history = [r for r in history
               if not (r.symbol == record.symbol and r.strategy == record.strategy)]
    history.append(record)
    history = history[-200:]
    save_bt_history(history)
    return record
