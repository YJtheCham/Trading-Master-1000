"""
策略推荐历史: JSON 文件
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.config import DATA_DIR

REC_HISTORY_FILE = DATA_DIR / "recommend_history.json"


@dataclass
class RecHistory:
    id: str
    symbol: str
    market: str
    stock_name: str
    current_price: float
    model_consensus: str
    avg_pct: float
    best_strategy: str
    best_ret: float
    best_sharpe: float
    best_maxdd: float
    total_models: int
    total_strats: int
    predicted_at: str


def load_rec_history() -> list[RecHistory]:
    if not REC_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(REC_HISTORY_FILE.read_text())
        return [RecHistory(**r) for r in data]
    except Exception:
        return []


def save_rec_history(records: list[RecHistory]):
    REC_HISTORY_FILE.write_text(
        json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))


def add_rec_history(symbol: str, market: str, stock_name: str,
                    current_price: float, model_consensus: str, avg_pct: float,
                    best_strategy: str, best_ret: float, best_sharpe: float,
                    best_maxdd: float, total_models: int, total_strats: int):
    record = RecHistory(
        id=f"rec_{market}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        symbol=symbol, market=market, stock_name=stock_name,
        current_price=current_price, model_consensus=model_consensus,
        avg_pct=avg_pct, best_strategy=best_strategy,
        best_ret=best_ret, best_sharpe=best_sharpe, best_maxdd=best_maxdd,
        total_models=total_models, total_strats=total_strats,
        predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    history = load_rec_history()
    history = [r for r in history
               if not (r.symbol == record.symbol and r.best_strategy == record.best_strategy)]
    history.append(record)
    history = history[-200:]
    save_rec_history(history)
    return record
