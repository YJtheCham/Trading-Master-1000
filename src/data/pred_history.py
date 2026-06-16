"""
预测历史存储: JSON 文件, 每行一条记录
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config import DATA_DIR

HISTORY_FILE = DATA_DIR / "prediction_history.json"


@dataclass
class PredictionRecord:
    id: str
    symbol: str
    market: str
    stock_name: str
    model: str
    predicted_at: str
    steps: int
    forecast: list[float] = field(default_factory=list)
    forecast_dates: list[str] = field(default_factory=list)
    last_price: float = 0.0
    final_prediction: float = 0.0
    direction: str = ""
    mape: float = 0.0

    @property
    def final_pct(self) -> float:
        if self.last_price:
            return (self.final_prediction - self.last_price) / self.last_price * 100
        return 0.0


def load_history() -> list[PredictionRecord]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text())
        return [PredictionRecord(**r) for r in data]
    except Exception:
        return []


def save_history(records: list[PredictionRecord]):
    HISTORY_FILE.write_text(
        json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))


def add_prediction(symbol: str, market: str, stock_name: str,
                   model: str, forecast: np.ndarray,
                   forecast_dates: list, last_price: float, mape: float) -> PredictionRecord:
    f_list = forecast.tolist() if hasattr(forecast, "tolist") else list(forecast)
    # 确保所有值是 JSON 序列化兼容的 Python 类型
    f_list = [float(v) for v in f_list]
    steps = len(f_list)
    final = f_list[-1] if steps > 0 else float(last_price)
    direction = "📈涨" if final > float(last_price) else "📉跌"
    fd_list = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
               for d in forecast_dates[:steps]]
    record = PredictionRecord(
        id=f"{market}_{symbol}_{model}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        symbol=symbol, market=market, stock_name=stock_name,
        model=model, steps=steps,
        predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        forecast=f_list,
        forecast_dates=fd_list,
        last_price=float(last_price),
        final_prediction=final,
        direction=direction,
        mape=float(mape),
    )
    history = load_history()
    # Dedup: 同股票同模型只保留最新一条
    history = [r for r in history
               if not (r.symbol == record.symbol and r.model == record.model)]
    history.append(record)
    # 最多保留 200 条
    history = history[-200:]
    save_history(history)
    return record
