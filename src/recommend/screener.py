"""
条件选股器: 用告警条件扫描多只股票, 返回匹配列表
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScreenerHit:
    symbol: str
    name: str
    market: str
    current_price: float
    direction: str = ""
    message: str = ""


def screen_market(
    condition: str, params: dict, market: str = "A",
    symbols: Optional[list[str]] = None,
    limit: int = 20
) -> list[ScreenerHit]:
    """用单个条件扫描多只股票, 返回命中的列表"""
    from src.alerts.conditions import evaluate
    from src.alerts.models import AlertRule
    from src.data.fetcher import fetch_data
    from src.data.stock_db import get_db, search_stocks

    db = get_db()
    stock_list = symbols or [code for code, _ in db.all_stocks(market)[:300]]

    hits = []
    for code in stock_list:
        try:
            # 只扫100天, 速度快
            df = fetch_data(code, market, period_days=120, use_cache=True)
            if df.empty or len(df) < 30:
                continue
            name = db.get_name(code, market) or code
            rule = AlertRule(symbol=code, market=market, condition=condition, params=params)
            triggered, msg, action = evaluate(rule, df)
            if triggered:
                price = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                direction = "📈" if price >= prev else "📉"
                hits.append(ScreenerHit(
                    symbol=code, name=name, market=market,
                    current_price=price, direction=direction, message=msg,
                ))
                if len(hits) >= limit:
                    break
        except Exception as e:
            logger.debug(f"筛选 {code} 失败: {e}")
    return hits
