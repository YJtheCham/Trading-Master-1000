"""
股票详情: 基本面 / 交易数据 / 板块 / 新闻
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_stock_info(symbol: str, market: str) -> dict:
    """获取股票基本信息 (行业/板块/市值等)"""
    info = {"symbol": symbol, "market": market, "name": "", "sector": "",
            "industry": "", "market_cap": "", "pe": "", "volume": ""}
    try:
        if market == "A":
            _fill_a_info(symbol, info)
        elif market == "US":
            _fill_us_info(symbol, info)
        elif market == "HK":
            _fill_hk_info(symbol, info)
    except Exception as e:
        logger.warning(f"获取 {market}:{symbol} 详情失败: {e}")
    return info


def _fill_a_info(symbol: str, info: dict):
    import akshare as ak
    # 实时行情 (含行业)
    df = ak.stock_zh_a_spot_em()
    row = df[df["代码"] == symbol]
    if row.empty:
        return
    r = row.iloc[0]
    info["name"] = str(r.get("名称", ""))
    info["sector"] = str(r.get("行业", ""))
    info["market_cap"] = _fmt_val(r.get("总市值", ""))
    info["pe"] = str(r.get("市盈率-动态", ""))
    info["volume"] = _fmt_val(r.get("成交量", ""))
    info["high"] = str(r.get("最高", ""))
    info["low"] = str(r.get("最低", ""))
    info["open"] = str(r.get("开盘", ""))
    info["pre_close"] = str(r.get("昨收", ""))
    info["amount"] = _fmt_val(r.get("成交额", ""))
    info["amplitude"] = str(r.get("振幅", ""))
    info["change_pct"] = str(r.get("涨跌幅", ""))
    info["change_amount"] = str(r.get("涨跌额", ""))


def _fill_us_info(symbol: str, info: dict):
    import yfinance as yf
    t = yf.Ticker(symbol)
    try:
        q = t.fast_info
        info["name"] = str(q.get("shortName") or q.get("longName") or "")
        info["sector"] = str(q.get("sector") or "")
        info["industry"] = str(q.get("industry") or "")
        mc = q.get("marketCap") or 0
        info["market_cap"] = _fmt_val(mc)
        info["pe"] = str(round(q.get("trailingPE") or 0, 2))
        info["volume"] = _fmt_val(q.get("regularMarketVolume") or 0)
        info["high"] = str(round(q.get("dayHigh") or 0, 2))
        info["low"] = str(round(q.get("dayLow") or 0, 2))
        info["open"] = str(round(q.get("regularMarketOpen") or 0, 2))
        info["pre_close"] = str(round(q.get("previousClose") or 0, 2))
        info["change_pct"] = str(round(q.get("regularMarketChangePercent") or 0, 2))
    except Exception:
        # fallback to history
        hist = t.history(period="5d")
        if not hist.empty:
            info["name"] = symbol


def _fill_hk_info(symbol: str, info: dict):
    import akshare as ak
    df = ak.stock_hk_spot_em()
    row = df[df["代码"] == symbol]
    if row.empty:
        return
    r = row.iloc[0]
    info["name"] = str(r.get("名称", ""))
    info["sector"] = str(r.get("行业", ""))
    info["market_cap"] = _fmt_val(r.get("总市值", ""))
    info["pe"] = str(r.get("市盈率", ""))
    info["volume"] = _fmt_val(r.get("成交量", ""))
    info["high"] = str(r.get("最高", ""))
    info["low"] = str(r.get("最低", ""))
    info["open"] = str(r.get("开盘", ""))
    info["change_pct"] = str(r.get("涨跌幅", ""))


def _fmt_val(val) -> str:
    """格式化大数字: 1234567 → 1.23亿"""
    try:
        v = float(val)
        if v >= 1e8:
            return f"{v/1e8:.2f}亿"
        elif v >= 1e4:
            return f"{v/1e4:.2f}万"
        return str(round(v, 2))
    except (ValueError, TypeError):
        return str(val)


# ─── 新闻 ─────────────────────────────────────────────────
def get_news(symbol: str, market: str, limit: int = 10) -> list[dict]:
    """获取最新新闻/公告"""
    try:
        if market == "A":
            return _news_a(symbol, limit)
        elif market == "US":
            return _news_us(symbol, limit)
        elif market == "HK":
            return _news_hk(symbol, limit)
    except Exception as e:
        logger.warning(f"获取 {symbol} 新闻失败: {e}")
    return []


def _news_a(symbol: str, limit: int = 10) -> list[dict]:
    import akshare as ak
    df = ak.stock_news_em(symbol=symbol)
    if df.empty:
        return []
    df = df.head(limit)
    news = []
    for _, r in df.iterrows():
        news.append({
            "title": str(r.get("新闻标题", "")),
            "time": str(r.get("发布时间", ""))[:16],
            "url": str(r.get("新闻内容", "")),
        })
    return news


def _news_us(symbol: str, limit: int = 10) -> list[dict]:
    import yfinance as yf
    t = yf.Ticker(symbol)
    try:
        articles = t.news or []
    except Exception:
        return []
    news = []
    for a in articles[:limit]:
        news.append({
            "title": a.get("title", ""),
            "time": datetime.fromtimestamp(a.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M") if a.get("providerPublishTime") else "",
            "url": a.get("link", ""),
        })
    return news


def _news_hk(symbol: str, limit: int = 10) -> list[dict]:
    import akshare as ak
    try:
        df = ak.stock_hk_news_em(symbol=symbol)
        if df.empty:
            return []
        news = []
        for _, r in df.head(limit).iterrows():
            news.append({
                "title": str(r.get("新闻标题", "")),
                "time": str(r.get("发布时间", ""))[:16],
                "url": str(r.get("新闻链接", "")),
            })
        return news
    except Exception:
        return []


# ─── 走势图 ───────────────────────────────────────────────
def get_recent_performance(df: pd.DataFrame) -> dict:
    """近期表现: 5日/20日/60日涨跌幅"""
    closes = df["Close"].values
    result = {}
    periods = {"5日": 5, "20日": 20, "60日": 60}
    for label, days in periods.items():
        if len(closes) > days:
            pct = (closes[-1] - closes[-days]) / closes[-days] * 100
            result[label] = f"{pct:+.1f}%"
        else:
            result[label] = "-"
    return result
