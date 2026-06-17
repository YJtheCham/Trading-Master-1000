"""
Wind MCP 数据源: 通过 Node.js CLI 调用 Wind 云 API
"""
import json, subprocess, os, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WIND_CLI_DIR = os.path.expanduser(
    "~/.config/opencode/skills/skills/wind-mcp-skill")

SKILL_DIR = os.path.expanduser(
    "~/.config/opencode/skills/skills/wind-mcp-skill")


def _call_wind(server: str, tool: str, params: dict) -> Optional[dict]:
    """调用 Wind MCP CLI"""
    if not os.path.isdir(WIND_CLI_DIR):
        return None
    try:
        cmd = ["node", "scripts/cli.mjs", "call", server, tool,
               json.dumps(params, ensure_ascii=False)]
        r = subprocess.run(cmd, cwd=WIND_CLI_DIR, capture_output=True,
                           text=True, timeout=15)
        data = json.loads(r.stdout)
        if data.get("isError"):
            return None
        text = data.get("content", [{}])[0].get("text", "{}")
        inner = json.loads(text)
        if inner.get("error"):
            return None
        return inner.get("data", {})
    except Exception as e:
        logger.warning(f"Wind MCP failed: {e}")
        return None


def get_price_indicators(windcode: str,
                         indexes: str = "中文简称,最新成交价,涨跌幅,成交量,市盈率(TTM),总市值") -> Optional[dict]:
    """获取实时行情指标"""
    data = _call_wind("stock_data", "get_stock_price_indicators",
                      {"windcode": windcode, "indexes": indexes})
    if not data or not data.get("rows"):
        return None
    cols = [c["name"] for c in data.get("columns", [])]
    vals = data["rows"][0]
    return dict(zip(cols, vals))


def get_kline(windcode: str, begin_date: str, end_date: str,
              indicators: str = "收盘价,开盘价,最高价,最低价,成交量") -> Optional[list]:
    """获取K线数据"""
    data = _call_wind("stock_data", "get_stock_kline",
                      {"windcode": windcode, "beginDate": begin_date,
                       "endDate": end_date, "indicators": indicators})
    if not data or not data.get("rows"):
        return None
    cols = [c["name"] for c in data.get("columns", [])]
    return [{cols[i]: row[i] for i in range(len(cols))} for row in data["rows"]]


def get_basic_info(windcode: str) -> Optional[dict]:
    """获取行业分类"""
    data = _call_wind("stock_data", "get_stock_basicinfo",
                      {"windcode": windcode, "question": "行业分类"})
    if not data:
        return None
    # 格式: data.data[0] 或 data.rows
    inner = data.get("data", [data])
    if isinstance(inner, list) and len(inner) > 0:
        inner = inner[0]
    rows = inner.get("rows", []) or data.get("rows", [])
    cols = [c["name"] for c in inner.get("columns", []) or data.get("columns", [])]
    if not rows or not cols:
        return None
    # 找匹配 windcode 的行
    for row in rows:
        if windcode in row:
            return dict(zip(cols, row))
    return None


def get_financial(windcode: str, report_date: str = "") -> Optional[dict]:
    """获取财务数据"""
    params = {"windcode": windcode,
              "question": "最近一期净资产收益率、营业收入、净利润、资产负债率、毛利率"}
    data = _call_wind("stock_data", "get_stock_financial", params)
    if not data:
        return None
    inner = data.get("data", [data])
    if isinstance(inner, list) and len(inner) > 0:
        inner = inner[0]
    rows = inner.get("rows", []) or data.get("rows", [])
    cols = [c["name"] for c in inner.get("columns", []) or data.get("columns", [])]
    if not rows or not cols:
        return None
    for row in rows:
        if windcode in row:
            return dict(zip(cols, row))
    return dict(zip(cols, rows[0])) if rows else None


def get_news(windcode: str, limit: int = 10) -> list[dict]:
    """获取最新公告/新闻"""
    data = _call_wind("financial_docs", "get_financial_news",
                      {"query": f"{windcode} 最新", "limit": str(limit)})
    if not data:
        return []
    inner = data.get("data", [data])
    if isinstance(inner, list) and len(inner) > 0:
        inner = inner[0]
    rows = inner.get("rows", []) or data.get("rows", [])
    cols = [c["name"] for c in inner.get("columns", []) or data.get("columns", [])]
    news = []
    for row in rows[:limit]:
        item = dict(zip(cols, row)) if cols else {}
        news.append({"title": item.get("标题", item.get("title", "")),
                     "time": item.get("时间", item.get("time", "")),
                     "source": item.get("来源", "")})
    return news


def get_stock_full(windcode: str) -> dict:
    """一次性获取完整数据"""
    full = {
        "windcode": windcode,
        "name": "", "price": "", "change_pct": "", "change_amt": "",
        "volume": "", "amount": "", "turnover": "",
        "pe_ttm": "", "pe_lyr": "", "pb": "", "dividend_yield": "",
        "market_cap": "", "high": "", "low": "", "open": "", "pre_close": "",
        "high_52w": "", "low_52w": "",
        "industry": "", "roe": "", "revenue": "", "net_profit": "",
        "debt_ratio": "", "gross_margin": "",
        "news": [],
    }
    # 行情指标
    indexes = ("中文简称,最新成交价,涨跌幅,涨跌额,成交量,成交额,换手率,"
               "市盈率(TTM),市盈率(LYR),市净率(LF),股息率,总市值1,总市值2,"
               "今日开盘价,今日最高价,今日最低价,前收盘价,52周最高,52周最低")
    pi = get_price_indicators(windcode, indexes)
    if pi:
        full["name"] = pi.get("中文简称", "")
        full["price"] = pi.get("最新成交价", "")
        full["change_pct"] = pi.get("涨跌幅", "")
        full["change_amt"] = pi.get("涨跌额", "")
        full["volume"] = pi.get("成交量", "")
        full["amount"] = pi.get("成交额", "")
        full["turnover"] = pi.get("换手率", "")
        full["pe_ttm"] = pi.get("市盈率(TTM)", "")
        full["pe_lyr"] = pi.get("市盈率(LYR)", "")
        full["pb"] = pi.get("市净率(LF)", "")
        full["dividend_yield"] = pi.get("股息率", "")
        full["market_cap"] = pi.get("总市值2", pi.get("总市值1", ""))
        full["high"] = pi.get("今日最高价", "")
        full["low"] = pi.get("今日最低价", "")
        full["open"] = pi.get("今日开盘价", "")
        full["pre_close"] = pi.get("前收盘价", "")
        full["high_52w"] = pi.get("52周最高", "")
        full["low_52w"] = pi.get("52周最低", "")

    # 行业
    bi = get_basic_info(windcode)
    if bi:
        full["industry"] = bi.get("所属WIND行业明细", bi.get("行业分类", ""))

    # 财务
    fi = get_financial(windcode)
    if fi:
        full["roe"] = fi.get("净资产收益率(ROE)", fi.get("净资产收益率", ""))
        full["revenue"] = fi.get("营业收入", "")
        full["net_profit"] = fi.get("净利润", "")
        full["debt_ratio"] = fi.get("资产负债率", "")
        full["gross_margin"] = fi.get("毛利率", "")

    # 新闻
    try:
        full["news"] = get_news(windcode, 5)
    except Exception:
        pass

    return full


def a_stock_to_windcode(symbol: str) -> str:
    """600519 → 600519.SH, 000001 → 000001.SZ"""
    s = symbol.strip()
    if "." in s:
        return s
    return s + (".SH" if s.startswith(("6", "9")) else ".SZ")
