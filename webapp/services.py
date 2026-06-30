"""
StockPredict 共享服务层 — Streamlit 和 NiceGUI 共用
从 webapp/app.py 提取，移除所有 Streamlit 依赖
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetcher import load_watchlist, save_watchlist, fetch_data, diagnose_sources
from src.utils.config import StockItem, DATA_DIR

# ─── 预设股票 ───────────────────────────────────────────────
PRESET_STOCKS = {
    "A-000001": {"symbol": "000001", "market": "A", "name": "平安银行"},
    "A-600519": {"symbol": "600519", "market": "A", "name": "贵州茅台"},
    "A-300750": {"symbol": "300750", "market": "A", "name": "宁德时代"},
    "HK-00700": {"symbol": "00700",  "market": "HK", "name": "腾讯控股"},
    "HK-09988": {"symbol": "09988",  "market": "HK", "name": "阿里巴巴"},
    "US-AAPL":  {"symbol": "AAPL",   "market": "US", "name": "Apple"},
    "US-TSLA":  {"symbol": "TSLA",   "market": "US", "name": "Tesla"},
    "US-MSFT":  {"symbol": "MSFT",   "market": "US", "name": "Microsoft"},
}


def watchlist_key(info: dict) -> str:
    return f"{info['market']}-{info['symbol']}"


def info_for(key: str, stock_names: dict = None) -> dict:
    if key in PRESET_STOCKS:
        return PRESET_STOCKS[key]
    parts = key.split("-", 1)
    if len(parts) == 2:
        market, symbol = parts
        name = ""
        if stock_names:
            name = stock_names.get(key, "")
        if not name:
            from src.data.stock_db import resolve_stock_name
            name = resolve_stock_name(symbol, market) or symbol
        return {"symbol": symbol, "market": market, "name": name}
    return {"symbol": key, "market": "A", "name": key}


# ─── 数据获取 ────────────────────────────────────────────────
def _mock_data(symbol: str, market: str, n: int = 500) -> pd.DataFrame:
    np.random.seed(abs(hash(f"{market}_{symbol}")) % (2**31))
    drift = {"A": 0.04, "HK": 0.03, "US": 0.05}.get(market, 0.03)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + drift)
    return pd.DataFrame({
        "Date": pd.date_range(datetime.now() - pd.Timedelta(days=n),
                              periods=n, freq="B"),
        "Close": prices,
        "Open": prices * (1 + np.random.randn(n) * 0.005),
        "High": prices * (1 + abs(np.random.randn(n)) * 0.01),
        "Low": prices * (1 - abs(np.random.randn(n)) * 0.01),
        "Volume": np.random.randint(1e6, 5e8, n),
    })


# Simple in-memory cache for get_data_for (replaces @st.cache_data)
_data_cache: dict = {}
_source_cache: dict = {}


def get_data_for(symbol: str, market: str, period_days: int = 500,
                 refresh: bool = False) -> pd.DataFrame:
    key = f"{market}:{symbol}:{period_days}"
    if not refresh and key in _data_cache:
        cached = _data_cache[key]
        if not cached.empty and len(cached) >= 5:
            return cached.copy()
        # Cache is stale/invalid — clear and re-fetch
        del _data_cache[key]
    try:
        df = fetch_data(symbol, market, period_days=period_days)
    except Exception:
        df = _mock_data(symbol, market)
    _data_cache[key] = df.copy()
    return df


def detect_source_name(symbol: str, market: str) -> str:
    key = f"src:{market}:{symbol}"
    if key in _source_cache:
        return _source_cache[key]
    from src.data.sources import get_sources
    for s in get_sources(market):
        try:
            r = s.run_historical(symbol, period_days=5, market=market)
            if r.success:
                _source_cache[key] = s.name
                return s.name
        except Exception:
            continue
    return "模拟数据"


def has_real_source(source_status: dict, market: str) -> bool:
    status = source_status.get(market, [])
    return any(s.get("available") and "模拟" not in s.get("name", "") for s in status)


def refresh_source_status() -> dict:
    return diagnose_sources()


def get_data_notify(symbol: str, market: str, name: str = "",
                    period_days: int = 500) -> tuple[pd.DataFrame, str]:
    """返回 (DataFrame, source_name)"""
    df = get_data_for(symbol, market, period_days)
    source = detect_source_name(symbol, market)
    return df, source


# ─── 自选管理 ────────────────────────────────────────────────
def add_to_watchlist(state: dict, symbol: str, market: str, name: str = "",
                     group: str = "默认") -> tuple[bool, str]:
    s = symbol.strip()
    if market == "US" and s.isdigit():
        if len(s) <= 5 and s.startswith("0"):
            market = "HK"
        elif len(s) <= 6:
            symbol = s.zfill(6)
            market = "A"
    key = f"{market}-{symbol}"

    wl = list(state.get("watchlist", []))
    if key in wl:
        return False, f"{name}({symbol}) 已在自选列表中"

    wl.append(key)
    state["watchlist"] = wl

    if "stock_names" not in state:
        state["stock_names"] = {}
    if "stock_groups" not in state:
        state["stock_groups"] = {}
    if "stock_order" not in state:
        state["stock_order"] = []

    if not name:
        from src.data.stock_db import resolve_stock_name
        name = resolve_stock_name(symbol, market) or symbol
    state["stock_names"][key] = name
    state["stock_groups"][key] = group
    if key not in state["stock_order"]:
        state["stock_order"].append(key)
    _persist_watchlist_from_state(state)
    return True, f"已添加 {market}:{symbol} {name}"


def remove_from_watchlist(state: dict, key: str) -> tuple[bool, str]:
    wl = list(state.get("watchlist", []))
    if key not in wl:
        return False, f"{key} 不在自选列表中"
    wl.remove(key)
    state["watchlist"] = wl
    state.get("stock_names", {}).pop(key, None)
    state.get("stock_groups", {}).pop(key, None)
    if "stock_order" in state and key in state["stock_order"]:
        state["stock_order"].remove(key)
    _persist_watchlist_from_state(state)
    name = state.get("stock_names", {}).get(key, key)
    return True, f"已移除 {name}"


def _persist_watchlist_from_state(state: dict):
    items = []
    for key in state.get("watchlist", []):
        parts = key.split("-", 1)
        if len(parts) == 2:
            m, s = parts
            n = state.get("stock_names", {}).get(key, s)
            g = state.get("stock_groups", {}).get(key, "默认")
            items.append(StockItem(symbol=s, market=m, name=n, group=g))
    save_watchlist(items)


def load_state_from_watchlist(state: dict):
    """从持久化 watchlist.json 初始化 state"""
    items = load_watchlist()
    if items:
        state["watchlist"] = list({f"{i.market}-{i.symbol}" for i in items})
    else:
        state["watchlist"] = list(f"{v['market']}-{v['symbol']}" for v in PRESET_STOCKS.values())
        for k, v in PRESET_STOCKS.items():
            state["stock_names"] = state.get("stock_names", {})
            state["stock_names"][k] = v["name"]

    state["stock_names"] = state.get("stock_names", {})
    for item in items:
        if item.name:
            key = f"{item.market}-{item.symbol}"
            state["stock_names"][key] = item.name

    state["stock_groups"] = state.get("stock_groups", {})
    for item in items:
        key = f"{item.market}-{item.symbol}"
        state["stock_groups"][key] = item.group or "默认"

    order_file = DATA_DIR / "watchlist_order.json"
    if order_file.exists():
        try:
            saved = json.loads(order_file.read_text())
            state["stock_order"] = [k for k in saved if k in state.get("watchlist", [])]
        except Exception:
            state["stock_order"] = list(state.get("watchlist", []))
    else:
        state["stock_order"] = list(state.get("watchlist", []))

    group_order_file = DATA_DIR / "group_order.json"
    if group_order_file.exists():
        try:
            state["group_order"] = json.loads(group_order_file.read_text())
        except Exception:
            state["group_order"] = sorted(set(state["stock_groups"].values()))
    else:
        state["group_order"] = sorted(set(state["stock_groups"].values()))

    state["active_group"] = "全部"
    state["selected_stock"] = None
    state["refresh_key"] = 0


# ─── 风控指标格式化 ─────────────────────────────────────────
PCT_KEYS = {"MaxDrawdown", "Volatility"}


def fmt_risk(k: str, v: float) -> str:
    if k in PCT_KEYS:
        return f"{v*100:.1f}%"
    if "VaR" in k or "CVaR" in k:
        return f"{v*100:.2f}%"
    if k == "SharpeRatio":
        return f"{v:.2f}"
    return f"{v:.4f}"


# ─── 缓存管理 ────────────────────────────────────────────────
def clear_all_caches():
    global _data_cache, _source_cache
    _data_cache.clear()
    _source_cache.clear()
