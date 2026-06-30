"""
数据获取入口: 多源回退 + 智能缓存 + 实时行情

对外 API (向后兼容):
  fetch_data(symbol, market, period_days, use_cache) -> DataFrame
  get_realtime_price(symbol, market) -> float | None
  load_watchlist / save_watchlist
  diagnose_sources() -> 数据源状态报告

策略:
  1. 按 priority 依次尝试各数据源
  2. 成功后缓存到 parquet
  3. 缓存有效期: 交易日当天有效, 非交易日用缓存
  4. 全部失败 → MockSource 兜底 + 警告
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from src.utils.config import CACHE_DIR, WATCHLIST_FILE, StockItem
from .sources import (
    MARKET_SOURCES, MockSource, check_all_sources, WIND_SOURCE,
)

logger = logging.getLogger(__name__)

# 市场交易时间 (用于缓存判断)
MARKET_HOURS = {
    "A":  (9, 15),   # 北京时间 15:00 收盘
    "HK": (16, 0),   # 16:00 收盘
    "US": (16, 0),   # 美东时间 16:00 收盘
}


def _cache_path(symbol: str, market: str) -> str:
    return str(CACHE_DIR / f"{market}_{symbol}.parquet")


def _is_trading_day(market: str) -> bool:
    """简单判断是否交易日: 周一到周五"""
    return datetime.now().weekday() < 5


def _cache_fresh(cache_path: str, market: str, max_age_hours: int = 24,
                 max_age_minutes: int = None) -> bool:
    """判断缓存是否有效

    额外检查: 如果缓存中最新日期不是今天 (或最近交易日),
    说明数据可能来自延迟源 (Tushare/Yahoo), 不应视为新鲜.
    """
    path = Path(cache_path)
    if not path.exists():
        return False

    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    now = datetime.now()

    if max_age_minutes is not None:
        return (now - mtime).total_seconds() < max_age_minutes * 60

    # 检查缓存数据最新日期是否够新
    try:
        df = pd.read_parquet(cache_path, columns=["Date"])
        if not df.empty:
            latest = pd.to_datetime(df["Date"].iloc[-1])
            if hasattr(latest, "date"):
                latest_date = latest.date()
            else:
                latest_date = latest
            # 数据最新日期距今超过 3 天 → 不新鲜
            if (now.date() - latest_date).days > 3:
                return False
            # 数据最新日期不是今天且距今超过1天 → 不新鲜（需要更新）
            if (now.date() - latest_date).days >= 1:
                return False
    except Exception:
        pass

    if _is_trading_day(market):
        if mtime.date() == now.date():
            return True
        if (now - mtime).days <= 2:
            return True
        return False

    return (now - mtime).total_seconds() < max_age_hours * 3600


def fetch_data(symbol: str, market: str, period_days: int = 730,
               use_cache: bool = True, max_age_minutes: int = None) -> pd.DataFrame:
    """获取历史数据, 多源回退 + Mock兜底

    Args:
        symbol: 股票代码
        market: A / HK / US
        period_days: 获取多少天数据
        use_cache: 是否使用本地缓存
    Returns:
        DataFrame (至少包含 Date, Close 列)
    Raises:
        Exception: 所有数据源(含Mock)都失败
    """
    cache_path = _cache_path(symbol, market)

    # 缓存命中
    if use_cache and _cache_fresh(cache_path, market, max_age_minutes=max_age_minutes):
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass  # 缓存损坏, 重新获取

    # 遍历数据源
    sources = MARKET_SOURCES.get(market, [])
    errors = []
    cached_df = None
    # 如果有缓存但不新鲜, 先读出来作为基准
    if Path(cache_path).exists():
        try:
            cached_df = pd.read_parquet(cache_path)
        except Exception:
            pass

    for source in sources:
        result = source.run_historical(symbol, period_days, market)
        if result.success and result.data is not None and not result.data.empty:
            df = result.data
            from .cleaning import clean_market_data, validate_data
            df = clean_market_data(df)
            issues = validate_data(df)
            if issues:
                logger.warning(f"{source.name} {symbol} 数据质量问题: {issues}")
            # 缓存写入策略: 只有比已有缓存更新才写入
            should_cache = True
            if cached_df is not None and not cached_df.empty and "Date" in cached_df.columns and "Date" in df.columns:
                try:
                    old_latest = pd.to_datetime(cached_df["Date"]).max()
                    new_latest = pd.to_datetime(df["Date"]).max()
                    today = datetime.now().date()
                    # 如果缓存数据有未来日期（Mock数据特征），无条件用新数据覆盖
                    if hasattr(old_latest, "date") and old_latest.date() > today:
                        logger.info(f"{source.name} 缓存含有未来日期({old_latest.date()}), 用新数据覆盖")
                    elif new_latest < old_latest:
                        should_cache = False
                        logger.info(f"{source.name} 数据({new_latest.date()})比缓存({old_latest.date()})旧, 不覆盖缓存")
                except Exception:
                    pass
            if should_cache:
                try:
                    df.to_parquet(cache_path, index=False)
                except Exception:
                    pass
            return df
        errors.append(f"{source.name}: {result.error}")

    error_msg = f"{market}:{symbol} 所有数据源失败:\n" + "\n".join(errors)
    raise ConnectionError(error_msg)


def get_realtime_price(symbol: str, market: str) -> float | None:
    """获取实时行情, 多源尝试"""
    sources = MARKET_SOURCES.get(market, [])
    for source in sources:
        try:
            result = source.fetch_realtime(symbol, market)
            if result is not None:
                return result
        except Exception:
            continue
    return None


class WindUnavailableError(ConnectionError):
    """Wind MCP 不可用, 需要实时数据但无法获取"""


def fetch_realtime_data(symbol: str, market: str, period_days: int = 120) -> pd.DataFrame:
    """获取实时数据 (优先 Wind MCP, VPS已部署Wind MCP CLI)

    Wind 不可用时抛出 WindUnavailableError, 不回退免费源,
    因为免费源无实时数据, 回退只会掩盖问题.
    """
    if market == "A":
        result = WIND_SOURCE.run_historical(symbol, period_days, market)
        if result.success and result.data is not None and not result.data.empty:
            df = result.data
            from .cleaning import clean_market_data
            df = clean_market_data(df)
            return df
        raise WindUnavailableError(
            f"Wind MCP 不可用 ({result.error}), 无法获取 {symbol} 实时数据")
    return fetch_data(symbol, market, period_days=period_days)


# ─── 诊断工具 ─────────────────────────────────────────────
def diagnose_sources() -> dict:
    """返回所有数据源的健康状态"""
    return {
        market: [
            {
                "name": h.source_name,
                "available": h.available,
                "latency_ms": h.latency_ms,
                "error": h.error,
            }
            for h in health_list
        ]
        for market, health_list in check_all_sources().items()
    }


def best_available_source(market: str) -> str:
    """返回当前市场最佳可用数据源名称"""
    from .sources import best_source
    return best_source(market).name


# ─── 自选管理 (保持不变) ───────────────────────────────────
def load_watchlist() -> list[StockItem]:
    if not WATCHLIST_FILE.exists():
        return []
    with open(WATCHLIST_FILE) as f:
        data = json.load(f)
    return [StockItem(**item) for item in data]


def save_watchlist(items: list[StockItem]):
    WATCHLIST_FILE.write_text(
        json.dumps([item.model_dump() for item in items],
                   ensure_ascii=False, indent=2)
    )
