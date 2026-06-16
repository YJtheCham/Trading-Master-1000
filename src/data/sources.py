"""
数据源抽象层: 多通道回退 + 重试 + 诊断

市场 → 有序数据源列表
  A:   东方财富 → 新浪财经 → Mock
  HK:  东方财富 → Yahoo Finance → Mock
  US:  Yahoo Finance → 东方财富(美股) → Mock
"""
import random
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 重试工具 (带 jitter) ─────────────────────────────────
def retry(max_attempts: int = 3, delay: float = 1.0,
          backoff: float = 2.0, jitter: float = 0.5):
    """指数退避重试 + 随机 jitter, 避免多实例同时重试导致雪崩"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts - 1:
                        base_wait = delay * (backoff ** attempt)
                        actual_wait = base_wait + random.uniform(0, jitter)
                        logger.debug(f"{func.__name__} 失败(第{attempt+1}次),"
                                     f" {actual_wait:.1f}s后重试: {e}")
                        time.sleep(actual_wait)
            raise last_err
        return wrapper
    return decorator


# ─── 数据源基类 ───────────────────────────────────────────
@dataclass
class SourceResult:
    success: bool
    data: Optional[pd.DataFrame] = None
    price: Optional[float] = None
    source_name: str = ""
    error: str = ""
    elapsed: float = 0.0


class BaseSource(ABC):
    """数据源基类"""
    name: str = "base"
    priority: int = 99

    @abstractmethod
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        ...

    def fetch_realtime(self, symbol: str, market: str = "") -> Optional[float]:
        return None

    def run_historical(self, symbol: str, period_days: int = 730,
                       market: str = "") -> SourceResult:
        from .ratelimit import get_limiter
        limiter = get_limiter(self.name)
        t0 = time.time()
        try:
            # 限流等待
            if not limiter.wait_and_consume(timeout=30):
                return SourceResult(False, error="请求被限流(超时30s)",
                                    source_name=self.name,
                                    elapsed=round(time.time() - t0, 3))
            df = self.fetch_historical(symbol, period_days, market)
            elapsed = time.time() - t0
            if df is not None and not df.empty:
                return SourceResult(True, data=df, source_name=self.name,
                                    elapsed=round(elapsed, 3))
            return SourceResult(False, error="空数据", source_name=self.name,
                                elapsed=round(elapsed, 3))
        except Exception as e:
            elapsed = time.time() - t0
            return SourceResult(False, error=str(e), source_name=self.name,
                                elapsed=round(elapsed, 3))


# ─── A股: 东方财富 ─────────────────────────────────────────
class EastMoneySource(BaseSource):
    name = "东方财富"
    priority = 1

    @retry(max_attempts=2, delay=0.5)
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        import akshare as ak
        start = (datetime.now() - timedelta(days=period_days)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return None
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def fetch_realtime(self, symbol: str, market: str = "") -> Optional[float]:
        import akshare as ak
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]
            return float(row.iloc[0]["最新价"]) if not row.empty else None
        except Exception:
            return None


# ─── A股: 新浪财经 ─────────────────────────────────────────
class SinaSource(BaseSource):
    name = "新浪财经"
    priority = 2

    @staticmethod
    def _a_symbol(symbol: str) -> str:
        """A股代码转新浪格式: 600xxx → sh600xxx, 00/30xxx → sz00xxx"""
        if symbol.startswith(("sh", "sz", "SH", "SZ")):
            return symbol.lower()
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return prefix + symbol

    @retry(max_attempts=2, delay=0.5)
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        import akshare as ak
        sym = self._a_symbol(symbol)
        df = ak.stock_zh_a_daily(symbol=sym,
                                 start_date=(datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d"),
                                 end_date=datetime.now().strftime("%Y-%m-%d"),
                                 adjust="qfq")
        if df.empty:
            return None
        df = df.rename(columns={"date": "Date", "open": "Open", "close": "Close",
                                "high": "High", "low": "Low", "volume": "Volume"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)


# ─── 港股: 东方财富 ───────────────────────────────────────
class HKEastMoneySource(BaseSource):
    name = "港股-东方财富"
    priority = 1

    @retry(max_attempts=2, delay=0.5)
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        import akshare as ak
        start = (datetime.now() - timedelta(days=period_days)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        df = ak.stock_hk_hist(symbol=symbol, period="daily",
                              start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return None
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def fetch_realtime(self, symbol: str, market: str = "") -> Optional[float]:
        import akshare as ak
        try:
            df = ak.stock_hk_spot_em()
            row = df[df["代码"] == symbol]
            return float(row.iloc[0]["最新价"]) if not row.empty else None
        except Exception:
            return None


# ─── 美股: Yahoo Finance ──────────────────────────────────
class YahooSource(BaseSource):
    name = "Yahoo Finance"
    priority = 1 if "US" in str(BaseSource) else 1  # will be set per market below

    @staticmethod
    def _yahoo_symbol(symbol: str, market: str) -> str:
        """转为 Yahoo Finance 可识别的代码格式"""
        s = symbol.strip()
        if market == "A":
            # 600519 → 600519.SS (上海), 000001 → 000001.SZ (深圳), 300750 → 300750.SZ
            if s.startswith(("6", "9")):
                s = s + ".SS"
            else:
                s = s + ".SZ"
        elif market == "HK":
            # 00700 → 0700.HK, 00001 → 0001.HK, 09988 → 9988.HK
            stripped = s.lstrip("0")
            if len(stripped) < 4:
                stripped = stripped.zfill(4)
            s = stripped + ".HK"
        return s

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        import yfinance as yf
        sym = self._yahoo_symbol(symbol, market)
        ticker = yf.Ticker(sym)
        df = ticker.history(period=f"{period_days}d")
        if df.empty:
            return None
        df = df.reset_index()
        col_map = {}
        for c in df.columns:
            c2 = str(c).capitalize()
            if c2 in ("Datetime", "Date"):
                col_map[c] = "Date"
            else:
                col_map[c] = c2
        df = df.rename(columns=col_map)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def fetch_realtime(self, symbol: str, market: str = "") -> Optional[float]:
        import yfinance as yf
        try:
            sym = self._yahoo_symbol(symbol, market)
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            return (info.get("lastPrice") or info.get("regularMarketPrice")
                    or info.get("previousClose") or None)
        except Exception:
            return None


# ─── 美股: 东方财富 (备用) ────────────────────────────────
class USEastMoneySource(BaseSource):
    name = "东方财富(美股)"
    priority = 2

    @retry(max_attempts=2, delay=0.5)
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        import akshare as ak
        df = ak.stock_us_hist(symbol=symbol, period="daily",
                              start_date=(datetime.now() - timedelta(days=period_days)).strftime("%Y%m%d"),
                              end_date=datetime.now().strftime("%Y%m%d"),
                              adjust="qfq")
        if df.empty:
            return None
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)


# ─── Tushare (A 股, 需 Token) ─────────────────────────────
class TushareSource(BaseSource):
    """Tushare Pro 数据源 (需配置 token)
    优势: 海外可用, 数据质量高, 支持复权
    缺点: 有调用频率限制 (200次/分钟), 需注册 token
    """
    name = "Tushare"
    priority = 1

    def __init__(self):
        super().__init__()
        self._pro = None

    def _get_pro(self):
        if self._pro is not None:
            return self._pro
        from src.utils.config import get_tushare_token
        import tushare as ts
        token = get_tushare_token()
        if not token:
            raise ValueError("未配置 Tushare Token，请设置环境变量 TUSHARE_TOKEN "
                             "或运行: stock config set-tushare-token <token>")
        self._pro = ts.pro_api(token)
        return self._pro

    @staticmethod
    def _ts_code(symbol: str) -> str:
        """600519 → 600519.SH, 000001 → 000001.SZ, 300750 → 300750.SZ"""
        s = symbol.strip()
        suffix = ".SH" if s.startswith(("6", "9")) else ".SZ"
        return s + suffix

    @retry(max_attempts=2, delay=0.5)
    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        import tushare as ts
        pro = self._get_pro()
        ts_code = self._ts_code(symbol)
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=period_days)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "trade_date": "Date", "open": "Open", "close": "Close",
            "high": "High", "low": "Low", "vol": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def fetch_realtime(self, symbol: str, market: str = "") -> Optional[float]:
        return None  # Tushare 不提供实时行情


# ─── Mock 数据源 (最终回退) ────────────────────────────────
class MockSource(BaseSource):
    name = "模拟数据"
    priority = 999

    def fetch_historical(self, symbol: str, period_days: int = 730,
                         market: str = "") -> Optional[pd.DataFrame]:
        np.random.seed(abs(hash(symbol)) % (2**31))
        n = min(period_days, 500)
        drift = {"A": 0.04, "HK": 0.03, "US": 0.05}.get(market, 0.03)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + drift)
        return pd.DataFrame({
            "Date": pd.date_range(datetime.now() - timedelta(days=n),
                                  periods=n, freq="B"),
            "Close": prices,
            "Open": prices * (1 + np.random.randn(n) * 0.005),
            "High": prices * (1 + abs(np.random.randn(n)) * 0.01),
            "Low": prices * (1 - abs(np.random.randn(n)) * 0.01),
            "Volume": np.random.randint(1e6, 5e8, n),
        })

    def fetch_realtime(self, symbol: str, market: str = "") -> Optional[float]:
        np.random.seed(abs(hash(symbol)) % (2**31))
        return round(100 + np.random.randn() * 20, 2)


# ─── 数据源注册 ────────────────────────────────────────────
# 优先级: Yahoo(海外可访问) > 东方财富(国内) > Mock
MARKET_SOURCES: dict[str, list[BaseSource]] = {
    "A":  [TushareSource(), YahooSource(), EastMoneySource(), SinaSource(), MockSource()],
    "HK": [YahooSource(), HKEastMoneySource(), MockSource()],
    "US": [YahooSource(), USEastMoneySource(), MockSource()],
}


def get_sources(market: str) -> list[BaseSource]:
    return MARKET_SOURCES.get(market, [MockSource()])


# ─── 数据源健康检查 ────────────────────────────────────────
@dataclass
class SourceHealth:
    source_name: str
    available: bool
    latency_ms: float
    error: str = ""


def _test_single_source(source: BaseSource, symbol: str, market: str,
                         timeout_sec: int) -> SourceHealth:
    """测试单个数据源 (在独立线程中运行)"""
    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_sec)
    t0 = time.time()
    try:
        result = source.run_historical(symbol, period_days=5, market=market)
        elapsed = (time.time() - t0) * 1000
        return SourceHealth(source.name, result.success,
                            round(elapsed, 1),
                            "" if result.success else result.error)
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return SourceHealth(source.name, False, round(elapsed, 1), str(e)[:80])
    finally:
        socket.setdefaulttimeout(old_timeout)


def check_all_sources(timeout_sec: int = 15) -> dict[str, list[SourceHealth]]:
    """并行检测所有数据源

    每个源一个独立线程, wait(FIRST_COMPLETED) 循环收集,
    超时未完成的记录为"超时"而非丢弃.
    """
    import concurrent.futures
    from functools import partial

    test_symbols = {"A": "000001", "HK": "00700", "US": "AAPL"}

    tasks = []
    for market, sources in MARKET_SOURCES.items():
        for source in sources:
            sym = test_symbols.get(market, "AAPL")
            fn = partial(_test_single_source, source, sym, market, timeout_sec)
            tasks.append((market, source.name, fn))

    seed: dict[str, list[SourceHealth]] = {m: [] for m in MARKET_SOURCES}

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=9)
    deadline = time.time() + len(tasks) * (timeout_sec + 1) + 5
    fut_map = {pool.submit(fn): (market, name) for market, name, fn in tasks}
    pending = set(fut_map.keys())

    while pending and time.time() < deadline:
        done, pending = concurrent.futures.wait(
            pending, timeout=max(0.5, deadline - time.time()),
            return_when=concurrent.futures.FIRST_COMPLETED)
        for fut in done:
            market, name = fut_map[fut]
            try:
                seed[market].append(fut.result(timeout=1))
            except Exception as e:
                seed[market].append(
                    SourceHealth(name, False, timeout_sec * 1000, str(e)[:80]))

    # 超时未完成的记录为超时
    for fut in pending:
        _, name = fut_map[fut]
        for m in seed:
            if not any(s.source_name == name for s in seed[m]):
                seed[m].append(SourceHealth(name, False, timeout_sec * 1000, "超时"))

    pool.shutdown(wait=False)
    return seed


def best_source(market: str) -> BaseSource:
    """返回该市场首个可用的数据源 (快速检测: 第一源最多等 6s)"""
    for s in get_sources(market):
        health = _test_single_source(s, "000001" if market == "A" else
                                     "00700" if market == "HK" else "AAPL",
                                     market, timeout_sec=6)
        if health.available:
            return s
    return MockSource()
