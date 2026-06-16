"""
请求频率控制: Token Bucket 算法, 数据源级别限流

防止因请求过快被 API 封禁。
"""
import time
import threading
from functools import wraps
from typing import Optional


class TokenBucket:
    """令牌桶限流器"""

    def __init__(self, rate: float, capacity: Optional[int] = None):
        """
        Args:
            rate: 每秒恢复的令牌数
            capacity: 最大令牌数 (默认 = rate)
        """
        self.rate = rate
        self.capacity = capacity or int(rate)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """消费 tokens 个令牌, 返回是否成功"""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: int = 1, timeout: float = 60) -> bool:
        """阻塞直到获取到令牌或超时"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.consume(tokens):
                return True
            time.sleep(max(0.01, 1.0 / self.rate / 2))
        return False


# ─── 各数据源限流配置 ─────────────────────────────────────
# rates: 每秒允许的请求数
SOURCE_LIMITS = {
    "东方财富": TokenBucket(rate=3, capacity=10),    # 东方财富: 3次/秒
    "新浪财经": TokenBucket(rate=5, capacity=20),    # 新浪: 5次/秒
    "Yahoo Finance": TokenBucket(rate=1, capacity=5), # Yahoo: 1次/秒 (严格)
    "Tushare": TokenBucket(rate=3, capacity=10),     # Tushare: 3次/秒
    "模拟数据": TokenBucket(rate=100, capacity=100), # Mock: 不限
    "港股-东方财富": TokenBucket(rate=3, capacity=10),
    "东方财富(美股)": TokenBucket(rate=3, capacity=10),
    "default": TokenBucket(rate=2, capacity=10),
}


def get_limiter(source_name: str) -> TokenBucket:
    return SOURCE_LIMITS.get(source_name, SOURCE_LIMITS["default"])


def rate_limited(source_name: str):
    """装饰器: 对被装饰函数应用限流"""
    limiter = get_limiter(source_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.wait_and_consume(timeout=30):
                raise RuntimeError(
                    f"{source_name} 请求过于频繁, 请稍后重试")
            return func(*args, **kwargs)
        return wrapper

    return decorator
