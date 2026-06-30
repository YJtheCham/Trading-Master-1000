"""
新闻获取 + 情绪分析: 多源新闻 → 日度情绪因子

支持 Finnhub (美股/港股) + 新浪 (A股) 两个源
VADER 做英文情绪, SnowNLP 做中文情绪
"""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "news"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NEWS_STORE = Path(__file__).resolve().parent.parent.parent / "data" / "news_data.parquet"


def _vader_sentiment(text: str) -> float:
    """VADER 英文情绪 [-1, 1]"""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except ImportError:
        return 0.0


def _snownlp_sentiment(text: str) -> float:
    """SnowNLP + 金融关键词 加权情绪 [-1, 1]"""
    score = 0.0
    try:
        from snownlp import SnowNLP
        score = SnowNLP(text).sentiments * 2 - 1
    except ImportError:
        pass

    pos_kw = ["增长", "上涨", "超预期", "突破", "利好", "涨停", "翻倍", "龙头", "订单",
              "中标", "量产", "扩产", "创新高", "净利润", "毛利率提升", "扭亏", "分红",
              "回购", "增持", "通过认证", "产能释放", "供不应求", "景气", "复苏"]
    neg_kw = ["下滑", "下跌", "亏损", "减持", "跌停", "暴雷", "罚款", "调查", "违约",
              "退市", "破产", "关停", "裁员", "诉讼", "商誉减值", "业绩预警", "不及预期",
              "产能过剩", "竞争加剧", "贸易战", "制裁"]

    kw = sum(1 for k in pos_kw if k in text) - sum(1 for k in neg_kw if k in text)
    if kw != 0:
        kw_score = min(1.0, abs(kw) * 0.3) * (1 if kw > 0 else -1)
        score = score * 0.4 + kw_score * 0.6

    return max(-1.0, min(1.0, score))


def compute_sentiment(text: str, market: str = "A") -> float:
    """统一情绪接口: A股中文→SnowNLP, 其他→VADER"""
    if not text or not text.strip():
        return 0.0
    if market == "A":
        return _snownlp_sentiment(text)
    return _vader_sentiment(text)


def fetch_finnhub_news(symbol: str, market: str, lookback_days: int = 7) -> pd.DataFrame:
    """Finnhub 免费新闻 API (美股/港股, 近7天)"""
    from src.utils.config import load_config
    cfg = load_config()
    api_key = cfg.get("finnhub_api_key", "")
    if not api_key:
        return pd.DataFrame()

    import requests
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if not isinstance(data, list):
            return pd.DataFrame()

        rows = []
        for item in data[:50]:
            headline = item.get("headline", "") or ""
            summary = item.get("summary", "") or ""
            text = headline + " " + summary
            sentiment = compute_sentiment(text, market)
            ts = item.get("datetime", 0)
            if ts:
                news_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                rows.append({"date": news_date, "sentiment": sentiment,
                             "headline": headline[:120]})
        if rows:
            df = pd.DataFrame(rows)
            daily = df.groupby("date").agg(
                news_sent_mean=("sentiment", "mean"),
                news_sent_std=("sentiment", "std"),
                news_count=("sentiment", "count"),
            ).reset_index()
            daily["news_sent_std"] = daily["news_sent_std"].fillna(0)
            return daily
    except Exception as e:
        logger.warning(f"Finnhub news failed for {symbol}: {e}")

    return pd.DataFrame()


def fetch_sina_news(symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    """新浪财经新闻爬取 (A股)"""
    try:
        import requests
        market_code = "sh" + symbol if symbol.startswith(("6", "9")) else "sz" + symbol
        url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{market_code}.phtml"
        resp = requests.get(url, timeout=10)
        resp.encoding = "gb2312"

        from html.parser import HTMLParser
        import re

        class NewsParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_link = False
                self.rows = []
                self.current = None

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "/corp/view/" in attrs_dict.get("href", ""):
                    self.in_link = True
                    self.current = {"text": "", "date": ""}

            def handle_endtag(self, tag):
                if tag == "a" and self.current:
                    self.in_link = False

            def handle_data(self, data):
                if self.in_link and self.current:
                    self.current["text"] += data.strip()
                if not self.in_link and self.current:
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", data)
                    if m:
                        self.current["date"] = m.group(1)
                        self.rows.append(self.current)
                        self.current = None

        parser = NewsParser()
        parser.feed(resp.text)

        if parser.rows:
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            filtered = [r for r in parser.rows if r["date"] >= cutoff]
            for r in filtered:
                r["sentiment"] = compute_sentiment(r["text"], "A")
            if filtered:
                df = pd.DataFrame(filtered)
                df["sentiment"] = df["sentiment"].astype(float)
                daily = df.groupby("date").agg(
                    news_sent_mean=("sentiment", "mean"),
                    news_sent_std=("sentiment", "std"),
                    news_count=("sentiment", "count"),
                ).reset_index()
                daily["news_sent_std"] = daily["news_sent_std"].fillna(0)
                return daily
    except Exception as e:
        logger.warning(f"新浪新闻抓取失败 {symbol}: {e}")

    return pd.DataFrame()


def get_news_sentiment(symbol: str, market: str, trade_dates: list,
                       lookback_days: int = 30) -> pd.DataFrame:
    """获取新闻情绪因子DataFrame, 与交易日期对齐 (无未来泄漏)"""
    cache_key = f"{market}_{symbol}_{lookback_days}"
    cache_file = CACHE_DIR / f"{cache_key}.parquet"

    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=4):
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    daily = pd.DataFrame()
    if market == "A":
        daily = fetch_sina_news(symbol, lookback_days)
    elif market in ("US", "HK"):
        daily = fetch_finnhub_news(symbol, market, lookback_days)

    if daily.empty:
        result = pd.DataFrame({
            "Date": pd.to_datetime(trade_dates),
            "news_sent_mean": 0.0,
            "news_sent_std": 0.0,
            "news_count": 0,
        })
        return result

    daily["Date"] = pd.to_datetime(daily["date"])
    result = pd.DataFrame({"Date": pd.to_datetime(trade_dates)})
    result = result.merge(
        daily[["Date", "news_sent_mean", "news_sent_std", "news_count"]],
        on="Date", how="left"
    )
    result[["news_sent_mean", "news_sent_std"]] = result[["news_sent_mean", "news_sent_std"]].fillna(0)
    result["news_count"] = result["news_count"].fillna(0).astype(int)

    # 关键: 日期D的新闻 → 影响D+1的预测 (shift 1)
    result["news_sent_mean"] = result["news_sent_mean"].shift(1, fill_value=0)
    result["news_sent_std"] = result["news_sent_std"].shift(1, fill_value=0)
    result["news_count"] = result["news_count"].shift(1, fill_value=0)

    try:
        result.to_parquet(cache_file)
        save_news_to_store(symbol, market, daily)
    except Exception:
        pass

    return result


def save_news_to_store(symbol: str, market: str, news_df: pd.DataFrame):
    """持久化消息面数据到 data/news_data.parquet"""
    if news_df.empty:
        return
    news_df = news_df.copy()
    news_df["symbol"] = symbol
    news_df["market"] = market
    news_df["saved_at"] = datetime.now().isoformat(timespec="minutes")

    if NEWS_STORE.exists():
        try:
            existing = pd.read_parquet(NEWS_STORE)
            key_cols = ["Date", "symbol", "market"]
            mask = ~existing.set_index(key_cols).index.isin(
                news_df.set_index(key_cols).index)
            existing = existing.loc[mask.values]
            news_df = pd.concat([existing, news_df], ignore_index=True)
        except Exception:
            pass

    news_df.to_parquet(NEWS_STORE, index=False)


def load_news_from_store(symbol: str, market: str,
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从持久化存储加载消息面数据"""
    if not NEWS_STORE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(NEWS_STORE)
        mask = (df["symbol"] == symbol) & (df["market"] == market)
        if start_date:
            mask &= df["Date"] >= pd.Timestamp(start_date)
        if end_date:
            mask &= df["Date"] <= pd.Timestamp(end_date)
        return df[mask].drop(columns=["symbol", "market", "saved_at"], errors="ignore")
    except Exception:
        return pd.DataFrame()
