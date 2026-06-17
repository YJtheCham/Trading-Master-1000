"""
iOS 移动端精简版 — 启动: streamlit run webapp/ios_app.py

只保留核心功能: 仪表盘(简洁卡片)、预测(单模型)、交易监控、自选管理
桌面端 webapp/app.py 保持全部功能不变
"""
import sys, time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 共享后端 (不修改 src/ 任何文件)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.factory import run_models, list_models
from src.data.fetcher import (
    fetch_data, diagnose_sources, load_watchlist, save_watchlist,
)
from src.data.tooltips import MODEL_TIPS, CONDITION_TIPS
from src.utils.config import get_tushare_token, save_config, load_config, StockItem
from src.data.sources import MockSource
from src.alerts import (
    AlertRule, add_rule, remove_rule, toggle_rule,
    load_rules, CONDITION_TYPES as CT,
)

# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="StockPredict", layout="wide", page_icon="📈")

# ─── iOS PWA 适配 ────────────────────────────────────────
st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="StockPredict">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  会话状态
# ═══════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "ios_page" not in st.session_state:
    st.session_state.ios_page = "仪表盘"
if "source_status" not in st.session_state:
    st.session_state.source_status = {}
if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

# ═══════════════════════════════════════════════════════════
#  CSS: 移动端优先 + iOS 底部 Tab 栏 + 触摸友好按钮
# ═══════════════════════════════════════════════════════════
base_css = """
    /* ── 全局按钮最小 44px (iOS HIG) ── */
    .stButton > button {
        min-height: 44px !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
        touch-action: manipulation;
    }
    .st-key-tab_bar button {
        min-height: 50px !important;
        border-radius: 12px !important;
        font-size: 0.8rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 2px !important;
        line-height: 1.2 !important;
        white-space: pre-line !important;
        padding: 6px 4px !important;
    }
    .main-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 2px; }
    .main-subtitle { font-size: 0.78rem; margin-bottom: 0.5rem; }
    hr { margin: 0.6rem 0; }

    /* ── 仪表盘卡片 ── */
    .st-key-dash button {
        min-height: 56px !important;
        border-radius: 14px !important;
        padding: 10px 12px !important;
        font-size: 0.82rem !important;
        line-height: 1.4 !important;
        white-space: pre-line !important;
    }
    .st-key-dash button p {
        font-weight: 700; font-size: 1.15rem; margin: 4px 0 0 0;
    }

    /* ── 主内容区 padding ── */
    .stMain, [data-testid="stAppViewContainer"] > section {
        padding: 0.5rem 0.75rem !important;
    }

    /* ── 底部内边距防止被固定 tab 栏遮挡 ── */
    [data-testid="stAppViewContainer"] {
        padding-bottom: 80px !important;
    }

    /* ── 标题缩小 ── */
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1rem !important; }

    /* ── 移动端: 2列卡片自适应 ── */
    @media (max-width: 768px) {
        body, [data-testid="stAppViewContainer"] {
            max-width: 100vw !important; overflow-x: hidden !important;
        }
        section[data-testid="stSidebar"] {
            min-width: 280px !important; max-width: 85vw !important; padding: 8px !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important; gap: 6px !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: calc(50% - 6px) !important;
            max-width: calc(50% - 6px) !important;
            flex: 0 0 calc(50% - 6px) !important;
        }
        .st-key-dash button { min-height: 48px !important; padding: 8px 6px !important; font-size: 0.72rem !important; }
        .st-key-dash button p { font-size: 0.95rem !important; }
        .stMetric { padding: 4px 8px !important; font-size: 0.78rem !important; }
        .js-plotly-plot { max-width: 100% !important; max-height: 280px !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            overflow-x: auto !important;
        }
    }
    @media (max-width: 480px) {
        [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: calc(50% - 4px) !important;
            max-width: calc(50% - 4px) !important;
            flex: 0 0 calc(50% - 4px) !important;
        }
        .js-plotly-plot { max-height: 220px !important; }
        h1 { font-size: 1.15rem !important; }
    }
"""

light_css = base_css + """
    .stApp { background: #f5f5f7; color: #1d1d1f; }
    .stMetric { background: #fff; border-radius: 10px; padding: 8px 12px; border: 1px solid #e5e5ea; }
    .st-key-tab_bar button {
        background: #fff; color: #8e8e93; border: none !important;
        box-shadow: none !important;
    }
    .st-key-tab_bar button:hover { color: #007aff; }
    .st-key-tab_bar button[kind="primary"] {
        color: #007aff !important; font-weight: 700 !important; background: #fff !important;
    }
    hr { border-color: #e5e5ea; }
    .main-subtitle { color: #8e8e93; }
    .st-key-dash button {
        background: #fff; color: #1d1d1f; border: 1px solid #e5e5ea !important;
    }
    .st-key-dash button p { color: #1d1d1f; }
    .ios-tab-spacer { border-top: 1px solid #e5e5ea; }
"""

dark_css = base_css + """
    .stApp, .stApp > header, .main, .stMain,
    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div {
        background: #000 !important; color: #f5f5f7 !important;
    }
    .stMetric { background: #1c1c1e; border: 1px solid #38383a; border-radius: 10px;
                padding: 8px 12px; color: #f5f5f7 !important; }
    .stMetric label, .stMetric div, .stMetric span { color: #f5f5f7 !important; }
    .stSidebar, [data-testid="stSidebar"] { background: #1c1c1e !important; }
    .stSidebar * { color: #f5f5f7 !important; }
    .st-key-tab_bar button {
        background: #000 !important; color: #8e8e93; border: none !important;
        box-shadow: none !important;
    }
    .st-key-tab_bar button:hover { color: #0a84ff; }
    .st-key-tab_bar button[kind="primary"] {
        color: #0a84ff !important; font-weight: 700 !important; background: #000 !important;
    }
    .stButton > button { background: #1c1c1e; color: #f5f5f7; border-color: #38383a; }
    .stButton > button:hover { border-color: #0a84ff; color: #fff; }
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
        background: #1c1c1e !important; color: #f5f5f7 !important; border-color: #38383a !important;
    }
    .stDataFrame, .stDataFrame * { background: #1c1c1e !important; color: #f5f5f7 !important; }
    .stExpander { background: #1c1c1e; border-color: #38383a; }
    .stExpander * { color: #f5f5f7 !important; }
    .stSelectbox label, .stNumberInput label { color: #f5f5f7 !important; }
    hr { border-color: #38383a; }
    .main-subtitle { color: #8e8e93; }
    .st-key-dash button {
        background: #1c1c1e; color: #f5f5f7; border: 1px solid #38383a !important;
    }
    .st-key-dash button p { color: #f5f5f7; }
    .ios-tab-spacer { border-top: 1px solid #38383a; }
"""

st.markdown(
    f"<style>{dark_css if st.session_state.dark_mode else light_css}</style>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════
#  预设股票
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
#  自选管理
# ═══════════════════════════════════════════════════════════
if "watchlist" not in st.session_state:
    saved = load_watchlist()
    if saved:
        st.session_state.watchlist = {f"{i.market}-{i.symbol}" for i in saved}
    else:
        st.session_state.watchlist = {"A-000001", "HK-00700", "US-AAPL"}

if "stock_names" not in st.session_state:
    st.session_state.stock_names = {}
    for k in st.session_state.watchlist:
        info = PRESET_STOCKS.get(k, {"name": k})
        st.session_state.stock_names[k] = info["name"]
    for item in load_watchlist():
        if item.name:
            key = f"{item.market}-{item.symbol}"
            st.session_state.stock_names[key] = item.name

if "stock_order" not in st.session_state:
    st.session_state.stock_order = list(st.session_state.watchlist)


# ─── 辅助函数 ─────────────────────────────────────────────
def info_for(key: str) -> dict:
    """根据 key 返回股票基本信息"""
    if key in PRESET_STOCKS:
        return PRESET_STOCKS[key]
    parts = key.split("-", 1)
    if len(parts) == 2:
        market, symbol = parts
        name = st.session_state.stock_names.get(key, "")
        if not name:
            from src.data.stock_db import resolve_stock_name
            name = resolve_stock_name(symbol, market) or symbol
        return {"symbol": symbol, "market": market, "name": name}
    return {"symbol": key, "market": "A", "name": key}


def _watchlist_key(symbol: str, market: str) -> str:
    return f"{market}-{symbol}"


def has_real_source(market: str) -> bool:
    status = st.session_state.source_status.get(market, [])
    return any(
        s.get("available") and "模拟" not in s.get("name", "")
        for s in status
    )


def refresh_sources():
    with st.spinner("检测数据源..."):
        st.session_state.source_status = diagnose_sources()


def _mock_data(symbol: str, market: str, n: int = 500) -> pd.DataFrame:
    np.random.seed(abs(hash(f"{market}_{symbol}")) % (2**31))
    drift = {"A": 0.04, "HK": 0.03, "US": 0.05}.get(market, 0.03)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + drift)
    return pd.DataFrame({
        "Date": pd.date_range(
            datetime.now() - pd.Timedelta(days=n), periods=n, freq="B"
        ),
        "Close": prices,
        "Open": prices * (1 + np.random.randn(n) * 0.005),
        "High": prices * (1 + abs(np.random.randn(n)) * 0.01),
        "Low": prices * (1 - abs(np.random.randn(n)) * 0.01),
        "Volume": np.random.randint(1e6, 5e8, n),
    })


@st.cache_data(ttl=60)
def get_data_for(symbol: str, market: str, period_days: int = 500) -> pd.DataFrame:
    try:
        df = fetch_data(symbol, market, period_days=period_days)
        return df
    except Exception:
        return _mock_data(symbol, market)


@st.cache_data(ttl=60)
def _detect_source_name(symbol: str, market: str) -> str:
    from src.data.sources import get_sources
    for s in get_sources(market):
        try:
            r = s.run_historical(symbol, period_days=3, market=market)
            if r.success:
                return s.name
        except Exception:
            continue
    return "模拟数据"


def get_data_notify(symbol: str, market: str, name: str = "",
                    period_days: int = 500) -> pd.DataFrame:
    df = get_data_for(symbol, market, period_days)
    source = _detect_source_name(symbol, market)
    if "模拟" in source:
        st.toast(f"⚠ {name} 使用模拟数据", icon="\U0001f504")
    else:
        st.toast(f"✅ {name} 来自 {source}", icon="\U0001f4e1")
    return df


def _add_to_watchlist(symbol: str, market: str, name: str = ""):
    s = symbol.strip()
    if market == "US" and s.isdigit():
        if len(s) <= 5 and s.startswith("0"):
            market = "HK"
        elif len(s) <= 6:
            symbol = s.zfill(6)
            market = "A"
    key = f"{market}-{symbol}"
    if key not in st.session_state.watchlist:
        st.session_state.watchlist.add(key)
        if not name:
            from src.data.stock_db import resolve_stock_name
            name = resolve_stock_name(symbol, market) or symbol
        st.session_state.stock_names[key] = name
        if key not in st.session_state.stock_order:
            st.session_state.stock_order.append(key)
        _persist_watchlist()
        st.toast(f"✅ 已添加 {market}:{symbol} {name}", icon="\U0001f4cb")
        return True
    return False


def _remove_from_watchlist(key: str):
    st.session_state.watchlist.discard(key)
    st.session_state.stock_names.pop(key, None)
    _persist_watchlist()


def _persist_watchlist():
    items = []
    for key in st.session_state.watchlist:
        parts = key.split("-", 1)
        if len(parts) == 2:
            m, s = parts
            n = st.session_state.stock_names.get(key, s)
            items.append(StockItem(symbol=s, market=m, name=n))
    save_watchlist(items)


# ═══════════════════════════════════════════════════════════
#  Sidebar (极简)
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="main-title">📈 StockPredict</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="main-subtitle">iOS 移动版</div>',
        unsafe_allow_html=True,
    )

    dm = st.toggle("\U0001f319 暗色模式", st.session_state.dark_mode, key="dark_toggle")
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        st.rerun()

    st.divider()

    if st.button("\U0001f504 刷新数据源", use_container_width=True):
        refresh_sources()
        st.rerun()

    with st.expander("\U0001f4e1 数据源", expanded=False):
        for m in ["A", "HK", "US"]:
            avail = has_real_source(m)
            icon = "\U0001f7e2" if avail else "\U0001f534"
            status = st.session_state.source_status.get(m, [])
            names = [
                s["name"]
                for s in status
                if s.get("available") and "模拟" not in s["name"]
            ]
            label = ", ".join(names) if names else "无可用"
            st.markdown(f"{icon} **{m}** → {label}")

    st.divider()

    with st.expander("\U0001f511 Tushare Token", expanded=False):
        current = get_tushare_token()
        if current:
            st.caption("✅ 已配置")
        token_input = st.text_input(
            "Token", value=current or "",
            placeholder="输入 Tushare Token",
            type="password", label_visibility="collapsed",
        )
        if token_input and token_input != current:
            cfg = load_config()
            cfg["tushare_token"] = token_input
            save_config(cfg)
            st.toast("Token 已保存", icon="✅")
            st.rerun()

    with st.expander("\U0001f916 LLM API (DeepSeek)", expanded=False):
        from src.utils.config import get_llm_key, get_llm_config
        current_key = get_llm_key()
        if current_key:
            st.caption("✅ DeepSeek Key 已配置")
        api_input = st.text_input(
            "API Key", value=current_key or "",
            placeholder="sk-xxx",
            type="password", label_visibility="collapsed",
            key="llm_key",
        )
        if api_input and api_input != current_key:
            cfg = load_config()
            cfg["llm_api_key"] = api_input
            save_config(cfg)
            st.toast("LLM Key 已保存", icon="\U0001f916")
            st.rerun()

    st.divider()
    st.caption(f"自选股 {len(st.session_state.watchlist)} 只")


# ═══════════════════════════════════════════════════════════
#  页面路由
# ═══════════════════════════════════════════════════════════
PAGE = st.session_state.ios_page

# ═══════════════════════════════════════════════════════════
#  🏠 仪表盘
# ═══════════════════════════════════════════════════════════
if PAGE == "仪表盘":
    st.title("\U0001f3e0 仪表盘")

    if not st.session_state.watchlist:
        st.warning("请在「自选管理」中添加股票")
    else:
        ordered = [
            k for k in st.session_state.stock_order
            if k in st.session_state.watchlist
        ]
        for k in st.session_state.watchlist:
            if k not in ordered:
                ordered.append(k)

        COLS_PER_ROW = 2  # 移动端友好: 每行 2 个
        for i in range(0, len(ordered), COLS_PER_ROW):
            row_stocks = ordered[i:i + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for col, key in zip(cols, row_stocks):
                info = info_for(key)
                try:
                    df = fetch_data(
                        info["symbol"], info["market"],
                        use_cache=True, max_age_minutes=5,
                    )
                except Exception:
                    df = _mock_data(info["symbol"], info["market"])
                if df is None or df.empty:
                    continue
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

                with col:
                    st.metric(
                        f"{info['name']} ({info['market']})",
                        f"{latest['Close']:.2f}",
                        f"{change:+.2f}%",
                        delta_color="normal" if change >= 0 else "inverse",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(
                            "📡 预测", key=f"dash_pred_{key}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_stock = key
                            st.session_state.ios_page = "预测"
                            st.rerun()
                    with c2:
                        if st.button(
                            "🔔 监控", key=f"dash_alert_{key}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_stock = key
                            st.session_state.ios_page = "交易监控"
                            st.rerun()

# ═══════════════════════════════════════════════════════════
#  📡 预测 (单模型)
# ═══════════════════════════════════════════════════════════
elif PAGE == "预测":
    st.title("📡 股价预测")
    st.caption("单模型快速预测")

    # 股票选择
    keys = sorted(st.session_state.watchlist)
    if not keys:
        st.warning("请先在「自选管理」中添加股票")
    else:
        default_idx = 0
        if st.session_state.selected_stock and st.session_state.selected_stock in keys:
            default_idx = keys.index(st.session_state.selected_stock)

        target = st.selectbox(
            "选择股票", keys,
            index=default_idx,
            format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
            key="pred_target",
        )

        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox(
                "模型", list_models(),
                index=0,
                help="选择一个预测模型",
                key="pred_model_single",
            )
        with col2:
            steps = st.slider("预测天数", 5, 60, 20, 5, key="pred_steps")

        if st.button("▶ 开始预测", type="primary", use_container_width=True):
            info = info_for(target)
            df = get_data_notify(info["symbol"], info["market"], info["name"])

            with st.spinner(f"正在用 {model_name} 预测 {info['name']}..."):
                results = run_models(df, model_names=[model_name], steps=steps)

            if model_name in results and len(results[model_name].forecast) > 0:
                r = results[model_name]
                st.divider()

                # 预测方向
                direction = "📈 看涨" if r.forecast[-1] > r.history[-1] else "📉 看跌"
                pct = (r.forecast[-1] - r.history[-1]) / r.history[-1] * 100
                cols = st.columns(3)
                cols[0].metric("当前价", f"{r.history[-1]:.2f}")
                cols[1].metric("预测末价", f"{r.forecast[-1]:.2f}")
                cols[2].metric("涨跌幅", f"{pct:+.1f}%")

                # 图表
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["Close"],
                    name="历史", line=dict(color="#007aff", width=1.5),
                ))
                fig.add_trace(go.Scatter(
                    x=r.forecast_dates, y=r.forecast,
                    name=f"{model_name} 预测",
                    line=dict(color="#ff9500", dash="dash", width=2.5),
                ))
                fig.update_layout(
                    height=300, hovermode="x unified",
                    title=f"{info['name']} — {model_name} 预测",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # 预测值表格
                st.caption("📋 预测值明细")
                fcast_df = pd.DataFrame({
                    "天数": [f"第{i+1}天" for i in range(len(r.forecast))],
                    "预测价": np.round(r.forecast, 2),
                    "日期": r.forecast_dates if len(r.forecast_dates) == len(r.forecast)
                            else [""] * len(r.forecast),
                })
                st.dataframe(fcast_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"模型 {model_name} 预测失败，请重试")

        # 模型说明
        with st.expander("📖 模型说明", expanded=False):
            st.markdown("""
| 模型 | 原理 | 适用场景 |
|------|------|---------|
| **ARIMA** | 自回归积分滑动平均 | 平稳序列, 短期预测 |
| **GBDT** | 梯度提升决策树 | 中短期趋势 |
| **XGBoost** | GBDT 工程优化版 | 精度更高 |
| **LSTM** | 长短期记忆网络 (PyTorch) | 长序列, 非线性模式 |
| **Transformer** | 自注意力机制 (PyTorch) | 全局趋势 |
""")

# ═══════════════════════════════════════════════════════════
#  🔔 交易监控
# ═══════════════════════════════════════════════════════════
elif PAGE == "交易监控":
    st.title("🔔 交易监控")
    st.caption("为自选股设置监控规则")

    from src.alerts.conditions import preview_notification
    from src.data.stock_db import search_stocks, resolve_stock_name

    if "alert_target" not in st.session_state:
        st.session_state.alert_target = None

    rules = load_rules()

    # ── 已设规则 ──
    st.subheader(f"📋 已设规则 ({len(rules)} 条)")
    if not rules:
        st.info("暂无规则, 在下方添加")
    else:
        for r in rules:
            cols = st.columns([0.3, 2, 3, 0.8, 0.8])
            name = st.session_state.stock_names.get(
                f"{r.market}-{r.symbol}", ""
            )
            if not name:
                name = resolve_stock_name(r.symbol, r.market) or ""
            with cols[0]:
                st.write("\U0001f7e2" if r.enabled else "\U0001f534")
            with cols[1]:
                st.write(f"**{r.symbol}** {name}" if name else f"**{r.symbol}**")
            with cols[2]:
                desc = CT.get(r.condition, r.condition)
                extra = ", ".join(f"{k}={v}" for k, v in r.params.items())
                st.write(f"{desc} | {extra}")
            with cols[3]:
                if st.button(
                    "⏸" if r.enabled else "▶",
                    key=f"tg_{r.uid}", use_container_width=True,
                ):
                    toggle_rule(r.uid)
                    st.rerun()
            with cols[4]:
                if st.button("✕", key=f"rm_{r.uid}", use_container_width=True):
                    remove_rule(r.uid)
                    st.rerun()

    st.divider()

    # ── 添加规则 ──
    with st.expander("➕ 添加规则", expanded=st.session_state.alert_target is not None):
        col1, col2 = st.columns([2, 1])
        with col1:
            search_q = st.text_input(
                "搜索股票 (代码/名称)", placeholder="例: 600519 / 茅台 / AAPL",
                key="alert_search",
            )
            if search_q:
                results = search_stocks(search_q, limit=5)
                if results:
                    selected_s = st.selectbox(
                        "选择", results,
                        format_func=lambda x: f"[{x[2]}] {x[0]} {x[1]}",
                        key="as_select",
                    )
                else:
                    selected_s = (search_q, "", "A")
                    st.caption(f"手动作为 [{selected_s[2]}] 添加")
            else:
                selected_s = None

        with col2:
            condition = st.selectbox(
                "条件", list(CT.keys()),
                format_func=lambda x: f"{CT[x]}", key="ac_cond",
            )

        params = {}
        if condition in ("above_ma", "below_ma"):
            params["window"] = st.number_input("MA窗口", 5, 120, 20, key="pw2")
        elif condition in ("above_price", "below_price"):
            params["threshold"] = st.number_input(
                "价格阈值", 0.0, 10000.0, 100.0, key="pt2",
            )
        elif condition in ("rsi_oversold", "rsi_overbought"):
            params["window"] = int(st.number_input("RSI窗口", 5, 30, 14, key="rw2"))
            params["level"] = int(st.number_input(
                "阈值", 10, 90,
                30 if "oversold" in condition else 70, key="rl2",
            ))
        elif condition == "volume_spike":
            params["ratio"] = st.number_input("倍数", 1.0, 10.0, 2.0, key="vr2")
        elif condition == "daily_change":
            params["direction"] = st.selectbox("方向", ["up", "down"], key="dd2")
            params["pct"] = st.number_input("百分比%", 1.0, 20.0, 5.0, key="dp2")
        elif condition in ("golden_cross", "death_cross", "ma_cross_combo"):
            params["short"] = int(st.number_input("短期MA", 5, 50, 20, key="gs2"))
            params["long"] = int(st.number_input("长期MA", 10, 200, 60, key="gl2"))
        elif condition in ("bollinger_upper", "bollinger_lower", "bollinger_combo"):
            params["window"] = int(st.number_input("窗口", 10, 50, 20, key="bw2"))
            params["std"] = int(st.number_input("标准差", 1, 4, 2, key="bs2"))
        elif condition == "rsi_combo":
            params["window"] = int(st.number_input("RSI窗口", 5, 30, 14, key="rc_w"))
            params["oversold"] = int(st.number_input("超卖阈值", 10, 40, 30, key="rc_o"))
            params["overbought"] = int(st.number_input("超买阈值", 60, 90, 70, key="rc_ob"))
        elif condition == "volume_breakout":
            params["lookback"] = int(st.number_input("回顾天数", 10, 60, 20, key="vb_l"))
            params["vol_ratio"] = st.number_input("成交量倍数", 1.5, 5.0, 2.0, key="vb_v")
        elif condition == "ma_triple":
            params["short"] = int(st.number_input("短期MA", 5, 30, 10, key="mt_s"))
            params["mid"] = int(st.number_input("中期MA", 15, 60, 30, key="mt_m"))
            params["long"] = int(st.number_input("长期MA", 30, 200, 60, key="mt_l"))
        elif condition == "ma_rsi_combo":
            params["ma_window"] = int(st.number_input("趋势MA", 20, 120, 60, key="mr_m"))
            params["rsi_window"] = int(st.number_input("RSI窗口", 5, 30, 14, key="mr_r"))
            params["oversold"] = int(st.number_input("RSI超卖", 10, 40, 30, key="mr_o"))
            params["overbought"] = int(st.number_input("RSI超买", 60, 90, 70, key="mr_ob"))

        # 通知预览
        if selected_s and condition:
            msg, action = preview_notification(condition, params, price=100.0)
            st.info(f"📢 **{msg}**\n\nℹ 操作建议: {action}")

        label = st.text_input("备注 (可选)", key="al_label")
        cooldown = st.number_input("冷却(分钟)", 10, 480, 60, key="al_cd")

        c1, c2 = st.columns(2)
        with c1:
            if selected_s and st.button(
                "✅ 添加", type="primary", use_container_width=True,
            ):
                code, n, m = selected_s
                add_rule(AlertRule(
                    symbol=code, market=m, condition=condition,
                    params={k: int(v) if isinstance(v, float) and v == int(v) else v
                            for k, v in params.items()},
                    label=label or n or code,
                    cooldown_minutes=int(cooldown),
                ))
                st.session_state.alert_target = None
                st.success("已添加")
                st.rerun()
        with c2:
            if st.button("取消", use_container_width=True):
                st.session_state.alert_target = None
                st.rerun()

# ═══════════════════════════════════════════════════════════
#  📋 自选管理
# ═══════════════════════════════════════════════════════════
elif PAGE == "自选管理":
    st.title("📋 自选管理")

    # ── 搜索添加 ──
    st.subheader("➕ 添加股票")
    from src.data.stock_db import search_stocks, resolve_stock_name

    search_q = st.text_input(
        "🔍 搜索 (代码/名称)",
        placeholder="例: 600519 / 茅台 / AAPL",
        key="wl_search",
    )

    if search_q:
        results = search_stocks(search_q, limit=10)
        if results:
            for code, name, market in results:
                key = f"{market}-{code}"
                already = key in st.session_state.watchlist
                btn_label = "✅ 已添加" if already else "➕ 添加"
                if st.button(
                    f"{btn_label} [{market}] {code} {name}",
                    key=f"wl_add_{key}",
                    use_container_width=True,
                    disabled=already,
                ):
                    _add_to_watchlist(code, market, name)
                    st.rerun()
        else:
            st.caption(f"未匹配: {search_q}")
            c1, c2, c3 = st.columns(3)
            if c1.button("手动 A 股", use_container_width=True):
                _add_to_watchlist(search_q, "A")
                st.rerun()
            if c2.button("手动 港股", use_container_width=True):
                _add_to_watchlist(search_q, "HK")
                st.rerun()
            if c3.button("手动 美股", use_container_width=True):
                _add_to_watchlist(search_q, "US")
                st.rerun()

    # ── 文件导入 ──
    with st.expander("📁 文件导入 (CSV/TXT/XLSX)", expanded=False):
        uploaded = st.file_uploader(
            "选择文件", type=["csv", "txt", "xlsx"],
            label_visibility="collapsed",
        )
        if uploaded:
            from src.data.stock_db import parse_stock_file
            import tempfile
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix,
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                codes = parse_stock_file(tmp_path)
                added = 0
                for code, name, market in codes:
                    if _add_to_watchlist(code, market, name):
                        added += 1
                if added:
                    st.success(f"导入 {added} 只股票")
                    st.rerun()
            except Exception as e:
                st.error(f"解析失败: {e}")
            Path(tmp_path).unlink(missing_ok=True)

    st.divider()

    # ── 当前自选列表 ──
    st.subheader(f"📋 自选列表 ({len(st.session_state.watchlist)})")
    if not st.session_state.watchlist:
        st.info("暂无自选股, 使用上方搜索添加")
    else:
        keys_to_show = [
            k for k in st.session_state.stock_order
            if k in st.session_state.watchlist
        ]
        for k in st.session_state.watchlist:
            if k not in keys_to_show:
                keys_to_show.append(k)

        for idx, key in enumerate(keys_to_show):
            parts = key.split("-", 1)
            if len(parts) != 2:
                continue
            market, symbol = parts
            name = st.session_state.stock_names.get(key, symbol)
            display = f"{symbol} {name}" if name and name != symbol else symbol

            c1, c2, c3, c4 = st.columns([3, 1, 0.8, 0.8])
            with c1:
                st.write(f"[{market}] **{display}**")
            with c2:
                if st.button(
                    "📡 预测", key=f"wl_pred_{key}", use_container_width=True,
                ):
                    st.session_state.selected_stock = key
                    st.session_state.ios_page = "预测"
                    st.rerun()
            with c3:
                if idx > 0:
                    if st.button("⬆", key=f"up_{key}", help="上移"):
                        st.session_state.stock_order.remove(key)
                        st.session_state.stock_order.insert(idx - 1, key)
                        st.rerun()
            with c4:
                if st.button("✕", key=f"del_{key}", help="删除"):
                    _remove_from_watchlist(key)
                    st.rerun()


# ═══════════════════════════════════════════════════════════
#  iOS 底部 Tab 栏 (touch-friendly, ≥44px)
# ═══════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="ios-tab-spacer"></div>', unsafe_allow_html=True)

TABS = [
    ("🏠", "仪表盘"),
    ("📡", "预测"),
    ("🔔", "交易监控"),
    ("📋", "自选管理"),
]

cols = st.columns(len(TABS), gap="small")
for col, (icon, label) in zip(cols, TABS):
    with col:
        is_active = (PAGE == label)
        if st.button(
            f"{icon}\n{label}",
            key=f"tab_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.ios_page = label
            st.rerun()
