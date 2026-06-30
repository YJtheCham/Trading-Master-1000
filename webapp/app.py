"""
启动: streamlit run webapp/app.py
"""
import sys, time, json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.factory import run_models, list_models
from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.strategies import (
    MovingAverageCrossStrategy, RollingPredictionStrategy,
    RSIStrategy, ChannelBreakoutStrategy, BollingerStrategy,
)
from src.models.gbdt import GBDTModel
from src.risk.metrics import calc_all_risk_metrics
from src.data.tooltips import MODEL_TIPS, RISK_TIPS, STRATEGY_TIPS, CONDITION_TIPS, PAGE_TIPS
from src.data.fetcher import (
    fetch_data, get_realtime_price, diagnose_sources,
)
from src.utils.config import get_tushare_token, save_config, load_config, DATA_DIR
from src.data.sources import MockSource

st.set_page_config(page_title="StockPredict", layout="wide", page_icon="📈")

# PWA / iPhone Safari 适配
st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="StockPredict">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">
""", unsafe_allow_html=True)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

base_css = """
    .stButton>button { border-radius:6px; transition:border-color 0.15s,background 0.15s; }
    .main-title { font-size:1.5rem; font-weight:600; margin-bottom:0; }
    .main-subtitle { font-size:0.85rem; margin-top:-0.3rem; }
    hr { margin:0.8rem 0; }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSegmentedControl"]) { width:100%; }
    div[data-testid="stSegmentedControl"] { width:100% !important; }
    div[data-testid="stSegmentedControl"] > div { width:100% !important; display:flex; }
    div[data-testid="stSegmentedControl"] button { flex:1; font-size:0.85rem; white-space:nowrap; }
    section[data-testid="stSidebar"] .stButton button { justify-content:flex-start; text-align:left; font-size:0.85rem; }
    /* ── 统一表格圆角/阴影 ── */
    [data-testid="stDataFrame"] { border-radius:8px; overflow:hidden; }
    [data-testid="stDataFrame"] > div { border-radius:8px; }
    [data-testid="stTable"] { border-radius:8px; overflow:hidden; }
    /* ── 统一 Expander 视觉 ── */
    .stExpander { border-radius:10px; margin:6px 0; }
    .stExpander > div[data-testid="stExpander"] { border-radius:10px; }
    /* ── 统一按钮风格 ── */
    .stButton>button[kind="primary"] { font-weight:600; }
"""

light_css = base_css + """
    .stApp { background:#fff; color:#212529; }
    .stMetric { background:#f8f9fa; border-radius:8px; padding:8px 12px; border:1px solid #e9ecef; box-shadow:0 1px 2px rgba(0,0,0,0.03); }
    div[data-testid="stSegmentedControl"] button[aria-selected="true"] { font-weight:600; background:#e8f0fe; border-color:#1f77b4; }
    hr { border-color:#e9ecef; }
    .main-subtitle { color:#6c757d; }
    /* ── 仪表盘: 卡片(metric)+按钮 整合成一体 ── */
    .st-key-dash button { border-radius:12px; padding:14px 10px; font-size:0.82rem; line-height:1.4; white-space:pre-line; min-height:95px; border:1px solid #dee2e6 !important; }
    .st-key-dash button p { font-weight:700; font-size:1.25rem; color:#212529; margin:4px 0 0 0; }
    /* 卡片自动换行 */
    [data-testid="stVerticalBlockBoundary"] > [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:8px; }
    [data-testid="stVerticalBlockBoundary"] > [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:200px; flex:1 1 200px; }
    /* ── 表格 ── */
    [data-testid="stDataFrame"] { box-shadow:0 1px 3px rgba(0,0,0,0.06); border:1px solid #e9ecef; }
    /* ── Expander ── */
    .stExpander { border:1px solid #e9ecef; }
    .stExpander:hover { border-color:#ced4da; }
    /* ── Selectbox 下拉 ── */
    [data-baseweb="popover"] [role="listbox"] { border-radius:8px; }
    /* ── Metric 统一阴影 ── */
    .stMetric:hover { box-shadow:0 2px 6px rgba(0,0,0,0.08); }
"""

dark_css = base_css + """
    .stApp, .stApp > header, .main, .stMain, [data-testid=\"stAppViewContainer\"], [data-testid=\"stAppViewContainer\"] > div { background:#0d1117 !important; color:#c9d1d9 !important; }
    .stMetric { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:8px 12px; color:#c9d1d9 !important; box-shadow:0 1px 2px rgba(0,0,0,0.3); }
    .stMetric label, .stMetric div, .stMetric span { color:#c9d1d9 !important; }
    .stMetric:hover { box-shadow:0 2px 6px rgba(0,0,0,0.5); }
    .stSidebar, [data-testid=\"stSidebar\"] { background:#161b22 !important; }
    .stSidebar * { color:#c9d1d9 !important; }
    div[data-testid=\"stSegmentedControl\"] button { background:#21262d; color:#c9d1d9; border-color:#30363d; }
    div[data-testid=\"stSegmentedControl\"] button[aria-selected=\"true\"] { background:#1f6feb; border-color:#1f6feb; color:#fff; }
    hr { border-color:#30363d; }
    .main-subtitle { color:#8b949e; }
    .stButton>button { background:#21262d; color:#c9d1d9; border-color:#30363d; }
    .stButton>button:hover { border-color:#1f6feb; color:#fff; }
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb=\"select\"] > div { background:#0d1117 !important; color:#c9d1d9 !important; border-color:#30363d !important; }
    /* ── 表格 ── */
    [data-testid="stDataFrame"] { box-shadow:0 1px 3px rgba(0,0,0,0.4); border:1px solid #30363d; }
    [data-testid="stDataFrame"] * { background:#161b22 !important; color:#c9d1d9 !important; }
    /* ── Expander ── */
    .stExpander { background:#161b22; border:1px solid #30363d; }
    .stExpander:hover { border-color:#484f58; }
    .stExpander * { color:#c9d1d9 !important; }
    /* ── Tabs ── */
    .stTabs [data-baseweb=\"tab-panel\"] { background:#0d1117; }
    .stTabs [data-baseweb=\"tab\"] { background:#161b22; color:#c9d1d9; }
    .stTabs [aria-selected=\"true\"] { background:#1f6feb !important; color:#fff !important; }
    /* ── 仪表盘卡片 ── */
    .st-key-dash button { border-radius:12px; padding:14px 10px; font-size:0.82rem; line-height:1.4; white-space:pre-line; min-height:95px; background:#161b22; border:1px solid #30363d !important; color:#c9d1d9; }
    .st-key-dash button p { font-weight:700; font-size:1.25rem; color:#c9d1d9; margin:4px 0 0 0; }
    /* ── Selectbox 下拉 ── */
    [data-baseweb="popover"] { background:#21262d !important; border:1px solid #30363d !important; border-radius:8px; }
    [data-baseweb="popover"] * { color:#c9d1d9 !important; }
    [data-baseweb="popover"] [role="option"]:hover { background:#30363d !important; }
    [data-baseweb="popover"] [role="option"][aria-selected="true"] { background:#1f6feb33 !important; }
    /* ── Alert / Notification ── */
    .stAlert, [data-testid=\"stInfoBox\"] { background:#161b22 !important; color:#c9d1d9 !important; border-color:#30363d !important; }
    [data-testid=\"stNotification\"] { background:#161b22 !important; }
"""

mobile_css = """
    @media (max-width: 768px) {
        /* === 防止页面横向溢出 === */
        body, [data-testid="stAppViewContainer"] { max-width:100vw !important; overflow-x:hidden !important; }

        /* === 侧边栏: 恢复原生 overlay 行为, 不占满全屏 === */
        section[data-testid="stSidebar"] { min-width:280px !important; max-width:85vw !important; padding:8px !important; }
        section[data-testid="stSidebar"] .stButton button { width:100%; }

        /* === 9个导航标签: 横向可滑动, 字号可读 === */
        div[data-testid="stSegmentedControl"] {
            width:100% !important; overflow-x:auto !important;
            -webkit-overflow-scrolling:touch; scrollbar-width:none;
        }
        div[data-testid="stSegmentedControl"]::-webkit-scrollbar { display:none; }
        div[data-testid="stSegmentedControl"] > div {
            width:max-content !important; min-width:100% !important;
            display:flex !important; flex-wrap:nowrap !important;
        }
        div[data-testid="stSegmentedControl"] button {
            font-size:0.72rem !important; padding:0.35rem 0.5rem !important;
            white-space:nowrap !important; flex-shrink:0 !important; border-radius:6px !important;
        }

        /* === 仪表盘卡片: 2列自动换行 (仅主内容区, 不影响侧边栏) === */
        [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] {
            flex-wrap:wrap !important; gap:6px !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width:calc(50% - 6px) !important; max-width:calc(50% - 6px) !important;
            flex:0 0 calc(50% - 6px) !important;
        }
        .st-key-dash button { min-height:80px !important; padding:10px 6px !important; font-size:0.72rem !important; }
        .st-key-dash button p { font-size:1rem !important; }
        .stMetric { padding:4px 8px !important; font-size:0.78rem !important; }

        /* === 主区域padding === */
        .stMain, [data-testid="stAppViewContainer"] > section { padding:0.5rem !important; }

        /* === 图表不溢出 === */
        .js-plotly-plot, .plotly { max-width:100% !important; overflow-x:auto !important; }
        .js-plotly-plot .plot-container { max-width:100% !important; }
        .js-plotly-plot { max-height:280px !important; }

        /* === 表格横向滚动 === */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { overflow-x:auto !important; }

        /* === 标题/文字缩小 === */
        h1 { font-size:1.3rem !important; }
        h2 { font-size:1.1rem !important; }
        h3 { font-size:1rem !important; }
        .streamlit-expanderHeader { font-size:0.85rem !important; }
        .stTabs [data-baseweb="tab"] { font-size:0.8rem !important; padding:0.4rem 0.6rem !important; }
    }
    @media (max-width: 480px) {
        [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width:calc(50% - 4px) !important; max-width:calc(50% - 4px) !important;
            flex:0 0 calc(50% - 4px) !important;
        }
        div[data-testid="stSegmentedControl"] button { font-size:0.68rem !important; padding:0.3rem 0.4rem !important; }
        .js-plotly-plot { max-height:220px !important; }
        .stMetric { padding:3px 6px !important; font-size:0.72rem !important; }
        h1 { font-size:1.15rem !important; }
    }
"""

st.markdown(f"<style>{dark_css if st.session_state.dark_mode else light_css}{mobile_css}</style>", unsafe_allow_html=True)

# ─── Plotly 暗色模板 ──────────────────────────────────────
def plotly_template():
    """返回当前主题对应的 Plotly 模板名"""
    return "plotly_dark" if st.session_state.dark_mode else "plotly_white"

def plotly_theme_colors():
    """返回当前主题下的线条/文字颜色"""
    if st.session_state.dark_mode:
        return {"line": "#c9d1d9", "grid": "#30363d", "paper": "#0d1117", "plot": "#0d1117", "font": "#c9d1d9"}
    return {"line": "#212529", "grid": "#e9ecef", "paper": "#fff", "plot": "#fff", "font": "#212529"}

# ─── 会话状态 ─────────────────────────────────────────────
if "source_status" not in st.session_state:
    st.session_state.source_status = {}
if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0

def has_real_source(market: str) -> bool:
    status = st.session_state.source_status.get(market, [])
    return any(s.get("available") and "模拟" not in s.get("name", "") for s in status)


def refresh_sources():
    """刷新数据源状态"""
    with st.spinner("检测数据源..."):
        st.session_state.source_status = diagnose_sources()


# ─── 数据获取: 真实 → Mock回退 ──────────────────────────
@st.cache_data(ttl=60, max_entries=32)
def get_data_for(symbol: str, market: str, period_days: int = 500,
                 _refresh_key: int = 0) -> pd.DataFrame:
    try:
        df = fetch_data(symbol, market, period_days=period_days)
        return df
    except Exception:
        return _mock_data(symbol, market)


@st.cache_data(ttl=300, max_entries=64)
def _detect_source_name(symbol: str, market: str) -> str:
    from src.data.sources import get_sources
    for s in get_sources(market):
        try:
            r = s.run_historical(symbol, period_days=5, market=market)
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
        st.toast(f"⚠️ {name} 使用模拟数据", icon="🔄")
    else:
        st.toast(f"✅ {name} 来自 {source}", icon="📡")
    return df

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

# ─── 预设股票 (快速参考) ─────────────────────────────────
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

if "watchlist" not in st.session_state:
    # 从持久化文件加载自选列表
    from src.data.fetcher import load_watchlist
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
    from src.data.fetcher import load_watchlist
    for item in load_watchlist():
        if item.name:
            key = f"{item.market}-{item.symbol}"
            st.session_state.stock_names[key] = item.name

if "stock_groups" not in st.session_state:
    st.session_state.stock_groups = {}
    from src.data.fetcher import load_watchlist
    for item in load_watchlist():
        key = f"{item.market}-{item.symbol}"
        st.session_state.stock_groups[key] = item.group or "默认"

if "stock_order" not in st.session_state:
    # 从持久化文件加载排序
    order_file = DATA_DIR / "watchlist_order.json"
    try:
        if order_file.exists():
            saved = json.loads(order_file.read_text())
            # 只保留仍在 watchlist 中的
            st.session_state.stock_order = [k for k in saved if k in st.session_state.watchlist]
        else:
            st.session_state.stock_order = list(st.session_state.watchlist)
    except Exception:
        st.session_state.stock_order = list(st.session_state.watchlist)

def _save_stock_order():
    try:
        order_file = DATA_DIR / "watchlist_order.json"
        order_file.write_text(json.dumps(st.session_state.stock_order, ensure_ascii=False))
    except Exception:
        pass
if "active_group" not in st.session_state:
    st.session_state.active_group = "全部"

if "group_order" not in st.session_state:
    order_file = DATA_DIR / "group_order.json"
    try:
        if order_file.exists():
            st.session_state.group_order = json.loads(order_file.read_text())
        else:
            st.session_state.group_order = sorted(set(st.session_state.stock_groups.values()))
    except Exception:
        st.session_state.group_order = sorted(set(st.session_state.stock_groups.values()))

def _save_group_order():
    try:
        (DATA_DIR / "group_order.json").write_text(
            json.dumps(st.session_state.group_order, ensure_ascii=False))
    except Exception:
        pass


# ─── 自选管理工具 ─────────────────────────────────────────
def _watchlist_key(info: dict) -> str:
    return f"{info['market']}-{info['symbol']}"

def _add_to_watchlist(symbol: str, market: str, name: str = "", group: str = "默认"):
    # 自动修正市场 + 补前导零
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
        st.session_state.stock_groups[key] = group
        if key not in st.session_state.stock_order:
            st.session_state.stock_order.append(key)
        _persist_watchlist()
        st.toast(f"✅ 已添加 {market}:{symbol} {name}", icon="📋")
        return True
    return False

def _remove_from_watchlist(key: str):
    st.session_state.watchlist.discard(key)
    st.session_state.stock_names.pop(key, None)
    st.session_state.stock_groups.pop(key, None)
    st.session_state.stock_order = [k for k in st.session_state.stock_order if k != key]
    _persist_watchlist()
    _save_stock_order()

def _persist_watchlist():
    from src.data.fetcher import save_watchlist
    from src.utils.config import StockItem
    items = []
    for key in st.session_state.watchlist:
        parts = key.split("-", 1)
        if len(parts) == 2:
            m, s = parts
            n = st.session_state.stock_names.get(key, s)
            g = st.session_state.stock_groups.get(key, "默认")
            items.append(StockItem(symbol=s, market=m, name=n, group=g))
    save_watchlist(items)

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">📈 StockPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">自选 · 预测 · 回测 · 风控</div>', unsafe_allow_html=True)

    dm = st.toggle("🌙 暗色模式", st.session_state.dark_mode, key="dark_toggle")
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        st.rerun()

    st.divider()

    if st.button("🔄 刷新数据源", use_container_width=True):
        refresh_sources()
        st.session_state.dash_refresh_queue = list(st.session_state.watchlist)
        st.rerun()
    with st.expander("📡 数据源", expanded=False):
        tushare_ok = bool(get_tushare_token())
        for m in ["A", "HK", "US"]:
            avail = has_real_source(m)
            icon = "🟢" if avail else "🔴"
            status = st.session_state.source_status.get(m, [])
            names = [s["name"] for s in status if s.get("available") and "模拟" not in s["name"]]
            label = ", ".join(names) if names else "无可用"
            st.markdown(f"{icon} **{m}** → {label}")
        if tushare_ok:
            st.caption("🔑 Tushare 已配置")

    st.divider()
    st.subheader("➕ 添加自选")

    # 搜索+添加
    from src.data.stock_db import search_stocks, resolve_stock_name
    search_q = st.text_input("🔍 搜索股票 (代码/名称)", placeholder="例: 600519 / 茅台 / AAPL")
    if search_q:
        results = search_stocks(search_q, limit=10)
        if results:
            for code, name, market in results:
                label = f"[{market}] {code} {name}" if name else f"[{market}] {code}"
                key = f"{market}-{code}"
                already = key in st.session_state.watchlist
                btn_label = "✅" if already else "➕"
                if st.button(f"{btn_label} {label}", key=f"add_{key}",
                             use_container_width=True,
                             disabled=already):
                    _add_to_watchlist(code, market, name, "默认")
                    st.rerun()
        else:
            # 逐一尝试三个市场, 第一个找到就停
            found = False
            for m in ["A", "HK", "US"]:
                name = resolve_stock_name(search_q, m)
                if name:
                    if st.button(f"➕ [{m}] {search_q} {name}", key=f"add_res_{search_q}",
                                 use_container_width=True, type="primary"):
                        _add_to_watchlist(search_q, m, name)
                        st.rerun()
                    found = True
                    break
            if not found:
                st.caption(f"未匹配: {search_q}")
                c1, c2, c3 = st.columns(3)
                if c1.button(f"手动添加 (A)", use_container_width=True):
                    _add_to_watchlist(search_q, "A")
                    st.rerun()
                if c2.button(f"手动添加 (HK)", use_container_width=True):
                    _add_to_watchlist(search_q, "HK")
                    st.rerun()
                if c3.button(f"手动添加 (US)", use_container_width=True):
                    _add_to_watchlist(search_q, "US")
                    st.rerun()

    # 文件导入
    with st.expander("📁 文件导入 (CSV/TXT/XLSX)", expanded=False):
        uploaded = st.file_uploader("选择文件", type=["csv", "txt", "xlsx"],
                                    label_visibility="collapsed")
        if uploaded:
            from src.data.stock_db import parse_stock_file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
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
    # 分组管理
    current_groups = [g for g in st.session_state.group_order
                      if g in set(st.session_state.stock_groups.values())]
    for g in set(st.session_state.stock_groups.values()):
        if g not in current_groups:
            current_groups.append(g)
    st.session_state.group_order = current_groups
    _save_group_order()

    display_groups = current_groups + ["全部"]
    active = st.selectbox("📂 分组", display_groups,
                           index=display_groups.index(st.session_state.active_group) if st.session_state.active_group in display_groups else 0,
                           key="grp_select")
    if active != st.session_state.active_group:
        st.session_state.active_group = active
        st.rerun()

    # 分组拖拽排序组件
    from src.components.group_drag import group_drag
    grp_drag_data = []
    for gname in current_groups:
        cnt = sum(1 for v in st.session_state.stock_groups.values() if v == gname)
        grp_drag_data.append({"name": gname, "count": cnt})
    new_order = group_drag(grp_drag_data, dark=st.session_state.get("dark_mode", False))
    if new_order and len(new_order) == len(current_groups):
        st.session_state.group_order = [str(g) for g in new_order]
        _save_group_order()

    with st.expander("⚙️ 分组管理", expanded=False):
        for idx, gname in enumerate(current_groups):
            cnt = sum(1 for v in st.session_state.stock_groups.values() if v == gname)
            gc1, gc2, gc3 = st.columns([4, 1, 1])
            with gc1:
                new_name = st.text_input(f"grpname_{idx}", value=gname, key=f"grp_rename_{idx}",
                                         label_visibility="collapsed")
            with gc2:
                st.caption(f"{cnt}只")
            with gc3:
                if new_name and new_name != gname and new_name.strip():
                    if st.button("✓", key=f"grp_apply_{idx}", help="确认改名"):
                        old_name = gname
                        st.session_state.group_order[idx] = new_name.strip()
                        for k, v in list(st.session_state.stock_groups.items()):
                            if v == old_name:
                                st.session_state.stock_groups[k] = new_name.strip()
                        _persist_watchlist()
                        _save_group_order()
                        st.rerun()

    st.subheader(f"📋 自选列表 ({len(st.session_state.watchlist)})")

    if st.session_state.watchlist:
        from src.alerts.engine import load_rules
        alert_rules_sidebar = load_rules()
        monitored_sidebar = set(
            f"{r.market}-{r.symbol}" for r in alert_rules_sidebar if r.enabled
        )
        keys_to_show = [k for k in st.session_state.stock_order if k in st.session_state.watchlist]
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
            group_tag = st.session_state.stock_groups.get(key, "默认")
            is_mon = key in monitored_sidebar
            label_prefix = "🟢" if is_mon else ""
            c1, c2, c3 = st.columns([3, 1, 0.8])
            with c1:
                if st.button(f"{label_prefix}[{market}] {display}", key=f"sidebar_sel_{key}", use_container_width=True):
                    st.session_state.selected_stock = key
                    st.session_state.page = "ℹ️ 自选详情"
                    st.rerun()
            with c2:
                group_opts = st.session_state.group_order + ["+新分组"]
                try:
                    gi = group_opts.index(group_tag)
                except ValueError:
                    gi = 0
                selected_grp = st.selectbox("_g", group_opts, index=gi,
                                            key=f"sidebar_grp_{key}", label_visibility="collapsed")
                if selected_grp == "+新分组":
                    new_g = st.text_input("_ng", key=f"sidebar_newg_{key}",
                                          placeholder="新分组名", label_visibility="collapsed")
                    if new_g and new_g != group_tag:
                        st.session_state.stock_groups[key] = new_g
                        if new_g not in st.session_state.group_order:
                            st.session_state.group_order.append(new_g)
                        _persist_watchlist()
                        _save_group_order()
                        st.rerun()
                elif selected_grp != group_tag:
                    st.session_state.stock_groups[key] = selected_grp
                    _persist_watchlist()
                    st.rerun()
            with c3:
                if st.button("✕", key=f"sidebar_del_{key}", help="删除"):
                    _remove_from_watchlist(key)
                    st.rerun()
    else:
        st.caption("暂无自选股, 上方搜索添加")

    st.divider()

    st.divider()
    with st.expander("🔑 Tushare Token", expanded=False):
        current = get_tushare_token()
        if current:
            st.caption("✅ 已配置")
        token_input = st.text_input("Token", value=current or "",
                                    placeholder="输入Tushare Token",
                                    type="password", label_visibility="collapsed")
        if token_input and token_input != current:
            cfg = load_config()
            cfg["tushare_token"] = token_input
            save_config(cfg)
            st.toast("Token 已保存", icon="✅")
            st.rerun()

    with st.expander("🤖 LLM API (DeepSeek)", expanded=False):
        from src.utils.config import get_llm_key, get_llm_config
        current_key = get_llm_key()
        llm_cfg = get_llm_config()
        if current_key:
            st.caption("✅ DeepSeek Key 已配置")
        api_input = st.text_input("API Key", value=current_key or "",
                                  placeholder="sk-xxx",
                                  type="password", label_visibility="collapsed",
                                  key="llm_key")
        if api_input and api_input != current_key:
            cfg = load_config()
            cfg["llm_api_key"] = api_input
            save_config(cfg)
            st.toast("LLM Key 已保存", icon="🤖")
            st.rerun()

    with st.expander("📱 Telegram 推送", expanded=False):
        from src.utils.config import get_telegram_config
        tg = get_telegram_config()
        if tg["token"] and tg["chat_id"]:
            st.caption("✅ Telegram 已配置")
        t1 = st.text_input("Bot Token", value=tg["token"] or "",
                           placeholder="123456:ABC...", type="password",
                           key="tg_tok", label_visibility="collapsed")
        t2 = st.text_input("Chat ID", value=tg["chat_id"] or "",
                           placeholder="-100xxx", key="tg_cid",
                           label_visibility="collapsed")
        if st.button("💾 保存 Telegram", key="tg_save_btn", use_container_width=True):
            cfg = load_config(); cfg["telegram_token"] = t1; cfg["telegram_chat_id"] = t2
            save_config(cfg); st.rerun()

    with st.expander("💬 微信推送 (PushPlus)", expanded=False):
        from src.utils.config import get_pushplus_token
        ppt = get_pushplus_token()
        if ppt: st.caption("✅ 已配置 (去 pushplus.plus 获取 Token)")
        pt = st.text_input("Token", value=ppt or "",
                           placeholder="xxxxx...", type="password",
                           key="ppt_key", label_visibility="collapsed")
        if st.button("💾 保存", key="ppt_save", use_container_width=True):
            cfg = load_config(); cfg["pushplus_token"] = pt; save_config(cfg); st.rerun()

    st.divider()
    st.caption(f"自选股 {len(st.session_state.watchlist)} 只")

# ═══════════════════════════════════════════════════════════
#  顶部导航
# ═══════════════════════════════════════════════════════════
PAGES = ["🏠 仪表盘", "🔮 预测", "⏪ 回测", "🛡️ 风控", "🔔 交易监控", "🔍 选股器", "💰 模拟交易", "🧠 策略推荐", "📊 因子库", "ℹ️ 自选详情"]

if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

selected = st.segmented_control(
    "导航", PAGES, default=st.session_state.page,
    selection_mode="single", label_visibility="collapsed",
)
if selected:
    st.session_state.page = selected
page = st.session_state.page
st.divider()

# ═══════════════════════════════════════════════════════════
def info_for(key: str) -> dict:
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


# ─── 风控指标格式化 ───────────────────────────────────────
PCT_KEYS = {"MaxDrawdown", "Volatility"}

def fmt_risk(k: str, v: float) -> str:
    if k in PCT_KEYS:
        return f"{v*100:.1f}%"
    if "VaR" in k or "CVaR" in k:
        return f"{v*100:.2f}%"
    if k == "SharpeRatio":
        return f"{v:.2f}"
    return f"{v:.4f}"

# ═══════════════════════════════════════════════════════════
#  📊 仪表盘
# ═══════════════════════════════════════════════════════════
if page == "🏠 仪表盘":
    st.title("🏠 仪表盘")
    st.caption("自选股概览 · 近期走势")

    if not st.session_state.watchlist:
        st.warning("请在左侧添加自选股")
        st.stop()

    ordered = [k for k in st.session_state.stock_order if k in st.session_state.watchlist]
    for k in st.session_state.watchlist:
        if k not in ordered:
            ordered.append(k)
    st.session_state.stock_order = ordered
    _save_stock_order()
    ag = st.session_state.active_group
    if ag != "全部":
        ordered = [k for k in ordered
                   if st.session_state.stock_groups.get(k, "默认") == ag]
    if not ordered:
        st.info(f"分组「{ag}」中暂无股票")
        st.stop()

    # 刷新触发: 并发刷新所有股票, 清 Streamlit 缓存
    if st.button("🔄 刷新数据", key="refresh_dash_btn", use_container_width=False):
        st.session_state.refresh_key = st.session_state.get("refresh_key", 0) + 1
        get_data_for.clear()
        _detect_source_name.clear()
        with st.spinner("正在刷新数据..."):
            import concurrent.futures
            def _refresh_one(key):
                info = info_for(key)
                try:
                    fetch_data(info["symbol"], info["market"], use_cache=False)
                    return (key, True)
                except Exception:
                    return (key, False)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                futs = {pool.submit(_refresh_one, k): k for k in ordered[:20]}
                for fut in concurrent.futures.as_completed(futs, timeout=180):
                    pass
        st.rerun()

    # ── 可拖拽 + 可点击 + 可分组卡片仪表盘 ──
    from src.components.sortable_cards import dash_cards
    from src.alerts.engine import load_rules
    alert_rules = load_rules()
    monitored_symbols = set(
        f"{r.market}-{r.symbol}" for r in alert_rules if r.enabled
    )

    all_groups = [g for g in st.session_state.group_order
                  if g in set(st.session_state.stock_groups.values())]
    for g in set(st.session_state.stock_groups.values()):
        if g not in all_groups:
            all_groups.append(g)
    cards = []
    for key in ordered:
        info = info_for(key)
        try:
            df = get_data_for(info["symbol"], info["market"], period_days=120,
                              _refresh_key=st.session_state.get("refresh_key", 0))
        except Exception:
            df = _mock_data(info["symbol"], info["market"])
        if df is None or df.empty:
            continue
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
        data_date = latest["Date"]
        if hasattr(data_date, "date"):
            data_date = data_date.date()
        today = datetime.now().date()
        
        # 更健壮的日期显示 - 添加严格检查
        try:
            if data_date == today:
                date_label = "📡 今日"
            elif hasattr(data_date, "strftime"):
                # 检查是否是合理的日期（不是01/01）
                month = data_date.month
                day = data_date.day
                if month == 1 and day == 1:
                    date_label = "📅 01/01 (异常日期)"
                else:
                    date_label = f"📅 {data_date.strftime('%m/%d')}收盘价"
            else:
                date_label = f"📅 {data_date}收盘价"
        except Exception:
            date_label = "📅 未知日期"
        cards.append({
            "key": key, "name": info["name"],
            "price": f"{latest['Close']:.2f}", "change": round(change, 2),
            "date_label": date_label, "dark": st.session_state.dark_mode,
            "group": st.session_state.stock_groups.get(key, "默认"),
            "monitored": key in monitored_symbols,
        })

    # 按分组顺序排列，每组内监控的排最前
    group_rank = {g: i for i, g in enumerate(all_groups)}
    cards.sort(key=lambda c: (
        group_rank.get(c["group"], 999),
        not c["monitored"],
        ordered.index(c["key"]) if c["key"] in ordered else 99
    ))

    result = dash_cards(cards, groups=all_groups,
                        height=max(400, len(cards) * 100 // 4 + 100))
    if result:
        if result["action"] == "click":
            st.session_state.selected_stock = result["key"]
            st.session_state.page = "ℹ️ 自选详情"
            st.rerun()
        elif result["action"] == "reorder":
            st.session_state.stock_order = [k for k in result["keys"]
                                            if k in st.session_state.watchlist]
            _save_stock_order()
            st.rerun()
        elif result["action"] == "group":
            key, grp = result["key"], result["group"]
            st.session_state.stock_groups[key] = grp
            _persist_watchlist()
            st.toast(f"✅ {info_for(key)['name']} → {grp}", icon="📂")
            st.rerun()
        elif result["action"] == "new_group":
            key, name = result.get("key"), result["name"]
            if key and name:
                st.session_state.stock_groups[key] = name
            _persist_watchlist()
            st.toast(f"✅ 新分组「{result['name']}」已创建", icon="📂")
            st.rerun()

    with st.expander("📈 自选股走势图", expanded=False):
        tab_names = [info_for(k)["name"] for k in sorted(st.session_state.watchlist)]
        tabs = st.tabs(tab_names)
        for tab, key in zip(tabs, sorted(st.session_state.watchlist)):
            with tab:
                info = info_for(key)
                try:
                    df = fetch_data(info["symbol"], info["market"], use_cache=True)
                except Exception:
                    df = _mock_data(info["symbol"], info["market"])
                from src.data.charting import kline_chart
                fig = kline_chart(df, title=info["name"], indicators=["ma"], height=300,
                                  template=plotly_template())
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📊 风控 & 详细指标"):
                    risk = calc_all_risk_metrics(df)
                    cols2 = st.columns(5)
                    for col2, (k, v) in zip(cols2, risk.items()):
                        col2.metric(k, fmt_risk(k, v), help=RISK_TIPS.get(k, ""))

# ═══════════════════════════════════════════════════════════
#  🔮 预测
# ═══════════════════════════════════════════════════════════
elif page == "🔮 预测":
    st.title("🔮 股价预测")

    tab_new, tab_hist, tab_batch, tab_opt = st.tabs(["🔮 新预测", "📜 历史记录", "📊 批量历史", "⚙️ 模型优化"])

    # ══════════════════════════════════════════════════════
    #  新预测
    # ══════════════════════════════════════════════════════
    with tab_new:
        mode = st.radio("模式", ["单股预测", "批量预测"], horizontal=True, key="pred_mode")

        if mode == "单股预测":
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            with col1:
                pred_grp = st.selectbox("分组",
                    ["全部"] + st.session_state.group_order,
                    key="pred_grp_sel")
            with col2:
                all_keys = st.session_state.watchlist or list(PRESET_STOCKS.keys())
                if pred_grp != "全部":
                    filtered = [k for k in all_keys if st.session_state.stock_groups.get(k, "默认") == pred_grp]
                else:
                    filtered = list(all_keys)
                target = st.selectbox("股票", filtered,
                                      format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
                                      key="pred_target")
            with col3:
                models_sel = st.multiselect("模型", list_models(),
                                            default=["arima", "gbdt", "xgboost"],
                                            key="pred_models")
            with col3:
                steps = st.slider("天数", 5, 90, 30, 5, key="pred_steps")
            with col4:
                run_btn = st.button("▶ 开始预测", type="primary", use_container_width=True)
            targets = [target] if run_btn and target else []
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                all_keys = sorted(st.session_state.watchlist)
                if st.session_state.get("batch_all_flag"):
                    st.session_state.batch_sel = all_keys
                    st.session_state.batch_all_flag = False
                pred_grp_batch = st.selectbox("分组筛选",
                    ["全部"] + st.session_state.group_order,
                    key="batch_grp_sel")
                if pred_grp_batch != "全部":
                    filtered_keys = [k for k in all_keys if st.session_state.stock_groups.get(k, "默认") == pred_grp_batch]
                else:
                    filtered_keys = all_keys
                targets = st.multiselect("批量选股", filtered_keys,
                                         format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
                                         key="batch_sel")
            with c2:
                if st.button("📋 全选", use_container_width=True):
                    st.session_state.batch_all_flag = True
                    st.rerun()
            c1, c2, c3 = st.columns(3)
            with c1:
                models_sel = st.multiselect("模型", list_models(),
                                            default=["arima", "gbdt", "xgboost"],
                                            key="batch_md")
            with c2:
                steps = st.slider("天数", 5, 90, 30, 5, key="batch_sp")
            with c3:
                run_btn = st.button("▶ 批量预测", type="primary", use_container_width=True,
                                    disabled=not targets)
            if not targets:
                targets = []

        if not models_sel:
            st.info("请选择模型后点击预测")
        elif not targets:
            st.info("请选择股票" if mode == "批量预测" else "点击「开始预测」运行")
        elif not run_btn:
            st.info("点击 ▶ 开始预测")
        else:
            st.session_state.batch_summary = []
            for tidx, tkey in enumerate(targets):
                info = info_for(tkey)
                df = get_data_notify(info["symbol"], info["market"], info["name"])
                with st.spinner(f"{tidx+1}/{len(targets)} {info['symbol']} {info['name']}..."):
                    source = _detect_source_name(info["symbol"], info["market"])
                    results = run_models(df, model_names=models_sel, steps=steps, data_source=source)

                from src.data.pred_history import add_prediction
                for name, r in results.items():
                    if len(r.forecast) == 0: continue
                    mape_val = r.metrics.get("MAPE", 0)
                    if isinstance(mape_val, str): mape_val = 0
                    add_prediction(info["symbol"], info["market"], info["name"],
                                   name, r.forecast, r.forecast_dates,
                                   float(r.history[-1]), float(mape_val),
                                   data_source=r.data_source,
                                   model_params=r.model_params)

                valid = [(name, r) for name, r in results.items() if len(r.forecast) > 0]
                if len(targets) == 1:
                    st.subheader("模型评估对比")
                    rows = []
                    for name, r in results.items():
                        m = r.metrics
                        if "error" in m:
                            rows.append({"模型": name, "状态": "❌"})
                        else:
                            direction = "📈涨" if r.forecast[-1] > r.history[-1] else "📉跌"
                            rows.append({"模型": name, "MAE": m.get("MAE","-"), "方向": direction,
                                         "预测末价": f"{r.forecast[-1]:.2f}"})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                with st.expander("🔧 模型参数详情", expanded=False):
                    for name, r in results.items():
                        if len(r.forecast):
                            st.caption(f"**{name}**")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                ds = r.data_source or "—"
                                st.write("📡 数据源:", ds)
                            with c2:
                                params = r.model_params.get("params", {}) if isinstance(r.model_params, dict) else {}
                                param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "默认"
                                st.write("⚙️ 参数:", param_str)
                            with c3:
                                feats = r.feature_names if r.feature_names else r.model_params.get("features", []) if isinstance(r.model_params, dict) else []
                                st.write(f"📊 因子: {len(feats)}个" if feats else "📊 因子: —")
                                if feats:
                                    with st.expander(f"查看因子 ({len(feats)})", expanded=False):
                                        st.caption(", ".join(str(f) for f in feats))
                if valid:
                    up = sum(1 for _, r in valid if r.forecast[-1] > r.history[-1])
                    prices = [r.forecast[-1] for _, r in valid]
                    avg_pct = np.mean([(r.forecast[-1]-r.history[-1])/r.history[-1]*100 for _, r in valid])
                    st.session_state.batch_summary.append({
                        "key": tkey,
                        "symbol": info["symbol"],
                        "name": info["name"],
                        "market": info["market"],
                        "up_count": f"{up}/{len(valid)}",
                        "price_range": f"{min(prices):.1f}-{max(prices):.1f}",
                        "avg_pct": avg_pct,
                        "current_price": float(df["Close"].iloc[-1]),
                        "results": results,
                        "df": df,
                        "info": info,
                    })

            # 保存批量历史
            if len(st.session_state.batch_summary) >= 1:
                from src.data.batch_history import add_batch_record
                details = []
                for item in st.session_state.batch_summary:
                    d = {"symbol": item["symbol"], "name": item["name"],
                         "market": item["market"], "key": item["key"],
                         "up_count": item["up_count"], "avg_pct": item["avg_pct"],
                         "price_range": item["price_range"],
                         "current_price": item["current_price"]}
                    forecasts = {}
                    for n, r in item["results"].items():
                        if len(r.forecast):
                            forecasts[n] = [round(float(x), 2) for x in r.forecast]
                    d["forecasts"] = forecasts
                    details.append(d)
                add_batch_record(steps, models_sel,
                                 [{k: v for k, v in item.items()
                                   if k not in ("results", "df", "info")}
                                  for item in st.session_state.batch_summary],
                                 details)

            # 批量简报表格
            if len(st.session_state.batch_summary) >= 1:
                st.divider()
                st.subheader("📊 批量预测简报")
                st.session_state.batch_summary.sort(key=lambda x: x["avg_pct"], reverse=True)
                for rank, item in enumerate(st.session_state.batch_summary, 1):
                    c1, c2, c3, c4, c5, c6 = st.columns([0.5, 1.5, 1, 1, 1, 1])
                    with c1: st.write(f"#{rank}")
                    with c2: st.write(f"{item['symbol']} {item['name']}")
                    with c3: st.write(f"看涨 {item['up_count']}")
                    with c4: st.write(f"末价 {item['price_range']}")
                    with c5: st.write(f"{item['avg_pct']:+.1f}%")
                    with c6:
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button("📋", key=f"bt_det_{item['key']}", use_container_width=True,
                                         help="查看预测详情"):
                                st.session_state.batch_detail = item["key"]
                                st.rerun()
                        with dc2:
                            if st.button("🔍", key=f"bt_rec_{item['key']}", use_container_width=True,
                                         help="策略推荐"):
                                st.session_state.rec_scan_target = item
                                st.session_state.page = "🧠 策略推荐"
                                st.rerun()

                # 展开选中股票的详情
                detail_key = st.session_state.get("batch_detail")
                if detail_key:
                    for item in st.session_state.batch_summary:
                        if item["key"] == detail_key:
                            st.divider()
                            info, results, df = item["info"], item["results"], item["df"]
                            st.subheader(f"📋 {info['symbol']} {info['name']} 预测详情")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                                                     name="历史", line=dict(color="black", width=1.5)))
                            cs = {"arima":"#E74C3C","gbdt":"#2ECC71","xgboost":"#F39C12",
                                  "lstm":"#9B59B6","transformer":"#1ABC9C"}
                            for n, r in results.items():
                                if len(r.forecast):
                                    fig.add_trace(go.Scatter(x=r.forecast_dates, y=r.forecast,
                                        name=n, line=dict(color=cs.get(n,"gray"), dash="dash", width=2)))
                            fig.update_layout(height=400, hovermode="x unified", template=plotly_template())
                            st.plotly_chart(fig, use_container_width=True)
                            dd = pd.DataFrame()
                            for n, r in results.items():
                                if len(r.forecast): dd[n] = r.forecast.round(2)
                            if not dd.empty:
                                dd.insert(0, "天数", [f"第{i+1}天" for i in range(len(dd))])
                                st.dataframe(dd, use_container_width=True, hide_index=True)
                            with st.expander("🔧 模型参数详情", expanded=False):
                                for n, r in results.items():
                                    if len(r.forecast):
                                        st.caption(f"**{n}**")
                                        c1, c2, c3 = st.columns(3)
                                        with c1:
                                            st.write("数据源:", r.data_source or "—")
                                        with c2:
                                            params = r.model_params.get("params", {}) if isinstance(r.model_params, dict) else {}
                                            st.write("参数:", ", ".join(f"{k}={v}" for k, v in params.items()) if params else "默认")
                                        with c3:
                                            feats = r.feature_names if r.feature_names else r.model_params.get("features", []) if isinstance(r.model_params, dict) else []
                                            st.write(f"因子数: {len(feats)}")
                                            if feats:
                                                with st.expander(f"查看因子列表 ({len(feats)}个)", expanded=False):
                                                    st.caption(", ".join(str(f) for f in feats))
                            if st.button("✕ 收起", key=f"bt_close_det_{detail_key}"):
                                st.session_state.batch_detail = None
                                st.rerun()
                            break

        with st.expander("📖 模型说明", expanded=False):
            st.markdown("""
| 模型 | 原理 | 适用场景 |
|------|------|---------|
| **ARIMA** | 自回归积分滑动平均,分析价格序列自身历史规律 | 平稳序列,短期预测较准 |
| **GBDT** | 梯度提升决策树,滞后价格/统计量作特征 | 中短期趋势,训练快 |
| **XGBoost** | GBDT 工程优化版,正则化防过拟合 | 精度更高 |
| **LSTM** | 长短期记忆网络(PyTorch) | 长序列,非线性模式 |
| **Transformer** | 自注意力机制(PyTorch) | 全局趋势,预测强 |
""")

    with tab_hist:
        from src.data.pred_history import load_history, PredictionRecord

        history = load_history()
        if history:
            for h in reversed(history):
                with st.expander(f"{h.predicted_at} | [{h.market}] {h.symbol} {h.stock_name} — {h.model} "
                                 f"预测{h.steps}天 → 末价{h.final_prediction:.2f} ({h.final_pct:+.1f}%)"):
                    detail = pd.DataFrame({
                        "天数": [f"第{i+1}天" for i in range(h.steps)],
                        "预测价": h.forecast,
                        "日期": [d.split('-')[1] + '/' + d.split('-')[2] if len(d.split('-')) == 3 else d 
                                for d in h.forecast_dates] if len(h.forecast_dates) == h.steps
                                else [""] * h.steps,
                    })
                    detail["预测价"] = detail["预测价"].round(2)
                    st.dataframe(detail, use_container_width=True, hide_index=True)

                    if h.data_source or h.model_params:
                        with st.expander("🔧 模型参数", expanded=False):
                            if h.data_source:
                                st.write("📡 数据源:", h.data_source)
                            if h.model_params:
                                params = h.model_params.get("params", {})
                                if params:
                                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                                    st.write("⚙️ 参数:", param_str)

                    c1, c2 = st.columns([1, 5])
                    with c1:
                        if st.button("🔄 一键重测", key=f"retest_{h.id}",
                                     use_container_width=True, type="primary"):
                            df = get_data_for(h.symbol, h.market)
                            with st.spinner(f"重测 {h.stock_name} {h.model}..."):
                                from src.models.factory import run_models
                                from src.data.pred_history import add_prediction
                                source = _detect_source_name(h.symbol, h.market)
                                results = run_models(df, model_names=[h.model], steps=h.steps, data_source=source)
                                if h.model in results and len(results[h.model].forecast) > 0:
                                    r = results[h.model]
                                    mape_val = r.metrics.get("MAPE", 0)
                                    if isinstance(mape_val, str):
                                        mape_val = 0
                                    add_prediction(h.symbol, h.market, h.stock_name,
                                                   h.model, r.forecast, r.forecast_dates,
                                                   float(r.history[-1]), float(mape_val),
                                                   data_source=r.data_source,
                                                   model_params=r.model_params)
                                    st.success(f"重测完成: 末价{r.forecast[-1]:.2f} "
                                                f"({(r.forecast[-1]-r.history[-1])/r.history[-1]*100:+.1f}%)")
                                    st.rerun()
                    with c2:
                        pass

    with tab_batch:
        from src.data.batch_history import load_batch_history

        bh = load_batch_history()
        if not bh:
            st.info("暂无批量预测历史")
        else:
            for b in reversed(bh):
                with st.expander(f"{b.predicted_at} | {len(b.summary)}只股票 "
                                 f"| 模型:{','.join(b.models)} | 预测{b.steps}天"):
                    # 按涨跌排序
                    items = sorted(b.summary, key=lambda x: x.get("avg_pct", 0), reverse=True)
                    for rank, item in enumerate(items, 1):
                        c1, c2, c3, c4, c5, c6 = st.columns([0.5, 1.5, 1, 1, 1, 1])
                        with c1: st.write(f"#{rank}")
                        with c2: st.write(f"{item['symbol']} {item['name']}")
                        with c3: st.write(f"看涨 {item['up_count']}")
                        with c4: st.write(f"末价 {item['price_range']}")
                        with c5: st.write(f"{item.get('avg_pct',0):+.1f}%")
                        with c6:
                            dc1, dc2 = st.columns(2)
                            with dc1:
                                if st.button("📋", key=f"bh_det_{b.id}_{item['key']}",
                                             use_container_width=True, help="查看详情"):
                                    st.session_state.batch_detail = item["key"]
                                    st.rerun()
                            with dc2:
                                if st.button("🔍", key=f"bh_rec_{b.id}_{item['key']}",
                                             use_container_width=True, help="策略推荐"):
                                    st.session_state.rec_scan_target = {
                                        "key": item["key"], "symbol": item["symbol"],
                                        "name": item["name"], "market": item.get("market", "A"),
                                        "up_count": item["up_count"],
                                    }
                                    st.session_state.page = "🧠 策略推荐"
                                    st.rerun()

                    # 展开详情
                    detail_key = st.session_state.get("batch_detail")
                    if detail_key:
                        for d in b.details:
                            if d.get("key") == detail_key:
                                st.divider()
                                st.caption(f"📋 {d['symbol']} {d['name']} 预测详情")
                                dd = pd.DataFrame(d.get("forecasts", {}))
                                if not dd.empty:
                                    dd.insert(0, "天数", [f"第{i+1}天" for i in range(len(dd))])
                                    st.dataframe(dd, use_container_width=True, hide_index=True)
                                if st.button("✕ 收起", key=f"bh_close_{b.id}"):
                                    st.session_state.batch_detail = None
                                    st.rerun()
                                break

    # ═══════════════════════════════════════════════════════════
    #  模型优化 (交叉验证 + 超参数优化)
    # ═══════════════════════════════════════════════════════════
    with tab_opt:
        opt_mode = st.radio("优化模式", ["🔬 交叉验证", "🎯 超参数优化", "📂 缓存查看"], 
                            horizontal=True, key="opt_mode")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            keys = st.session_state.watchlist or list(PRESET_STOCKS.keys())
            target = st.selectbox("股票", keys,
                                  format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
                                  key="opt_target")
        
        info = info_for(target)
        df = get_data_for(info["symbol"], info["market"], period_days=500)
        
        if opt_mode == "🔬 交叉验证":
            with col2:
                st.subheader("🔬 交叉验证")
                c1, c2, c3 = st.columns(3)
                with c1:
                    model_name = st.selectbox("模型", ["gbdt", "xgboost", "lstm", "transformer"], key="cv_model")
                with c2:
                    n_splits = st.slider("折数", 3, 10, 5, 1, key="cv_splits")
                with c3:
                    use_cache_cv = st.checkbox("使用缓存", value=True, key="cv_cache")
            
            if st.button("▶ 开始交叉验证", type="primary", use_container_width=True):
                from src.models import MODEL_REGISTRY
                from src.models.factory import list_models
                
                model_cls = MODEL_REGISTRY.get(model_name)
                if model_cls:
                    model = model_cls()
                    with st.spinner(f"正在执行 {n_splits} 折交叉验证..."):
                        result = model.cross_validate(df, n_splits=n_splits, use_cache=use_cache_cv)
                    
                    st.success("交叉验证完成！")
                    
                    metrics_df = pd.DataFrame({
                        "指标": ["MAE", "RMSE", "MAPE", "方向准确率"],
                        "平均值": [
                            result.avg_metrics.get("MAE", 0),
                            result.avg_metrics.get("RMSE", 0),
                            result.avg_metrics.get("MAPE", 0),
                            result.avg_metrics.get("Direction_Accuracy", 0)
                        ],
                        "标准差": [
                            result.std_metrics.get("MAE", 0),
                            result.std_metrics.get("RMSE", 0),
                            result.std_metrics.get("MAPE", 0),
                            result.std_metrics.get("Direction_Accuracy", 0)
                        ]
                    })
                    metrics_df["平均值"] = metrics_df["平均值"].round(4)
                    metrics_df["标准差"] = metrics_df["标准差"].round(4)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    with col_b:
                        fold_data = []
                        for i, m in enumerate(result.fold_metrics):
                            fold_data.append({
                                "折数": f"Fold {i+1}",
                                "MAE": m["MAE"],
                                "RMSE": m["RMSE"],
                                "MAPE": m["MAPE"],
                                "方向准确率": m["Direction_Accuracy"]
                            })
                        fold_df = pd.DataFrame(fold_data)
                        st.dataframe(fold_df, use_container_width=True, hide_index=True)
                    
                    # 可视化各折性能
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(rows=2, cols=2, 
                                      subplot_titles=("各折 MAE", "各折 RMSE", "各折 MAPE", "各折 方向准确率"))
                    
                    folds = [f"Fold {i+1}" for i in range(n_splits)]
                    fig.add_trace(go.Bar(x=folds, y=[m["MAE"] for m in result.fold_metrics], 
                                        name="MAE", marker_color="steelblue"), row=1, col=1)
                    fig.add_trace(go.Bar(x=folds, y=[m["RMSE"] for m in result.fold_metrics], 
                                        name="RMSE", marker_color="orange"), row=1, col=2)
                    fig.add_trace(go.Bar(x=folds, y=[m["MAPE"] for m in result.fold_metrics], 
                                        name="MAPE", marker_color="green"), row=2, col=1)
                    fig.add_trace(go.Bar(x=folds, y=[m["Direction_Accuracy"] for m in result.fold_metrics], 
                                        name="方向准确率", marker_color="purple"), row=2, col=2)
                    
                    fig.update_layout(height=500, showlegend=False, 
                                    template="plotly_dark" if st.session_state.dark_mode else "plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif opt_mode == "🎯 超参数优化":
            with col2:
                st.subheader("🎯 超参数优化")
                c1, c2, c3 = st.columns(3)
                with c1:
                    model_name = st.selectbox("模型", ["gbdt", "xgboost", "lstm", "transformer"], key="hpo_model")
                with c2:
                    n_trials = st.slider("试验次数", 10, 100, 50, 10, key="hpo_trials")
                with c3:
                    metric = st.selectbox("优化指标", ["rmse", "mae", "mape"], key="hpo_metric")
            
            if st.button("▶ 开始优化", type="primary", use_container_width=True):
                from src.models.optimization import get_tuner
                
                with st.spinner(f"正在优化 {model_name}（{n_trials}次试验）..."):
                    tuner = get_tuner(model_name, n_trials=n_trials, metric=metric, n_splits=3)
                    result = tuner.tune(df)
                
                st.success(f"优化完成！最佳{metric.upper()}: {result.best_score:.4f}")
                
                # 最佳参数
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("🏆 最佳参数")
                    params_df = pd.DataFrame({
                        "参数": list(result.best_params.keys()),
                        "值": [f"{v:.4f}" if isinstance(v, float) else str(v) 
                               for v in result.best_params.values()]
                    })
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                with col_b:
                    st.subheader("📊 优化统计")
                    stats_df = pd.DataFrame({
                        "指标": ["试验次数", "最佳分数", "优化时间(秒)", "最佳试验编号"],
                        "值": [result.n_trials, f"{result.best_score:.4f}", 
                               f"{result.optimization_time:.2f}", result.best_trial]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # 可视化优化过程
                if result.all_trials:
                    import plotly.graph_objects as go
                    
                    trials_data = pd.DataFrame(result.all_trials)
                    trials_data["trial"] = trials_data["trial"] + 1
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trials_data["trial"],
                        y=trials_data["score"],
                        mode="lines+markers",
                        name="分数",
                        line=dict(color="steelblue", width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[result.best_trial + 1],
                        y=[result.best_score],
                        mode="markers",
                        name="最佳",
                        marker=dict(color="red", size=15, symbol="star")
                    ))
                    
                    fig.update_layout(
                        title="超参数优化过程",
                        xaxis_title="试验次数",
                        yaxis_title=metric.upper(),
                        template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif opt_mode == "📂 缓存查看":
            with col2:
                st.subheader("📂 缓存管理")
                
                # 交叉验证缓存
                st.write("**交叉验证缓存**")
                cv_cache_dir = Path("data/cache/cv")
                if cv_cache_dir.exists():
                    cv_files = list(cv_cache_dir.glob("*.pkl"))
                    if cv_files:
                        cv_data = []
                        for f in cv_files:
                            import pickle
                            try:
                                with open(f, "rb") as file:
                                    cvr = pickle.load(file)
                                cv_data.append({
                                    "模型": cvr.model_name,
                                    "折数": cvr.n_splits,
                                    "平均MAE": cvr.avg_metrics.get("MAE", 0),
                                    "文件": f.name
                                })
                            except:
                                pass
                        
                        if cv_data:
                            cv_df = pd.DataFrame(cv_data)
                            st.dataframe(cv_df, use_container_width=True, hide_index=True)
                            
                            if st.button("🗑️ 清空CV缓存", key="clear_cv"):
                                for f in cv_files:
                                    f.unlink()
                                st.rerun()
                    else:
                        st.info("暂无CV缓存")
                else:
                    st.info("暂无CV缓存")
                
                st.divider()
                
                # HPO缓存
                st.write("**超参数优化缓存**")
                hpo_cache_dir = Path("data/cache/tuning")
                if hpo_cache_dir.exists():
                    hpo_files = list(hpo_cache_dir.glob("*.json"))
                    if hpo_files:
                        hpo_data = []
                        for f in hpo_files:
                            try:
                                import json
                                with open(f) as file:
                                    data = json.load(file)
                                hpo_data.append({
                                    "模型": f.stem.split("_")[0],
                                    "指标": data.get("metric_name", "rmse"),
                                    "最佳分数": data.get("best_score", 0),
                                    "试验次数": data.get("n_trials", 0),
                                    "文件": f.name
                                })
                            except:
                                pass
                        
                        if hpo_data:
                            hpo_df = pd.DataFrame(hpo_data)
                            st.dataframe(hpo_df, use_container_width=True, hide_index=True)
                            
                            if st.button("🗑️ 清空HPO缓存", key="clear_hpo"):
                                for f in hpo_files:
                                    f.unlink()
                                st.rerun()
                    else:
                        st.info("暂无HPO缓存")
                else:
                    st.info("暂无HPO缓存")

# ═══════════════════════════════════════════════════════════
#  📈 回测
# ═══════════════════════════════════════════════════════════
elif page == "⏪ 回测":
    st.title("⏪ 回测引擎")
    tab_bt, tab_bt_hist = st.tabs(["📈 新回测", "📜 历史记录"])

    STG_LIST = [
        "双均线交叉(5/20)", "双均线交叉(10/30)", "双均线交叉(20/60)",
        "RSI均值回归(14)", "通道突破(20/10)", "布林带(20/2)",
        "滚动预测(月频)", "滚动预测(周频)",
    ]

    strategy_map = {
        "双均线交叉(5/20)":  MovingAverageCrossStrategy(5, 20),
        "双均线交叉(10/30)": MovingAverageCrossStrategy(10, 30),
        "双均线交叉(20/60)": MovingAverageCrossStrategy(20, 60),
        "RSI均值回归(14)":   RSIStrategy(14, 30, 70),
        "通道突破(20/10)":   ChannelBreakoutStrategy(20, 10),
        "布林带(20/2)":      BollingerStrategy(20, 2),
        "滚动预测(月频)":    RollingPredictionStrategy(
            GBDTModel(), warmup=200, retrain_freq=20,
            threshold_buy=0.015, threshold_sell=-0.015),
        "滚动预测(周频)":    RollingPredictionStrategy(
            GBDTModel(), warmup=200, retrain_freq=5,
            threshold_buy=0.01, threshold_sell=-0.01),
    }

    # ══════════════════════════════════════════════════════
    #  新回测
    # ══════════════════════════════════════════════════════
    with tab_bt:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            bt_grp = st.selectbox("分组",
                ["全部"] + st.session_state.group_order,
                key="bt_grp_sel")
        with col2:
            keys = st.session_state.watchlist or list(PRESET_STOCKS.keys())
            if bt_grp != "全部":
                filtered = [k for k in keys if st.session_state.stock_groups.get(k, "默认") == bt_grp]
            else:
                filtered = list(keys)
            target = st.selectbox("股票", filtered,
                                  format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
                                  key="bt_target")
        with col2:
            strategy_name = st.selectbox("策略", STG_LIST, key="bt_strat")
        with col3:
            capital = st.number_input("初始资金", 10000, 1_000_000, 100_000, step=10000)
        with col4:
            run_btn = st.button("▶ 运行回测", type="primary", use_container_width=True)

        # 时间范围
        info = info_for(target)
        try:
            df_raw = get_data_for(info["symbol"], info["market"])
        except Exception:
            df_raw = None
        if df_raw is not None and not df_raw.empty:
            d_min = df_raw["Date"].min().date()
            d_max = df_raw["Date"].max().date()
            st.caption(f"数据范围: {d_min} ~ {d_max} 共{len(df_raw)}条")
            col_s, col_e = st.columns(2)
            with col_s:
                start_date = st.date_input("开始日", d_min,
                                           min_value=d_min, max_value=d_max, key="bt_start")
            with col_e:
                end_date = st.date_input("结束日", d_max,
                                         min_value=d_min, max_value=d_max, key="bt_end")
        else:
            start_date = end_date = None

        with st.expander("📖 策略说明", expanded=False):
            st.markdown("""
| 策略 | 逻辑 | 适用场景 |
|------|------|---------|
| **双均线(5/20)** | 短线交叉信号 | 短线交易,捕捉快速趋势 |
| **双均线(10/30)** | 中短线交叉 | 过滤噪音比5/20更稳 |
| **双均线(20/60)** | 经典金叉死叉 | 中长期趋势,可靠性高 |
| **RSI均值回归** | 超卖买入/超买卖出 | 震荡市效果好 |
| **通道突破** | 突破N日高点买入 | 强势趋势市 |
| **布林带** | 下轨买入/上轨卖出 | 均值回归,区间内低买高卖 |
| **滚动预测(月频)** | GBDT月频重训 | 中长线持仓 |
| **滚动预测(周频)** | GBDT周频重训 | 中短线调仓 |
""")

        if not run_btn:
            st.info("选择参数后点击「运行回测」")
        else:
            df = df_raw
            if start_date and end_date:
                df = df_raw[(df_raw["Date"] >= pd.Timestamp(start_date)) &
                            (df_raw["Date"] <= pd.Timestamp(end_date))]
                if df.empty:
                    st.error("选定时间范围内无数据")
                    st.stop()

            strat = strategy_map[strategy_name]

            with st.spinner("回测进行中..."):
                cfg = BacktestConfig(initial_capital=capital, market=info["market"])
                result = BacktestEngine(df, strat, cfg).run()

            from src.data.bt_history import add_bt_record
            add_bt_record(info["symbol"], info["market"], info["name"],
                          strategy_name, capital,
                          result.total_return, result.annual_return,
                          result.sharpe_ratio, result.max_drawdown,
                          result.win_rate, result.profit_factor,
                          result.total_trades, result.total_fees)

            st.divider()
            st.subheader("📊 绩效指标")
            cols = st.columns(5)
            for col, (k, v) in zip(cols, result.metrics.items()):
                col.metric(k, v)

            st.divider()
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06,
                                subplot_titles=("组合净值", "回撤曲线", "持仓市值"))
            dates = result.equity_curve.index
            fig.add_trace(go.Scatter(x=dates, y=result.equity_curve.values,
                                     name="组合净值", line=dict(color="black")), row=1, col=1)
            fig.add_hline(y=capital, line_dash="dash", line_color="gray", row=1, col=1)
            peak = np.maximum.accumulate(result.equity_curve.values)
            dd = (result.equity_curve.values - peak) / peak * 100
            fig.add_trace(go.Scatter(x=dates, y=dd, name="回撤(%)",
                                     fill="tozeroy", line=dict(color="red")), row=2, col=1)
            fig.add_trace(go.Scatter(x=dates, y=result.holdings_curve.values,
                                     name="持仓市值", fill="tozeroy",
                                     line=dict(color="#1f77b4")), row=3, col=1)
            fig.update_layout(height=600, hovermode="x unified",
                               title=f"{info['name']} — {strategy_name}",
                               template=plotly_template())
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader(f"📝 交易记录 ({result.total_trades} 笔)")
            tdf = result.trades_df()
            if not tdf.empty:
                st.dataframe(tdf, use_container_width=True, hide_index=True)
            else:
                st.info("无交易记录")

            with st.expander("🔧 策略参数", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("📡 数据源:", _detect_source_name(info["symbol"], info["market"]))
                    st.write("💰 初始资金:", f"{capital:,.0f}")
                with c2:
                    st.write("🎯 策略:", strategy_name)
                    st.write("📅 回测区间:", f"{start_date} ~ {end_date}")
                with c3:
                    cfg_params = {
                        "手续费率": f"{cfg.fee_rate*100:.2f}%",
                        "印花税": f"{cfg.stamp_tax*100:.2f}%",
                        "滑点": f"{cfg.slippage*100:.2f}%",
                    }
                    if hasattr(strat, '__dict__'):
                        strat_params = {k: v for k, v in strat.__dict__.items() 
                                       if not k.startswith('_') and k != 'name'}
                        if strat_params:
                            st.write("⚙️ 策略参数:", ", ".join(f"{k}={v}" for k, v in strat_params.items()))
                    st.write("🛠️ 配置:", ", ".join(f"{k}={v}" for k, v in cfg_params.items()))

    # ══════════════════════════════════════════════════════
    #  历史记录
    # ══════════════════════════════════════════════════════
    with tab_bt_hist:
        from src.data.bt_history import load_bt_history

        history = load_bt_history()
        if not history:
            st.info("暂无回测历史, 先做一个回测吧")
        else:
            for h in reversed(history):
                with st.expander(f"{h.predicted_at} | [{h.market}] {h.symbol} {h.stock_name} "
                                 f"— {h.strategy} | 收益{h.total_return*100:+.1f}% "
                                 f"夏普{h.sharpe:.2f} 回撤{h.max_dd*100:.1f}% 交易{h.total_trades}笔"):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("总收益", f"{h.total_return*100:.2f}%")
                    c2.metric("夏普", f"{h.sharpe:.2f}")
                    c3.metric("最大回撤", f"{h.max_dd*100:.2f}%")
                    c4.metric("胜率", f"{h.win_rate*100:.1f}%")
                    c5.metric("交易次数", str(h.total_trades))

                    if st.button("🔄 一键重测", key=f"bt_retest_{h.id}",
                                 use_container_width=True, type="primary"):
                        df = get_data_for(h.symbol, h.market, h.stock_name)
                        strat_map = {k: v for k, v in strategy_map.items()}
                        s = strat_map.get(h.strategy)
                        if s:
                            cfg = BacktestConfig(initial_capital=h.capital, market=h.market)
                            r2 = BacktestEngine(df, s, cfg).run()
                            from src.data.bt_history import add_bt_record
                            add_bt_record(h.symbol, h.market, h.stock_name,
                                          h.strategy, h.capital,
                                          r2.total_return, r2.annual_return,
                                          r2.sharpe_ratio, r2.max_drawdown,
                                          r2.win_rate, r2.profit_factor,
                                          r2.total_trades, r2.total_fees)
                            st.success(f"重测完成: 收益{r2.total_return*100:+.1f}%")
                            st.rerun()

# ═══════════════════════════════════════════════════════════
#  ⚠️ 风控
# ═══════════════════════════════════════════════════════════
elif page == "🛡️ 风控":
    st.title("🛡️ 风险分析")
    st.caption("VaR · 回撤 · 波动率 · 夏普比率")

    keys = st.session_state.watchlist or list(PRESET_STOCKS.keys())
    target = st.selectbox("选择股票", keys,
                           format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}")
    info = info_for(target)
    df = get_data_notify(info["symbol"], info["market"], info["name"])
    prices = df["Close"].values
    returns = np.diff(prices) / prices[:-1]

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 核心指标")
        risk = calc_all_risk_metrics(df)
        for k, v in risk.items():
            st.metric(k, fmt_risk(k, v))
        st.divider()
        st.subheader("收益率分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns * 100, nbinsx=50,
                                   marker_color="#1f77b4", opacity=0.7))
        fig.update_layout(height=300, xaxis_title="日收益率(%)", yaxis_title="频次",
                          template=plotly_template())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 回撤分析")
        peak = np.maximum.accumulate(prices)
        dd = (prices - peak) / peak * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=dd, fill="tozeroy",
                                  line=dict(color="red"), name="回撤%"))
        fig.update_layout(height=300, yaxis_title="回撤(%)", template=plotly_template())
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🎯 VaR 分析")
        confidence = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01)
        var_val = float(np.percentile(returns, (1 - confidence) * 100))
        cvar_val = float(returns[returns <= np.percentile(returns, (1 - confidence) * 100)].mean())
        cols = st.columns(2)
        cols[0].metric(f"VaR ({confidence:.0%})", f"{var_val*100:.2f}%")
        cols[1].metric(f"CVaR ({confidence:.0%})", f"{cvar_val*100:.2f}%")

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=returns * 100, nbinsx=50,
                                    marker_color="#1f77b4", opacity=0.7, name="日收益率"))
        fig2.add_vline(x=var_val * 100, line_dash="dash", line_color="red",
                       annotation_text=f"VaR {var_val*100:.2f}%")
        fig2.update_layout(height=250, xaxis_title="日收益率(%)", template=plotly_template())
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("滚动波动率 (20日)")
    vol = pd.Series(returns).rolling(20).std() * np.sqrt(252) * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["Date"][1:], y=vol.values,
                               line=dict(color="orange"), name="年化波动率%"))
    fig3.update_layout(height=300, yaxis_title="波动率(%)", template=plotly_template())
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  🔔 提醒
# ═══════════════════════════════════════════════════════════
elif page == "🔔 交易监控":
    st.title("🔔 交易监控")
    st.caption("为自选股设置监控策略, 触发时发送通知")

    from src.alerts import (AlertRule, add_rule, remove_rule, toggle_rule,
                             load_rules, get_engine, CONDITION_TYPES as CT)
    from src.data.stock_db import resolve_stock_name, search_stocks
    from src.alerts.settings import (MonitorSettings, load_settings,
                                      save_settings, MARKET_NAMES)
    from src.alerts.conditions import preview_notification

    if "alert_target" not in st.session_state:
        st.session_state.alert_target = None
    rules = load_rules()

    # ── 自选股列表 + 添加策略按钮 ──
    with st.expander("📋 自选股票策略", expanded=True):
        wk = sorted(st.session_state.watchlist)
        if not wk:
            st.info("暂无自选股, 请在侧边栏添加")
        else:
            for key in wk:
                parts = key.split("-", 1)
                if len(parts) != 2:
                    continue
                market, symbol = parts
                name = st.session_state.stock_names.get(key, symbol)
                stock_rules = [r for r in rules if r.symbol == symbol and r.market == market]
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.write(f"[{market}] **{symbol}** {name}")
                with c2:
                    n = len(stock_rules)
                    st.write(f"策略: {n} 条" if n else "无策略")
                with c3:
                    if st.button("＋ 添加", key=f"as_{key}", use_container_width=True):
                        st.session_state.alert_target = (symbol, market, name)
                        st.rerun()

    st.divider()

    # ── 添加规则 (自动预填选中的股票) ──
    with st.expander("➕ 添加规则", expanded=st.session_state.alert_target is not None):
        target = st.session_state.alert_target
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            if target:
                code, m, n = target
                st.text_input("股票", value=f"[{m}] {code} {n}", disabled=True, key="at_fixed")
                selected_s = (code, n, m)
            else:
                search_q = st.text_input("搜索股票", placeholder="例: 茅台 / AAPL",
                                         key="alert_search")
                if search_q:
                    results = search_stocks(search_q, limit=5)
                    selected_s = (st.selectbox("选择", results,
                                   format_func=lambda x: f"[{x[2]}] {x[0]} {x[1]}",
                                   key="as_select")
                                  ) if results else (search_q, "", "A")
                else:
                    selected_s = None
        with col2:
            if selected_s:
                market_override = st.selectbox("市场", ["A", "HK", "US"],
                                               index=["A","HK","US"].index(selected_s[2]),
                                               key="am_override")
        with col3:
            condition = st.selectbox("条件", list(CT.keys()),
                                     format_func=lambda x: f"{CT[x]}", key="ac_cond")

        params = {}
        if condition in ("above_ma", "below_ma"):
            params["window"] = st.number_input("MA窗口", 5, 120, 20, key="pw2")
        elif condition in ("above_price", "below_price"):
            params["threshold"] = st.number_input("价格阈值", 0.0, 10000.0, 100.0, key="pt2")
        elif condition in ("rsi_oversold", "rsi_overbought"):
            w = st.number_input("RSI窗口", 5, 30, 14, key="rw2")
            lv = st.number_input("阈值", 10, 90, 30 if "oversold" in condition else 70, key="rl2")
            params.update(window=int(w), level=int(lv))
        elif condition == "volume_spike":
            params["ratio"] = st.number_input("倍数", 1.0, 10.0, 2.0, key="vr2")
        elif condition == "daily_change":
            params["direction"] = st.selectbox("方向", ["up", "down"], key="dd2")
            params["pct"] = st.number_input("百分比%", 1.0, 20.0, 5.0, key="dp2")
        elif condition in ("golden_cross", "death_cross"):
            params["short"] = st.number_input("短期MA", 5, 50, 20, key="gs2")
            params["long"] = st.number_input("长期MA", 10, 200, 60, key="gl2")
        elif condition in ("bollinger_upper", "bollinger_lower"):
            params["window"] = st.number_input("窗口", 10, 50, 20, key="bw2")
            params["std"] = st.number_input("标准差", 1, 4, 2, key="bs2")
        elif condition == "ma_cross_combo":
            params["short"] = st.number_input("短期MA", 5, 50, 20, key="mc_s")
            params["long"] = st.number_input("长期MA", 10, 200, 60, key="mc_l")
        elif condition == "rsi_combo":
            params["window"] = st.number_input("RSI窗口", 5, 30, 14, key="rc_w")
            params["oversold"] = st.number_input("超卖阈值", 10, 40, 30, key="rc_o")
            params["overbought"] = st.number_input("超买阈值", 60, 90, 70, key="rc_ob")
        elif condition == "bollinger_combo":
            params["window"] = st.number_input("布林窗口", 10, 50, 20, key="bc_w")
            params["std"] = st.number_input("标准差", 1, 4, 2, key="bc_s")
        elif condition == "ma_rsi_combo":
            params["ma_window"] = st.number_input("趋势MA", 20, 120, 60, key="mr_m")
            params["rsi_window"] = st.number_input("RSI窗口", 5, 30, 14, key="mr_r")
            params["oversold"] = st.number_input("RSI超卖", 10, 40, 30, key="mr_o")
            params["overbought"] = st.number_input("RSI超买", 60, 90, 70, key="mr_ob")
        elif condition == "volume_breakout":
            params["lookback"] = st.number_input("回顾天数", 10, 60, 20, key="vb_l")
            params["vol_ratio"] = st.number_input("成交量倍数", 1.5, 5.0, 2.0, key="vb_v")
        elif condition == "ma_triple":
            params["short"] = st.number_input("短期MA", 5, 30, 10, key="mt_s")
            params["mid"] = st.number_input("中期MA", 15, 60, 30, key="mt_m")
            params["long"] = st.number_input("长期MA", 30, 200, 60, key="mt_l")

        # 通知预览
        if selected_s and condition:
            msg, action = preview_notification(condition, params, price=100.0)
            st.info(f"📢 触发时将通知:\n  **{msg}**\n  ℹ️ 操作建议: {action}")

        label = st.text_input("备注 (可选)", placeholder="例: 突破买入", key="al_label")
        cooldown = st.number_input("冷却(分钟)", 10, 480, 60, key="al_cd")

        cols_btn = st.columns([1, 1, 4])
        with cols_btn[0]:
            if selected_s and st.button("✅ 添加", type="primary", use_container_width=True):
                code, name, m = selected_s
                m = market_override if "market_override" in dir() else m
                add_rule(AlertRule(
                    symbol=code, market=m, condition=condition,
                    params={k: int(v) if isinstance(v, float) and v == int(v) else v
                            for k, v in params.items()},
                    label=label or name or code,
                    cooldown_minutes=int(cooldown),
                ))
                st.session_state.alert_target = None
                st.success("已添加")
                st.rerun()
        with cols_btn[1]:
            if st.button("取消", use_container_width=True):
                st.session_state.alert_target = None
                st.rerun()

    st.divider()

    # ── 已设规则列表 ──
    st.subheader(f"📋 已设规则 ({len(rules)} 条)")
    if not rules:
        st.info("暂无规则, 从上方自选股添加")
    else:
            for r in rules:
                cols = st.columns([1, 2, 3, 1, 1, 1])
                name = st.session_state.stock_names.get(f"{r.market}-{r.symbol}", "")
                if not name:
                    from src.data.stock_db import resolve_stock_name
                    name = resolve_stock_name(r.symbol, r.market)
                with cols[0]:
                    st.write("🟢" if r.enabled else "🔴")
                with cols[1]:
                    st.write(f"**{r.symbol}** {name}" if name else f"**{r.symbol}** ({r.market})")
                with cols[2]:
                    desc = CT.get(r.condition, r.condition)
                    extra = ", ".join(f"{k}={v}" for k, v in r.params.items())
                    st.write(f"{desc} | {extra}")
                with cols[3]:
                    st.write(r.label[:15])
                with cols[4]:
                    if st.button("⏸" if r.enabled else "▶", key=f"tg_{r.uid}",
                                 use_container_width=True):
                        toggle_rule(r.uid)
                        st.rerun()
                with cols[5]:
                    if st.button("✕", key=f"rm_{r.uid}", use_container_width=True):
                        remove_rule(r.uid)
                        st.rerun()

    st.divider()

    # ── 设置 + 监控控制 ──
    with st.expander("⚙️ 监控设置", expanded=False):
        settings = load_settings()
        col1, col2 = st.columns(2)
        with col1:
            market = st.selectbox("参考市场时段", ["A", "HK", "US"],
                                  index=["A","HK","US"].index(settings.market),
                                  format_func=lambda x: MARKET_NAMES.get(x, x))
            interval = st.number_input("检查间隔(分钟)", 1, 60,
                                       settings.interval_minutes, key="si")
            trade_only = st.checkbox("仅交易日(周一到周五)", settings.trade_days_only)
        with col2:
            use_custom = st.checkbox("自定义时段", False)
            if use_custom:
                custom_s = st.text_input("开始 HH:MM", settings.custom_start or "09:30")
                custom_e = st.text_input("结束 HH:MM", settings.custom_end or "15:00")
            else:
                custom_s = None
                custom_e = None
        if st.button("💾 保存设置", use_container_width=True):
            save_settings(MonitorSettings(
                market=market, interval_minutes=int(interval),
                custom_start=custom_s, custom_end=custom_e,
                trade_days_only=trade_only,
            ))
            st.toast("设置已保存", icon="⚙️")

    # 监控进程控制 (独立进程, 不依赖 Web)
    import subprocess, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    daemon_script = os.path.join(root, "tools", "monitor_daemon.py")

    def _daemon_running():
        try:
            r = subprocess.run(["pgrep", "-f", "monitor_daemon.py"],
                               capture_output=True, text=True, timeout=3)
            return bool(r.stdout.strip())
        except Exception:
            return False

    running = _daemon_running()
    col1, col2, col3 = st.columns(3)
    with col1:
        if running:
            st.success("🟢 后台监控运行中")
        else:
            st.warning("🔴 监控已停止")
    with col2:
        if st.button("▶ 启动", use_container_width=True, disabled=running):
            subprocess.Popen(["python3", "-u", daemon_script],
                             cwd=root, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            st.toast("监控已启动 (后台进程)", icon="🟢")
            st.rerun()
    with col3:
        if st.button("⏹ 停止", use_container_width=True, disabled=not running):
            subprocess.run(["pkill", "-f", "monitor_daemon.py"], timeout=5)
            st.toast("监控已停止", icon="🔴")
            st.rerun()

    # ── 连接健康状态 ──
    st.divider()
    st.subheader("📡 连接状态监控")
    from src.alerts.health import summary as health_summary, get_health
    hs = health_summary()
    recent_checks = get_health().recent(60)

    status_map = {
        "ok": ("🟢 正常", "success"),
        "degraded": ("🟡 降级", "warning"),
        "down": ("🔴 断连", "error"),
        "unknown": ("⚪ 未知", "info"),
    }
    status_label, status_type = status_map.get(hs["status"], ("⚪ 未知", "info"))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("状态", status_label)
    with c2:
        st.metric("近1h查询", str(hs["recent_checks"]))
    with c3:
        st.metric("成功率", f"{hs['success_rate']}%")
    with c4:
        fail_count = hs["consecutive_failures"]
        fail_label = f"⚠️ {fail_count}" if fail_count > 0 else "0"
        st.metric("连续失败", fail_label)
    with c5:
        lat = hs.get("avg_latency_ms", 0)
        st.metric("平均延迟", f"{lat:.0f}ms" if lat else "-")

    if hs.get("last_error"):
        st.error(f"最近错误: {hs['last_error']}")

    if recent_checks:
        with st.expander("📋 最近查询记录", expanded=False):
            import pandas as pd
            checks_df = pd.DataFrame(reversed(recent_checks))
            display_cols = ["checked_at", "symbol", "market", "success", "latency_ms", "error"]
            show_cols = [c for c in display_cols if c in checks_df.columns]
            checks_df = checks_df[show_cols].head(30)
            checks_df = checks_df.rename(columns={
                "checked_at": "时间", "symbol": "代码", "market": "市场",
                "success": "成功", "latency_ms": "延迟(ms)", "error": "错误"
            })
            st.dataframe(checks_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  🔍 选股器
# ═══════════════════════════════════════════════════════════
elif page == "🔍 选股器":
    st.title("🔍 条件选股器")
    st.caption("组合条件, 扫描全市场")

    from src.recommend.screener import screen_market
    from src.alerts.models import CONDITION_TYPES as CT, AlertRule
    from src.data.stock_db import get_db

    SCOPE_MAP = {"自选股": "watchlist", "全部A股": "all_a",
                 "创业板(300)": "gem", "科创板(688)": "star",
                 "沪深主板": "main_board"}

    col1, col2 = st.columns([2, 1])
    with col1: scope = st.selectbox("选股范围", list(SCOPE_MAP.keys()), key="scr_scope")
    with col2: limit = st.slider("最多结果", 5, 100, 30, key="scr_limit")

    # 条件管理
    if "scr_conds" not in st.session_state:
        st.session_state.scr_conds = [{"cond": "golden_cross", "params": {"short": 20, "long": 60}, "logic": "AND"}]

    for idx, cdata in enumerate(st.session_state.scr_conds):
        cond = cdata["cond"]
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            new_cond = st.selectbox(f"条件{idx+1}", list(CT.keys()),
                                    index=list(CT.keys()).index(cond) if cond in CT else 0,
                                    format_func=lambda x: CT[x], key=f"scr_c_{idx}")
            cdata["cond"] = new_cond
        with c2:
            # 参数
            if new_cond in ("above_ma", "below_ma"):
                cdata["params"]["window"] = st.number_input("MA窗口", 5, 120, cdata["params"].get("window", 20), key=f"scp_w_{idx}")
            elif new_cond in ("golden_cross", "death_cross", "ma_cross_combo"):
                cdata["params"]["short"] = st.number_input("短期MA", 5, 50, cdata["params"].get("short", 20), key=f"scp_s_{idx}")
                cdata["params"]["long"] = st.number_input("长期MA", 10, 200, cdata["params"].get("long", 60), key=f"scp_l_{idx}")
            elif "rsi" in new_cond:
                cdata["params"]["window"] = st.number_input("RSI窗口", 5, 30, int(cdata["params"].get("window", 14)), key=f"scp_rw_{idx}")
                cdata["params"]["level"] = st.number_input("阈值", 10, 90, int(cdata["params"].get("level", 30)), key=f"scp_rl_{idx}")
            elif new_cond in ("bollinger_upper", "bollinger_lower", "bollinger_combo"):
                cdata["params"]["window"] = st.number_input("窗口", 10, 50, int(cdata["params"].get("window", 20)), key=f"scp_bw_{idx}")
                cdata["params"]["std"] = st.number_input("标准差", 1, 4, int(cdata["params"].get("std", 2)), key=f"scp_bs_{idx}")
            elif "volume" in new_cond:
                cdata["params"]["ratio"] = st.number_input("倍数", 1.5, 5.0, float(cdata["params"].get("ratio", 2.0)), key=f"scp_vr_{idx}")
            elif new_cond in ("above_price", "below_price"):
                cdata["params"]["threshold"] = st.number_input("价格", 0.0, 10000.0, float(cdata["params"].get("threshold", 100.0)), key=f"scp_pt_{idx}")
            elif new_cond == "alpha120":
                cdata["params"]["threshold"] = st.number_input("偏离", 0.001, 0.1, float(cdata["params"].get("threshold", 0.02)), 0.005, format="%.4f", key=f"scp_a120_{idx}")
            elif new_cond == "alpha006":
                cdata["params"]["threshold"] = st.number_input("相关", 0.1, 1.0, float(cdata["params"].get("threshold", 0.3)), 0.1, key=f"scp_a006_{idx}")
            elif new_cond == "alpha053":
                cdata["params"]["threshold_up"] = st.number_input("涨阈值", 1.01, 1.2, float(cdata["params"].get("threshold_up", 1.05)), 0.01, key=f"scp_a53u_{idx}")
                cdata["params"]["threshold_dn"] = st.number_input("跌阈值", 0.8, 0.99, float(cdata["params"].get("threshold_dn", 0.95)), 0.01, key=f"scp_a53d_{idx}")
            elif new_cond == "alpha015":
                cdata["params"]["window"] = st.number_input("窗口", 10, 60, int(cdata["params"].get("window", 20)), key=f"scp_a15w_{idx}")
                cdata["params"]["threshold"] = st.number_input("阈值", 0.1, 0.8, float(cdata["params"].get("threshold", 0.3)), 0.05, key=f"scp_a15t_{idx}")
            else:
                st.caption("无额外参数")
        with c3:
            if len(st.session_state.scr_conds) > 1:
                cdata["logic"] = st.selectbox("逻辑", ["AND", "OR"], index=0 if cdata.get("logic") == "AND" else 1, key=f"scr_lg_{idx}")

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        if len(st.session_state.scr_conds) < 4 and st.button("➕ 添加条件", use_container_width=True):
            st.session_state.scr_conds.append({"cond": "above_ma", "params": {"window": 20}, "logic": "AND"})
            st.rerun()
    with cbtn2:
        if len(st.session_state.scr_conds) > 1 and st.button("➖ 删除最后", use_container_width=True):
            st.session_state.scr_conds.pop()
            st.rerun()

    if st.button("🔍 开始扫描", type="primary", use_container_width=True, key="scr_scan"):
        scope_type = SCOPE_MAP[scope]
        if scope_type == "watchlist":
            scan_list = [k.split("-", 1)[1] for k in st.session_state.watchlist if k.startswith("A-")]
        else:
            db = get_db()
            all_a = [code for code, _ in db.all_stocks("A")]
            if scope_type == "all_a":     scan_list = all_a[:500]
            elif scope_type == "gem":     scan_list = [c for c in all_a if c.startswith("300")]
            elif scope_type == "star":    scan_list = [c for c in all_a if c.startswith("688")]
            elif scope_type == "main_board": scan_list = [c for c in all_a if c.startswith(("600","000","002"))]
            else: scan_list = all_a[:500]

        # 逐个条件扫描, 然后组合
        all_hits = {}
        for cdata in st.session_state.scr_conds:
            hits = screen_market(cdata["cond"], cdata["params"], "A", scan_list, limit * 5)
            for h in hits:
                key = f"{h.market}-{h.symbol}"
                if key not in all_hits:
                    all_hits[key] = {"hit": h, "conds": []}
                all_hits[key]["conds"].append(cdata["cond"])

        # 应用 AND/OR 逻辑
        final = []
        for key, val in all_hits.items():
            and_conditions = [c for c in st.session_state.scr_conds if c["logic"] == "AND"]
            or_conditions = [c for c in st.session_state.scr_conds if c["logic"] == "OR"]
            and_ok = all(c["cond"] in val["conds"] for c in and_conditions) if and_conditions else True
            or_ok = any(c["cond"] in val["conds"] for c in or_conditions) if or_conditions else True
            if and_ok and or_ok:
                final.append(val["hit"])
                if len(final) >= limit:
                    break

        hits = final

        if hits:
            st.success(f"找到 {len(hits)} 只符合条件的股票")
            cols = st.columns(3)
            for i, h in enumerate(hits):
                with cols[i % 3]:
                    st.metric(f"{h.direction} {h.symbol} {h.name}",
                              f"{h.current_price:.2f}", h.message[:30])
                    key = f"{h.market}-{h.symbol}"
                    if key not in st.session_state.watchlist:
                        if st.button(f"➕ 加到自选", key=f"scr_add_{h.symbol}",
                                     use_container_width=True):
                            _add_to_watchlist(h.symbol, h.market, h.name)
                            st.rerun()
        else:
            st.warning("未找到符合条件的股票")

# ═══════════════════════════════════════════════════════════
#  💰 模拟交易
# ═══════════════════════════════════════════════════════════
elif page == "💰 模拟交易":
    st.title("💰 模拟交易")

    if "paper_account" not in st.session_state:
        st.session_state.paper_account = {"cash": 100000.0, "positions": {}}
        st.session_state.paper_history = []

    acc = st.session_state.paper_account
    hist = st.session_state.paper_history

    c1, c2, c3 = st.columns(3)
    c1.metric("现金", f"{acc['cash']:.2f}")
    pos_value = sum(float(v.get("shares", 0)) * float(v.get("price", 0))
                    for v in acc["positions"].values())
    c2.metric("持仓市值", f"{pos_value:.2f}")
    c3.metric("总资产", f"{acc['cash'] + pos_value:.2f}")

    st.divider()
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        keys = st.session_state.watchlist or list(PRESET_STOCKS.keys())
        sym = st.selectbox("股票", keys,
                           format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
                           key="paper_sym")  # line 870
    with col2:
        action = st.selectbox("操作", ["买入", "卖出"], key="paper_act")
    with col3:
        shares = st.number_input("数量(股)", 1, 100000, 100, key="paper_sh")
    with col4:
        if st.button("✅ 执行", type="primary", use_container_width=True):
            info = info_for(sym)
            df = get_data_for(info["symbol"], info["market"], info["name"])
            price = float(df["Close"].iloc[-1])
            key = sym
            if action == "买入":
                cost = price * shares * 1.0003
                if cost <= acc["cash"]:
                    acc["cash"] -= cost
                    pos = acc["positions"].get(key, {"shares": 0, "price": 0, "name": info["name"]})
                    old_val = pos["shares"] * pos["price"]
                    pos["shares"] += shares
                    pos["price"] = (old_val + price * shares) / pos["shares"]
                    pos["name"] = info["name"]
                    acc["positions"][key] = pos
                    hist.append(f"📅 买入 {info['name']} {shares}股 @{price:.2f}")
                    st.success(f"买入 {info['name']} {shares}股 @{price:.2f}")
                else:
                    st.error("资金不足")
            else:
                pos = acc["positions"].get(key)
                if pos and pos["shares"] >= shares:
                    gross = price * shares
                    fee = gross * 0.0003
                    stamp_tax = gross * 0.001
                    net = gross - fee - stamp_tax
                    pnl = (price - pos["price"]) * shares - fee - stamp_tax
                    acc["cash"] += net
                    pos["shares"] -= shares
                    if pos["shares"] == 0:
                        del acc["positions"][key]
                    else:
                        acc["positions"][key] = pos
                    hist.append(f"📅 卖出 {info['name']} {shares}股 @{price:.2f} 盈亏{pnl:+.2f}")
                    st.success(f"卖出 {info['name']} {shares}股 @{price:.2f} 盈亏{pnl:+.2f}")
                else:
                    st.error("持仓不足")
            st.rerun()

    if acc["positions"]:
        st.divider()
        st.subheader("📋 当前持仓")
        pos_rows = []
        for key, pos in acc["positions"].items():
            info = info_for(key)
            df = get_data_for(info["symbol"], info["market"], info["name"])
            cur_price = float(df["Close"].iloc[-1])
            pnl = (cur_price - pos["price"]) * pos["shares"]
            pos_rows.append({"股票": pos["name"], "成本价": f"{pos['price']:.2f}",
                             "现价": f"{cur_price:.2f}", "股数": pos["shares"],
                             "盈亏": f"{pnl:+.2f}",
                             "收益率": f"{(cur_price/pos['price']-1)*100:+.1f}%"})
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

    if hist:
        with st.expander("📜 交易记录"):
            for h in reversed(hist[-20:]):
                st.write(h)

# ═══════════════════════════════════════════════════════════
#  📊 因子库
# ═══════════════════════════════════════════════════════════
elif page == "📊 因子库":
    st.title("📊 因子库")

    # Load factor definitions
    import json
    factor_file = Path(__file__).resolve().parent.parent / "data" / "factors.json"
    if factor_file.exists():
        factor_data = json.loads(factor_file.read_text())
    else:
        factor_data = {"factors": [], "models": [], "strategies": []}

    # ── Tab: 因子列表 ──
    tab1, tab2, tab3 = st.tabs(["🧬 因子列表", "🤖 模型", "✅ 数据源验证"])

    with tab1:
        types = sorted(set(f["type"] for f in factor_data["factors"]))
        sel_type = st.selectbox("因子类型", ["全部"] + types, key="factor_type")
        filtered = factor_data["factors"]
        if sel_type != "全部":
            filtered = [f for f in filtered if f["type"] == sel_type]

        rows = []
        for f in filtered:
            bias = "✅" if not f.get("forward_bias") else "⚠️"
            rows.append({
                "ID": f["id"], "因子": f["name"], "类型": f["type"],
                "描述": f["desc"], "回溯窗口": f["lookback"], "未来泄漏": bias
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={"描述": st.column_config.TextColumn(width="large")})
        st.caption(f"共 {len(filtered)} 个因子 · 全部 ✅ 无未来函数泄漏 · 目标变量已对齐 shift+1")

    with tab2:
        st.subheader("预测模型")
        for m in factor_data["models"]:
            badge = "🟢 轻量" if m.get("light") else "🟡 重模型(PyTorch)"
            st.markdown(f"**{m['name']}** {badge} — {m['desc']}")

        st.subheader("回测策略")
        for s in factor_data["strategies"]:
            st.markdown(f"- **{s['name']}**: {s['desc']}")

    with tab3:
        st.subheader("📡 数据源连通性验证")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 验证全部", use_container_width=True, type="primary"):
                with st.spinner("检测中..."):
                    results = {}
                    # Tushare
                    try:
                        from src.utils.config import get_tushare_token
                        import tushare as ts
                        pro = ts.pro_api(get_tushare_token())
                        df = pro.daily(ts_code="000001.SZ", start_date="20260101", end_date="20260105")
                        results["Tushare"] = ("✅", f"{len(df)}行" if not df.empty else "空")
                    except Exception as e:
                        results["Tushare"] = ("❌", str(e)[:60])

                    # Wind MCP
                    try:
                        from src.data.sources import WIND_SOURCE
                        r = WIND_SOURCE.run_historical("000001", period_days=5, market="A")
                        results["Wind MCP"] = ("✅", "可用") if r.success else ("❌", r.error[:60])
                    except Exception as e:
                        results["Wind MCP"] = ("❌", str(e)[:60])

                    # Yahoo
                    try:
                        from src.data.sources import YahooSource
                        s = YahooSource()
                        r = s.run_historical("AAPL", period_days=5, market="US")
                        results["Yahoo Finance"] = ("✅", "可用") if r.success else ("❌", r.error[:60])
                    except Exception as e:
                        results["Yahoo Finance"] = ("❌", str(e)[:60])

                    # Finnhub
                    try:
                        from src.data.sources import FinnhubSource
                        s = FinnhubSource()
                        p = s.fetch_realtime("AAPL", "US")
                        results["Finnhub"] = ("✅", f"${p}") if p else ("❌", "无数据")
                    except Exception as e:
                        results["Finnhub"] = ("❌", str(e)[:60])

                    # News Sentiment
                    try:
                        from src.data.news_fetcher import compute_sentiment
                        s1 = compute_sentiment("revenue growth beats expectations", "US")
                        s2 = compute_sentiment("业绩超预期营收大幅增长", "A")
                        ok = abs(s1) > 0.01 or abs(s2) > 0.01
                        results["新闻情绪"] = ("✅", f"EN={s1:.2f} CN={s2:.2f}") if ok else ("⚠️", "分数=0, 需安装vaderSentiment/snownlp")
                    except Exception as e:
                        results["新闻情绪"] = ("❌", str(e)[:60])

                    # News fetch
                    try:
                        from src.data.news_fetcher import fetch_finnhub_news, fetch_sina_news
                        fn = fetch_finnhub_news("AAPL", "US", lookback_days=3)
                        sn = fetch_sina_news("002025", lookback_days=3)
                        results["Finnhub新闻"] = ("✅", f"{len(fn)}条") if not fn.empty else ("⚠️", "空")
                        results["新浪新闻"] = ("✅", f"{len(sn)}条") if not sn.empty else ("⚠️", "空")
                    except Exception as e:
                        results["Finnhub新闻"] = ("❌", str(e)[:60])

                    st.session_state.factor_check = results

        if "factor_check" in st.session_state:
            for name, (status, detail) in st.session_state.factor_check.items():
                st.markdown(f"{status} **{name}**: {detail}")

        st.divider()
        st.caption("消息面数据存储在 data/news_data.parquet，自动增量更新")


# ═══════════════════════════════════════════════════════════
#  🧠 策略推荐
# ═══════════════════════════════════════════════════════════
elif page == "🧠 策略推荐":
    st.title("🧠 策略推荐")
    tab_rec_new, tab_rec_hist = st.tabs(["🔍 新扫描", "📜 历史"])

    from src.recommend.engine import (scan_predictions, scan_strategies,
                                       generate_report)
    from src.recommend.advisor import analyze_with_llm
    from src.utils.config import get_llm_key
    from src.alerts import AlertRule, add_rule, CONDITION_TYPES as CT
    from src.alerts.models import CONDITION_TYPES

    # ══════════════════════════════════════════════════════
    #  新扫描
    # ══════════════════════════════════════════════════════
    with tab_rec_new:
        # 批量预测跳转过来的
        if st.session_state.get("rec_scan_target"):
            bt = st.session_state.pop("rec_scan_target")
            st.success(f"来自批量预测: {bt['symbol']} {bt['name']} (看涨{bt['up_count']})  →  正在自动扫描...")
            # 预选股票
            if bt["key"] in st.session_state.watchlist:
                st.session_state.rec_target = bt["key"]
            # 直接触发扫描
            st.session_state.rec_auto_scan = True
        else:
            target_preselected = None

        # 初始化 session_state
        if "rec_result" not in st.session_state:
            st.session_state.rec_result = None
        if "batch_summary" not in st.session_state:
            st.session_state.batch_summary = []

        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            rec_grp = st.selectbox("分组",
                ["全部"] + st.session_state.group_order,
                key="rec_grp_sel")
        with col2:
            keys = st.session_state.watchlist or list(PRESET_STOCKS.keys())
            if rec_grp != "全部":
                filtered = [k for k in keys if st.session_state.stock_groups.get(k, "默认") == rec_grp]
            else:
                filtered = list(keys)
            target = st.selectbox("选择股票", filtered,
                                  format_func=lambda x: f"{info_for(x)['symbol']} {info_for(x)['name']}",
                                  key="rec_target")
        with col2:
            pred_steps = st.slider("预测天数", 10, 90, 30, 5, key="rec_steps",
                                   help="预测模型预测到未来多少天")
        with col3:
            capital = st.number_input("回测资金", 10000, 1_000_000, 100_000, step=10000, key="rec_cap")

        run = st.button("🔍 全面扫描 + AI 分析", type="primary",
                        use_container_width=True, key="rec_scan")
        if st.session_state.get("rec_auto_scan"):
            run = True
            st.session_state.rec_auto_scan = False

        # 如果没有新扫描且没有缓存结果 → 显示提示
        if not run and st.session_state.rec_result is None:
            st.info("选好股票和预测周期, 点击扫描")
        else:

            # ── 执行扫描 ──
            if run:
                import gc
                info = info_for(target)
                from src.data.fetcher import fetch_data
                df = fetch_data(info["symbol"], info["market"], period_days=250)
                cutoff = pd.Timestamp.now() - timedelta(days=540)
                df_full = df.copy()
                # 统一时区: 去掉时区信息避免比较失败
                if hasattr(df["Date"].dtype, "tz") and df["Date"].dtype.tz is not None:
                    df["Date"] = df["Date"].dt.tz_localize(None)
                df = df[df["Date"] >= cutoff]
                if df.empty or len(df) < 60:
                    df = df_full
                if df.empty:
                    st.error("数据不足，无法扫描")
                with st.spinner("运行全部预测模型 + 全部回测策略..."):
                    source = _detect_source_name(info["symbol"], info["market"])
                    models = scan_predictions(df, steps=pred_steps, data_source=source)
                    gc.collect()
                    strategies = scan_strategies(df, info["symbol"], info["market"], capital)
                    gc.collect()
                    risk = calc_all_risk_metrics(df)
                    cur_price = float(df["Close"].iloc[-1])
                    report = generate_report(info["name"], info["symbol"], info["market"],
                                              models, strategies, cur_price, risk, pred_steps)
                # 持久化
                valid_m = [m for m in models if not m.error]
                up = sum(1 for m in valid_m if m.pct_change > 0) if valid_m else 0
                avg_pct = np.mean([m.pct_change for m in valid_m]) if valid_m else 0
                best_s = None
                for s in strategies:
                    if not s.error and (not best_s or s.total_return > best_s.total_return):
                        best_s = s
                st.session_state.rec_result = {
                    "symbol": info["symbol"], "market": info["market"], "name": info["name"],
                    "cur_price": cur_price, "report": report,
                    "up": up, "avg_pct": avg_pct,
                    "best_strategy": best_s.strategy if best_s else "-",
                    "best_ret": best_s.total_return if best_s else 0,
                    "best_sharpe": best_s.sharpe if best_s else 0,
                    "best_maxdd": best_s.max_dd if best_s else 0,
                    "pred_steps": pred_steps,
                }
                # 保存历史
                from src.data.rec_history import add_rec_history
                add_rec_history(info["symbol"], info["market"], info["name"],
                                cur_price, f"{up}/{len(valid_m)} 看涨", avg_pct,
                                best_s.strategy if best_s else "-",
                                best_s.total_return if best_s else 0,
                                best_s.sharpe if best_s else 0,
                                best_s.max_dd if best_s else 0,
                                len(models), len(strategies),
                                report=report,
                                models_data=[{"model": m.model, "direction": m.direction,
                                              "final_price": m.final_price if not m.error else 0,
                                              "pct_change": m.pct_change if not m.error else 0,
                                              "mape": m.mape if not m.error else 0,
                                              "error": m.error} for m in models],
                                strategies_data=[{"strategy": s.strategy,
                                                  "total_return": s.total_return,
                                                  "sharpe": s.sharpe, "max_dd": s.max_dd,
                                                  "win_rate": s.win_rate, "total_trades": s.total_trades,
                                                  "error": s.error} for s in strategies])
                st.rerun()
    
            # ── 显示结果 (从历史记录 + session_state 指针) ──
            res = st.session_state.rec_result
            if res is not None:
                from src.data.rec_history import load_rec_history
                hist = load_rec_history()
                h = None
                for r in reversed(hist):
                    if r.symbol == res["symbol"] and r.market == res["market"]:
                        h = r
                        break
                if h is None:
                    st.warning("历史记录未找到，请重新扫描")
                    st.session_state.rec_result = None
                else:
                    st.subheader("🔮 多模型预测共识", help="各模型对未来走势的预测方向与价格")
                    cols = st.columns(3)
                    cols[0].metric("当前价", f"{h.current_price:.2f}")
                    cols[1].metric("模型共识", h.model_consensus)
                    cols[2].metric("平均预测涨跌", f"{res['avg_pct']:+.1f}%")
                    m_rows = [{"模型": m.get("model",""),
                               "方向": m.get("direction","") if not m.get("error") else f"❌ {m.get('error','')[:60]}",
                               "预测末价": f"{m.get('final_price',0):.2f}" if not m.get("error") else "-",
                               "涨跌幅": f"{m.get('pct_change',0):+.1f}%" if not m.get("error") else "-",
                               "MAPE": f"{m.get('mape',0):.1f}%" if not m.get("error") else "-"}
                              for m in h.models_data]
                    st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

                    with st.expander("🔧 模型参数详情", expanded=False):
                        for m in h.models_data:
                            if m.get("error"):
                                st.caption(f"**{m.get('model','')}**: ❌ {m.get('error')}")
                                continue
                            st.caption(f"**{m.get('model','')}**")
                            st.write(f"数据源: {res.get('name','')} | 预测天数: {res.get('pred_steps',30)}")

                    # ── 回测 ──
                    st.subheader("📈 策略回测对比", help="各策略在历史数据上的回测绩效")
                    s_rows = []
                    for s in h.strategies_data:
                        if s.get("error"):
                            s_rows.append({"策略": s.get("strategy",""), "状态": "❌"})
                        else:
                            s_rows.append({"策略": s.get("strategy",""),
                                           "收益": f"{s.get('total_return',0)*100:+.1f}%",
                                           "夏普": f"{s.get('sharpe',0):.2f}",
                                           "回撤": f"{s.get('max_dd',0)*100:.1f}%",
                                           "胜率": f"{s.get('win_rate',0)*100:.0f}%",
                                           "交易": s.get("total_trades",0)})
                    st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

                    # ── 风控 ──
                    with st.expander("📊 风控指标"):
                        st.markdown(h.report.split("### 风控指标")[1].split("###")[0] if "### 风控指标" in h.report else "暂无风控数据")

                    # ── 自动生成交易监控 ──
                    st.divider()
                    st.subheader("⚡ 添加到交易监控", help="一条组合策略=一条监控规则, 同时覆盖买入和卖出信号")

                    STRAT_TO_COND = {
                        "双均线(5/20)":  ("ma_cross_combo", {"short": 5, "long": 20}),
                        "双均线(10/30)": ("ma_cross_combo", {"short": 10, "long": 30}),
                        "双均线(20/60)": ("ma_cross_combo", {"short": 20, "long": 60}),
                        "RSI(14)":       ("rsi_combo", {"window": 14, "oversold": 30, "overbought": 70}),
                        "通道突破(20/10)":("volume_breakout", {"lookback": 20, "vol_ratio": 2.0}),
                        "布林带(20/2)":  ("bollinger_combo", {"window": 20, "std": 2}),
                        "滚动预测(月频)": ("ma_cross_combo", {"short": 20, "long": 60}),
                        "滚动预测(周频)": ("ma_cross_combo", {"short": 5, "long": 20}),
                    }

                    best_strategy = res.get("best_strategy", "")
                    best_ret = res.get("best_ret", 0)
                    best_sharpe = res.get("best_sharpe", 0)
                    mapped = STRAT_TO_COND.get(best_strategy)
                    if mapped and best_ret > 0:
                        cond, params = mapped
                        desc = CONDITION_TYPES.get(cond, cond)
                        st.info(f"推荐策略 **{best_strategy}** → 组合条件 **{cond}** ({desc})")

                        if st.button("✅ 一键添加到交易监控", use_container_width=True, type="primary",
                                     key="rec_add_alert"):
                            add_rule(AlertRule(
                                symbol=res["symbol"], market=res["market"],
                                condition=cond, params=params,
                                label=f"推荐策略: {best_strategy}",
                            ))
                            st.success(f"已添加监控规则: {desc} ({res['symbol']})")
                            st.session_state.page = "🔔 交易监控"
                            st.rerun()

                    else:
                        suggest_conds = [("above_ma", "上穿20日均线", {"window": 20})]
                        selected = []
                        for cond, desc, params in suggest_conds:
                            if st.checkbox(desc, True, key=f"as_{cond}", help=CONDITION_TIPS.get(cond, "")):
                                selected.append((cond, params))
                        if selected and st.button("✅ 添加选中条件", use_container_width=True, type="primary"):
                            for cond, params in selected:
                                add_rule(AlertRule(
                                    symbol=res["symbol"], market=res["market"],
                                    condition=cond, params=params,
                                    label=f"推荐: {res['name']}",
                                ))
                            st.success(f"已添加 {len(selected)} 条监控规则")
                            st.session_state.page = "🔔 交易监控"
                            st.rerun()

                    # ── AI 分析 ──
                    if get_llm_key():
                        with st.expander("🤖 AI 综合分析 (三维度)", expanded=False):
                            ai_result = analyze_with_llm(h.report)
                            if ai_result:
                                st.success(ai_result)
                            else:
                                st.warning("AI 调用失败, 请检查 API Key 和网络")

            # ══════════════════════════════════════════════════════
            #  历史
            # ══════════════════════════════════════════════════════
    with tab_rec_hist:
        from src.data.rec_history import load_rec_history

        history = load_rec_history()
        if not history:
            st.info("暂无推荐历史")
        else:
            for h in reversed(history):
                with st.expander(f"{h.predicted_at} | {h.stock_name} ({h.market}:{h.symbol}) "
                                 f"— 共识{h.model_consensus} 最佳{h.best_strategy}"):
                    cols = st.columns(5)
                    cols[0].metric("当前价", f"{h.current_price:.2f}")
                    cols[1].metric("模型共识", h.model_consensus)
                    cols[2].metric("最佳策略", h.best_strategy)
                    cols[3].metric("最佳收益", f"{h.best_ret*100:+.1f}%")
                    cols[4].metric("最佳夏普", f"{h.best_sharpe:.2f}")

                    if h.models_data:
                        st.caption("🔮 模型预测")
                        m_rows_h = [{"模型": m.get("model",""),
                                     "方向": m.get("direction","") if not m.get("error") else f"❌ {m.get('error','')[:60]}",
                                     "预测末价": f"{m.get('final_price',0):.2f}" if not m.get("error") else "-",
                                     "涨跌幅": f"{m.get('pct_change',0):+.1f}%" if not m.get("error") else "-",
                                     "MAPE": f"{m.get('mape',0):.1f}%" if not m.get("error") else "-"}
                                    for m in h.models_data]
                        st.dataframe(pd.DataFrame(m_rows_h), use_container_width=True, hide_index=True)

                    if h.strategies_data:
                        st.caption("📈 策略对比")
                        sdf = pd.DataFrame([{
                            "策略": s.get("strategy",""), "收益": f"{s.get('total_return',0)*100:+.1f}%",
                            "夏普": f"{s.get('sharpe',0):.2f}", "回撤": f"{s.get('max_dd',0)*100:.1f}%",
                            "胜率": f"{s.get('win_rate',0)*100:.0f}%", "交易": s.get("total_trades",0),
                            "状态": "❌" if s.get("error") else "✅",
                        } for s in h.strategies_data])
                        st.dataframe(sdf, use_container_width=True, hide_index=True)

                    if h.report:
                        with st.expander("📋 完整报告"):
                            st.markdown(h.report)
                    else:
                        with st.expander("📋 扫描摘要"):
                            st.write(f"**共识**: {h.model_consensus}")
                            st.write(f"**最佳策略**: {h.best_strategy} (收益{h.best_ret*100:+.1f}%, 夏普{h.best_sharpe:.2f})")
                            st.write(f"**模型数**: {h.total_models}, **策略数**: {h.total_strats}")

# ═══════════════════════════════════════════════════════════
#  ℹ️ 自选详情
# ═══════════════════════════════════════════════════════════
elif page == "ℹ️ 自选详情":
    key = st.session_state.get("selected_stock")
    if not key or key not in st.session_state.watchlist:
        st.warning("请在左侧自选列表点击一只股票查看详情")
        st.session_state.selected_stock = None
        st.stop()

    parts = key.split("-", 1)
    if len(parts) != 2:
        st.error("无效的股票")
        st.stop()
    market, symbol = parts
    name = st.session_state.stock_names.get(key, symbol)
    st.title(f"ℹ️ {name} ({market}:{symbol})")

    # Wind 全量数据
    from src.data.wind_source import a_stock_to_windcode, get_stock_full
    from src.data.stock_info import get_recent_performance

    wc = a_stock_to_windcode(symbol) if market == "A" else symbol
    with st.spinner("加载 Wind 数据..."):
        wd = get_stock_full(wc) if market == "A" else {}

    def _w(key, default="-"):
        v = wd.get(key, "")
        return v if v else default

    def _fmt(v, unit="auto"):
        """格式化大数字: 1.55e+12 → 1.55万亿"""
        if not v or v == "-":
            return "-"
        try:
            n = float(v)
        except (ValueError, TypeError):
            return str(v)
        if unit == "vol":
            if n >= 1e8: return f"{n/1e8:.2f}亿"
            if n >= 1e4: return f"{n/1e4:.0f}万"
            return f"{n:.0f}"
        if unit == "cap":
            if n >= 1e12: return f"{n/1e12:.2f}万亿"
            if n >= 1e8: return f"{n/1e8:.2f}亿"
            return f"{n:.0f}"
        return str(v)

    # ═══ 交易数据 ═══
    st.subheader("📊 交易数据")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("最新价", _w("price"))
    with col2: st.metric("涨跌幅", f"{_w('change_pct')}%" if wd.get("change_pct") else "-")
    with col3: st.metric("涨跌额", _w("change_amt"))
    with col4: st.metric("换手率", f"{_w('turnover')}%" if wd.get("turnover") else "-")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("开盘", _w("open"))
    with col2: st.metric("最高", _w("high"))
    with col3: st.metric("最低", _w("low"))
    with col4: st.metric("昨收", _w("pre_close"))

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("成交量", _fmt(_w("volume"), "vol"))
    with col2: st.metric("成交额", _fmt(_w("amount"), "vol"))
    with col3: st.metric("52周最高", _w("high_52w"))
    with col4: st.metric("52周最低", _w("low_52w"))

    # ═══ 基本面 ═══
    st.divider()
    st.subheader("💰 基本面")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("总市值", _fmt(_w("market_cap"), "cap"))
    with col2: st.metric("市盈率(TTM)", _w("pe_ttm"))
    with col3: st.metric("市盈率(LYR)", _w("pe_lyr"))
    with col4: st.metric("市净率", _w("pb"))

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("股息率", f"{_w('dividend_yield')}%" if wd.get("dividend_yield") else "-")
    with col2: st.metric("ROE", f"{_w('roe')}%" if wd.get("roe") else "-")
    with col3: st.metric("毛利率", f"{_w('gross_margin')}%" if wd.get("gross_margin") else "-")
    with col4: st.metric("资产负债率", f"{_w('debt_ratio')}%" if wd.get("debt_ratio") else "-")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("营业收入", _fmt(_w("revenue"), "cap"))
    with col2: st.metric("净利润", _fmt(_w("net_profit"), "cap"))
    with col3: st.metric("行业", _w("industry"))
    with col4: st.metric("市场", market)

    # ═══ 走势图 ═══
    st.divider()
    df = get_data_for(symbol, market)
    if df is not None and not df.empty:
        st.subheader("📈 近期走势")
        perf = get_recent_performance(df)
        cols = st.columns(len(perf))
        for col, (k, v) in zip(cols, perf.items()):
            col.metric(k, v)
        from src.data.charting import kline_chart
        fig = kline_chart(df, indicators=["ma", "macd"], height=350, template=plotly_template())
        st.plotly_chart(fig, use_container_width=True)

    # ═══ 最新消息 ═══
    st.divider()
    st.subheader("📰 最新消息")
    wind_news = wd.get("news", [])
    if wind_news:
        for n in wind_news[:8]:
            title = n.get("title", "") or n.get("标题", "")
            t = n.get("time", "") or n.get("时间", "")
            src = n.get("source", "") or n.get("来源", "")
            st.markdown(f"- {title}  *{t}*")
    else:
        try:
            from src.data.stock_info import get_news
            news = get_news(symbol, market, 8)
            if news:
                for n in news:
                    st.markdown(f"- **{n['title']}** ({n['time']})")
            else:
                st.info("暂无最新消息")
        except Exception:
            st.info("暂无最新消息")

    # ═══ 已设策略 ═══
    st.divider()
    st.subheader("⚙️ 交易策略")
    from src.alerts import load_rules, CONDITION_TYPES as CT
    rules = [r for r in load_rules() if r.symbol == symbol and r.market == market]
    if rules:
        for r in rules:
            desc = CT.get(r.condition, r.condition)
            extra = ", ".join(f"{k}={v}" for k, v in r.params.items())
            st.write(f"{'🟢' if r.enabled else '🔴'} **{desc}** | {extra} | {r.label}")
    else:
        st.info("未设置交易策略")

    if st.button("← 返回仪表盘"):
        st.session_state.page = "🏠 仪表盘"
        st.rerun()
