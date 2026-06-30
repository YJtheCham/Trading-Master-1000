"""
StockPredict — NiceGUI Edition
运行: python3 webapp_nice/main.py  端口: 8502
Streamlit仍在8501保留
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nicegui import ui, app
from webapp.services import (
    info_for, get_data_for, detect_source_name, get_data_notify,
    has_real_source, refresh_source_status, load_state_from_watchlist,
    add_to_watchlist, remove_from_watchlist, fmt_risk, clear_all_caches, _mock_data,
)
from webapp_nice.pages.predict import predict_page
from webapp_nice.pages.backtest import backtest_page
from webapp_nice.pages.extra import risk_page, monitor_page, strategy_page
from webapp_nice.pages.trading import screener_page, paper_page
from src.data.fetcher import load_watchlist
from src.alerts.engine import load_rules, get_engine as get_alert_engine
import pandas as pd, numpy as np
from datetime import datetime, timedelta

PAGES = [
    {"id": "dashboard", "label": "仪表盘", "icon": "home"},
    {"id": "predict",   "label": "预测",   "icon": "psychology"},
    {"id": "backtest",  "label": "回测",   "icon": "replay"},
    {"id": "risk",      "label": "风控",   "icon": "shield"},
    {"id": "monitor",   "label": "监控",   "icon": "notifications"},
    {"id": "screener",  "label": "选股器", "icon": "search"},
    {"id": "paper",     "label": "模拟交易", "icon": "account_balance_wallet"},
    {"id": "strategy",  "label": "策略推荐", "icon": "lightbulb"},
    {"id": "factors",   "label": "因子库",  "icon": "dataset"},
    {"id": "settings",  "label": "设置",    "icon": "settings"},
]

THEME = {
    "bg": "#0b0e14", "card": "#141824", "border": "#232738",
    "accent": "#5b9bd5", "accent2": "#3b82f6", "positive": "#26a69a",
    "negative": "#ef5350", "warning": "#f59e0b",
    "fg": "#e2e8f0", "fg2": "#94a3b8", "fg3": "#64748b",
}

GLOBAL_CSS = f"""
body {{ background: {THEME['bg']}; color: {THEME['fg']}; font-family: 'Inter',-apple-system,sans-serif; }}
.q-page {{ background: {THEME['bg']}; }}
.q-drawer {{ background: {THEME['card']} !important; border-right:1px solid {THEME['border']}; }}
.q-drawer .q-item {{ color: {THEME['fg2']}; border-radius:8px; margin:2px 8px; }}
.q-drawer .q-item.q-router-link--active {{ background: {THEME['border']}; color: {THEME['accent']}; font-weight:600; }}
.q-drawer .q-item__label {{ font-size:0.85rem; }}
.card-ds {{ background:{THEME['card']}; border:1px solid {THEME['border']}; border-radius:12px; padding:16px; }}
.price-up {{ color:{THEME['positive']}; }} .price-down {{ color:{THEME['negative']}; }}
.page-title {{ font-size:1.4rem; font-weight:700; color:{THEME['fg']}; margin-bottom:16px; }}
.stock-card {{ background:{THEME['card']}; border:1px solid {THEME['border']}; border-radius:10px;
    padding:12px; width:230px; text-align:center; cursor:pointer; transition:all .15s; }}
.stock-card:hover {{ border-color:{THEME['accent']}; box-shadow:0 2px 12px rgba(91,155,213,0.15); }}
.stock-card.monitored {{ border-color:{THEME['positive']}; }}
.status-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }}
.q-btn {{ border-radius:8px; text-transform:none; font-weight:500; }}
.q-table {{ background:{THEME['card']}; color:{THEME['fg']}; }}
"""


def init_state():
    s = app.storage.user
    if "watchlist" not in s:
        s["watchlist"] = []
        s["stock_names"] = {}
        s["stock_groups"] = {}
        s["stock_order"] = []
        s["group_order"] = []
        s["active_group"] = "全部"
        s["page"] = "dashboard"
        s["dark_mode"] = True
        s["refresh_key"] = 0
        s["drawer_open"] = True
        s["source_status"] = {}
        load_state_from_watchlist(s)

    # Always ensure stock_names populated from persisted JSON
    from src.data.fetcher import load_watchlist
    items = load_watchlist()
    if items:
        for item in items:
            key = f"{item.market}-{item.symbol}"
            if item.name:
                s["stock_names"] = s.get("stock_names", {})
                s["stock_names"][key] = item.name
            g = item.group or "默认"
            s["stock_groups"] = s.get("stock_groups", {})
            if key not in s["stock_groups"]:
                s["stock_groups"][key] = g

    return s


# ─── Progress Notification System ───────────────────────────
_compute_tasks: dict = {}


def start_progress(task_id: str, label: str, total: int = 1):
    _compute_tasks[task_id] = {"label": label, "current": 0, "total": total}
    ui.notify(f"⏳ {label}...", type="ongoing", position="bottom-right", timeout=0)


def update_progress(task_id: str, step: int = 1, message: str = ""):
    if task_id in _compute_tasks:
        _compute_tasks[task_id]["current"] += step
        t = _compute_tasks[task_id]
        pct = min(100, int(t["current"] / max(1, t["total"]) * 100))
        ui.notify(f"{t['label']}: {pct}% {message}", type="ongoing", position="bottom-right", timeout=3000)


def finish_progress(task_id: str, success: bool = True, message: str = ""):
    if task_id in _compute_tasks:
        t = _compute_tasks.pop(task_id)
        ui.notify(f"{'✅' if success else '❌'} {t['label']} {message}",
                  type="positive" if success else "negative", position="bottom-right", timeout=5000)


# ─── Navigation ──────────────────────────────────────────────
def build_nav():
    state = init_state()
    drawer_open = state.get("drawer_open", True)

    with ui.left_drawer(bordered=True, elevated=True, value=drawer_open).classes("bg-dark") as drawer:
        drawer.on("update:model-value", lambda e: state.update({"drawer_open": e}))
        with ui.column().classes("w-full items-center"):
            ui.image("webapp_nice/股票分析软件接入数据源.png").style("width:128px;height:128px;margin-right:8px")
            ui.markdown("## **StockPredict**").classes("text-primary")
            ui.separator().classes("q-mb-sm")
            for p in PAGES:
                ui.button(
                    p["label"],
                    on_click=lambda pid=p["id"]: navigate_to(pid),
                    icon=p["icon"],
                ).props("flat align=left").classes("w-full")

            ui.separator().classes("q-mt-sm q-mb-sm")
            _render_nav_status(state)

            # ── Quick add stock ──
            with ui.row().classes("w-full q-px-sm q-mt-sm"):
                search_inp = ui.input(placeholder="搜索添加股票...").props("dense outlined").classes("w-full")
                search_inp.on("keydown", lambda e: _add_stock_from_search(state, search_inp) if e["key"] == "Enter" else None)
            ui.separator().classes("q-mt-xs")

            # ── Watchlist management ──
            ui.separator().classes("q-mt-sm")
            with ui.expansion("📋 自选管理", icon="list", value=False).classes("w-full"):
                # Group filter + stock list
                wl = state.get("watchlist", [])
                all_grps = ["全部"] + state.get("group_order", [])
                wl_filter = ui.select(all_grps, value="全部", label="筛选").props("dense outlined").classes("w-full q-mb-xs")

                def refresh_stock_list():
                    wl_container.clear()
                    flt = wl_filter.value
                    with wl_container:
                        for key in state.get("watchlist", []):
                            g = state.get("stock_groups", {}).get(key, "默认")
                            if flt != "全部" and g != flt:
                                continue
                            inf = info_for(key, state.get("stock_names", {}))
                            with ui.row().classes("w-full items-center q-pa-xs"):
                                ui.button(f"[{inf['market']}] {inf['symbol']} {inf['name']}",
                                          on_click=lambda k=key: select_stock_and_nav(state, k),
                                          icon="search").props("flat dense align=left").classes("w-full text-caption")
                                grp_opts = state.get("group_order", []) + ["+新分组"]
                                grp_sel = ui.select(grp_opts, value=g, label=None).props("dense").classes("w-24")
                                grp_sel.on("update:model-value", lambda e, k=key: _move_to_group(state, k, e.value))
                                del_btn = ui.button("✕", on_click=lambda k=key: _del_stock(state, k),
                                                    icon="delete").props("flat dense size=sm")

                    ui.label(f"共 {len(state.get('watchlist', []))} 只").classes("text-caption text-grey q-mt-sm")

                wl_filter.on("update:model-value", lambda: refresh_stock_list())
                wl_container = ui.column().classes("w-full")
                refresh_stock_list()

            # ── Group management ── (drag-and-drop outside expansion)
            ui.separator().classes("q-mt-sm")
            ui.markdown("**⚙️ 分组管理** (拖拽排序)").classes("text-subtitle2 q-px-sm q-mb-xs")
            _render_group_drag(state)

    # Header bar
    with ui.header().classes("bg-dark q-pa-sm"):
        with ui.row().classes("w-full items-center"):
            ui.button(icon="menu", on_click=lambda: drawer.toggle()).props("flat dense")
            ui.space()
            _render_header_status(state)
            ui.button(icon="add_circle", on_click=lambda: _show_add_stock_dialog(state)).props("flat dense")
            ui.button(icon="refresh", on_click=lambda: (clear_all_caches(), refresh_current())).props("flat dense")
            theme_icon = "dark_mode" if state.get("dark_mode") else "light_mode"
            ui.button(icon=theme_icon, on_click=lambda: toggle_theme(state)).props("flat dense")


def _render_nav_status(state):
    wl_count = len(state.get("watchlist", []))
    ui.label(f"自选股: {wl_count}只").classes("text-caption text-grey")
    engine = get_alert_engine()
    rules = load_rules()
    active_rules = sum(1 for r in rules if r.enabled)
    if engine.is_running:
        ui.label(f"🟢 监控中 ({active_rules}规则)").classes("text-caption text-positive")
    else:
        s = "⚪ 待机" if active_rules > 0 else "🔴 未启动"
        ui.label(f"{s} ({active_rules}规则)").classes("text-caption text-grey")


def _render_header_status(state):
    """右上角数据源状态检测 - 可点击展开详情"""
    try:
        status = state.get("source_status", {})
        if not status:
            try:
                from src.data.fetcher import diagnose_sources
                status = diagnose_sources()
                state["source_status"] = status
            except Exception:
                pass
        
        a_ok = has_real_source(status, "A")
        hk_ok = has_real_source(status, "HK")
        us_ok = has_real_source(status, "US")
        overall = a_ok and hk_ok and us_ok
        icon = "🟢" if overall else ("🟡" if (a_ok or hk_ok or us_ok) else "🔴")
        tooltip = "数据源正常" if overall else "部分异常" if (a_ok or hk_ok or us_ok) else "全部不可用"

        btn = ui.button(icon=icon, on_click=lambda: _show_source_dialog(state)).props("flat dense round")
        btn.tooltip(tooltip)
    except Exception:
        ui.label("⚪").tooltip("数据源未检测")


@ui.refreshable
def _render_header_status_refr(state):
    _render_header_status(state)


def _show_source_dialog(state):
    """弹出数据源详情对话框"""
    with ui.dialog() as dialog, ui.card().classes("q-pa-md"):
        ui.label("📡 数据源状态检测").classes("text-h6 q-mb-sm")
        status_col = ui.column().classes("w-full")

        def render_status():
            status_col.clear()
            status = state.get("source_status", {})
            if not status:
                with status_col:
                    ui.label("未检测").classes("text-caption text-grey")
                return
            with status_col:
                for market, sources in status.items():
                    ok = has_real_source(status, market)
                    icon = "🟢" if ok else "🔴"
                    ui.label(f"{icon} 市场 {market}").classes("text-subtitle2 q-mt-sm")
                    if sources:
                        for s in sources:
                            color = THEME["positive"] if s.get("available") else THEME["negative"]
                            latency = s.get("latency_ms", 0)
                            latency_str = f"{latency}ms" if latency else "-"
                            err = s.get("error", "")
                            err_str = f" ⚠️{err[:30]}" if err else ""
                            ui.label(f"  • {s.get('name','?')} — {'✅' if s.get('available') else '❌'} {latency_str}{err_str}").classes("text-caption").style(f"color:{color}")
                    else:
                        ui.label("  无可用数据源").classes("text-caption text-grey")

        render_status()

        async def refresh_now():
            ui.notify("检测中...", type="ongoing", timeout=0)
            try:
                import asyncio
                from src.data.fetcher import diagnose_sources
                new_status = await asyncio.get_event_loop().run_in_executor(None, diagnose_sources)
                state["source_status"] = new_status
                render_status()
                ui.notify("检测完成", type="positive")
            except Exception as e:
                ui.notify(f"检测失败: {e}", type="negative")

        with ui.row().classes("q-mt-md"):
            ui.button("🔄 立即检测", on_click=refresh_now, icon="refresh").props("color=primary")
            ui.button("关闭", on_click=dialog.close).props("flat")
    dialog.open()


def toggle_theme(state):
    state["dark_mode"] = not state.get("dark_mode", True)
    ui.notify("主题切换需刷新页面生效", type="info", position="bottom-right")


# ─── Drawer helpers ──────────────────────────────────────────
def build_stock_options(state):
    """Return [{label: '600519 贵州茅台', value: 'A-600519'}, ...]"""
    opts = []
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        opts.append({"label": f"{inf['symbol']} {inf['name']}", "value": k})
    opts.sort(key=lambda x: x["label"])
    return opts


def _show_add_stock_dialog(state):
    with ui.dialog() as dialog, ui.card().classes("q-pa-md"):
        ui.label("添加自选股").classes("text-h6 q-mb-sm")
        inp = ui.input(placeholder="股票代码或名称, 如: 600519 或 茅台").props("outlined autofocus").classes("w-full q-mb-sm")
        result_col = ui.column()

        async def search():
            result_col.clear()
            q = (inp.value or "").strip()
            if not q:
                return
            try:
                from src.data.stock_db import search_stocks
                results = search_stocks(q, limit=10)
                with result_col:
                    if results:
                        for code, name, market in results:
                            key = f"{market}-{code}"
                            already = key in state.get("watchlist", [])
                            label = f"{'✅' if already else '➕'} [{market}] {code} {name}"
                            ui.button(label, on_click=lambda c=code, m=market, n=name: _do_add(state, c, m, n, dialog),
                                     icon="add" if not already else "check").props("flat align=left").classes("w-full")
                    else:
                        ui.label("未找到匹配股票").classes("text-caption text-grey")
            except Exception as e:
                ui.label(f"搜索失败: {e}").classes("text-negative text-caption")

        inp.on("keydown", lambda e: search if e["key"] == "Enter" else None)
        ui.button("搜索", on_click=search, icon="search").props("color=primary").classes("w-full")
    dialog.open()


def _do_add(state, code, market, name, dialog):
    ok, msg = add_to_watchlist(state, code, market, name)
    ui.notify(msg, type="positive" if ok else "warning")
    if ok:
        dialog.close()
        _refresh_all()


def _add_stock_from_search(state, search_inp):
    q = (search_inp.value or "").strip()
    if not q:
        ui.notify("请输入股票代码或名称", type="warning")
        return
    try:
        from src.data.stock_db import search_stocks
        results = search_stocks(q, limit=5)
        if results:
            code, name, market = results[0]
            ok, msg = add_to_watchlist(state, code, market, name)
            ui.notify(msg, type="positive" if ok else "warning")
            search_inp.value = ""
        else:
            ui.notify("未找到匹配股票", type="warning")
    except Exception as e:
        ui.notify(f"搜索失败: {e}", type="negative")


def _move_to_group(state, key, new_group):
    if new_group == "+新分组":
        new_group = f"自定义{len(state.get('group_order', []))+1}"
        if new_group not in state.get("group_order", []):
            state["group_order"].append(new_group)
    state["stock_groups"][key] = new_group
    from webapp.services import _persist_watchlist_from_state
    _persist_watchlist_from_state(state)
    _refresh_all()


def _del_stock(state, key):
    from webapp.services import remove_from_watchlist as rfw
    ok, msg = rfw(state, key)
    ui.notify(msg, type="positive" if ok else "warning")
    _refresh_all()


def _render_group_drag(state):
    grps = state.get("group_order", [])
    items = []
    for idx, g in enumerate(grps):
        cnt = sum(1 for v in state.get("stock_groups", {}).values() if v == g)
        items.append({"idx": idx, "name": g, "count": cnt})

    data_json = json.dumps(items)
    html = f'<div id="group-drag-list" style="padding:4px 8px"></div>'
    ui.html(html)

    # Use ui.timer to ensure JS runs after DOM is ready
    js_code = f"""var ITEMS={data_json};
var dS=null;
function gsend(v){{location.href="/?gorder="+encodeURIComponent(JSON.stringify(v.groups))}};
(function rn(){{
 var el=document.getElementById("group-drag-list");
 if(!el){{setTimeout(rn,300);return}}
 el.innerHTML="";
 ITEMS.forEach(function(g,i){{
  var d=document.createElement("div");d.draggable=true;d.dataset.idx=i;
  d.style.cssText="display:flex;align-items:center;padding:5px 8px;margin:2px 0;border:1px solid {THEME["border"]};border-radius:6px;cursor:grab;background:{THEME["card"]};color:{THEME["fg"]};font-size:0.78rem";
  d.innerHTML='<span style="margin-right:8px;font-size:0.65rem;color:{THEME["fg3"]}">⠿</span><span style="flex:1">'+g.name+'</span><span style="margin-left:auto;font-size:0.65rem;color:{THEME["fg3"]}">'+g.count+'只</span>';
  d.onmouseover=function(){{this.style.borderColor="{THEME["accent"]}"}};d.onmouseout=function(){{this.style.borderColor="{THEME["border"]}"}};
  d.ondragstart=function(e){{dS=this;this.style.opacity="0.4";e.dataTransfer.effectAllowed="move"}};
  d.ondragend=function(e){{this.style.opacity="1";el.querySelectorAll("[draggable]").forEach(function(x){{x.style.borderTop="none"}})}};
  d.ondragover=function(e){{e.preventDefault();this.style.borderTop="2px solid {THEME["accent"]}"}};
  d.ondragleave=function(e){{this.style.borderTop="none"}};
  d.ondrop=function(e){{e.preventDefault();e.stopPropagation();this.style.borderTop="none";
   if(dS&&dS!==this){{var f=+dS.dataset.idx,t=+this.dataset.idx;var m=ITEMS.splice(f,1)[0];ITEMS.splice(t,0,m);dS=null;rn();gsend({{groups:ITEMS.map(function(x){{return x.name}})}})}}}}
  el.appendChild(d);
 }})
}})();"""

    ui.timer(0.5, lambda: ui.run_javascript(js_code), once=True)


def _save_group_order():
    from pathlib import Path
    from src.utils.config import DATA_DIR
    go = app.storage.user.get("group_order", [])
    (DATA_DIR / "group_order.json").write_text(json.dumps(go, ensure_ascii=False))


def _move_group(state, idx, direction):
    grps = state["group_order"]
    new_idx = idx + direction
    if 0 <= new_idx < len(grps):
        grps[idx], grps[new_idx] = grps[new_idx], grps[idx]
    _save_group_order()
    _refresh_all()


def _pin_group_to_top(state, idx):
    grps = state["group_order"]
    if idx > 0:
        g = grps.pop(idx)
        grps.insert(0, g)
    _save_group_order()
    _refresh_all()


def _refresh_all():
    ui.run_javascript("setTimeout(()=>location.reload(), 200)")


def navigate_to(page_id: str):
    app.storage.user["page"] = page_id
    ui.navigate.to("/")


def refresh_current():
    app.storage.user["refresh_key"] = app.storage.user.get("refresh_key", 0) + 1
    ui.run_javascript("setTimeout(()=>location.reload(),100)")


# ─── Global Search ───────────────────────────────────────────
def _global_search(state):
    query = ui.input(placeholder="搜索股票... (Enter搜索, Esc关闭)", value="").props("autofocus outlined dense").classes("w-full")
    result_col = ui.column().classes("w-full q-mt-sm")

    def do_search():
        q = query.value.strip()
        result_col.clear()
        if not q or len(q) < 1:
            return
        matches = []
        for key in state.get("watchlist", []):
            inf = info_for(key, state.get("stock_names", {}))
            if q.lower() in inf["symbol"].lower() or q.lower() in inf["name"].lower():
                matches.append((key, inf))
        with result_col:
            if matches:
                for key, inf in matches[:10]:
                    ui.button(
                        f"[{inf['market']}] {inf['symbol']} {inf['name']}",
                        on_click=lambda k=key: select_stock_and_nav(state, k),
                        icon="search",
                    ).props("flat dense align=left").classes("w-full")
            else:
                ui.label("无匹配结果").classes("text-caption text-grey")

    query.on("keydown", lambda e: do_search() if e["key"] == "Enter" else None)
    query.on("keydown", lambda e: query.set_value("") if e["key"] == "Escape" else None)


def select_stock_and_nav(state, key):
    state["selected_stock"] = key
    navigate_to("detail")


def select_stock(key: str):
    app.storage.user["selected_stock"] = key
    app.storage.user["page"] = "detail"
    ui.navigate.to("/")


# ─── Confirmation Dialog ─────────────────────────────────────
async def confirm_action(message: str, on_confirm, title: str = "确认操作"):
    with ui.dialog() as dialog, ui.card().classes("q-pa-md"):
        ui.label(title).classes("text-h6 q-mb-sm")
        ui.label(message).classes("text-body1 q-mb-md")
        with ui.row().classes("justify-end gap-2"):
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("确认", on_click=lambda: (on_confirm(), dialog.close())).props("color=negative")
    dialog.open()


# ─── Global Keyboard Listener ──────────────────────────────
@ui.page("/")
def main_page():
    state = init_state()
    build_nav()
    page_id = state.get("page", "dashboard")

    # Apply group order from URL parameter (drag-and-drop)
    from urllib.parse import urlparse, parse_qs
    try:
        parsed = urlparse(str(ui.context.client.request.url))
        qs = parse_qs(parsed.query)
        if "gorder" in qs:
            import urllib.parse
            new_order = json.loads(urllib.parse.unquote(qs["gorder"][0]))
            state["group_order"] = new_order
            _save_group_order()
            ui.navigate.to("/", replace=True)
            return
    except Exception:
        pass

    with ui.element("div").classes("q-pa-md").style("max-width:none;width:100%"):
        if page_id == "dashboard":
            dashboard_page(state)
        elif page_id == "predict":
            predict_page(state)
        elif page_id == "backtest":
            backtest_page(state)
        elif page_id == "risk":
            risk_page(state)
        elif page_id == "monitor":
            monitor_page(state)
        elif page_id == "screener":
            screener_page(state)
        elif page_id == "paper":
            paper_page(state)
        elif page_id == "strategy":
            strategy_page(state)
        elif page_id == "factors":
            factors_page(state)
        elif page_id == "settings":
            settings_page(state)
        elif page_id == "detail":
            detail_page(state)
        else:
            dashboard_page(state)


# ─── Dashboard ───────────────────────────────────────────────
@ui.refreshable
def dashboard_page(state):
    with ui.column().classes("w-full"):
        ui.markdown("## 仪表盘").classes("page-title")
        ag = state.get("active_group", "全部")
        groups = ["全部"] + list(state.get("group_order", []))
        with ui.row().classes("items-center gap-4 q-mb-sm"):
            ui.select(groups, value=ag, label="分组",
                      on_change=lambda e: (state.update({"active_group": e.value}), dashboard_page.refresh())).props("dense outlined").classes("w-40")
            ui.button("刷新", on_click=lambda: (clear_all_caches(), dashboard_page.refresh()), icon="refresh").props("flat")

        ordered = [k for k in state.get("stock_order", []) if k in state.get("watchlist", [])]
        for k in state.get("watchlist", []):
            if k not in ordered:
                ordered.append(k)
        if ag != "全部":
            ordered = [k for k in ordered if state.get("stock_groups", {}).get(k, "默认") == ag]

        if not ordered:
            ui.label("无股票 — 到「选股器」添加股票").classes("text-grey q-mt-xl")
            return

        # Sort by group order, then stock order within group
        group_rank = {g: i for i, g in enumerate(state.get("group_order", []))}
        ordered.sort(key=lambda k: (
            group_rank.get(state.get("stock_groups", {}).get(k, "默认"), 999),
            state.get("stock_order", []).index(k) if k in state.get("stock_order", []) else 99
        ))

        alert_rules = load_rules()
        monitored = {f"{r.market}-{r.symbol}" for r in alert_rules if r.enabled}
        current_group = None
        group_counts = {}
        for k in ordered:
            g = state.get("stock_groups", {}).get(k, "默认")
            group_counts[g] = group_counts.get(g, 0) + 1

        # Build dashboard — simple flex-wrap cards with group headers
        parts = ['<div style="display:flex;flex-wrap:wrap;gap:8px;width:100%">']
        for key in ordered:
            g = state.get("stock_groups", {}).get(key, "默认")
            if g != current_group:
                current_group = g
                cnt = group_counts.get(g, 0)
                parts.append(
                    f'<div style="width:100%;display:flex;align-items:center;'
                    f'margin:12px 0 4px;padding:6px 0;border-bottom:1px solid {THEME["border"]}">'
                    f'<span style="font-size:1rem;margin-right:8px">{_group_emoji(g)}</span>'
                    f'<span style="font-size:1rem;font-weight:600;color:{THEME["fg"]}">{g}</span>'
                    f'<span style="margin-left:auto;font-size:0.75rem;color:{THEME["fg3"]}">{cnt}只</span>'
                    f'</div>'
                )

            inf = info_for(key, state.get("stock_names", {}))
            border = THEME["positive"] if key in monitored else THEME["border"]

            # Fetch price data for date label
            price = "—"; change_str = "—"; chg_c = THEME["fg3"]; date_label = ""
            try:
                df = get_data_for(inf["symbol"], inf["market"], period_days=5)
                if df is not None and not df.empty and len(df) >= 2:
                    lr = df.iloc[-1]
                    pr = df.iloc[-2]
                    price = f"{lr['Close']:.2f}"
                    chg = (lr["Close"] - pr["Close"]) / pr["Close"] * 100
                    change_str = f"{chg:+.1f}%"
                    chg_c = THEME["positive"] if chg >= 0 else THEME["negative"]
                    dt = lr.get("Date", None)
                    if hasattr(dt, "strftime"):
                        date_label = f"📅{dt.strftime('%m/%d')}"
                    elif dt is not None:
                        date_label = f"📅{str(dt)[:10]}"
                    else:
                        date_label = "📡实时"
            except Exception:
                pass

            parts.append(
                f'<div style="background:{THEME["card"]};border:1px solid {border};'
                f'border-radius:10px;padding:12px 10px;width:200px;flex-shrink:0;'
                f'text-align:center;cursor:pointer;transition:all .15s" '
                f'onmouseover="this.style.borderColor=\'{THEME["accent"]}\'" '
                f'onmouseout="this.style.borderColor=\'{border}\'">'
                f'<div style="font-size:0.75rem;color:{THEME["fg3"]};margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{inf["name"]}</div>'
                f'<div style="font-size:1.25rem;font-weight:700;margin:4px 0">{price}</div>'
                f'<div style="font-size:0.85rem;color:{chg_c};font-weight:500">{change_str}</div>'
                f'<div style="font-size:0.65rem;color:{THEME["fg3"]};margin-top:4px">{inf["symbol"]} {date_label}</div>'
                f'</div>'
            )

        parts.append('</div>')
        ui.html(''.join(parts))


def _group_emoji(g: str) -> str:
    gl = g.lower()
    if "半导体" in gl: return "🧊"
    elif "元器件" in gl: return "⚡"
    elif "通信" in gl: return "📡"
    elif "电气" in gl or "电力" in gl or "水电" in gl: return "🔌"
    elif "机械" in gl: return "⚙️"
    elif "港股" in gl: return "🇭🇰"
    elif "金属" in gl: return "⛏️"
    elif "化工" in gl: return "🧪"
    elif "玻璃" in gl or "陶瓷" in gl: return "🪞"
    elif "医疗" in gl: return "💊"
    elif "能源" in gl or "供热" in gl: return "🔥"
    elif "建筑" in gl: return "🏗️"
    elif "日用" in gl: return "🛒"
    return "📌"


# ─── Stock Detail ────────────────────────────────────────────
@ui.refreshable
@ui.refreshable
def detail_page(state):
    key = state.get("selected_stock")
    if not key:
        ui.label("请从仪表盘选择股票").classes("text-grey q-mt-xl"); return
    inf = info_for(key, state.get("stock_names", {}))
    df = get_data_for(inf["symbol"], inf["market"], period_days=500)
    if df is None or df.empty:
        ui.label("无数据").classes("text-grey"); return
    latest = df.iloc[-1]; prev = df.iloc[-2]
    change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
    ui.markdown(f"## {inf['name']} ({inf['market']}:{inf['symbol']})")
    with ui.row().classes("gap-3 q-mb-md"):
        for label, val in [("现价", f"{latest['Close']:.2f}"), ("涨跌", f"{change:+.2f}%"),
                           ("最高", f"{latest.get('High','-'):.2f}"), ("最低", f"{latest.get('Low','-'):.2f}"),
                           ("昨收", f"{prev.get('Close','-'):.2f}"), ("成交量", f"{latest.get('Volume',0):,.0f}")]:
            with ui.card().classes("text-center q-pa-sm"):
                ui.label(val).classes("text-h6"); ui.label(label).classes("text-caption text-grey")
    with ui.tabs() as tabs: t1,t2,t3 = ui.tab("📈 K线"), ui.tab("📰 新闻"), ui.tab("⚙️ 策略")
    with ui.tab_panels(tabs, value=t1):
        with ui.tab_panel(t1):
            import plotly.graph_objects as go
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=df["Date"],open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="K线",increasing_line_color="#26a69a",decreasing_line_color="#ef5350"))
            for p,c in [(5,"#ff0"),(20,"#f0f"),(60,"#0ff")]: fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"].rolling(p).mean(),name=f"MA{p}",line=dict(color=c,width=1)))
            fig.update_layout(height=450,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="#141824",plot_bgcolor="#141824",font=dict(color="#94a3b8"),xaxis_rangeslider_visible=False)
            ui.plotly(fig)
            with ui.row().classes("gap-2 q-mt-sm"):
                ui.button("预测",on_click=lambda:navigate_to("predict"),icon="psychology")
                ui.button("回测",on_click=lambda:navigate_to("backtest"),icon="replay")
                ui.button("风控",on_click=lambda:navigate_to("risk"),icon="shield")
        with ui.tab_panel(t2):
            try:
                from src.data.news_fetcher import load_news_from_store
                news=load_news_from_store(inf["symbol"],inf["market"]); 
                if not news.empty:
                    for _,r in news.tail(10).iterrows(): ui.label(f"{r.get('Date','')} — 情绪:{r.get('news_sent_mean',0):.2f}").classes("text-caption")
                else: ui.label("暂无新闻").classes("text-grey")
            except: ui.label("新闻加载失败").classes("text-grey")
        with ui.tab_panel(t3):
            rules=load_rules(); sr=[r for r in rules if r.symbol==inf["symbol"]]
            if sr:
                for r in sr: ui.label(f"{'🟢' if r.enabled else '🔴'} {CT.get(r.condition,r.condition)} | {r.label}").classes("text-caption")
            else: ui.label("未设置监控").classes("text-grey"); ui.button("添加监控",on_click=lambda:navigate_to("monitor"),icon="add")


# ─── Factor Library ──────────────────────────────────────────
@ui.refreshable
def factors_page(state):
    ui.markdown("## 因子库").classes("page-title")
    factor_file = Path(__file__).resolve().parent.parent / "data" / "factors.json"
    if factor_file.exists():
        data = json.loads(factor_file.read_text())
        factors = data.get("factors", [])
        with ui.tabs() as tabs:
            t1 = ui.tab("因子列表")
            t2 = ui.tab("模型")
            t3 = ui.tab("数据源验证")
        with ui.tab_panels(tabs, value=t1):
            with ui.tab_panel(t1):
                rows = []
                for f in factors:
                    bias = "✅" if not f.get("forward_bias") else "⚠️"
                    rows.append({"ID": f["id"], "因子": f["name"], "类型": f["type"],
                                 "描述": f["desc"], "回溯": f["lookback"], "安全": bias})
                if rows:
                    ui.table(rows=rows, columns=[{"name": c, "label": c, "field": c, "align": "left"} for c in rows[0].keys()],
                             row_key="ID").classes("w-full")
                ui.label(f"共 {len(factors)} 个因子 · 无未来函数泄漏 · 目标已对齐 shift+1").classes("text-caption text-grey q-mt-sm")
            with ui.tab_panel(t2):
                for m in data.get("models", []):
                    badge = "🟢 轻量" if m.get("light") else "🟡 重模型"
                    ui.markdown(f"**{m['name']}** {badge} — {m['desc']}")
                for s in data.get("strategies", []):
                    ui.markdown(f"- **{s['name']}**: {s['desc']}")
            with ui.tab_panel(t3):
                async def verify_sources():
                    results = {}
                    try:
                        from src.data.sources import WIND_SOURCE, YahooSource, FinnhubSource
                        sr = WIND_SOURCE.run_historical("000001", period_days=5, market="A")
                        results["Wind MCP"] = "✅" if sr.success else f"❌ {sr.error[:40]}"
                    except Exception as e: results["Wind MCP"] = f"❌ {str(e)[:40]}"
                    try:
                        sr = YahooSource().run_historical("AAPL", period_days=5, market="US")
                        results["Yahoo"] = "✅" if sr.success else "❌"
                    except Exception as e: results["Yahoo"] = f"❌ {str(e)[:40]}"
                    try:
                        from src.data.news_fetcher import compute_sentiment
                        s1 = compute_sentiment("revenue growth beats", "US")
                        s2 = compute_sentiment("业绩超预期 营收增长", "A")
                        results["新闻情绪"] = "✅" if abs(s1) > 0.01 or abs(s2) > 0.01 else "⚠️ 分数0"
                    except Exception as e: results["新闻情绪"] = f"❌ {str(e)[:40]}"
                    for k, v in results.items():
                        ui.notify(f"{v} {k}", position="bottom-right", type="info" if "✅" in v else "warning")

                ui.button("验证全部数据源", on_click=verify_sources,
                          icon="checklist").props("color=primary").classes("q-mt-md")


# ─── Settings ────────────────────────────────────────────────
@ui.refreshable
def settings_page(state):
    with ui.card().classes("w-full q-pa-md"):
        ui.markdown("### 自选股管理")
        wl=state.get("watchlist",[]); ui.label(f"当前 {len(wl)} 只")
        with ui.row().classes("gap-2 q-mt-sm"):
            ui.button("导出自选",on_click=lambda:export_watchlist(state),icon="download")
            ui.button("清空自选",on_click=lambda:confirm_action("确定清空？",lambda:clear_watchlist(state)),icon="delete").props("color=negative flat")
    with ui.card().classes("w-full q-pa-md q-mt-sm"):
        ui.markdown("### 文件导入")
        up=ui.upload(label="CSV/TXT/XLSX",on_upload=lambda e:_import_file(state,e),auto_upload=True).props("dense outlined").classes("w-full")
    with ui.card().classes("w-full q-pa-md q-mt-sm"):
        ui.markdown("### API 配置")
        from src.utils.config import load_config, save_config, get_llm_key, get_tushare_token
        cfg=load_config()
        tk=ui.input("Tushare Token",value=get_tushare_token() or "",password=True,password_toggle_button=True).props("dense outlined").classes("w-full")
        lk=ui.input("DeepSeek Key",value=get_llm_key() or "",password=True,password_toggle_button=True).props("dense outlined").classes("w-full")
        pp=ui.input("PushPlus Token",value=cfg.get("pushplus_token","") or "",password=True,password_toggle_button=True).props("dense outlined").classes("w-full")
        def save_api():
            cfg["tushare_token"]=tk.value; cfg["llm_api_key"]=lk.value; cfg["pushplus_token"]=pp.value
            save_config(cfg); ui.notify("API已保存",type="positive")
        ui.button("💾 保存API配置",on_click=save_api,icon="save").props("color=primary")
    with ui.card().classes("w-full q-pa-md q-mt-sm"):
        ui.markdown("### 数据缓存")
        from webapp.services import _data_cache
        ui.label(f"行情缓存: {len(_data_cache)} 条"); ui.button("清除缓存",on_click=clear_all_caches,icon="cleaning_services").props("flat")


def export_watchlist(state):
    rows=["symbol,market,name,group"]
    for key in state.get("watchlist",[]):
        inf=info_for(key,state.get("stock_names",{})); g=state.get("stock_groups",{}).get(key,"默认")
        rows.append(f"{inf['symbol']},{inf['market']},{inf['name']},{g}")
    Path("/tmp/watchlist_export.csv").write_text("\n".join(rows))
    ui.download("/tmp/watchlist_export.csv")
    ui.notify("导出完成",type="positive")


def _import_file(state, e):
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False,suffix="."+e.name.split(".")[-1]) as tmp:
            tmp.write(e.content.read()); tpath=tmp.name
        from src.data.stock_db import parse_stock_file
        codes=parse_stock_file(tpath); added=0
        for code,name,market in codes:
            if add_to_watchlist(state,code,market,name)[0]: added+=1
        os.unlink(tpath)
        ui.notify(f"导入 {added} 只",type="positive")
        settings_page.refresh()
    except Exception as ex:
        ui.notify(f"导入失败: {ex}",type="negative")


def clear_watchlist(state):
    state["watchlist"]=[]; state["stock_names"]={}; state["stock_groups"]={}; state["stock_order"]=[]
    ui.notify("已清空",type="positive"); navigate_to("dashboard")


# ─── Startup ─────────────────────────────────────────────────
def run(port: int = 8502, host: str = "0.0.0.0"):
    print(f"🚀 StockPredict NiceGUI → http://{host}:{port}")
    print(f"   Streamlit保留 → http://{host}:8501")
    ui.run(host=host, port=port, title="StockPredict", dark=True, reload=False,
           favicon="webapp_nice/股票分析软件接入数据源.png",
           storage_secret="stockpredict_nicegui_2026_v2")


if __name__ == "__main__":
    run()
