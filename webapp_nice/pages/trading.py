"""
选股器 + 模拟交易页面 — NiceGUI
"""
from nicegui import ui
from webapp.services import info_for, get_data_for, add_to_watchlist, fmt_risk
from src.recommend.screener import screen_market
from src.alerts.models import CONDITION_TYPES as CT
from src.data.stock_db import get_db
import pandas as pd
import numpy as np

SCOPE_MAP = {
    "自选股": "watchlist", "全部A股": "all_a",
    "创业板(300)": "gem", "科创板(688)": "star", "沪深主板": "main_board",
}


# ─── Screener Page ───────────────────────────────────────────
@ui.refreshable
def screener_page(state):
    with ui.column().classes("w-full"):
        ui.markdown("## 🔍 条件选股器").classes("page-title")
        ui.label("组合技术条件, 扫描全市场").classes("text-caption text-grey q-mb-md")

        scope = ui.select(list(SCOPE_MAP.keys()), value="自选股", label="选股范围")
        limit = ui.slider(min=5, max=100, value=30, step=5)
        ui.label().bind_text_from(limit, "value", backward=lambda v: f"最多: {v}只").classes("text-caption")

        # Condition management
        if "scr_conds" not in state:
            state["scr_conds"] = [{"cond": "golden_cross", "params": {"short": 20, "long": 60}, "logic": "AND"}]

        conds = state["scr_conds"]
        cond_containers = []

        for idx, cdata in enumerate(conds):
            with ui.card().classes("w-full q-pa-sm"):
                with ui.row().classes("items-center gap-2"):
                    cond_val = cdata["cond"]
                    all_conds = {f"{k}: {v}": k for k, v in CT.items()}
                    cond_input = ui.select(
                        options=all_conds, value=f"{cond_val}: {CT[cond_val]}",
                        label=f"条件 {idx+1}",
                    ).classes("w-60").props("dense")
                    cond_input.on("update:model-value", lambda e, i=idx: _update_cond(state, i, all_conds[e.value]))

                    # Condition-specific params
                    params_card = ui.card().classes("w-full q-pa-xs")
                    _render_cond_params(params_card, cdata, idx, state)
                    cond_containers.append(params_card)

                    if len(conds) > 1:
                        logic_val = ui.select(["AND", "OR"], value=cdata.get("logic", "AND"),
                                              label="逻辑").classes("w-20").props("dense")
                        logic_val.on("update:model-value", lambda e, i=idx: _update_logic(state, i, e.value))

        with ui.row().classes("gap-2 q-mt-sm"):
            if len(conds) < 4:
                ui.button("➕ 添加条件", on_click=lambda: _add_cond(state), icon="add")
            if len(conds) > 1:
                ui.button("➖ 删除最后", on_click=lambda: _remove_cond(state), icon="remove")

        result_container = ui.column().classes("w-full q-mt-md")

        async def run_scan():
            result_container.clear()
            scope_type = SCOPE_MAP[scope.value]
            if scope_type == "watchlist":
                scan_list = [k.split("-", 1)[1] for k in state.get("watchlist", []) if k.startswith("A-")]
            else:
                db = get_db()
                all_a = [code for code, _ in db.all_stocks("A")]
                if scope_type == "all_a":
                    scan_list = all_a[:500]
                elif scope_type == "gem":
                    scan_list = [c for c in all_a if c.startswith("300")]
                elif scope_type == "star":
                    scan_list = [c for c in all_a if c.startswith("688")]
                elif scope_type == "main_board":
                    scan_list = [c for c in all_a if c.startswith(("600", "000", "002"))]
                else:
                    scan_list = all_a[:500]

            if not scan_list:
                ui.notify("无股票可扫描", type="warning")
                return

            with result_container:
                ui.label(f"扫描中: {len(scan_list)}只...").classes("text-caption")

            conds_cur = state["scr_conds"]
            all_hits = {}
            for cdata in conds_cur:
                hits = screen_market(cdata["cond"], cdata["params"], "A", scan_list, limit.value * 5)
                for h in hits:
                    key = f"{h.market}-{h.symbol}"
                    if key not in all_hits:
                        all_hits[key] = {"hit": h, "conds": []}
                    all_hits[key]["conds"].append(cdata["cond"])

            final = []
            for key, val in all_hits.items():
                and_ok = all(c["cond"] in val["conds"] for c in conds_cur if c.get("logic") == "AND")
                or_ok = any(c["cond"] in val["conds"] for c in conds_cur if c.get("logic") == "OR") if any(c.get("logic") == "OR" for c in conds_cur) else True
                if and_ok and or_ok:
                    final.append(val["hit"])
                    if len(final) >= limit.value:
                        break

            with result_container:
                result_container.clear()
                if final:
                    ui.label(f"找到 {len(final)} 只符合条件的股票").classes("text-positive text-subtitle1 q-mt-md")
                    for h in final[:30]:
                        with ui.card().classes("w-full q-pa-sm"):
                            with ui.row().classes("items-center"):
                                ui.label(f"{h.direction} {h.symbol} {h.name}").classes("text-subtitle2")
                                ui.space()
                                ui.label(f"{h.current_price:.2f}").classes("text-h6")
                            ui.label(h.message[:60]).classes("text-caption text-grey")
                            key = f"{h.market}-{h.symbol}"
                            if key not in state.get("watchlist", []):
                                ui.button("➕ 加到自选",
                                          on_click=lambda k=key, s=h.symbol, m=h.market, n=h.name: _scr_add(state, s, m, n, k),
                                          icon="add").props("flat dense size=sm")
                else:
                    ui.notify("未找到符合条件的股票", type="warning")

        ui.button("🔍 开始扫描", on_click=run_scan,
                  icon="search").props("color=primary").classes("q-mt-md w-full")
    ui.separator()


def _update_cond(state, idx, new_cond):
    if idx < len(state.get("scr_conds", [])):
        state["scr_conds"][idx]["cond"] = new_cond
        old_params = state["scr_conds"][idx]["params"]
        state["scr_conds"][idx]["params"] = _default_params_for(new_cond, old_params)
    screener_page.refresh()


def _update_logic(state, idx, value):
    if idx < len(state.get("scr_conds", [])):
        state["scr_conds"][idx]["logic"] = value


def _add_cond(state):
    state.setdefault("scr_conds", [])
    state["scr_conds"].append({"cond": "above_ma", "params": {"window": 20}, "logic": "AND"})
    screener_page.refresh()


def _remove_cond(state):
    if len(state.get("scr_conds", [])) > 1:
        state["scr_conds"].pop()
        screener_page.refresh()


def _scr_add(state, symbol, market, name, key):
    ok, msg = add_to_watchlist(state, symbol, market, name)
    ui.notify(msg, type="positive" if ok else "warning")
    if ok:
        screener_page.refresh()


def _default_params_for(cond, old=None):
    base = old or {}
    if cond in ("above_ma", "below_ma"):
        return {"window": base.get("window", 20)}
    if cond in ("golden_cross", "death_cross", "ma_cross_combo"):
        return {"short": base.get("short", 20), "long": base.get("long", 60)}
    if "rsi" in cond:
        return {"window": int(base.get("window", 14)), "level": int(base.get("level", 30))}
    if "bollinger" in cond:
        return {"window": int(base.get("window", 20)), "std": int(base.get("std", 2))}
    if "volume" in cond:
        return {"ratio": float(base.get("ratio", 2.0))}
    if cond in ("above_price", "below_price"):
        return {"threshold": float(base.get("threshold", 100.0))}
    if cond == "alpha120":
        return {"threshold": float(base.get("threshold", 0.02))}
    if cond == "alpha006":
        return {"threshold": float(base.get("threshold", 0.3))}
    if cond == "alpha053":
        return {"threshold_up": float(base.get("threshold_up", 1.05)),
                "threshold_dn": float(base.get("threshold_dn", 0.95))}
    if cond == "alpha015":
        return {"window": int(base.get("window", 20)), "threshold": float(base.get("threshold", 0.3))}
    return {}


def _render_cond_params(container, cdata, idx, state):
    container.clear()
    cond = cdata["cond"]
    params = cdata["params"]
    with container:
        if cond in ("above_ma", "below_ma"):
            v = ui.number("MA窗口", value=int(params.get("window", 20)), min=5, max=120)
            v.on("update:model-value", lambda e: params.update({"window": int(e.value)}))
        elif cond in ("golden_cross", "death_cross", "ma_cross_combo"):
            s = ui.number("短期", value=int(params.get("short", 20)), min=5, max=50)
            l = ui.number("长期", value=int(params.get("long", 60)), min=10, max=200)
            s.on("update:model-value", lambda e: params.update({"short": int(e.value)}))
            l.on("update:model-value", lambda e: params.update({"long": int(e.value)}))
        elif "rsi" in cond:
            w = ui.number("RSI窗口", value=int(params.get("window", 14)), min=5, max=30)
            lv = ui.number("阈值", value=int(params.get("level", 30)), min=10, max=90)
            w.on("update:model-value", lambda e: params.update({"window": int(e.value)}))
            lv.on("update:model-value", lambda e: params.update({"level": int(e.value)}))
        elif "bollinger" in cond:
            w = ui.number("窗口", value=int(params.get("window", 20)), min=10, max=50)
            s = ui.number("标准差", value=int(params.get("std", 2)), min=1, max=4)
            w.on("update:model-value", lambda e: params.update({"window": int(e.value)}))
            s.on("update:model-value", lambda e: params.update({"std": int(e.value)}))
        elif "volume" in cond:
            r = ui.number("倍数", value=float(params.get("ratio", 2.0)), min=1.5, max=5.0, step=0.1)
            r.on("update:model-value", lambda e: params.update({"ratio": float(e.value)}))
        elif cond in ("above_price", "below_price"):
            t = ui.number("价格", value=float(params.get("threshold", 100.0)), min=0.0, max=10000.0)
            t.on("update:model-value", lambda e: params.update({"threshold": float(e.value)}))
        elif cond == "alpha120":
            t = ui.number("偏离", value=float(params.get("threshold", 0.02)), min=0.001, max=0.1, step=0.005)
            t.on("update:model-value", lambda e: params.update({"threshold": float(e.value)}))
        elif cond == "alpha053":
            u = ui.number("涨阈值", value=float(params.get("threshold_up", 1.05)), min=1.01, max=1.2, step=0.01)
            d = ui.number("跌阈值", value=float(params.get("threshold_dn", 0.95)), min=0.8, max=0.99, step=0.01)
            u.on("update:model-value", lambda e: params.update({"threshold_up": float(e.value)}))
            d.on("update:model-value", lambda e: params.update({"threshold_dn": float(e.value)}))
        else:
            ui.label("无额外参数").classes("text-caption text-grey")


# ─── Paper Trading Page ──────────────────────────────────────
@ui.refreshable
def paper_page(state):
    with ui.column().classes("w-full"):
        ui.markdown("## 💰 模拟交易").classes("page-title")

        if "paper_account" not in state:
            state["paper_account"] = {"cash": 100000.0, "positions": {}}
        if "paper_history" not in state:
            state["paper_history"] = []

        acc = state["paper_account"]
        hist = state["paper_history"]

        # Account summary
        with ui.row().classes("gap-4 q-mb-md"):
            with ui.card().classes("card-container w-40"):
                ui.label("现金").classes("text-caption text-grey")
                ui.label(f"¥{acc['cash']:,.2f}").classes("text-h5 price-up")
            pos_value = sum(float(v.get("shares", 0)) * float(v.get("price", 0)) for v in acc["positions"].values())
            with ui.card().classes("card-container w-40"):
                ui.label("持仓市值").classes("text-caption text-grey")
                ui.label(f"¥{pos_value:,.2f}").classes("text-h5")
            with ui.card().classes("card-container w-40"):
                ui.label("总资产").classes("text-caption text-grey")
                total = acc["cash"] + pos_value
                pnl = total - 100000
                color = "price-up" if pnl >= 0 else "price-down"
                ui.label(f"¥{total:,.2f}").classes("text-h5")
                ui.label(f"盈亏 {pnl:+,.2f}").classes(f"text-subtitle2 {color}")

        # Trade form
        stock_items, key_map = [], {}
        for k in state.get("watchlist", []):
            inf = info_for(k, state.get("stock_names", {}))
            d = f"{inf['symbol']} {inf['name']}"; stock_items.append(d); key_map[d] = k

        with ui.row().classes("items-end gap-2 q-mt-sm"):
            sym_sel = ui.select(options=stock_items, label="股票", value=None).props("dense outlined").classes("w-56")
            action_sel = ui.select(["买入", "卖出"], value="买入", label="操作").classes("w-20")
            shares_inp = ui.number("数量(股)", value=100, min=1, max=100000, step=100).classes("w-30")

        async def execute_trade():
            if not sym_sel.value:
                ui.notify("请选择股票", type="warning")
                return
            key = key_map.get(sym_sel.value)
            if not key:
                ui.notify("请选择股票", type="warning")
                return
            inf = info_for(key, state.get("stock_names", {}))
            df = get_data_for(inf["symbol"], inf["market"], period_days=10)
            if df is None or df.empty:
                ui.notify("无法获取价格", type="negative")
                return
            price = float(df["Close"].iloc[-1])

            if action_sel.value == "买入":
                cost = price * shares_inp.value * 1.0003
                if cost <= acc["cash"]:
                    acc["cash"] -= cost
                    pos = acc["positions"].get(key, {"shares": 0, "price": 0, "name": inf["name"]})
                    old_val = pos["shares"] * pos["price"]
                    pos["shares"] += shares_inp.value
                    pos["price"] = (old_val + price * shares_inp.value) / pos["shares"] if pos["shares"] > 0 else price
                    pos["name"] = inf["name"]
                    acc["positions"][key] = pos
                    hist.append(f"买入 {inf['name']} {shares_inp.value}股 @{price:.2f}")
                    ui.notify(f"买入 {inf['name']} {shares_inp.value}股 @{price:.2f}", type="positive")
                else:
                    ui.notify("资金不足", type="negative")
            else:
                pos = acc["positions"].get(key)
                if pos and pos["shares"] >= shares_inp.value:
                    gross = price * shares_inp.value
                    fee = gross * 0.0003
                    stamp = gross * 0.001
                    net = gross - fee - stamp
                    pnl = (price - pos["price"]) * shares_inp.value - fee - stamp
                    acc["cash"] += net
                    pos["shares"] -= shares_inp.value
                    if pos["shares"] == 0:
                        del acc["positions"][key]
                    else:
                        acc["positions"][key] = pos
                    hist.append(f"卖出 {inf['name']} {shares_inp.value}股 @{price:.2f} 盈亏{pnl:+.2f}")
                    ui.notify(f"卖出 {inf['name']} {shares_inp.value}股 @{price:.2f}", type="positive")
                else:
                    ui.notify("持仓不足", type="negative")
            paper_page.refresh()

        ui.button("✅ 执行交易", on_click=execute_trade,
                  icon="check").props("color=primary").classes("q-mt-sm")

        # Positions table
        if acc["positions"]:
            ui.markdown("### 📋 当前持仓").classes("q-mt-md")
            pos_rows = []
            for key, pos in acc["positions"].items():
                inf = info_for(key, state.get("stock_names", {}))
                df = get_data_for(inf["symbol"], inf["market"], period_days=10)
                cur_price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else pos["price"]
                pnl = (cur_price - pos["price"]) * pos["shares"]
                pos_rows.append({
                    "股票": pos.get("name", ""), "成本价": f"{pos['price']:.2f}",
                    "现价": f"{cur_price:.2f}", "股数": str(pos.get("shares", 0)),
                    "盈亏": f"{pnl:+,.2f}", "收益率": f"{(cur_price/pos['price']-1)*100:+.1f}%",
                })
            ui.table(rows=pos_rows, columns=[
                {"name": c, "label": c, "field": c} for c in pos_rows[0].keys()
            ]).classes("w-full")

        # History
        if hist:
            with ui.expansion("📜 交易记录", icon="receipt").classes("w-full q-mt-md"):
                with ui.column():
                    for h in reversed(hist[-20:]):
                        ui.label(h).classes("text-caption")
    ui.separator()
