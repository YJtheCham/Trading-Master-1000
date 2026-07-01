from nicegui import ui
from webapp.services import info_for, get_data_for, get_data_notify, fmt_risk
from src.alerts.engine import load_rules, add_rule, toggle_rule, remove_rule, get_engine as get_alert_engine
from src.alerts.models import AlertRule, CONDITION_TYPES as CT
from src.alerts.settings import load_settings as load_alert_settings, save_settings
from src.risk.metrics import calc_all_risk_metrics
from src.recommend.engine import scan_predictions, scan_strategies, generate_report
import pandas as pd, numpy as np, plotly.graph_objects as go
from datetime import timedelta


@ui.refreshable
def risk_page(state):
    stock_items, key_map = [], {}
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        d = f"{inf['symbol']} {inf['name']}"; stock_items.append(d); key_map[d] = k
    sel = ui.select(options=stock_items, label="股票", value=stock_items[0] if stock_items else None).props("dense outlined").classes("w-56")
    result_col = ui.column()
    progress = ui.spinner(size="lg"); progress.visible = False

    async def run():
        result_col.clear()
        if not sel.value: return
        progress.visible = True
        try:
            key = key_map.get(sel.value)
            if not key: return
            inf = info_for(key, state.get("stock_names", {}))
            df, _ = get_data_notify(inf["symbol"], inf["market"], inf["name"], period_days=500)
            risk = calc_all_risk_metrics(df)
            with result_col:
                for k, v in risk.items():
                    color = "negative" if (isinstance(v,(int,float)) and v<0) else "positive"
                    with ui.row().classes("items-center gap-2"):
                        ui.label(k).classes("text-caption w-24")
                        ui.linear_progress(value=min(abs(v)/0.5,1.0) if isinstance(v,(int,float)) else 0.5).props(f"color={color}").classes("w-60 q-mx-sm")
                        ui.label(fmt_risk(k, v)).classes("text-subtitle2")
                returns = df["Close"].pct_change().dropna()
                if len(returns) > 0:
                    var95 = np.percentile(returns, 5)
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name="日收益分布", marker_color="#4fc3f7"))
                    fig.add_vline(x=var95, line_dash="dash", line_color="#ef5350", annotation_text=f"VaR 95%: {var95*100:.2f}%")
                    fig.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="#141824", plot_bgcolor="#141824", font=dict(color="#94a3b8"))
                    ui.plotly(fig)
                cummax = df["Close"].cummax()
                dd = (df["Close"] - cummax) / cummax * 100
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=dd, fill="tozeroy", name="回撤%", line=dict(color="#ef5350")))
                fig2.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="#141824", plot_bgcolor="#141824", font=dict(color="#94a3b8"))
                ui.plotly(fig2)
        except Exception as e:
            ui.notify(f"风险分析失败: {e}", type="negative")
        finally:
            progress.visible = False

    ui.button("📊 分析风险", on_click=run, icon="shield").props("color=primary")


@ui.refreshable
def monitor_page(state):
    with ui.tabs() as tabs:
        t_conn, t_rules, t_add, t_set = ui.tab("📡 连接状态"), ui.tab("📋 规则列表"), ui.tab("➕ 添加规则"), ui.tab("⚙️ 设置")
    with ui.tab_panels(tabs, value=t_conn):
        with ui.tab_panel(t_conn): _mc(state)
        with ui.tab_panel(t_rules): _mr(state)
        with ui.tab_panel(t_add): _ma(state)
        with ui.tab_panel(t_set): _ms(state)


def _mc(state):
    """连接状态监控面板"""
    from src.alerts.health import summary as health_summary, get_health

    hs = health_summary()
    recent = get_health().recent(60)

    status_map = {
        "ok": ("🟢 正常", "text-positive"),
        "degraded": ("🟡 降级", "text-orange"),
        "down": ("🔴 断连", "text-negative"),
        "unknown": ("⚪ 未知", "text-grey"),
    }
    status_label, status_cls = status_map.get(hs["status"], ("⚪ 未知", "text-grey"))

    ui.label("📡 数据源连接状态监控").classes("text-h6 q-mb-sm")

    with ui.row().classes("w-full q-mb-sm items-center"):
        with ui.card().classes("q-pa-sm"):
            ui.label("状态").classes("text-caption text-grey")
            ui.label(status_label).classes(f"text-subtitle2 {status_cls}")
        with ui.card().classes("q-pa-sm"):
            ui.label("近1h查询").classes("text-caption text-grey")
            ui.label(str(hs["recent_checks"])).classes("text-subtitle2")
        with ui.card().classes("q-pa-sm"):
            ui.label("成功率").classes("text-caption text-grey")
            ui.label(f"{hs['success_rate']}%").classes("text-subtitle2")
        with ui.card().classes("q-pa-sm"):
            fail = hs["consecutive_failures"]
            ui.label("连续失败").classes("text-caption text-grey")
            ui.label(f"⚠️ {fail}" if fail > 0 else "0").classes("text-subtitle2")
        with ui.card().classes("q-pa-sm"):
            lat = hs.get("avg_latency_ms", 0)
            ui.label("平均延迟").classes("text-caption text-grey")
            ui.label(f"{lat:.0f}ms" if lat else "-").classes("text-subtitle2")

    if hs.get("last_error"):
        ui.label(f"⚠️ 最近错误: {hs['last_error']}").classes("text-caption text-negative q-mb-sm")

    with ui.row().classes("q-mb-sm"):
        ui.button("🔄 刷新", on_click=monitor_page.refresh, icon="refresh").props("flat dense color=primary")

    if recent:
        with ui.expansion("📋 最近查询记录", icon="list").classes("w-full"):
            import pandas as pd
            df = pd.DataFrame(reversed(recent))
            display_cols = ["checked_at", "symbol", "market", "success", "latency_ms", "error"]
            show_cols = [c for c in display_cols if c in df.columns]
            df = df[show_cols].head(30)
            renames = {
                "checked_at": "时间", "symbol": "代码", "market": "市场",
                "success": "成功", "latency_ms": "延迟(ms)", "error": "错误"
            }
            df = df.rename(columns=renames)
            ui.table.from_pandas(df).props("dense flat")


def _mr(state):
    rules = load_rules(); settings = load_alert_settings(); engine = get_alert_engine()
    ui.label(f"{sum(1 for r in rules if r.enabled)}/{len(rules)} 启用 | {settings.summary} | {'🟢运行中' if engine.is_running else '🔴已停止'}").classes("text-subtitle2 q-mb-sm")
    for r in rules:
        desc = CT.get(r.condition, r.condition)
        with ui.card().classes("w-full q-pa-sm"):
            with ui.row().classes("items-center"):
                ui.label(f"{'🟢' if r.enabled else '🔴'} [{r.market}] {r.symbol} — {desc} | {r.label}").classes("text-subtitle2")
                ui.space()
                ui.button("⏸" if r.enabled else "▶", on_click=lambda uid=r.uid: (toggle_rule(uid), monitor_page.refresh()), icon="pause" if r.enabled else "play_arrow").props("flat dense")
                ui.button("✕", on_click=lambda uid=r.uid: (remove_rule(uid), monitor_page.refresh()), icon="delete").props("flat dense")


def _ma(state):
    stock_items, key_map = [], {}
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        d = f"[{inf['market']}] {inf['symbol']} {inf['name']}"; stock_items.append(d); key_map[d] = k
    code_sel = ui.select(options=stock_items, label="股票", value=None).props("dense outlined").classes("w-full")
    cond_sel = ui.select(options=list(CT.keys()), value="golden_cross", label="条件类型").props("dense outlined").classes("w-full")
    params_card = ui.card().classes("q-pa-sm"); _render_cond_guided(params_card, cond_sel, "golden_cross")
    cond_sel.on("update:model-value", lambda e: _render_cond_guided(params_card, cond_sel, cond_sel.value))
    label_inp = ui.input(label="标签 (可选)").props("dense outlined").classes("w-full")

    async def do_add():
        if not code_sel.value: ui.notify("请选择股票", type="warning"); return
        key = key_map.get(code_sel.value)
        if not key: ui.notify("请选择股票", type="warning"); return
        inf = info_for(key, state.get("stock_names", {}))
        rule = AlertRule(symbol=inf["symbol"], market=inf["market"], condition=cond_sel.value, params={}, label=label_inp.value or "")
        add_rule(rule); ui.notify(f"已添加 {inf['symbol']}", type="positive"); monitor_page.refresh()

    ui.button("➕ 添加规则", on_click=do_add, icon="add").props("color=primary").classes("w-full q-mt-sm")


def _render_cond_guided(card, cond_sel, cond):
    card.clear()
    with card:
        if cond in ("above_ma","below_ma"): ui.number("MA窗口", value=20, min=5, max=120).props("dense")
        elif cond in ("golden_cross","death_cross","ma_cross_combo"):
            with ui.row(): ui.number("短期MA",value=5,min=2,max=50).props("dense"); ui.number("长期MA",value=20,min=5,max=200).props("dense")
        elif "rsi" in cond:
            with ui.row(): ui.number("窗口",value=14,min=5,max=30).props("dense"); ui.number("阈值",value=30,min=10,max=90).props("dense")
        elif "bollinger" in cond:
            with ui.row(): ui.number("窗口",value=20,min=10,max=50).props("dense"); ui.number("标准差",value=2,min=1,max=4).props("dense")
        elif "volume" in cond: ui.number("倍数",value=2.0,min=1.5,max=5.0,step=0.1).props("dense")
        elif cond in ("above_price","below_price"): ui.number("价格",value=100.0,min=0.0,max=10000.0).props("dense")
        elif cond=="alpha120": ui.number("偏离",value=0.02,min=0.001,max=0.1,step=0.005).props("dense")
        elif cond=="alpha053":
            with ui.row(): ui.number("涨阈值",value=1.05,min=1.01,max=1.2,step=0.01).props("dense"); ui.number("跌阈值",value=0.95,min=0.8,max=0.99,step=0.01).props("dense")
        else: ui.label("无额外参数").classes("text-caption")


def _ms(state):
    engine = get_alert_engine()
    ui.button("▶ 启动" if not engine.is_running else "⏹ 停止", on_click=lambda: (_toggle_engine(), monitor_page.refresh()), icon="play_arrow" if not engine.is_running else "stop")
    ui.label(f"当前: {'🟢运行中' if engine.is_running else '🔴已停止'}").classes("text-subtitle2")

def _toggle_engine():
    engine = get_alert_engine()
    if engine.is_running: engine.stop()
    else: engine.start()


@ui.refreshable
def strategy_page(state):
    with ui.tabs() as tabs: t_new, t_hist = ui.tab("🔍 新扫描"), ui.tab("📜 历史")
    with ui.tab_panels(tabs, value=t_new):
        with ui.tab_panel(t_new): _strategy_new(state)
        with ui.tab_panel(t_hist): _strategy_hist(state)


def _strategy_new(state):
    stock_items, key_map = [], {}
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        d = f"{inf['symbol']} {inf['name']}"; stock_items.append(d); key_map[d] = k

    with ui.row().classes("items-end gap-2 q-mb-sm"):
        sel = ui.select(options=stock_items, label="选择股票", value=stock_items[0] if stock_items else None).props("dense outlined").classes("w-56")
        steps_sl = ui.slider(min=10, max=90, value=30, step=5)
        ui.label().bind_text_from(steps_sl, "value", backward=lambda v: f"{v}天").classes("text-caption")
        cap_inp = ui.number(label="资金", value=100000, min=10000, step=10000).props("dense outlined").classes("w-24")
    result_col = ui.column()
    progress = ui.spinner(size="lg"); progress.visible = False
    progress_label = ui.label("").classes("text-caption text-grey")
    progress_label.visible = False

    async def run():
        result_col.clear()
        if not sel.value: ui.notify("请选择股票", type="warning"); return
        progress.visible = True
        progress_label.visible = True
        progress_label.text = "获取数据中..."
        try:
            target = key_map.get(sel.value)
            if not target: ui.notify("请选择股票", type="warning"); return
            inf = info_for(target, state.get("stock_names", {}))
            df, src = get_data_notify(inf["symbol"], inf["market"], inf["name"], period_days=250)
            cutoff = pd.Timestamp.now() - timedelta(days=540)
            df_full = df.copy()
            if hasattr(df["Date"].dtype,"tz") and df["Date"].dtype.tz is not None: df["Date"] = df["Date"].dt.tz_localize(None)
            df = df[df["Date"]>=cutoff]
            if df.empty or len(df)<60: df = df_full
            if df.empty or len(df)<30: ui.notify("数据不足", type="negative"); return
            progress_label.text = "运行模型预测..."
            models = scan_predictions(df, steps=steps_sl.value, data_source=src, light=True)
            import gc; gc.collect()
            progress_label.text = "回测策略..."
            strategies = scan_strategies(df, inf["symbol"], inf["market"], cap_inp.value, light=True)
            gc.collect()
            progress_label.text = "计算风险指标..."
            risk = calc_all_risk_metrics(df)
            cur_price = float(df["Close"].iloc[-1])
            report = generate_report(inf["name"], inf["symbol"], inf["market"], models, strategies, cur_price, risk, steps_sl.value)
            valid_m = [m for m in models if not m.error]
            up = sum(1 for m in valid_m if m.pct_change > 0) if valid_m else 0
            avg_pct = np.mean([m.pct_change for m in valid_m]) if valid_m else 0
            best_s = None
            for s in strategies:
                if not s.error and (not best_s or s.total_return > best_s.total_return):
                    best_s = s
            from src.data.rec_history import add_rec_history
            add_rec_history(inf["symbol"], inf["market"], inf["name"], cur_price, f"{up}/{len(valid_m)} 看涨", avg_pct,
                            best_s.strategy if best_s else "-", best_s.total_return if best_s else 0,
                            best_s.sharpe if best_s else 0, best_s.max_dd if best_s else 0,
                            len(models), len(strategies), report=report,
                            models_data=[{"model": m.model, "direction": m.direction, "final_price": m.final_price if not m.error else 0,
                                          "pct_change": m.pct_change if not m.error else 0, "mape": m.mape if not m.error else 0, "error": m.error} for m in models],
                            strategies_data=[{"strategy": s.strategy, "total_return": s.total_return, "sharpe": s.sharpe, "max_dd": s.max_dd,
                                              "win_rate": s.win_rate, "total_trades": s.total_trades, "error": s.error} for s in strategies])
            with result_col:
                ui.markdown("### 🔮 多模型预测共识").classes("q-mt-md")
                with ui.row().classes("gap-3"):
                    with ui.card().classes("text-center q-pa-sm"):
                        ui.label(f"{cur_price:.2f}").classes("text-h6")
                        ui.label("当前价").classes("text-caption text-grey")
                    with ui.card().classes("text-center q-pa-sm"):
                        ui.label(f"{up}/{len(valid_m)} 看涨").classes("text-h6")
                        ui.label("模型共识").classes("text-caption text-grey")
                    with ui.card().classes("text-center q-pa-sm"):
                        ui.label(f"{avg_pct:+.1f}%").classes("text-h6")
                        ui.label("平均预测涨跌").classes("text-caption text-grey")
                model_rows = []
                for m in models:
                    if m.error: model_rows.append({"模型": m.model, "状态": f"❌ {m.error[:40]}"})
                    else: model_rows.append({"模型": m.model, "方向": m.direction, "预测价": f"{m.final_price:.2f}", "涨跌": f"{m.pct_change:+.1f}%", "MAPE": f"{m.mape:.1f}%" if m.mape else "-"})
                ui.table(rows=model_rows, columns=[{"name":c,"label":c,"field":c} for c in model_rows[0].keys()]).classes("w-full")
                ui.markdown("### 📈 策略回测对比").classes("q-mt-md")
                strat_rows = []
                for s in strategies:
                    if s.error: strat_rows.append({"策略": s.strategy, "状态": "❌"})
                    else: strat_rows.append({"策略": s.strategy, "收益": f"{s.total_return*100:+.1f}%", "Sharpe": f"{s.sharpe:.2f}", "最大回撤": f"{s.max_dd*100:.1f}%", "胜率": f"{s.win_rate*100:.1f}%"})
                strat_rows.sort(key=lambda x:float(x["收益"].replace("%","").replace("+","")),reverse=True)
                ui.table(rows=strat_rows, columns=[{"name":c,"label":c,"field":c} for c in strat_rows[0].keys()]).classes("w-full")
                with ui.expansion("📊 风控指标", icon="shield").classes("w-full q-mt-md"):
                    for k, v in risk.items():
                        with ui.row().classes("items-center gap-2"):
                            ui.label(k).classes("text-caption w-24")
                            ui.label(fmt_risk(k, v)).classes("text-subtitle2")
                ui.markdown("### ⚡ 添加到交易监控").classes("q-mt-md")
                STRAT_TO_COND = {
                    "双均线(5/20)": ("ma_cross_combo", {"short": 5, "long": 20}),
                    "双均线(10/30)": ("ma_cross_combo", {"short": 10, "long": 30}),
                    "双均线(20/60)": ("ma_cross_combo", {"short": 20, "long": 60}),
                    "RSI(14)": ("rsi_combo", {"window": 14, "oversold": 30, "overbought": 70}),
                    "通道突破(20/10)": ("volume_breakout", {"lookback": 20, "vol_ratio": 2.0}),
                    "布林带(20/2)": ("bollinger_combo", {"window": 20, "std": 2}),
                    "滚动预测(月频)": ("ma_cross_combo", {"short": 20, "long": 60}),
                    "滚动预测(周频)": ("ma_cross_combo", {"short": 5, "long": 20}),
                }
                best_strategy = best_s.strategy if best_s else ""
                best_ret = best_s.total_return if best_s else 0
                mapped = STRAT_TO_COND.get(best_strategy)
                if mapped and best_ret > 0:
                    cond, params = mapped
                    desc = CT.get(cond, cond)
                    ui.label(f"推荐策略 **{best_strategy}** → 组合条件 **{cond}** ({desc})").classes("text-subtitle2")
                    async def add_alert():
                        add_rule(AlertRule(symbol=inf["symbol"], market=inf["market"], condition=cond, params=params, label=f"推荐策略: {best_strategy}"))
                        ui.notify(f"已添加监控规则: {desc} ({inf['symbol']})", type="positive")
                    ui.button("✅ 一键添加到交易监控", on_click=add_alert, icon="add").props("color=primary").classes("q-mt-sm")
                from src.utils.config import get_llm_key
                if get_llm_key():
                    with ui.expansion("🤖 AI 综合分析", icon="psychology").classes("w-full q-mt-md") as ai_exp:
                        ai_container = ui.column().classes("w-full")
                        with ai_container:
                            ui.spinner(size="md").classes("q-mt-sm")
                            ui.label("AI 分析中...").classes("text-caption text-grey")
                        try:
                            progress_label.text = "AI 综合分析中..."
                            from src.recommend.advisor import analyze_with_llm
                            ai_result = analyze_with_llm(report)
                            ai_container.clear()
                            with ai_container:
                                if ai_result:
                                    ui.markdown(ai_result)
                                else:
                                    ui.label("AI 调用失败, 请检查 API Key 和网络").classes("text-warning")
                        except Exception as ai_err:
                            ai_container.clear()
                            with ai_container:
                                ui.label(f"AI 分析失败: {ai_err}").classes("text-warning")
        except Exception as e:
            ui.notify(f"扫描失败: {e}", type="negative")
        finally:
            progress.visible = False
            progress_label.visible = False

    ui.button("🔍 全面扫描", on_click=run, icon="search").props("color=primary")


def _strategy_hist(state):
    from src.data.rec_history import load_rec_history
    history = load_rec_history()
    if not history:
        ui.label("暂无推荐历史").classes("text-grey")
        return
    for h in reversed(history[-20:]):
        with ui.expansion(f"{h.predicted_at} | {h.stock_name} ({h.market}:{h.symbol}) — 共识{h.model_consensus} 最佳{h.best_strategy}", icon="history").classes("w-full"):
            with ui.row().classes("gap-3"):
                with ui.card().classes("text-center q-pa-sm"):
                    ui.label(f"{h.current_price:.2f}").classes("text-h6"); ui.label("当前价").classes("text-caption text-grey")
                with ui.card().classes("text-center q-pa-sm"):
                    ui.label(f"{h.model_consensus}").classes("text-h6"); ui.label("模型共识").classes("text-caption text-grey")
                with ui.card().classes("text-center q-pa-sm"):
                    ui.label(f"{h.best_strategy}").classes("text-h6"); ui.label("最佳策略").classes("text-caption text-grey")
                with ui.card().classes("text-center q-pa-sm"):
                    ui.label(f"{h.best_ret*100:+.1f}%").classes("text-h6"); ui.label("最佳收益").classes("text-caption text-grey")
                with ui.card().classes("text-center q-pa-sm"):
                    ui.label(f"{h.best_sharpe:.2f}").classes("text-h6"); ui.label("最佳夏普").classes("text-caption text-grey")
            if h.models_data:
                ui.markdown("🔮 模型预测").classes("q-mt-sm")
                m_rows = [{"模型": m.get("model",""), "方向": m.get("direction","") if not m.get("error") else f"❌ {m.get('error','')[:60]}",
                           "预测末价": f"{m.get('final_price',0):.2f}" if not m.get("error") else "-",
                           "涨跌幅": f"{m.get('pct_change',0):+.1f}%" if not m.get("error") else "-",
                           "MAPE": f"{m.get('mape',0):.1f}%" if not m.get("error") else "-"} for m in h.models_data]
                ui.table(rows=m_rows, columns=[{"name":c,"label":c,"field":c} for c in m_rows[0].keys()]).classes("w-full")
            if h.strategies_data:
                ui.markdown("📈 策略对比").classes("q-mt-sm")
                s_rows = [{"策略": s.get("strategy",""), "收益": f"{s.get('total_return',0)*100:+.1f}%", "夏普": f"{s.get('sharpe',0):.2f}",
                           "回撤": f"{s.get('max_dd',0)*100:.1f}%", "胜率": f"{s.get('win_rate',0)*100:.0f}%", "交易": s.get("total_trades",0),
                           "状态": "❌" if s.get("error") else "✅"} for s in h.strategies_data]
                ui.table(rows=s_rows, columns=[{"name":c,"label":c,"field":c} for c in s_rows[0].keys()]).classes("w-full")
            if h.report:
                with ui.expansion("📋 完整报告", icon="description"):
                    ui.markdown(h.report)
