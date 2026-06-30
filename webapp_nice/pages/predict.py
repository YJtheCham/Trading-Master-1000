from nicegui import ui
from webapp.services import info_for, get_data_for, get_data_notify
from src.models.factory import run_models, list_models
import pandas as pd, numpy as np, plotly.graph_objects as go


def predict_page(state):
    with ui.tabs() as tabs:
        tab_new = ui.tab("🔮 新预测")
        tab_hist = ui.tab("📜 历史")
        tab_batch = ui.tab("📊 批量历史")
        tab_opt = ui.tab("⚙️ 优化")
    with ui.tab_panels(tabs, value=tab_new):
        with ui.tab_panel(tab_new): _predict_new(state)
        with ui.tab_panel(tab_hist): _predict_hist(state)
        with ui.tab_panel(tab_batch): _predict_batch(state)
        with ui.tab_panel(tab_opt): _predict_opt(state)


def _predict_new(state):
    stock_items, key_map = [], {}
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        d = f"{inf['symbol']} {inf['name']}"; stock_items.append(d); key_map[d] = k
    with ui.row().classes("items-end gap-2 q-mb-sm"):
        stock_sel = ui.select(options=stock_items, label="股票", value=stock_items[0] if stock_items else None).props("dense outlined").classes("w-56")
        model_sel = ui.select(options=list_models(), label="模型", value=["arima","gbdt","xgboost"], multiple=True).props("use-chips dense outlined").classes("w-40")
        steps_sl = ui.slider(min=5, max=90, value=30, step=5)
        ui.label().bind_text_from(steps_sl, "value", backward=lambda v: f"{v}天").classes("text-caption")
    col = ui.column()
    progress = ui.spinner(size="lg").classes("q-ml-md"); progress.visible = False
    progress_label = ui.label("").classes("text-caption text-grey q-ml-md")
    progress_label.visible = False

    async def run():
        col.clear()
        tgt = key_map.get(stock_sel.value)
        if not tgt: ui.notify("请选股", type="warning"); return
        progress.visible = True
        progress_label.visible = True
        progress_label.text = "获取数据中..."
        try:
            inf = info_for(tgt, state.get("stock_names", {}))
            df, src = get_data_notify(inf["symbol"], inf["market"], inf["name"], period_days=500)
            progress_label.text = "运行模型预测..."
            res = run_models(df, model_names=model_sel.value, steps=steps_sl.value, data_source=src)
            from src.data.pred_history import add_prediction
            for n, r in res.items():
                if len(r.forecast) == 0: continue
                mape_val = r.metrics.get("MAPE", 0)
                if isinstance(mape_val, str): mape_val = 0
                add_prediction(inf["symbol"], inf["market"], inf["name"], n,
                               r.forecast.tolist() if hasattr(r.forecast, 'tolist') else r.forecast,
                               r.forecast_dates, float(r.history[-1]), float(mape_val),
                               data_source=r.data_source, model_params=r.model_params)
            valid = [(n, r) for n, r in res.items() if len(r.forecast) > 0]
            with col:
                rows = []
                for n, r in res.items():
                    m = r.metrics
                    if "error" in m: rows.append({"模型": n, "状态": "❌", "方向": "-", "预测末价": "-"})
                    else:
                        d = "📈涨" if r.forecast[-1] > r.history[-1] else "📉跌"
                        rows.append({"模型": n, "状态": "✅", "方向": d, "预测末价": f"{r.forecast[-1]:.2f}",
                                     "MAPE": f"{m.get('MAPE','-'):.1f}%" if m.get('MAPE') else "-"})
                ui.table(rows=rows, columns=[{"name": c, "label": c, "field": c} for c in rows[0].keys()])
                if valid:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df["Close"].values[-60:], name="历史", line=dict(color="#888")))
                    cs = {"arima": "#E74C3C", "gbdt": "#2ECC71", "xgboost": "#3498DB", "lstm": "#9B59B6", "transformer": "#1ABC9C"}
                    for n, r in valid:
                        fc = r.forecast if hasattr(r.forecast, 'tolist') else np.array(r.forecast)
                        h = r.history if hasattr(r.history, 'tolist') else np.array(r.history)
                        fig.add_trace(go.Scatter(y=np.concatenate([[h[-1]], fc]), name=n, line=dict(color=cs.get(n, "#ccc"))))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#141824", plot_bgcolor="#141824", font=dict(color="#94a3b8"))
                    ui.plotly(fig)
        except Exception as e:
            ui.notify(f"预测失败: {e}", type="negative")
        finally:
            progress.visible = False
            progress_label.visible = False

    ui.button("▶ 开始预测", icon="play_arrow", on_click=run).props("color=primary").classes("q-mt-sm")


def _predict_hist(state):
    try:
        from src.data.pred_history import load_history
        h = load_history()
        if not h: ui.label("暂无记录").classes("text-grey"); return
        rows = []
        for r in h[-50:]:
            rows.append({"时间": r.predicted_at, "股票": f"{r.symbol} {r.stock_name}", "模型": r.model,
                         "价格": f"{r.last_price:.2f}", "MAPE": f"{r.mape:.1f}%"})
        ui.table(rows=rows, columns=[{"name": c, "label": c, "field": c} for c in rows[0].keys()])
    except Exception as e: ui.label(f"加载失败: {e}").classes("text-grey")


def _predict_batch(state):
    try:
        from src.data.batch_history import load_batch_history
        records = load_batch_history()
        if not records: ui.label("暂无批量记录").classes("text-grey"); return
        for rec in records[-10:]:
            s = rec.summary
            ui.markdown(f"**{rec.predicted_at}** — {len(s)}只").classes("text-subtitle2")
            if s:
                rows = [{"股票": x.get("name",""), "共识": x.get("up_count",""), "涨跌": f'{x.get("avg_pct",0):+.1f}%'} for x in s]
                ui.table(rows=rows, columns=[{"name": c, "label": c, "field": c} for c in rows[0].keys()])
            ui.separator()
    except Exception as e: ui.label(f"加载失败: {e}").classes("text-grey")


def _predict_opt(state):
    with ui.tabs() as otabs:
        tcv = ui.tab("🔬 CV")
        thpo = ui.tab("🎯 HPO")
    with ui.tab_panels(otabs, value=tcv):
        with ui.tab_panel(tcv): _cv_tab(state)
        with ui.tab_panel(thpo): _hpo_tab(state)


def _stock_opts(state):
    d = {}
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        d[f"{inf['symbol']} {inf['name']}"] = k
    return d


def _cv_tab(state):
    m = ui.select(options=["arima", "gbdt", "xgboost"], value="gbdt", label="模型").props("dense outlined")
    s = ui.select(options=[3, 5], value=5, label="折数").props("dense outlined")
    sel = ui.select(options=_stock_opts(state), value=None, label="股票").props("dense outlined")
    col = ui.column()

    async def run():
        col.clear()
        if not sel.value: ui.notify("请选股", type="warning"); return
        pk = _stock_opts(state)[sel.value]
        inf = info_for(pk, state.get("stock_names", {}))
        df = get_data_for(inf["symbol"], inf["market"], period_days=500)
        from src.models.factory import _get_registry
        cls = _get_registry().get(m.value)
        if not cls: ui.notify("模型不可用", type="negative"); return
        model = cls()
        try:
            cvr = model.cross_validate(df, n_splits=s.value)
            with col:
                ui.label(f"CV完成 — MAPE: {cvr.avg_mape:.2f}% ± {cvr.std_mape:.2f}").classes("text-subtitle1")
                rows = [{"Fold": i+1, "MAPE": f"{v:.2f}%"} for i, v in enumerate(cvr.fold_mapes)]
                ui.table(rows=rows, columns=[{"name": c, "label": c, "field": c} for c in ["Fold", "MAPE"]])
        except Exception as e: ui.notify(f"CV失败: {e}", type="negative")

    ui.button("🔬 运行CV", on_click=run, icon="science").props("color=primary")


def _hpo_tab(state):
    m = ui.select(options=["gbdt", "xgboost"], value="gbdt", label="模型").props("dense outlined")
    t = ui.slider(min=10, max=100, value=30, step=10); ui.label().bind_text_from(t, "value", backward=lambda v: f"{v}次").classes("text-caption")
    sel = ui.select(options=_stock_opts(state), value=None, label="股票").props("dense outlined")
    col = ui.column()

    async def run():
        col.clear()
        if not sel.value: ui.notify("请选股", type="warning"); return
        pk = _stock_opts(state)[sel.value]
        inf = info_for(pk, state.get("stock_names", {}))
        df = get_data_for(inf["symbol"], inf["market"], period_days=250)
        from src.models.factory import _get_registry
        cls = _get_registry().get(m.value)
        if not cls: ui.notify("模型不可用", type="negative"); return
        model = cls()
        try:
            best = model.hyperopt(df, n_trials=t.value)
            with col:
                ui.label(f"最优参数 — MAPE: {best.get('mape',0):.2f}%").classes("text-subtitle1")
                rows = [{"参数": k, "值": str(v)} for k, v in best.get("params", {}).items()]
                if rows: ui.table(rows=rows, columns=[{"name": c, "label": c, "field": c} for c in ["参数", "值"]])
        except Exception as e: ui.notify(f"HPO失败: {e}", type="negative")

    ui.button("🎯 运行HPO", on_click=run, icon="tune").props("color=primary")
