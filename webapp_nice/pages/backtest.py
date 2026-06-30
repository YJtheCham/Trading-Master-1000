from nicegui import ui
from webapp.services import info_for, get_data_for, get_data_notify
from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.strategies import (MovingAverageCrossStrategy, RSIStrategy, ChannelBreakoutStrategy, BollingerStrategy, RollingPredictionStrategy)
from src.models.gbdt import GBDTModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STRATEGIES = {
    "双均线(5/20)": lambda: MovingAverageCrossStrategy(5,20),
    "双均线(10/30)": lambda: MovingAverageCrossStrategy(10,30),
    "双均线(20/60)": lambda: MovingAverageCrossStrategy(20,60),
    "RSI(14)": lambda: RSIStrategy(14,30,70),
    "通道突破(20/10)": lambda: ChannelBreakoutStrategy(20,10),
    "布林带(20/2)": lambda: BollingerStrategy(20,2),
    "滚动预测(月频)": lambda: RollingPredictionStrategy(GBDTModel(),warmup=200,retrain_freq=20,threshold_buy=0.015,threshold_sell=-0.015),
    "滚动预测(周频)": lambda: RollingPredictionStrategy(GBDTModel(),warmup=200,retrain_freq=5,threshold_buy=0.01,threshold_sell=-0.01),
}

def backtest_page(state):
    stock_items, key_map = [], {}
    for k in state.get("watchlist", []):
        inf = info_for(k, state.get("stock_names", {}))
        d = f"{inf['symbol']} {inf['name']}"; stock_items.append(d); key_map[d] = k

    with ui.tabs() as tabs: tab_new = ui.tab("📈 新回测")
    with ui.tab_panels(tabs, value=tab_new):
        with ui.tab_panel(tab_new):
            with ui.row().classes("items-end gap-2 q-mb-sm"):
                stock_sel = ui.select(options=stock_items, label="股票", value=stock_items[0] if stock_items else None).props("dense outlined").classes("w-56")
                strat_sel = ui.select(options=list(STRATEGIES.keys()), value="双均线(5/20)", label="策略").props("dense outlined").classes("w-40")
                cap_inp = ui.number(label="资金", value=100000, min=10000, step=10000).props("dense outlined").classes("w-24")
            result_col = ui.column()
            progress = ui.spinner(size="lg"); progress.visible = False
            progress_label = ui.label("").classes("text-caption text-grey")
            progress_label.visible = False

            async def run():
                result_col.clear()
                if not stock_sel.value: ui.notify("请选股票", type="warning"); return
                target = key_map.get(stock_sel.value)
                if not target: ui.notify("请选股票", type="warning"); return
                progress.visible = True
                progress_label.visible = True
                progress_label.text = "获取数据中..."
                try:
                    inf = info_for(target, state.get("stock_names", {}))
                    df, src = get_data_notify(inf["symbol"], inf["market"], inf["name"], period_days=500)
                    if df.empty or len(df)<60: ui.notify("数据不足", type="negative"); return
                    progress_label.text = "运行回测..."
                    strat = STRATEGIES[strat_sel.value]()
                    cfg = BacktestConfig(initial_capital=cap_inp.value, market=inf["market"])
                    engine = BacktestEngine(df, strat, cfg)
                    result = engine.run()
                    with result_col:
                        with ui.row().classes("gap-3"):
                            for label,val in [("总收益",f"{result.total_return*100:+.1f}%"),("Sharpe",f"{result.sharpe_ratio:.2f}"),
                                              ("最大回撤",f"{result.max_drawdown*100:.1f}%"),("胜率",f"{result.win_rate*100:.1f}%"),
                                              ("交易次数",str(result.total_trades)),("盈利因子",f"{result.profit_factor:.2f}")]:
                                with ui.card().classes("text-center q-pa-sm"): ui.label(val).classes("text-h6"); ui.label(label).classes("text-caption text-grey")
                        if hasattr(engine,'equity_curve') and len(engine.equity_curve)>0:
                            eq = engine.equity_curve
                            peak = [max(eq[:i+1]) for i in range(len(eq))]
                            dd = [(eq[i]-peak[i])/peak[i]*100 for i in range(len(eq))]
                            fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.05)
                            fig.add_trace(go.Scatter(y=eq,name="净值",line=dict(color="#4fc3f7")),row=1,col=1)
                            fig.add_hline(y=cap_inp.value,line_dash="dash",line_color="#888",annotation_text="初始",row=1,col=1)
                            fig.add_trace(go.Scatter(y=dd,name="回撤%",fill="tozeroy",line=dict(color="#ef5350")),row=2,col=1)
                            fig.update_layout(height=400,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="#141824",plot_bgcolor="#141824",font=dict(color="#94a3b8"))
                            ui.plotly(fig)
                except Exception as e:
                    ui.notify(f"回测失败: {e}", type="negative")
                finally:
                    progress.visible = False
                    progress_label.visible = False

            ui.button("▶ 运行回测", on_click=run, icon="play_arrow").props("color=primary").classes("q-mt-sm")
