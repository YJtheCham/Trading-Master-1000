"""
K线图 + 技术指标渲染
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def kline_chart(df: pd.DataFrame, title: str = "",
                indicators: list[str] | None = None,
                height: int = 600) -> go.Figure:
    """K线图 + 成交量 + 可选技术指标

    indicators: ["macd", "rsi", "ma", "bollinger"]
    """
    if indicators is None:
        indicators = ["ma", "macd"]

    # 计算子图数量: K线 + 成交量 + 额外指标
    extra = sum(1 for i in indicators if i in ("macd", "rsi"))
    rows = 2 + extra
    row_heights = [0.5, 0.2] + [0.15] * extra
    subplot_titles = [title] + ["成交量"] + [i.upper() for i in indicators if i in ("macd", "rsi")]
    vertical_spacing = 0.04

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="K线", increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a",
    ), row=1, col=1)

    # MA 均线
    if "ma" in indicators:
        for period, color in [(5, "#ff9800"), (20, "#e91e63"), (60, "#2196f3")]:
            if len(df) >= period:
                ma = df["Close"].rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=ma, name=f"MA{period}",
                    line=dict(color=color, width=1), opacity=0.7,
                ), row=1, col=1)

    # 布林带
    if "bollinger" in indicators and len(df) >= 20:
        ma20 = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        upper = ma20 + 2 * std
        lower = ma20 - 2 * std
        fig.add_trace(go.Scatter(
            x=df["Date"], y=upper, name="BB upper",
            line=dict(color="gray", width=0.5, dash="dot"), opacity=0.5,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Date"], y=lower, name="BB lower",
            line=dict(color="gray", width=0.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)", opacity=0.5,
        ), row=1, col=1)

    # 成交量
    colors = ["#ef5350" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#26a69a"
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="成交量",
                         marker_color=colors, opacity=0.5), row=2, col=1)

    row_idx = 3
    # MACD
    if "macd" in indicators and len(df) >= 26:
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        fig.add_trace(go.Bar(x=df["Date"], y=histogram, name="MACD Hist",
                             marker_color=["#ef5350" if v >= 0 else "#26a69a"
                                          for v in histogram]), row=row_idx, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=macd, name="MACD",
                                 line=dict(color="#2196f3", width=1)),
                      row=row_idx, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=signal, name="Signal",
                                 line=dict(color="#ff9800", width=1)),
                      row=row_idx, col=1)
        row_idx += 1

    # RSI
    if "rsi" in indicators and len(df) >= 14:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=df["Date"], y=rsi, name="RSI",
                                 line=dict(color="#9c27b0", width=1)),
                      row=row_idx, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      opacity=0.3, row=row_idx, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      opacity=0.3, row=row_idx, col=1)

    fig.update_layout(
        height=height, hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    return fig
