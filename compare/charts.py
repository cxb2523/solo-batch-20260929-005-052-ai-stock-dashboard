"""Plotly figures: legacy single-stock chart and the multi-code compare view."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_advanced_chart(data, symbol):
    """Advanced candlestick chart with technical indicators (legacy signature)."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'Volume', 'MACD', 'RSI & Stochastic'),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )

    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]

    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma_col],
                          line=dict(color=colors[i], width=1.5), name=ma_name),
                row=1, col=1
            )

    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'],
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper',
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'],
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )

    volume_colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444'
                    for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'],
              marker_color=volume_colors, name='Volume', opacity=0.7),
        row=2, col=1
    )

    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume_SMA'],
                      line=dict(color='white', width=1), name='Vol SMA'),
            row=2, col=1
        )

    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'],
                      line=dict(color='#007aff', width=2), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_signal'],
                      line=dict(color='#ff9500', width=2), name='Signal'),
            row=3, col=1
        )

        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_histogram'],
                  marker_color=histogram_colors, name='Histogram', opacity=0.6),
            row=3, col=1
        )

    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'],
                      line=dict(color='#af52de', width=2), name='RSI'),
            row=4, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)

    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_K'],
                      line=dict(color='#ffcc00', width=1.5), name='Stoch %K'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_D'],
                      line=dict(color='#ff6600', width=1.5), name='Stoch %D'),
            row=4, col=1
        )

    fig.update_layout(
        title=f'{symbol} - Complete Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )

    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    return fig


def _error_rows(errors: dict[str, str]) -> list[list[str]]:
    return [[code, errors[code]] for code in sorted(errors)]


def create_compare_chart(
    normalized: pd.DataFrame,
    errors: dict[str, str] | None = None,
    *,
    title: str = "Multi-Code Compare (normalized to first close = 1.0)",
):
    """Overlay of normalized trends plus a per-row error panel.

    Failures never abort the figure: each bad code gets one row in the error
    table under the overlay.
    """
    errors = errors or {}
    rows = max(2, min(8, len(errors) + 1))
    height_ratio = 1.0 - 0.05 * rows
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[max(0.55, height_ratio), min(0.45, 0.05 * rows)],
        vertical_spacing=0.09,
        subplot_titles=(title, None),
    )

    for code in normalized.columns:
        col = normalized[code].dropna()
        if col.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=col.index,
                y=col.values,
                mode="lines",
                name=code,
                hovertemplate=f"{code}<br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    fig.update_yaxes(title_text="Relative level (first close = 1.0)", row=1, col=1)
    fig.update_xaxes(title_text="Trading day (union, forward-filled)", row=1, col=1)

    if errors:
        table_values = list(zip(*_error_rows(errors)))
        header = ["Code", "Error"]
        cells = [list(table_values[0]), list(table_values[1])]
        fill_color = [["#3a1212"] * len(cells[0])] * 2
    else:
        header = ["Code", "Error"]
        cells = [["-"], ["No per-code errors"]]
        fill_color = [["#123a1a"], ["#123a1a"]]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header],
                fill_color="#222",
                align="left",
                font=dict(color="white"),
            ),
            cells=dict(
                values=cells,
                fill_color=fill_color,
                align="left",
                font=dict(color="#ff8080" if errors else "#80ff9a"),
                height=24,
            ),
            columnwidth=[0.12, 0.88],
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=90, b=20, l=60, r=20),
    )
    return fig
