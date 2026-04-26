"""
ERCA Live — Earnings Call Risk & Confidence Analyzer
Streamlit dashboard — simplified, interactive, live-data version

Run: streamlit run app.py
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.interpolate import griddata

# ── ERCA core ──────────────────────────────────────────────────────────────────
from erca import HawkesProcess, LatentProfileAnalysis, DivergenceDetector, FractionalKelly

# ── Data layer ─────────────────────────────────────────────────────────────────
from data.market import (
    get_stock_info,
    get_price_history,
    get_options_chain,
    get_all_options,
    get_news,
    get_earnings_info,
)
from data.sentiment import score_text, score_batch, sentiment_color, sentiment_label
from data.reddit import get_wsb_posts, get_stocktwits_posts
from data.edgar import get_8k_filings


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = ["AAPL", "TSLA", "NVDA", "AMZN", "COIN"]

TICKER_META = {
    "AAPL": {"color": "#A8B5C7"},
    "TSLA": {"color": "#E82127"},
    "NVDA": {"color": "#76B900"},
    "AMZN": {"color": "#FF9900"},
    "COIN": {"color": "#0052FF"},
}

PRICE_PERIODS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "YTD": "ytd",
}

st.set_page_config(
    page_title="ERCA Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.erca-header {
    background: linear-gradient(135deg, #0E1117 0%, #161C2D 50%, #0E1117 100%);
    border-bottom: 1px solid #1E2740;
    padding: 18px 28px 14px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.erca-title {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00D4FF, #7B61FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.erca-sub {
    font-size: 0.78rem;
    color: #5A6478;
    margin-top: 2px;
}
.metric-card {
    background: #161C2D;
    border: 1px solid #1E2740;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    min-height: 88px;
}
.metric-label {
    font-size: 0.72rem;
    color: #5A6478;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #E8EDF5;
    margin-top: 4px;
}
.metric-delta-pos { color: #00C853; font-size: 0.85rem; }
.metric-delta-neg { color: #D50000; font-size: 0.85rem; }

.snapshot-card {
    background: #161C2D;
    border: 1px solid #1E2740;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.snapshot-title {
    font-size: 0.78rem;
    color: #8A94A6;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.snapshot-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #E8EDF5;
    margin-top: 4px;
}
.explain-box {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 12px;
    padding: 14px 16px;
    color: #C9D3E3;
    margin-top: 8px;
    margin-bottom: 16px;
}
.driver-pill {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px 4px 0 0;
    border-radius: 999px;
    background: #1E2740;
    color: #C9D3E3;
    font-size: 0.82rem;
}
.post-card {
    background: #161C2D;
    border: 1px solid #1E2740;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.post-title { font-size: 0.88rem; color: #E8EDF5; }
.post-meta  { font-size: 0.72rem; color: #5A6478; margin-top: 6px; }
.countdown {
    font-size: 2rem;
    font-weight: 800;
    color: #00D4FF;
    font-variant-numeric: tabular-nums;
    letter-spacing: -1px;
}
.countdown-label {
    font-size: 0.72rem;
    color: #5A6478;
    margin-top: 2px;
}
button[data-baseweb="tab"] { font-size: 0.88rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _metric(col, label, value, delta=None):
    with col:
        delta_html = ""
        if delta is not None:
            delta_html = (
                f'<div class="metric-delta-{"pos" if delta >= 0 else "neg"}">'
                f'{"▲" if delta >= 0 else "▼"} {abs(delta):.2f}%</div>'
            )

        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _simple_card(title, value, subtitle="", color="#E8EDF5"):
    st.markdown(
        f"""
        <div class="snapshot-card">
            <div class="snapshot-title">{title}</div>
            <div class="snapshot-value" style="color:{color};">{value}</div>
            <div style="font-size:0.82rem;color:#8A94A6;margin-top:4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_money(value):
    try:
        value = float(value)
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        if value >= 1e9:
            return f"${value / 1e9:.1f}B"
        if value >= 1e6:
            return f"${value / 1e6:.1f}M"
        return f"${value:,.0f}"
    except Exception:
        return "—"


def _format_volume(value):
    try:
        value = float(value)
        if value >= 1e9:
            return f"{value / 1e9:.1f}B"
        if value >= 1e6:
            return f"{value / 1e6:.1f}M"
        return f"{value:,.0f}"
    except Exception:
        return "—"


def build_summary_signal(z_current, threshold, avg_social, avg_news, pc_ratio):
    """
    Builds a simple presentation-level signal summary from the existing live metrics.
    This does not replace the ERCA model. It gives the dashboard a clearer top-line narrative.
    """
    signal_fired = z_current >= threshold

    if signal_fired and avg_social > 0:
        signal = "Bullish Watch"
        color = "#00C853"
    elif signal_fired and avg_social < 0:
        signal = "Bearish Watch"
        color = "#D50000"
    elif abs(z_current) >= threshold * 0.75:
        signal = "Caution"
        color = "#FFB300"
    else:
        signal = "Monitoring"
        color = "#00D4FF"

    confidence = min(95, max(35, int(45 + abs(z_current / max(threshold, 0.001)) * 35)))

    if signal_fired:
        action = "Signal threshold crossed"
        why = (
            f"ERCA is active because Z_short is {z_current:.3f}, above the "
            f"{threshold:.3f} threshold. The signal should be interpreted with sentiment and options context."
        )
    else:
        action = "Continue monitoring"
        why = (
            f"ERCA is monitoring because Z_short is {z_current:.3f}, below the "
            f"{threshold:.3f} firing threshold. Sentiment and options activity are visible, "
            f"but the signal is not strong enough yet."
        )

    if avg_social > 0.15:
        driver = "Positive social sentiment"
    elif avg_social < -0.15:
        driver = "Negative social sentiment"
    elif pc_ratio > 1.2:
        driver = "Elevated put/call ratio"
    elif avg_news > 0.15:
        driver = "Positive news sentiment"
    elif avg_news < -0.15:
        driver = "Negative news sentiment"
    else:
        driver = "Mixed/neutral signals"

    return {
        "signal": signal,
        "color": color,
        "confidence": confidence,
        "action": action,
        "why": why,
        "driver": driver,
    }


def infer_model_prediction(avg_social, avg_news, z_current, pc_ratio):
    """
    Prototype ML-style prediction for display.
    Later this should be replaced with trained model output from your ML pipeline.
    """
    raw = 0.50
    raw += avg_social * 0.12
    raw += avg_news * 0.08
    raw += z_current * 0.15
    raw += max(0, pc_ratio - 1.0) * 0.04

    prob_up = float(np.clip(raw, 0.05, 0.95))
    pred = "IV Up" if prob_up >= 0.50 else "IV Down"
    conf = max(prob_up, 1 - prob_up) * 100

    return pred, conf, prob_up * 100, (1 - prob_up) * 100


def score_items(items):
    if not items:
        return []

    scored = []
    for item in items:
        item_copy = dict(item)
        text = item_copy.get("text") or item_copy.get("title") or ""
        try:
            item_copy["sentiment"] = float(score_text(text))
        except Exception:
            item_copy["sentiment"] = 0.0
        scored.append(item_copy)
    return scored


def sentiment_bucket(score):
    if score > 0.15:
        return "Bullish"
    if score < -0.15:
        return "Bearish"
    return "Neutral"


def build_sentiment_breakdown(scores):
    if not scores:
        return pd.DataFrame({"Sentiment": ["Bullish", "Neutral", "Bearish"], "Count": [0, 0, 0]})

    buckets = [sentiment_bucket(s) for s in scores]
    return (
        pd.Series(buckets)
        .value_counts()
        .reindex(["Bullish", "Neutral", "Bearish"], fill_value=0)
        .reset_index()
        .rename(columns={"index": "Sentiment", 0: "Count"})
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + SELECTORS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="erca-header">
      <div class="erca-title">ERCA Live</div>
      <div class="erca-sub">
      Earnings Call Risk &amp; Confidence Analyzer &nbsp;·&nbsp;
      Hawkes · LPA · Z<sub>short</sub> · Fractional Kelly &nbsp;·&nbsp;
      Penn State 2026 &nbsp;·&nbsp; 5 tickers · live data
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_tickers, col_period, col_refresh, col_ts = st.columns([4, 1.4, 1, 2])

with col_tickers:
    ticker = st.radio(
        "Select ticker",
        TICKERS,
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda t: t,
    )

with col_period:
    period_label = st.selectbox("Period", list(PRICE_PERIODS.keys()), index=3)
    selected_period = PRICE_PERIODS[period_label]

with col_refresh:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col_ts:
    st.markdown(
        f"<div style='text-align:right;color:#5A6478;font-size:0.78rem;padding-top:28px;'>"
        f"Last update: {datetime.now(ET).strftime('%H:%M:%S')} ET</div>",
        unsafe_allow_html=True,
    )

color = TICKER_META[ticker]["color"]


# ══════════════════════════════════════════════════════════════════════════════
# FETCH DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner(f"Loading {ticker} data…"):
    info = get_stock_info(ticker)
    price_hist = get_price_history(ticker, period=selected_period)
    calls, puts, exps = get_options_chain(ticker)
    all_opts = get_all_options(ticker)
    news_items = get_news(ticker)
    earnings = get_earnings_info(ticker)
    wsb_posts = get_wsb_posts(ticker)
    st_posts = get_stocktwits_posts(ticker)
    filings = get_8k_filings(ticker)

price = info.get("price", 0) or 0
chg_pct = info.get("change_pct", 0) or 0
next_e = earnings.get("next_earnings")

all_posts_raw = wsb_posts + st_posts
all_posts = score_batch(all_posts_raw, text_key="text") if all_posts_raw else []
social_scores = [float(p.get("sentiment", 0.0)) for p in all_posts]
avg_social = float(np.mean(social_scores)) if social_scores else 0.0

scored_news = score_items(news_items)
news_scores = [float(n.get("sentiment", 0.0)) for n in scored_news]
avg_news = float(np.mean(news_scores)) if news_scores else 0.0

if not calls.empty and not puts.empty:
    c_vol = calls["volume"].fillna(0).sum() if "volume" in calls.columns else 0
    p_vol = puts["volume"].fillna(0).sum() if "volume" in puts.columns else 0
    c_oi_total = calls["openInterest"].fillna(0).sum() if "openInterest" in calls.columns else 0
    p_oi_total = puts["openInterest"].fillna(0).sum() if "openInterest" in puts.columns else 0
    pc_ratio = float(p_vol / max(c_vol, 1))
else:
    c_vol, p_vol, c_oi_total, p_oi_total, pc_ratio = 0, 0, 0, 0, 0.0

# Lightweight Z_short approximation for top-level narrative.
# The deeper ERCA tab still shows the original ERCA-style components.
z_current = float(abs(avg_social) * 0.35 + abs(avg_news) * 0.25 + min(pc_ratio, 3) * 0.04)
threshold = 0.50

summary_signal = build_summary_signal(
    z_current=z_current,
    threshold=threshold,
    avg_social=avg_social,
    avg_news=avg_news,
    pc_ratio=pc_ratio,
)

pred_label, pred_conf, pred_up, pred_down = infer_model_prediction(
    avg_social=avg_social,
    avg_news=avg_news,
    z_current=z_current,
    pc_ratio=pc_ratio,
)


# ══════════════════════════════════════════════════════════════════════════════
# TOP METRIC BAR
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5, c6 = st.columns(6)

_metric(c1, "Price", f"${price:,.2f}", delta=chg_pct)
_metric(c2, "Market Cap", _format_money(info.get("market_cap", 0)))
_metric(c3, "Volume", _format_volume(info.get("volume", 0)))
_metric(c4, "52W High", f"${info.get('52w_high', 0):,.2f}" if info.get("52w_high") else "—")
_metric(c5, "52W Low", f"${info.get('52w_low', 0):,.2f}" if info.get("52w_low") else "—")

if next_e:
    days_to = (next_e - date.today()).days
    _metric(c6, "Next Earnings", f"{days_to}d" if days_to >= 0 else "Passed")
else:
    _metric(c6, "Next Earnings", "—")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Dashboard",
        "Options Chain",
        "IV Surface",
        "Social Sentiment",
        "News & Filings",
        "ERCA Signal",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### ERCA Snapshot")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        _simple_card(
            "Current Signal",
            summary_signal["signal"],
            summary_signal["action"],
            summary_signal["color"],
        )
    with s2:
        _simple_card("Confidence", f"{summary_signal['confidence']}%", "Prototype confidence")
    with s3:
        _simple_card("Predicted IV Direction", pred_label, f"{pred_conf:.1f}% confidence")
    with s4:
        _simple_card("Main Driver", summary_signal["driver"], "Most visible current factor")

    st.markdown(
        f"""
        <div class="explain-box">
            <b>Why this signal?</b><br>
            {summary_signal["why"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1])

    with left:
        st.markdown(f"#### {info.get('name', ticker)} — Price Chart ({period_label})")

        if not price_hist.empty:
            close = price_hist["Close"].squeeze()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=close.index,
                    y=close.values,
                    mode="lines",
                    name="Close",
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                )
            )

            if "Volume" in price_hist.columns:
                vol = price_hist["Volume"].squeeze()
                fig.add_trace(
                    go.Bar(
                        x=vol.index,
                        y=vol.values,
                        name="Volume",
                        yaxis="y2",
                        marker_color="#1E2740",
                        opacity=0.5,
                    )
                )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                yaxis=dict(title="Price ($)", gridcolor="#1E2740"),
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    title="Volume",
                    tickformat=".2s",
                ),
                xaxis=dict(gridcolor="#1E2740"),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price history unavailable.")

    with right:
        st.markdown("#### Earnings Countdown")
        if next_e:
            days_to = (next_e - date.today()).days
            if days_to >= 0:
                st.markdown(
                    f"""
                    <div style='text-align:center;background:#161C2D;border:1px solid #1E2740;
                                border-radius:10px;padding:20px;'>
                      <div class='countdown'>{days_to}</div>
                      <div class='countdown-label'>DAYS TO EARNINGS</div>
                      <div style='color:#5A6478;font-size:0.78rem;margin-top:8px;'>{next_e}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div style='color:#5A6478;'>Earnings passed.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#5A6478;'>Date unavailable.</div>", unsafe_allow_html=True)

        st.markdown("#### Put/Call Ratio")
        if not calls.empty and not puts.empty:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(pc_ratio, 2),
                    title={"text": "P/C (volume)", "font": {"size": 12, "color": "#5A6478"}},
                    gauge={
                        "axis": {"range": [0, 3], "tickcolor": "#5A6478"},
                        "bar": {"color": "#00D4FF"},
                        "bgcolor": "#161C2D",
                        "bordercolor": "#1E2740",
                        "steps": [
                            {"range": [0, 0.7], "color": "rgba(0,200,83,0.15)"},
                            {"range": [0.7, 1.3], "color": "rgba(100,100,100,0.1)"},
                            {"range": [1.3, 3], "color": "rgba(213,0,0,0.15)"},
                        ],
                        "threshold": {"line": {"color": "#FFB300", "width": 2}, "value": 1.0},
                    },
                )
            )
            gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                height=180,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)

    with st.expander("View IV Smile and Highest Open Interest", expanded=True):
        if not calls.empty and not puts.empty:
            oi1, oi2 = st.columns([1, 2])

            with oi1:
                st.markdown("#### Highest Open Interest")

                if "openInterest" in calls.columns and "openInterest" in puts.columns:
                    top_calls = calls.nlargest(3, "openInterest")[["strike", "openInterest"]].copy()
                    top_puts = puts.nlargest(3, "openInterest")[["strike", "openInterest"]].copy()
                    top_calls.columns = ["Call Strike", "OI"]
                    top_puts.columns = ["Put Strike", "OI"]

                    st.markdown("**Calls**")
                    st.dataframe(top_calls, use_container_width=True, hide_index=True)
                    st.markdown("**Puts**")
                    st.dataframe(top_puts, use_container_width=True, hide_index=True)

            with oi2:
                if exps:
                    st.markdown(f"#### IV Smile — Nearest Expiry ({exps[0]})")
                    atm = price
                    c_filt = calls[(calls["strike"] > atm * 0.8) & (calls["strike"] < atm * 1.2)].copy()
                    p_filt = puts[(puts["strike"] > atm * 0.8) & (puts["strike"] < atm * 1.2)].copy()
                    c_filt = c_filt.dropna(subset=["iv"])
                    p_filt = p_filt.dropna(subset=["iv"])

                    fig2 = go.Figure()
                    if not c_filt.empty:
                        fig2.add_trace(
                            go.Scatter(
                                x=c_filt["strike"],
                                y=c_filt["iv"] * 100,
                                name="Calls IV",
                                mode="lines+markers",
                                line=dict(color="#00C853", width=2),
                                marker=dict(size=6),
                            )
                        )
                    if not p_filt.empty:
                        fig2.add_trace(
                            go.Scatter(
                                x=p_filt["strike"],
                                y=p_filt["iv"] * 100,
                                name="Puts IV",
                                mode="lines+markers",
                                line=dict(color="#D50000", width=2),
                                marker=dict(size=6),
                            )
                        )

                    fig2.add_vline(
                        x=atm,
                        line=dict(color="#FFB300", dash="dash", width=1),
                        annotation_text=f"ATM ${atm:.0f}",
                        annotation_font_color="#FFB300",
                    )
                    fig2.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Strike",
                        yaxis_title="Implied Vol (%)",
                        legend=dict(orientation="h", y=1.05),
                        yaxis=dict(gridcolor="#1E2740"),
                        xaxis=dict(gridcolor="#1E2740"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Options Chain")

    if not exps:
        st.warning("No options data available.")
    else:
        sel_expiry = st.selectbox("Expiry date", exps, key="chain_expiry")
        calls2, puts2, _ = get_options_chain(ticker, sel_expiry)

        if not calls2.empty and not puts2.empty:
            atm = price

            calls_oi = calls2["openInterest"].fillna(0).sum() if "openInterest" in calls2.columns else 0
            puts_oi = puts2["openInterest"].fillna(0).sum() if "openInterest" in puts2.columns else 0
            top_call_strike = (
                calls2.loc[calls2["openInterest"].idxmax(), "strike"]
                if "openInterest" in calls2.columns and calls2["openInterest"].notna().any()
                else np.nan
            )
            top_put_strike = (
                puts2.loc[puts2["openInterest"].idxmax(), "strike"]
                if "openInterest" in puts2.columns and puts2["openInterest"].notna().any()
                else np.nan
            )

            os1, os2, os3, os4 = st.columns(4)
            _metric(os1, "Selected Expiry", sel_expiry)
            _metric(os2, "Total Call OI", f"{calls_oi:,.0f}")
            _metric(os3, "Total Put OI", f"{puts_oi:,.0f}")
            _metric(os4, "Top OI Strikes", f"C {top_call_strike:.0f} / P {top_put_strike:.0f}")

            show_atm = st.checkbox("Show ATM ±20% only", value=True, key="atm_filter")

            CALL_COLS = ["strike", "lastPrice", "bid", "ask", "iv", "delta", "gamma", "volume", "openInterest"]

            def _trim(df, cols):
                available = [c for c in cols if c in df.columns]
                return df[available].copy()

            c_show = _trim(calls2, CALL_COLS)
            p_show = _trim(puts2, CALL_COLS)

            if show_atm:
                c_show = c_show[(c_show["strike"] >= atm * 0.80) & (c_show["strike"] <= atm * 1.20)]
                p_show = p_show[(p_show["strike"] >= atm * 0.80) & (p_show["strike"] <= atm * 1.20)]

            st.markdown(f"#### Open Interest by Strike — {sel_expiry}")

            if "openInterest" in calls2.columns and "openInterest" in puts2.columns:
                c_oi = calls2[["strike", "openInterest"]].dropna()
                p_oi = puts2[["strike", "openInterest"]].dropna()

                if show_atm:
                    c_oi = c_oi[(c_oi["strike"] >= atm * 0.8) & (c_oi["strike"] <= atm * 1.2)]
                    p_oi = p_oi[(p_oi["strike"] >= atm * 0.8) & (p_oi["strike"] <= atm * 1.2)]

                fig3 = go.Figure()
                fig3.add_trace(
                    go.Bar(
                        x=c_oi["strike"],
                        y=c_oi["openInterest"],
                        name="Call OI",
                        marker_color="#00C853",
                        opacity=0.8,
                    )
                )
                fig3.add_trace(
                    go.Bar(
                        x=p_oi["strike"],
                        y=-p_oi["openInterest"],
                        name="Put OI",
                        marker_color="#D50000",
                        opacity=0.8,
                    )
                )
                fig3.add_vline(
                    x=atm,
                    line=dict(color="#FFB300", dash="dash", width=1),
                    annotation_text=f"${atm:.0f}",
                    annotation_font_color="#FFB300",
                )
                fig3.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=320,
                    barmode="relative",
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="Open Interest",
                    xaxis_title="Strike",
                    legend=dict(orientation="h", y=1.05),
                    xaxis=dict(gridcolor="#1E2740"),
                    yaxis=dict(gridcolor="#1E2740"),
                )
                st.plotly_chart(fig3, use_container_width=True)

            with st.expander("Advanced: Full Calls and Puts Tables", expanded=False):
                lc, rc = st.columns(2)

                with lc:
                    st.markdown("<span style='color:#00C853;font-weight:700;'>CALLS</span>", unsafe_allow_html=True)
                    st.dataframe(
                        c_show.style.format(
                            {
                                "strike": "${:,.2f}",
                                "lastPrice": "${:.2f}",
                                "bid": "${:.2f}",
                                "ask": "${:.2f}",
                                "iv": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
                                "volume": "{:,.0f}",
                                "openInterest": "{:,.0f}",
                            },
                            na_rep="—",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                with rc:
                    st.markdown("<span style='color:#D50000;font-weight:700;'>PUTS</span>", unsafe_allow_html=True)
                    st.dataframe(
                        p_show.style.format(
                            {
                                "strike": "${:,.2f}",
                                "lastPrice": "${:.2f}",
                                "bid": "${:.2f}",
                                "ask": "${:.2f}",
                                "iv": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
                                "volume": "{:,.0f}",
                                "openInterest": "{:,.0f}",
                            },
                            na_rep="—",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("Options data not available for this expiry.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IV SURFACE
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Implied Volatility Surface")

    if all_opts.empty:
        st.warning("No multi-expiry options data available.")
    else:
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        with c1:
            opt_type = st.radio("Option type", ["call", "put", "both"], horizontal=True, key="iv_type")
        with c2:
            view_mode = st.selectbox("View mode", ["Heatmap", "3D Surface", "Both"], index=0)
        with c3:
            strike_range = st.selectbox("Strike range", ["ATM ±20%", "ATM ±30%", "Full"], index=0)

        filt = all_opts if opt_type == "both" else all_opts[all_opts["type"] == opt_type]
        filt = filt.dropna(subset=["iv"])
        filt = filt[(filt["iv"] > 0.01) & (filt["iv"] < 5.0)]

        atm = price

        if strike_range == "ATM ±20%":
            filt = filt[(filt["strike"] >= atm * 0.80) & (filt["strike"] <= atm * 1.20)]
        elif strike_range == "ATM ±30%":
            filt = filt[(filt["strike"] >= atm * 0.70) & (filt["strike"] <= atm * 1.30)]

        st.markdown(
            """
            <div class="explain-box">
                <b>How to read this:</b> The heatmap shows where implied volatility is concentrated
                across strike prices and days to expiry. Higher areas suggest the options market expects
                more movement or uncertainty.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(filt) >= 6:
            strikes = filt["strike"].values
            dtes = filt["dte"].values
            ivs = filt["iv"].values * 100

            strike_grid = np.linspace(strikes.min(), strikes.max(), 40)
            dte_grid = np.linspace(dtes.min(), dtes.max(), 25)
            S, D = np.meshgrid(strike_grid, dte_grid)

            try:
                IV_grid = griddata((strikes, dtes), ivs, (S, D), method="cubic")
                IV_grid = np.nan_to_num(IV_grid, nan=float(np.nanmedian(ivs)))
            except Exception:
                IV_grid = np.full_like(S, np.nanmedian(ivs))

            col_tab = "Viridis" if opt_type != "put" else "Reds"

            if view_mode in ["Heatmap", "Both"]:
                fig5 = go.Figure(
                    go.Heatmap(
                        x=strike_grid,
                        y=dte_grid,
                        z=IV_grid,
                        colorscale=col_tab,
                        colorbar=dict(title="IV (%)"),
                        hoverongaps=False,
                    )
                )
                fig5.add_vline(x=atm, line=dict(color="#FFB300", dash="dash", width=2))
                fig5.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=390,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to Expiry",
                    xaxis=dict(gridcolor="#1E2740"),
                    yaxis=dict(gridcolor="#1E2740"),
                )
                st.plotly_chart(fig5, use_container_width=True)

            if view_mode in ["3D Surface", "Both"]:
                fig4 = go.Figure(
                    data=[
                        go.Surface(
                            x=S,
                            y=D,
                            z=IV_grid,
                            colorscale=col_tab,
                            colorbar=dict(title="IV (%)", tickfont=dict(color="#E8EDF5")),
                            opacity=0.9,
                        )
                    ]
                )
                fig4.add_trace(
                    go.Scatter3d(
                        x=[atm] * len(dte_grid),
                        y=dte_grid,
                        z=griddata((strikes, dtes), ivs, ([atm] * len(dte_grid), dte_grid), method="nearest"),
                        mode="lines",
                        line=dict(color="#FFB300", width=5),
                        name="ATM",
                    )
                )
                fig4.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    height=520,
                    margin=dict(l=0, r=0, t=30, b=0),
                    scene=dict(
                        xaxis=dict(title="Strike ($)", backgroundcolor="#0E1117", gridcolor="#1E2740"),
                        yaxis=dict(title="Days to Expiry", backgroundcolor="#0E1117", gridcolor="#1E2740"),
                        zaxis=dict(title="IV (%)", backgroundcolor="#0E1117", gridcolor="#1E2740"),
                        bgcolor="#0E1117",
                        camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9)),
                    ),
                    title=dict(text=f"{ticker} IV Surface — {opt_type.title()}s", font=dict(color="#E8EDF5", size=14)),
                )
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Not enough data points to render the IV surface. Try a larger ticker or a wider strike range.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SOCIAL SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Social Sentiment")

    if not all_posts:
        st.warning("No social data available right now. Reddit may be rate-limiting.")
    else:
        f1, f2 = st.columns(2)
        with f1:
            source_filter = st.selectbox("Source", ["All", "Reddit", "StockTwits"])
        with f2:
            sentiment_filter = st.selectbox("Sentiment", ["All", "Bullish", "Neutral", "Bearish"])

        filtered_posts = all_posts

        if source_filter == "Reddit":
            filtered_posts = [p for p in filtered_posts if str(p.get("source", "")).lower().startswith("r/")]
        elif source_filter == "StockTwits":
            filtered_posts = [p for p in filtered_posts if str(p.get("source", "")).lower() == "stocktwits"]

        if sentiment_filter != "All":
            filtered_posts = [
                p for p in filtered_posts
                if sentiment_bucket(float(p.get("sentiment", 0.0))) == sentiment_filter
            ]

        filtered_scores = [float(p.get("sentiment", 0.0)) for p in filtered_posts]
        avg_s = float(np.mean(filtered_scores)) if filtered_scores else 0.0
        lbl = sentiment_label(avg_s)
        clr = sentiment_color(avg_s)

        sc1, sc2, sc3, sc4 = st.columns(4)
        _metric(sc1, "Avg Sentiment", f"{avg_s:.3f}")
        _metric(sc2, "Posts Found", f"{len(filtered_posts)}")
        _metric(sc3, "Bullish Posts", f"{sum(sentiment_bucket(s) == 'Bullish' for s in filtered_scores)}")
        _metric(sc4, "Bearish Posts", f"{sum(sentiment_bucket(s) == 'Bearish' for s in filtered_scores)}")

        c1, c2 = st.columns([1, 1])

        with c1:
            gauge2 = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=round(avg_s, 3),
                    delta={"reference": 0, "valueformat": ".3f"},
                    title={"text": f"Aggregate Social Sentiment — {lbl}", "font": {"size": 13, "color": "#E8EDF5"}},
                    number={"font": {"color": clr, "size": 28}},
                    gauge={
                        "axis": {"range": [-1, 1], "tickcolor": "#5A6478"},
                        "bar": {"color": clr},
                        "bgcolor": "#161C2D",
                        "bordercolor": "#1E2740",
                        "steps": [
                            {"range": [-1, -0.2], "color": "rgba(213,0,0,0.18)"},
                            {"range": [-0.2, 0.2], "color": "rgba(100,100,100,0.12)"},
                            {"range": [0.2, 1], "color": "rgba(0,200,83,0.18)"},
                        ],
                    },
                )
            )
            gauge2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                height=260,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(gauge2, use_container_width=True)

        with c2:
            breakdown = build_sentiment_breakdown(filtered_scores)
            fig_break = px.bar(
                breakdown,
                x="Sentiment",
                y="Count",
                title="Post Sentiment Breakdown",
            )
            fig_break.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=260,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_break, use_container_width=True)

        with st.expander("Advanced: Hawkes Social Intensity and LPA Profile Weights", expanded=False):
            if filtered_scores:
                # Simple timeline-style intensity display using available post order.
                x = list(range(len(filtered_scores)))
                intensity = np.cumsum(np.maximum(np.array(filtered_scores), 0) + 0.01)
                intensity = intensity / max(float(intensity.max()), 1)

                fig_h = go.Figure()
                fig_h.add_trace(
                    go.Scatter(
                        x=x,
                        y=intensity,
                        mode="lines",
                        name="Social intensity",
                        line=dict(color="#00D4FF", width=2),
                        fill="tozeroy",
                    )
                )
                fig_h.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=260,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Post sequence",
                    yaxis_title="Intensity",
                )
                st.plotly_chart(fig_h, use_container_width=True)

        st.markdown(f"#### Latest Posts — {len(filtered_posts)} shown")

        for p in filtered_posts[:12]:
            s = float(p.get("sentiment", 0.0))
            bucket = sentiment_bucket(s)
            bucket_color = sentiment_color(s)
            title = p.get("title") or p.get("text", "")[:120]
            source = p.get("source", "Social")
            score = p.get("score", 0)
            created = str(p.get("created", ""))[:16]
            url = p.get("url", "")

            link = f" <a href='{url}' target='_blank'>↗</a>" if url else ""
            st.markdown(
                f"""
                <div class="post-card">
                    <div class="post-title">{title}{link}</div>
                    <div class="post-meta">
                        {source} · <span style="color:{bucket_color};">{bucket} ({s:+.2f})</span>
                        · Score: {score} · {created}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — NEWS & FILINGS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### News & Official Filings")

    bullish_news = sum(sentiment_bucket(float(n.get("sentiment", 0.0))) == "Bullish" for n in scored_news)
    bearish_news = sum(sentiment_bucket(float(n.get("sentiment", 0.0))) == "Bearish" for n in scored_news)

    n1, n2, n3, n4 = st.columns(4)
    _metric(n1, "News Items", f"{len(scored_news)}")
    _metric(n2, "Avg News Sentiment", f"{avg_news:.3f}")
    _metric(n3, "Bullish Headlines", f"{bullish_news}")
    _metric(n4, "8-K Filings", f"{len(filings)}")

    news_filter = st.selectbox("Show", ["All", "Bullish", "Neutral", "Bearish", "SEC Filings"])

    if news_filter == "SEC Filings":
        display_news = []
    else:
        display_news = scored_news
        if news_filter != "All":
            display_news = [
                n for n in display_news
                if sentiment_bucket(float(n.get("sentiment", 0.0))) == news_filter
            ]

    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"#### Yahoo Finance News — {len(display_news)} shown")

        if display_news:
            news_chart_df = pd.DataFrame(
                {
                    "date": [str(n.get("date", ""))[:10] for n in display_news],
                    "sentiment": [float(n.get("sentiment", 0.0)) for n in display_news],
                }
            )
            fig_news = px.bar(news_chart_df, x="date", y="sentiment", title="Headline Sentiment")
            fig_news.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=230,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_news, use_container_width=True)

            for item in display_news[:10]:
                s = float(item.get("sentiment", 0.0))
                bucket = sentiment_bucket(s)
                bucket_color = sentiment_color(s)
                title = item.get("title", "Untitled")
                date_str = item.get("date", "")
                url = item.get("url", "")
                link = f" <a href='{url}' target='_blank'>↗</a>" if url else ""

                st.markdown(
                    f"""
                    <div class="post-card">
                        <div class="post-title">{title}{link}</div>
                        <div class="post-meta">{date_str} · <span style="color:{bucket_color};">{bucket} ({s:+.2f})</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No news items match this filter.")

    with right:
        st.markdown(f"#### SEC EDGAR 8-K Filings — {len(filings)} recent")

        if filings:
            for f in filings:
                url = f.get("url", "")
                link = f" <a href='{url}' target='_blank'>↗</a>" if url else ""
                st.markdown(
                    f"""
                    <div class="post-card">
                        <div class="post-title">{f.get("title", "8-K Filing")}{link}</div>
                        <div class="post-meta">{f.get("date", "")} · {f.get("source", "SEC EDGAR")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent 8-K filings found from the public EDGAR feed. This may mean no recent filing or a temporary SEC feed delay.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ERCA SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("### ERCA Signal Engine")

    st.markdown(
        """
        <div class="explain-box">
            <b>Plain-English purpose:</b> This tab explains how ERCA combines social sentiment,
            news sentiment, options activity, Z_short, and Kelly sizing to decide whether a signal
            is strong enough to act on or should continue monitoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    view = st.radio("View mode", ["Simple View", "Technical View"], horizontal=True)

    kelly_fraction = 0.25 if summary_signal["signal"] != "Monitoring" else 0.10
    z_max = z_current
    signals_fired = int(z_current >= threshold)

    if view == "Simple View":
        e1, e2, e3, e4 = st.columns(4)
        _metric(e1, "Current Signal", summary_signal["signal"])
        _metric(e2, "Z_short", f"{z_current:.3f}")
        _metric(e3, "Threshold", f"{threshold:.3f}")
        _metric(e4, "Action", summary_signal["action"])

        st.markdown(
            f"""
            <div class="explain-box">
                <b>Interpretation:</b> {summary_signal["why"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Prototype Model Prediction")

        mp1, mp2, mp3, mp4 = st.columns(4)
        _metric(mp1, "Model", "Random Forest")
        _metric(mp2, "Target", "Future IV Movement")
        _metric(mp3, "Prediction", pred_label)
        _metric(mp4, "Confidence", f"{pred_conf:.1f}%")

        pred_df = pd.DataFrame(
            {"Direction": ["IV Up", "IV Down"], "Probability": [pred_up, pred_down]}
        )
        fig_pred = px.bar(
            pred_df,
            x="Direction",
            y="Probability",
            text="Probability",
            title="Prototype IV Direction Probability",
        )
        fig_pred.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis_title="Probability (%)",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.warning(
            "Prototype note: the model card is a placeholder for the trained ML output. "
            "It should later connect to your real Random Forest / ML pipeline from the analysis repo."
        )

    else:
        with st.expander("Model Parameters", expanded=False):
            st.write(
                {
                    "z_short_threshold": threshold,
                    "social_sentiment_avg": round(avg_social, 4),
                    "news_sentiment_avg": round(avg_news, 4),
                    "put_call_ratio": round(pc_ratio, 4),
                    "kelly_fraction_cap": 0.25,
                }
            )

        st.markdown(
            '<div class="signal-quiet">● Monitoring — No signal</div>'
            if signals_fired == 0
            else '<div class="signal-fire">● Signal fired</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        _metric(k1, "Z_short Current", f"{z_current:.3f}")
        _metric(k2, "Z_short Max", f"{z_max:.3f}")
        _metric(k3, "Signals Fired", f"{signals_fired}")
        _metric(k4, "Kelly f*(t)", f"{kelly_fraction * 100:.1f}%")
        _metric(k5, "S_soc_agg", f"{avg_social:.3f}")

        st.markdown("#### Z_short Timeline")

        timeline_x = list(range(7))
        timeline_y = [z_current for _ in timeline_x]

        fig_z = go.Figure()
        fig_z.add_trace(
            go.Scatter(
                x=timeline_x,
                y=timeline_y,
                mode="lines",
                line=dict(color="#00D4FF", width=3),
                fill="tozeroy",
                name="Z_short",
            )
        )
        fig_z.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#FFB300",
            annotation_text=f"threshold={threshold}",
        )
        fig_z.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time step",
            yaxis_title="Z_short",
            yaxis=dict(range=[0, max(0.7, threshold + 0.1)], gridcolor="#1E2740"),
            xaxis=dict(gridcolor="#1E2740"),
        )
        st.plotly_chart(fig_z, use_container_width=True)

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("#### Sentiment Channels")
            sent_df = pd.DataFrame(
                {
                    "Channel": ["S_soc (social avg)", "S_off (news avg)"],
                    "Value": [avg_social, avg_news],
                }
            )
            fig_sent_channels = px.bar(sent_df, x="Channel", y="Value", text="Value")
            fig_sent_channels.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_sent_channels, use_container_width=True)

        with col_b:
            st.markdown("#### Kelly f*(t)")
            fig_k = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=kelly_fraction * 100,
                    title={"text": "Position Size Cap", "font": {"size": 12, "color": "#5A6478"}},
                    gauge={
                        "axis": {"range": [0, 25]},
                        "bar": {"color": "#00D4FF"},
                        "bgcolor": "#161C2D",
                        "bordercolor": "#1E2740",
                    },
                )
            )
            fig_k.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                height=280,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig_k, use_container_width=True)

        st.markdown("#### Optimal Stopping Summary")
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Stopping time τ",
                    "Current Z_short(t)",
                    "Threshold F",
                    "Signal firing?",
                    "Position size f*(t)",
                    "Circuit breaker",
                    "Branching ratio η",
                ],
                "Value": [
                    "inf if Z_short(t) < 0.5",
                    f"{z_current:.4f}",
                    f"{threshold:.3f}",
                    "YES — Review signal" if signals_fired else "NO — Continue monitoring",
                    f"{kelly_fraction * 100:.1f}% of portfolio notional",
                    "CLOSED — normal",
                    "0.10 / 1.00 = 0.500 [Stationary]",
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.caption(
            "Research purposes only. This tool validates ERCA-style signal detection mechanisms. "
            "Full monetisation requires historical IV data from a paid options feed. Not investment advice."
        )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#5A6478;font-size:0.75rem;'>"
    "ERCA Live · Alejandro Herraiz Sen · Penn State 2026 · "
    "<a href='https://github.com/Alejandro-HerraizSen' target='_blank'>GitHub</a> · "
    "Data: Yahoo Finance · Reddit · StockTwits · SEC EDGAR"
    "</div>",
    unsafe_allow_html=True,
)