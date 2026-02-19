"""Streamlit dashboard for gamma exposure analytics.

Replaces matplotlib plots with an interactive Plotly heatmap.
Run with: streamlit run dashboard.py
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytz
import streamlit as st
import plotly.graph_objects as go

from main import (
    _load_broker_client,
    BrokerConfigurationError,
    GammaExposureScheduler,
)
from gamma_analysis import calculate_gamma_exposure, get_per_strike_details


# --- Session-state client (persist across reruns) ---
@st.cache_resource
def get_authenticated_client():
    """Load broker and authenticate once per session."""
    try:
        broker_name, auth_module, client_module, secrets = _load_broker_client(
            os.environ.get("BROKER")
        )
        scheduler = GammaExposureScheduler(os.environ.get("BROKER"))
        scheduler.authenticate()
        default_symbol = getattr(secrets, "option_symbol", "$SPX.X")
        return scheduler.client, broker_name, default_symbol, client_module
    except BrokerConfigurationError as e:
        st.error(str(e))
        return None, None, "$SPX.X", None


def generate_gex_interpretation(
    spot_price: float,
    total_gex: float,
    king_strike: Optional[float],
    top_positive: List[float],
    top_negative: List[float],
    per_strike_gex: Dict[float, float],
) -> str:
    """Generate a 3-4 sentence interpretation of the GEX dashboard."""
    sentences = []

    # Sentence 1: Overall context
    net = "positive" if total_gex > 0 else "negative"
    sentences.append(
        f"With spot at ${spot_price:.0f}, total gamma exposure stands at ${total_gex:.2f}B ({net}), "
        f"indicating dealers are {'long gamma overall and may dampen volatility' if total_gex > 0 else 'short gamma and may amplify directional moves'}."
    )

    # Sentence 2: King Node
    if king_strike is not None:
        king_gex = per_strike_gex.get(king_strike, 0)
        king_role = "support" if king_gex > 0 else "resistance"
        above_below = "above" if spot_price > king_strike else "below"
        sentences.append(
            f"The King Node at ${king_strike:.0f} (${abs(king_gex):.2f}B {'positive' if king_gex > 0 else 'negative'}) "
            f"acts as strong {king_role}; with spot {above_below} this level, expect price to be "
            f"{'attracted toward' if king_gex > 0 else 'repelled from'} ${king_strike:.0f} as dealers hedge."
        )
    else:
        sentences.append("No dominant King Node is present in the current strike range.")

    # Sentence 3: Gatekeepers
    supp = [s for s in top_positive if s != king_strike][:2]
    res = [s for s in top_negative if s != king_strike][:2]
    gatekeeper_parts = []
    if supp:
        supp_str = ", ".join(f"${s:.0f}" for s in supp)
        gatekeeper_parts.append(f"support gatekeepers at {supp_str}")
    if res:
        res_str = ", ".join(f"${s:.0f}" for s in res)
        gatekeeper_parts.append(f"resistance gatekeepers at {res_str}")
    if gatekeeper_parts:
        sentences.append(
            "Key " + " and ".join(gatekeeper_parts) + " "
            "add additional layers of dealer hedging that can slow or reverse moves as price approaches these strikes."
        )

    # Sentence 4: Near-term implication
    references = [s for s in ([king_strike] if king_strike else []) + supp + res if s]
    if references:
        nearest = min(references, key=lambda s: abs(s - spot_price))
        direction = "up" if spot_price < nearest else "down"
        sentences.append(
            f"The nearest major gamma level to spot is ${nearest:.0f}; "
            f"a break {'above' if direction == 'up' else 'below'} could accelerate moves as dealer hedging flips."
        )

    return " ".join(sentences)


def generate_trader_suggestions(
    spot_price: float,
    total_gex: float,
    king_strike: Optional[float],
    top_positive: List[float],
    top_negative: List[float],
    per_strike_gex: Dict[float, float],
) -> str:
    """Generate 3-4 sentence trading suggestions based on GEX profile."""
    sentences = []

    supp = [s for s in top_positive if s != king_strike][:2]
    res = [s for s in top_negative if s != king_strike][:2]
    key_levels = [s for s in ([king_strike] if king_strike else []) + supp + res if s]

    # Suggestion 1: Key levels to watch
    if key_levels:
        levels_str = ", ".join(f"${s:.0f}" for s in sorted(key_levels, reverse=True))
        sentences.append(
            f"Traders should watch {levels_str} as primary decision points; "
            f"expect potential reversals or pauses at positive-gamma strikes and breakout acceleration through negative-gamma resistance."
        )
    else:
        sentences.append("Monitor the nearest high-OI strikes for potential support and resistance.")

    # Suggestion 2: Fade vs follow based on total GEX
    if total_gex > 0:
        sentences.append(
            "With positive total GEX, consider fading extended moves—mean reversion toward the King Node may offer better risk/reward than chasing breakouts."
        )
    else:
        sentences.append(
            "With negative total GEX, be cautious fading breakouts; momentum can extend as dealer hedging amplifies directional moves."
        )

    # Suggestion 3: Near-spot trade idea
    if king_strike is not None:
        dist = spot_price - king_strike
        if abs(dist) < 50:
            sentences.append(
                f"Spot is near the King Node at ${king_strike:.0f}; range-bound behavior is likely until a decisive break—tighten stops or reduce size until direction is clear."
            )
        elif dist > 0:
            sentences.append(
                f"With spot above the King Node, a pullback toward ${king_strike:.0f} could present a long entry if it holds as support; invalidation below suggests follow-through lower."
            )
        else:
            sentences.append(
                f"With spot below the King Node, a rally toward ${king_strike:.0f} may stall as resistance; a breakout above would signal dealer hedging has flipped and could extend higher."
            )

    # Suggestion 4: General risk reminder
    sentences.append(
        "As always, use proper position sizing and respect key gamma levels as both profit targets and stop-loss references."
    )

    return " ".join(sentences)


def fetch_options_and_gex(
    client,
    option_symbol: str,
    strike_count: int,
    previous_gamma: Optional[Dict[float, float]] = None,
    client_module=None,
) -> Tuple[Optional[Tuple[Dict, float, Dict[float, float], Dict[float, Dict], float, datetime.date]], Optional[str]]:
    """Fetch option chain, compute GEX.
    Returns (result_tuple, error_message). Result is None on failure."""
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    if now.weekday() == 4 and now.hour >= 16:
        use_date = (now + timedelta(days=3)).date()
    else:
        use_date = now.date() + timedelta(days=1 if now.hour >= 16 else 0)

    options_source = getattr(client, "get_option_chain", None)
    if not options_source:
        return None, "Client has no get_option_chain method"

    contract_type_all = None
    if client_module:
        try:
            options_source = getattr(client_module, "Options", None) or getattr(client, "Options", None)
            if options_source is not None:
                contract_type_all = getattr(options_source, "ContractType", None)
                if contract_type_all is not None:
                    contract_type_all = getattr(contract_type_all, "ALL", contract_type_all)
        except Exception:
            pass

    kwargs = {
        "symbol": option_symbol,
        "from_date": use_date,
        "to_date": use_date,
        "strike_count": strike_count,
    }
    if contract_type_all is not None:
        kwargs["contract_type"] = contract_type_all

    try:
        r = client.get_option_chain(**kwargs)
    except Exception as e:
        return None, str(e)

    if r.status_code != 200:
        body = r.text
        try:
            err = r.json()
            body = err.get("message", err.get("error", body))
        except Exception:
            pass
        return None, f"API error {r.status_code}: {body}"

    data = r.json()
    (
        total_gex,
        per_strike_gex,
        _change_in_gamma,
        _largest_changes,
        spot_price,
    ) = calculate_gamma_exposure(data, previous_gamma or {})
    details = get_per_strike_details(data)
    return (data, total_gex, per_strike_gex, details, spot_price, use_date), None


# --- Page config and client init (before sidebar so we have default_symbol) ---
st.set_page_config(
    page_title="Gamma Exposure Dashboard",
    page_icon="📊",
    layout="wide",
)

client, broker_name, default_symbol, client_module = get_authenticated_client()
if client is None:
    st.stop()

with st.sidebar:
    st.title("Gamma Exposure")
    ticker = st.text_input(
        "Ticker",
        value=default_symbol,
        key="ticker",
        help=f"Default {default_symbol} for {broker_name}. Use $SPX for Schwab, $SPX.X for TDA.",
    )
    strike_count = st.slider("Strike count", min_value=10, max_value=100, value=50, key="strikes")
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=15,
        key="refresh",
    )
    strike_range = st.slider(
        "Strike range (± from spot)",
        min_value=200,
        max_value=1500,
        value=800,
        step=50,
        key="strike_range",
        help="Only show strikes within ± this many points from spot",
    )
    if st.button("Refresh now"):
        st.rerun()

st.caption(f"Using {broker_name} API")

# --- Fetch data ---
if "previous_gex" not in st.session_state:
    st.session_state.previous_gex = {}

result, err_msg = fetch_options_and_gex(
    client, ticker, strike_count, st.session_state.previous_gex, client_module
)
if result is None:
    st.error(f"Failed to fetch option chain: {err_msg}")
    st.stop()

_data, total_gex, per_strike_gex, strike_details, spot_price, exp_date = result
st.session_state.previous_gex = per_strike_gex.copy()

last_update = datetime.now(pytz.timezone("US/Eastern")).strftime("%b %d, %Y %H:%M:%S ET")
exp_date_str = exp_date.strftime("%b %d, %Y")
st.caption(f"Options expiring **{exp_date_str}**  •  Data as of {last_update}")

# --- Identify King Node and Gatekeepers ---
sorted_by_abs = sorted(
    per_strike_gex.items(),
    key=lambda x: abs(x[1]),
    reverse=True,
)
king_strike = sorted_by_abs[0][0] if sorted_by_abs else None

top_positive = [s for s, g in sorted(per_strike_gex.items(), key=lambda x: x[1], reverse=True)[:3]]
top_negative = [s for s, g in sorted(per_strike_gex.items(), key=lambda x: x[1])[:3]]
gatekeeper_strikes = set(top_positive + top_negative)

# --- Build heatmap with focused strike range (spot ± strike_range) ---
all_strikes = sorted(per_strike_gex.keys(), reverse=True)
strikes = [s for s in all_strikes if abs(s - spot_price) <= strike_range]
if not strikes:
    strikes = all_strikes  # fallback if filter too tight

gex_values = [per_strike_gex[s] for s in strikes]

# Single column heatmap (y=strikes as floats for numeric axis + correct range)
z = np.array([[g] for g in gex_values])
max_abs = max(abs(g) for g in gex_values) if gex_values else 1

gex_colorscale = [
    [0.0, "#4B0082"],
    [0.25, "#8B0000"],
    [0.5, "#F5F5F5"],
    [0.75, "#228B22"],
    [1.0, "#006400"],
]

customdata = np.zeros((len(strikes), 1, 4))
cell_texts = []
for i, s in enumerate(strikes):
    d = strike_details.get(s, {})
    customdata[i, 0, :] = [s, per_strike_gex[s], d.get("oi", 0), d.get("gamma_sum", 0)]
    g = gex_values[i]
    t = f"${g:.2f}B"
    if s == king_strike:
        t += "\n👑 King Node"
    elif s in gatekeeper_strikes:
        t += "\nGatekeeper"
    cell_texts.append([t])

fig = go.Figure(
    data=go.Heatmap(
        z=z,
        x=["GEX"],
        y=strikes,
        text=cell_texts,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=gex_colorscale,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        hoverongaps=False,
        customdata=customdata,
        hovertemplate="Strike: %{customdata[0]:.0f}<br>GEX: $%{z:.3f}B<br>OI: %{customdata[2]:,.0f}<br>Gamma×Vol: %{customdata[3]:,.2f}<extra></extra>",
    )
)

# Red horizontal line at spot
fig.add_shape(
    type="line",
    x0=-0.5,
    x1=0.5,
    y0=spot_price,
    y1=spot_price,
    line=dict(color="red", width=2, dash="dash"),
    xref="x",
    yref="y",
)
fig.add_annotation(
    x=-0.5,
    y=spot_price,
    text=f"Spot ${spot_price:.1f}",
    showarrow=False,
    font=dict(color="red", size=10),
    xref="x",
    yref="y",
    xanchor="right",
)

# Constrain y-axis to strike range (numeric y = strikes, so range applies correctly)
y_min, y_max = min(strikes), max(strikes)
y_pad = max(10, (y_max - y_min) * 0.03)
fig.update_layout(
    title=f"Gamma Exposure by Strike — Expiring {exp_date_str}",
    xaxis_title="",
    yaxis_title="Strike Price",
    yaxis=dict(
        autorange="reversed",
        range=[y_max + y_pad, y_min - y_pad],
        dtick=50,
    ),
    height=400 + len(strikes) * 14,
    margin=dict(l=100, r=140),
)

st.plotly_chart(fig, use_container_width=True)

# --- Summary stats ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total GEX", f"${total_gex:.3f}B")
with col2:
    st.metric("Spot", f"${spot_price:.2f}")
with col3:
    st.metric("King Node", f"${king_strike:.0f}" if king_strike else "—")
with col4:
    st.metric("Last update", last_update)

# --- Interpretation ---
interpretation = generate_gex_interpretation(
    spot_price, total_gex, king_strike, top_positive, top_negative, per_strike_gex
)
trader_suggestions = generate_trader_suggestions(
    spot_price, total_gex, king_strike, top_positive, top_negative, per_strike_gex
)
st.markdown("---")
st.markdown("**Interpretation**")
# Escape $ to avoid Streamlit's LaTeX math mode; render as HTML for proper paragraph formatting
safe_text = interpretation.replace("$", "&#36;")
st.markdown(f"<p style='line-height: 1.6;'>{safe_text}</p>", unsafe_allow_html=True)
st.markdown("**Trading Suggestions**")
safe_suggestions = trader_suggestions.replace("$", "&#36;")
st.markdown(f"<p style='line-height: 1.6;'>{safe_suggestions}</p>", unsafe_allow_html=True)

# --- Auto-refresh ---
st.markdown("---")
st.caption(f"Auto-refreshing every {refresh_interval} seconds.")
time.sleep(refresh_interval)
st.rerun()
