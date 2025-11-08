# financial_dashboard.py
# Run with:  python financial_dashboard.py
# Requires: dash, requests
#   pip install dash requests

import os
import requests
from datetime import datetime
from urllib.parse import urlencode
from dash import Dash, html, dcc, Input, Output, State, callback_context

APP_TITLE = "Bloomberg‑Style Financial Dashboard"

# --- Secure key loading (DO NOT HARDCODE KEYS) ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
MARKETSTACK_ACCESS_KEY = os.getenv("MARKETSTACK_ACCESS_KEY")

# --------------- API HELPERS ---------------
def fetch_alpha_vantage_overview(ticker: str, api_key: str):
    """
    Fetch fundamental overview from Alpha Vantage (function=OVERVIEW).
    Extracts: P/E Ratio (PERatio), Dividend Yield (DividendYield), 52W High/Low.
    Returns (data, error): data is a dict or None; error is a string or None.
    """
    if not api_key:
        return None, "Missing Alpha Vantage API key. Set ALPHA_VANTAGE_API_KEY in your environment."

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        # Alpha Vantage returns {} or {"Note": "..."} on issues/limits
        if not data or ("Note" in data) or ("Information" in data) or ("Error Message" in data):
            note = data.get("Note") or data.get("Information") or data.get("Error Message") or "No data returned."
            return None, f"Alpha Vantage overview error: {note}"

        parsed = {
            "pe_ratio": safe_float(data.get("PERatio")),
            "dividend_yield": safe_float(data.get("DividendYield")),  # often a decimal (e.g., 0.015 = 1.5%)
            "fifty_two_week_high": safe_float(data.get("52WeekHigh")),
            "fifty_two_week_low": safe_float(data.get("52WeekLow")),
            "currency": data.get("Currency", ""),
            "name": data.get("Name", ""),
        }
        return parsed, None

    except requests.RequestException as e:
        return None, f"Alpha Vantage network error: {e}"
    except ValueError as e:
        return None, f"Alpha Vantage parse error: {e}"


def fetch_marketstack_eod(ticker: str, access_key: str):
    """
    Fetch latest EOD prices from Marketstack (limit=2 to compute daily % change).
    Extracts: last closing price, daily percentage change.
    Returns (data, error): data is a dict or None; error is a string or None.
    """
    if not access_key:
        return None, "Missing Marketstack access key. Set MARKETSTACK_ACCESS_KEY in your environment."

    base_url = "http://api.marketstack.com/v1/eod"
    params = {
        "access_key": access_key,
        "symbols": ticker,
        "limit": 2,
        # You can add 'sort': 'DESC' to ensure most recent first, but Marketstack defaults to most recent.
    }

    try:
        # Marketstack free plan uses HTTP (not HTTPS) for base URL
        url = f"{base_url}?{urlencode(params)}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json() or {}
        data_list = payload.get("data", [])
        if not data_list:
            err_msg = payload.get("error", {}).get("message", "No EOD data returned.")
            return None, f"Marketstack EOD error: {err_msg}"

        # Expect most recent first
        latest = data_list[0]
        prev = data_list[1] if len(data_list) > 1 else None

        last_close = safe_float(latest.get("close"))
        prev_close = safe_float(prev.get("close")) if prev else None

        pct_change = None
        if last_close is not None and prev_close not in (None, 0):
            pct_change = ((last_close - prev_close) / prev_close) * 100.0

        parsed = {
            "last_close": last_close,
            "prev_close": prev_close,
            "pct_change": pct_change,
            "latest_date": latest.get("date"),
            "exchange": latest.get("exchange") or payload.get("exchange", ""),
            "symbol": latest.get("symbol") or ticker,
        }
        return parsed, None

    except requests.RequestException as e:
        return None, f"Marketstack network error: {e}"
    except ValueError as e:
        return None, f"Marketstack parse error: {e}"


def safe_float(val):
    try:
        if val in (None, "", "None"):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def fmt_num(val, decimals=2):
    return f"{val:,.{decimals}f}" if isinstance(val, (int, float)) else "—"


def fmt_pct(val, decimals=2):
    return f"{val:.{decimals}f}%" if isinstance(val, (int, float)) else "—"


def parse_iso_date(iso_str):
    if not iso_str:
        return "—"
    try:
        # Marketstack returns ISO8601 (e.g., "2024-09-20T00:00:00+0000" or "2024-09-20T00:00:00Z")
        # Normalize by stripping timezone for pretty print
        if iso_str.endswith("Z"):
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        elif "+" in iso_str[10:] or "-" in iso_str[10:]:
            # has timezone offset
            dt = datetime.fromisoformat(iso_str)
        else:
            dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso_str


# --------------- DASH APP ---------------

external_fonts = [
    # Google Fonts for Roboto and Source Code Pro
    "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Source+Code+Pro:wght@400;600;700&display=swap"
]

app = Dash(__name__, external_stylesheets=external_fonts)
server = app.server

# --- Global styles (Bloomberg-esque) ---
BG_BLACK = "#000000"
AMBER = "#FFB300"       # Primary Data Text
TEAL = "#00CED1"        # Labels, borders, titles
LIME = "#32CD32"        # Positive change
ORANGERED = "#FF4500"   # Negative change

CARD_STYLE = {
    "backgroundColor": "#0B0B0B",
    "border": f"1px solid {TEAL}",
    "boxShadow": f"0 0 12px rgba(0, 206, 209, 0.25)",
    "borderRadius": "12px",
    "padding": "18px 20px",
    "margin": "10px 0",
}

TITLE_STYLE = {
    "color": TEAL,
    "fontFamily": "'Roboto','Helvetica Neue',sans-serif",
    "fontWeight": 700,
    "fontSize": "22px",
    "marginBottom": "10px",
    "letterSpacing": "0.5px",
}

LABEL_STYLE = {
    "color": TEAL,
    "fontFamily": "'Roboto','Helvetica Neue',sans-serif",
    "fontSize": "14px",
    "opacity": 0.9,
}

VALUE_STYLE = {
    "color": AMBER,
    "fontFamily": "'Source Code Pro','Roboto Mono',monospace",
    "fontSize": "26px",
    "fontWeight": 600,
}

SMALL_VALUE_STYLE = {
    "color": AMBER,
    "fontFamily": "'Source Code Pro','Roboto Mono',monospace",
    "fontSize": "18px",
    "fontWeight": 600,
}

HEADER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "marginBottom": "10px",
}

app.layout = html.Div(
    style={
        "backgroundColor": BG_BLACK,
        "minHeight": "100vh",
        "padding": "20px 24px",
        "color": TEAL,
        "fontFamily": "'Roboto','Helvetica Neue',sans-serif",
    },
    children=[
        # Inline CSS for body defaults (use dcc.Markdown for <style>)
        dcc.Markdown(f"""
<style>
html, body, #_dash-app-content {{
    background-color: {BG_BLACK} !important;
}}
.mono {{
    font-family: 'Source Code Pro','Roboto Mono',monospace !important;
}}
.teal-border {{
    border: 1px solid {TEAL};
    box-shadow: 0 0 10px rgba(0,206,209,0.2);
    border-radius: 12px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 14px;
}}
.btn {{
    background: {TEAL};
    color: #001314;
    padding: 10px 16px;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    cursor: pointer;
}}
.btn:active {{
    transform: translateY(1px);
}}
input, .dash-input {{
    background: #0F0F0F;
    border: 1px solid {TEAL};
    color: {AMBER};
    padding: 10px 12px;
    border-radius: 10px;
    font-family: 'Source Code Pro','Roboto Mono',monospace;
    font-size: 16px;
    outline: none;
}}
.hint {{
    color: {TEAL}; opacity: 0.8; font-size: 12px;
}}
</style>
""", dangerously_allow_html=True),
        html.Div(style=HEADER_STYLE, children=[
            html.H1(APP_TITLE, style={"color": AMBER, "fontWeight": 800, "margin": 0}),
            html.Div("High‑contrast, data‑dense dashboard", style={"color": TEAL, "opacity": 0.8})
        ]),
        html.Div(
            style={**CARD_STYLE, "display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div(children=[html.Div("Ticker", style=LABEL_STYLE),
                                   dcc.Input(id="ticker-input", type="text", value="IBM",
                                             debounce=True, className="dash-input", style={"width": "120px", "textTransform": "uppercase"})]),
                html.Button("Fetch Data", id="fetch-button", n_clicks=0, className="btn"),
                html.Div(
                    className="hint",
                    children="Note: Set ALPHA_VANTAGE_API_KEY and MARKETSTACK_ACCESS_KEY as environment variables before running."
                ),
            ],
        ),
        html.Div(className="grid", children=[
            html.Div(style=CARD_STYLE, children=[
                html.Div("Price Summary", style=TITLE_STYLE),
                dcc.Loading(color=TEAL, type="dot", children=[
                    html.Div(id="price-summary", children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
                            html.Div([html.Div("Last Close", style=LABEL_STYLE),
                                      html.Div("—", id="last-close", style=VALUE_STYLE)]),
                            html.Div([html.Div("Daily % Change", style=LABEL_STYLE),
                                      html.Div("—", id="pct-change", style=VALUE_STYLE)]),
                        ]),
                        html.Div(style={"marginTop": "10px", "display": "flex", "gap": "20px"}, children=[
                            html.Div([html.Div("As of (EOD)", style=LABEL_STYLE),
                                      html.Div("—", id="last-date", style=SMALL_VALUE_STYLE)]),
                            html.Div([html.Div("Exchange", style=LABEL_STYLE),
                                      html.Div("—", id="exchange", style=SMALL_VALUE_STYLE)]),
                            html.Div([html.Div("Symbol", style=LABEL_STYLE),
                                      html.Div("—", id="echo-symbol", style=SMALL_VALUE_STYLE)]),
                        ]),
                        html.Div(id="marketstack-error", style={"color": ORANGERED, "marginTop": "8px", "fontSize": "12px"})
                    ])
                ]),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Key Fundamentals", style=TITLE_STYLE),
                dcc.Loading(color=TEAL, type="dot", children=[
                    html.Div(id="fundamentals", children=[
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                            html.Div([html.Div("P/E Ratio", style=LABEL_STYLE),
                                      html.Div("—", id="pe-ratio", style=VALUE_STYLE)]),
                            html.Div([html.Div("Dividend Yield", style=LABEL_STYLE),
                                      html.Div("—", id="div-yield", style=VALUE_STYLE)]),
                            html.Div([html.Div("52‑Week High", style=LABEL_STYLE),
                                      html.Div("—", id="high-52", style=VALUE_STYLE)]),
                            html.Div([html.Div("52‑Week Low", style=LABEL_STYLE),
                                      html.Div("—", id="low-52", style=VALUE_STYLE)]),
                        ]),
                        html.Div(style={"marginTop": "10px"}, children=[
                            html.Div([html.Span("Company: ", style=LABEL_STYLE),
                                      html.Span("—", id="company-name", style=SMALL_VALUE_STYLE)]),
                            html.Div([html.Span("Currency: ", style=LABEL_STYLE),
                                      html.Span("—", id="currency", style=SMALL_VALUE_STYp0]';E)]),
                        ]),
                        html.Div(id="alphavantage-error", style={"color": ORANGERED, "marginTop": "8px", "fontSize": "12px"})
                    ])
                ]),
            ]),
        ]),
        html.Div(style={"marginTop": "8px", "fontSize": "12px", "opacity": 0.75}, children=[
            html.Span("Theme: ", style=LABEL_STYLE),
            html.Span("Deep Black #000000 background, Amber #FFB300 data, Teal #00CED1 labels; Lime #32CD32 for ≥0% change, Orange‑Red #FF4500 for <0%.", className="mono")
        ]),
    ],
)

@app.callback(
    Output("last-close", "children"),
    Output("pct-change", "children"),
    Output("pct-change", "style"),
    Output("last-date", "children"),
    Output("exchange", "children"),
    Output("echo-symbol", "children"),
    Output("marketstack-error", "children"),
    Output("pe-ratio", "children"),
    Output("div-yield", "children"),
    Output("high-52", "children"),
    Output("low-52", "children"),
    Output("company-name", "children"),
    Output("currency", "children"),
    Output("alphavantage-error", "children"),
    Input("fetch-button", "n_clicks"),
    State("ticker-input", "value"),
    prevent_initial_call=False
)
def update_data(n_clicks, ticker_value):
    # Ensure ticker uppercase and trimmed
    ticker = (ticker_value or "").strip().upper() or "IBM"

    # Fetch Marketstack
    ms_data, ms_err = fetch_marketstack_eod(ticker, MARKETSTACK_ACCESS_KEY)
    if ms_data:
        last_close_val = fmt_num(ms_data.get("last_close"))
        pct_val = ms_data.get("pct_change")
        pct_text = fmt_pct(pct_val) if pct_val is not None else "—"
        pct_style = dict(VALUE_STYLE)
        if isinstance(pct_val, (int, float)):
            pct_style["color"] = LIME if pct_val >= 0 else ORANGERED
        last_date = parse_iso_date(ms_data.get("latest_date"))
        exch = ms_data.get("exchange") or "—"
        sym = ms_data.get("symbol") or ticker
        ms_err_text = ""
    else:
        last_close_val = "—"
        pct_text = "—"
        pct_style = dict(VALUE_STYLE)
        last_date = "—"
        exch = "—"
        sym = ticker
        ms_err_text = ms_err or ""

    # Fetch Alpha Vantage
    av_data, av_err = fetch_alpha_vantage_overview(ticker, ALPHA_VANTAGE_API_KEY)
    if av_data:
        pe = fmt_num(av_data.get("pe_ratio"))
        # DividendYield from AV is a decimal fraction; show as percent
        dy_raw = av_data.get("dividend_yield")
        dy = fmt_pct(dy_raw * 100.0 if isinstance(dy_raw, (int, float)) else dy_raw)  # if already %, fmt_pct handles
        hi_52 = fmt_num(av_data.get("fifty_two_week_high"))
        lo_52 = fmt_num(av_data.get("fifty_two_week_low"))
        comp = av_data.get("name") or "—"
        ccy = av_data.get("currency") or "—"
        av_err_text = ""
    else:
        pe, dy, hi_52, lo_52, comp, ccy = "—", "—", "—", "—", "—", "—"
        av_err_text = av_err or ""

    return (
        last_close_val,
        pct_text,
        pct_style,
        last_date,
        exch,
        sym,
        ms_err_text,
        pe,
        dy,
        hi_52,
        lo_52,
        comp,
        ccy,
        av_err_text,
    )


if __name__ == "__main__":
    # Host on 0.0.0.0 for container friendliness; default port 8050
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
