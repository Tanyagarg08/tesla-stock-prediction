import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="TSLA · Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Lazy import TF so sidebar renders immediately ─────────────────────────────
@st.cache_resource(show_spinner="Loading model... (first time only)")
def load_model_cached():
    from tensorflow.keras.models import load_model as _load
    model  = _load('tsla_lstm_best.keras')
    scaler = joblib.load('tsla_scaler.pkl')
    return model, scaler

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('TSLA.csv', parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_inr(val):
    """Indian number formatting: ₹55,24,312"""
    val = int(round(float(val)))
    s   = str(abs(val))
    if len(s) > 3:
        last3 = s[-3:]
        rest  = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        s = ",".join(reversed(groups)) + "," + last3
    return ("−" if val < 0 else "") + "₹" + s

def make_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i, 0])
        y.append(data[i + horizon - 1, 0])
    return np.array(X)[..., np.newaxis], np.array(y)

# ── Plot constants ─────────────────────────────────────────────────────────────
PANEL = "#08111f"
BG    = "#050d1a"
GRID  = "#0e2035"
TEAL  = "#00d4c8"
WHITE = "#f0f4f8"
SLATE = "#8899aa"
AMBER = "#f5a623"
RED   = "#ff5c5c"
GREEN = "#00c87a"

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": SLATE,
    "font.size": 9,
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif !important; }
.stApp                      { background: #050d1a; }
.block-container            { padding: 1.2rem 2rem 3rem 2rem !important; }
#MainMenu, footer, header   { visibility: hidden; }

/* sidebar */
[data-testid="stSidebar"]   { background: #08111f !important; border-right: 1px solid #0e2035 !important; }
[data-testid="stSidebar"] * { color: #8899aa !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="radio"]  > div
                            { background: #050d1a !important; border-color: #0e2035 !important; border-radius: 4px !important; }
.sb-head  { font-family:'JetBrains Mono',monospace; font-size:.82rem; font-weight:600;
            color:#00d4c8 !important; letter-spacing:.1em;
            padding-bottom:.9rem; border-bottom:1px solid #0e2035; margin-bottom:1rem; }
.sb-sec   { font-size:.58rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
            color:#3d5570 !important; margin:1rem 0 .35rem; }
.rate-box { background:rgba(0,212,200,.06); border:1px solid rgba(0,212,200,.18);
            border-radius:4px; padding:.45rem .7rem; font-size:.7rem;
            font-family:'JetBrains Mono',monospace; color:#00d4c8 !important; margin-top:.5rem; }

/* ticker bar */
.ticker   { background:#08111f; border-bottom:1px solid #0e2035;
            display:flex; align-items:center; gap:1.8rem; flex-wrap:wrap;
            padding:.5rem 0 .5rem 0; margin-bottom:1.4rem; }
.ti       { display:flex; flex-direction:column; gap:1px; }
.ti-l     { font-size:.56rem; font-weight:700; letter-spacing:.12em;
            text-transform:uppercase; color:#8899aa; }
.ti-v     { font-family:'JetBrains Mono',monospace; font-size:.84rem;
            font-weight:600; color:#f0f4f8; }
.ti-v.tl  { color:#00d4c8; }
.ti-v.up  { color:#00c87a; }
.ti-v.dn  { color:#ff5c5c; }
.tsep     { width:1px; height:26px; background:#0e2035; flex-shrink:0; }

/* section label */
.sec  { font-size:.6rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
        color:#8899aa; border-bottom:1px solid #0e2035;
        padding-bottom:.35rem; margin:1.4rem 0 .8rem; }

/* metric strip */
.mstrip { display:grid; grid-template-columns:repeat(4,1fr);
          gap:1px; background:#0e2035; border:1px solid #0e2035;
          border-radius:4px; overflow:hidden; margin-bottom:1.2rem; }
.mcell  { background:#08111f; padding:.9rem 1.1rem; }
.ml     { font-size:.58rem; font-weight:700; letter-spacing:.12em;
          text-transform:uppercase; color:#8899aa; margin-bottom:.3rem; }
.mv     { font-family:'JetBrains Mono',monospace; font-size:1.25rem;
          font-weight:600; color:#f0f4f8; line-height:1; margin-bottom:.15rem; }
.mv.g   { color:#00c87a; }
.mv.w   { color:#f5a623; }
.mv.b   { color:#ff5c5c; }
.ms     { font-size:.6rem; color:#3d5570; }

/* forecast table */
.fct    { width:100%; border-collapse:collapse; font-size:.76rem; }
.fct th { font-size:.57rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
          color:#8899aa; padding:.35rem .55rem; border-bottom:1px solid #0e2035; text-align:left; }
.fct td { font-family:'JetBrains Mono',monospace; padding:.42rem .55rem;
          border-bottom:1px solid #0e2035; color:#f0f4f8; }
.fct tr:last-child td { border-bottom:none; }
.fct tr:hover td      { background:#0a1828; }
.up { color:#00c87a !important; }
.dn { color:#ff5c5c !important; }

/* disclaimer */
.disc { margin-top:1.8rem; padding:.65rem .9rem; border-left:2px solid #f5a623;
        background:rgba(245,166,35,.04); font-size:.7rem; color:#7a6535; line-height:1.5; }

/* page title */
.ptitle { font-size:1.5rem; font-weight:700; color:#f0f4f8;
          letter-spacing:-.02em; line-height:1.2; margin-bottom:.15rem; }
.psub   { font-size:.78rem; color:#8899aa; margin-bottom:1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar (renders immediately, before model loads) ─────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-head">📈 TSLA PREDICTOR</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio("", ["LSTM", "SimpleRNN"], label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Prediction Horizon</div>', unsafe_allow_html=True)
    horizon = st.select_slider("", options=[1, 5, 10],
                                format_func=lambda x: f"{x}-day",
                                label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Days to Forecast</div>', unsafe_allow_html=True)
    n_future = st.slider("", 5, 60, 15, 5, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Chart History (days)</div>', unsafe_allow_html=True)
    history_days = st.slider("", 60, 400, 150, 30, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">USD → INR Rate</div>', unsafe_allow_html=True)
    usd_inr = st.number_input("", value=83.5, step=0.5,
                               min_value=50.0, max_value=150.0,
                               label_visibility="collapsed")
    st.markdown(f'<div class="rate-box">$1 USD = ₹{usd_inr:.2f}</div>',
                unsafe_allow_html=True)

LOOKBACK = 60

# ── Load data first (fast), then model ───────────────────────────────────────
df = load_data()

# Show spinner only in main area while model loads
model_obj, scaler = load_model_cached()

# ── Compute predictions ───────────────────────────────────────────────────────
scaled    = scaler.transform(df[['Adj Close']])
split_idx = int(len(scaled) * 0.8)
test_data = scaled[split_idx:]

X_te, y_te  = make_sequences(test_data, LOOKBACK, horizon)
preds_usd   = scaler.inverse_transform(model_obj.predict(X_te, verbose=0)).flatten()
actual_usd  = scaler.inverse_transform(y_te.reshape(-1,1)).flatten()
test_dates  = df.index[split_idx + LOOKBACK : split_idx + LOOKBACK + len(preds_usd)]

preds   = preds_usd  * usd_inr
actuals = actual_usd * usd_inr

rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
mae  = float(mean_absolute_error(actuals, preds))
r2   = float(r2_score(actuals, preds))
mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

last_inr  = float(df['Adj Close'].iloc[-1]) * usd_inr
prev_inr  = float(df['Adj Close'].iloc[-2]) * usd_inr
day_chg   = last_inr - prev_inr
day_pct   = day_chg / prev_inr * 100
chg_cls   = "up" if day_chg >= 0 else "dn"
chg_arrow = "▲" if day_chg >= 0 else "▼"

# ── Ticker bar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ticker">
  <div class="ti"><span class="ti-l">Ticker</span><span class="ti-v tl">TSLA</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">Last Close</span><span class="ti-v">{fmt_inr(last_inr)}</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">Day Change</span>
    <span class="ti-v {chg_cls}">{chg_arrow} {fmt_inr(abs(day_chg))} ({abs(day_pct):.2f}%)</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">As of</span><span class="ti-v">{df.index[-1].strftime('%d %b %Y')}</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">Active Model</span><span class="ti-v tl">{model_choice}</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">Horizon</span><span class="ti-v">{horizon}-day</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">Currency</span><span class="ti-v tl">INR ₹</span></div>
  <div class="tsep"></div>
  <div class="ti"><span class="ti-l">FX Rate</span><span class="ti-v">$1 = ₹{usd_inr:.2f}</span></div>
</div>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ptitle">Tesla Stock Forecast</div>
<div class="psub">{model_choice} · {horizon}-day horizon · Indian Rupee (₹) · Training data: Jun 2010 – Feb 2020</div>
""", unsafe_allow_html=True)

# ── Metric strip ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Test Set Performance</div>', unsafe_allow_html=True)

def grade(v, good, ok, hi=False):
    if hi: return "g" if v>=good else "w" if v>=ok else "b"
    return "g" if v<=good else "w" if v<=ok else "b"

st.markdown(f"""
<div class="mstrip">
  <div class="mcell">
    <div class="ml">RMSE</div>
    <div class="mv {grade(rmse,3000,5000)}">{fmt_inr(rmse)}</div>
    <div class="ms">Root mean squared error</div>
  </div>
  <div class="mcell">
    <div class="ml">MAE</div>
    <div class="mv {grade(mae,2000,3500)}">{fmt_inr(mae)}</div>
    <div class="ms">Mean absolute error</div>
  </div>
  <div class="mcell">
    <div class="ml">R² Score</div>
    <div class="mv {grade(r2,0.80,0.65,hi=True)}">{r2:.4f}</div>
    <div class="ms">Variance explained</div>
  </div>
  <div class="mcell">
    <div class="ml">MAPE</div>
    <div class="mv {grade(mape,10,20)}">{mape:.2f}%</div>
    <div class="ms">Mean abs. % error</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Actual vs Predicted ───────────────────────────────────────────────────────
st.markdown('<div class="sec">Actual vs Predicted — Test Period</div>', unsafe_allow_html=True)

n_show = min(history_days, len(test_dates))
fig, ax = plt.subplots(figsize=(13, 3.8))
ax.plot(test_dates[-n_show:], actuals[-n_show:],
        color=TEAL, linewidth=1.6, label="Actual", zorder=3)
ax.plot(test_dates[-n_show:], preds[-n_show:],
        color=RED, linewidth=1.3, linestyle="--", alpha=0.85,
        label=f"{model_choice} Predicted", zorder=3)
ax.fill_between(test_dates[-n_show:], actuals[-n_show:], preds[-n_show:],
                alpha=0.07, color=AMBER)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"₹{x/1000:.0f}K" if x >= 1000 else f"₹{int(x)}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Price (₹)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(pad=1)
st.pyplot(fig, use_container_width=True)
plt.close()

# ── History + Residuals ───────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="medium")

with c1:
    st.markdown('<div class="sec">Full Price History</div>', unsafe_allow_html=True)
    hist_inr = df['Adj Close'] * usd_inr
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(df.index, hist_inr, color=TEAL, linewidth=1.0)
    ax2.fill_between(df.index, hist_inr, alpha=0.08, color=TEAL)
    ax2.axvline(df.index[split_idx], color=AMBER, linewidth=0.9, linestyle=":")
    ax2.text(df.index[split_idx], float(hist_inr.max()) * 0.88,
             "  Train | Test", color=AMBER, fontsize=7.5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1000:.0f}K" if x >= 1000 else f"₹{int(x)}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_ylabel("Adj Close (₹)", fontsize=9)
    fig2.tight_layout(pad=1)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with c2:
    st.markdown('<div class="sec">Error Distribution</div>', unsafe_allow_html=True)
    res = actuals - preds
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.hist(res, bins=40, color=TEAL, edgecolor=PANEL, linewidth=0.3, alpha=0.8)
    ax3.axvline(0, color=WHITE, linewidth=1.1, linestyle="--", alpha=0.4)
    ax3.axvline(res.mean(), color=RED, linewidth=1.1,
                label=f"Mean = {fmt_inr(res.mean())}")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1000:.0f}K" if abs(x)>=1000 else f"₹{int(x)}"))
    ax3.set_xlabel("Residual (₹)", fontsize=9)
    ax3.set_ylabel("Count", fontsize=9)
    ax3.legend(fontsize=7.5)
    fig3.tight_layout(pad=1)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── Forward Forecast ──────────────────────────────────────────────────────────
st.markdown(f'<div class="sec">Forward Forecast — Next {n_future} Trading Days</div>',
            unsafe_allow_html=True)

window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1).copy()
fp_usd = []
for _ in range(n_future):
    p = model_obj.predict(window, verbose=0)[0, 0]
    fp_usd.append(p)
    window = np.append(window[:, 1:, :], [[[p]]], axis=1)

fp_inr       = scaler.inverse_transform(np.array(fp_usd).reshape(-1,1)).flatten() * usd_inr
future_dates = pd.bdate_range(df.index[-1], periods=n_future + 1)[1:]

fc1, fc2 = st.columns([3, 2], gap="medium")

with fc1:
    fig4, ax4 = plt.subplots(figsize=(8, 3.3))
    h60 = df['Adj Close'].iloc[-60:] * usd_inr
    ax4.plot(h60.index, h60.values, color=TEAL, linewidth=1.4, label="Historical")
    ax4.plot(future_dates, fp_inr, color=GREEN, linewidth=1.4,
             linestyle="--", marker="o", markersize=3, label="Forecast")
    unc = np.array([rmse * np.sqrt(i+1) * 0.35 for i in range(n_future)])
    ax4.fill_between(future_dates, fp_inr - unc, fp_inr + unc,
                     alpha=0.1, color=GREEN)
    ax4.axvline(df.index[-1], color=AMBER, linewidth=0.8, linestyle=":")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1000:.0f}K" if x>=1000 else f"₹{int(x)}"))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax4.set_ylabel("Price (₹)", fontsize=9)
    ax4.legend(fontsize=8)
    fig4.tight_layout(pad=1)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

with fc2:
    base = float(df['Adj Close'].iloc[-1]) * usd_inr
    rows = ""
    for i, (d, price) in enumerate(zip(future_dates[:14], fp_inr[:14])):
        prev  = fp_inr[i-1] if i > 0 else base
        chg   = price - prev
        pct   = chg / prev * 100
        cls   = "up" if chg >= 0 else "dn"
        arrow = "▲" if chg >= 0 else "▼"
        rows += f"""<tr>
          <td>{d.strftime('%d %b')}</td>
          <td>{fmt_inr(price)}</td>
          <td class="{cls}">{arrow} {abs(pct):.2f}%</td>
        </tr>"""

    st.markdown(f"""
    <table class="fct">
      <thead><tr><th>Date</th><th>Forecast (₹)</th><th>Change</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disc">
  <b>Educational use only.</b> Model trained on TSLA data 2010–2020. INR prices use a
  fixed FX rate set in the sidebar. Not financial advice.
</div>
""", unsafe_allow_html=True)