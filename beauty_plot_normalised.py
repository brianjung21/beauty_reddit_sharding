from pathlib import Path
from typing import List, Optional
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# ---------------- Config ----------------
# Look ONLY in ./data relative to this script
DATA_DIR = (Path(__file__).parent / "data").resolve()

PIVOT_FILE = "beauty_alias_pivoted_brand_counts.csv"   # wide: date + one column per brand
RAW_MATCHES = "reddit_matches_raw.csv"                 # long: date, keyword, alias, subreddit, ...

DEFAULT_TOPN = 5

# ---------------- File helpers ----------------
def find_file(fname: str) -> Path:
    p = DATA_DIR / fname
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find {fname} in: {DATA_DIR}")

@st.cache_data(show_spinner=False)
def load_pivot() -> pd.DataFrame:
    p = find_file(PIVOT_FILE)
    df = pd.read_csv(p, parse_dates=["date"])
    # coerce numeric for brand columns
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_raw_matches_or_none() -> Optional[pd.DataFrame]:
    try:
        p = find_file(RAW_MATCHES)
    except FileNotFoundError:
        return None
    try:
        raw = pd.read_csv(p, parse_dates=["date"])
        keep = ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments"]
        cols = [c for c in keep if c in raw.columns]
        out = raw[cols].copy()
        # normalize types
        out["keyword"] = out["keyword"].astype(str)
        if "subreddit" in out.columns:
            out["subreddit"] = out["subreddit"].astype(str)
        return out.sort_values(["date", "score"], ascending=[True, False])
    except Exception:
        return None

# ---------------- Transform helpers ----------------
def resample_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df.set_index("date").asfreq("D").fillna(0).reset_index()
    # Weekly sums
    return df.set_index("date").resample("W").sum().reset_index()

def to_long(df: pd.DataFrame, brands: List[str], value_col: str = "mentions") -> pd.DataFrame:
    cols = ["date"] + brands
    sub = df[cols].copy()
    return sub.melt(id_vars="date", var_name="brand", value_name=value_col)

def default_top_brands(df_freq: pd.DataFrame, topn: int) -> List[str]:
    totals = df_freq.drop(columns=["date"]).sum(axis=0).sort_values(ascending=False)
    return totals.head(topn).index.tolist()

def apply_rolling(long_df: pd.DataFrame, win: int) -> pd.DataFrame:
    if long_df.empty:
        long_df["smoothed"] = long_df.get("value", 0)
        return long_df
    out = []
    for b, g in long_df.groupby("brand", as_index=False):
        g = g.sort_values("date").copy()
        g["smoothed"] = g["value"].rolling(win, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)

# ---------------- Normalization modes ----------------
def compute_sov(long_df: pd.DataFrame) -> pd.DataFrame:
    """Share of Voice = brand_mentions / total_mentions_that_period (in %)"""
    totals = long_df.groupby("date", as_index=False)["mentions"].sum().rename(columns={"mentions": "total"})
    x = long_df.merge(totals, on="date", how="left")
    x["value"] = np.where(x["total"] > 0, (x["mentions"] / x["total"]) * 100.0, 0.0)
    return x[["date", "brand", "value"]]

# --- SoV over a window (ratio of sums, bounded 0–100%) ---
def compute_window_sov(df_window: pd.DataFrame, brands_to_show: List[str]) -> pd.DataFrame:
    """
    df_window: wide frame with columns: date + brand columns (counts)
    Returns per-brand SoV over the window in percent:
        100 * sum(brand_mentions) / sum(total_mentions_all_brands)
    Only returns rows for brands_to_show.
    """
    if df_window.empty:
        return pd.DataFrame({"brand": [], "value": []})
    totals_by_date = df_window.drop(columns=["date"]).sum(axis=1).rename("total")
    totals_df = pd.DataFrame({"date": df_window["date"], "total": totals_by_date})
    brands = [b for b in brands_to_show if b in df_window.columns]
    if not brands:
        return pd.DataFrame({"brand": [], "value": []})
    m = df_window[["date"] + brands].melt("date", var_name="brand", value_name="mentions")
    m = m.merge(totals_df, on="date", how="left")
    sums = m.groupby("brand", as_index=False).agg({"mentions": "sum", "total": "sum"})
    sums["value"] = (sums["mentions"] / sums["total"]).fillna(0) * 100.0
    return sums[["brand", "value"]]

# ---------------- UI ----------------
st.set_page_config(page_title="Reddit Brand Mentions — Normalized", layout="wide")
st.title("Reddit Brand Mentions — Normalized Views")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    # Quick cache-clear to ensure latest CSVs are loaded
    if st.button("🔄 Refresh from disk"):
        st.cache_data.clear()

# Load data
try:
    df = load_pivot()  # wide daily brand counts
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Show loaded range so you can verify latest dates are present
st.caption(f"Loaded data range: {df['date'].min().date()} → {df['date'].max().date()} (UTC)")

raw = load_raw_matches_or_none()  # for drilldowns (optional)

all_brands = [c for c in df.columns if c != "date"]

with st.sidebar:
    freq_label = st.radio("Frequency", ["Daily", "Weekly"], index=0)
    freq = "D" if freq_label == "Daily" else "W"

    norm_mode = st.radio(
        "Normalization",
        ["Raw counts", "Share of Voice (SoV)"],
        index=1  # default to SoV to reduce platform drift
    )

    # Resample counts to chosen frequency before picking defaults
    df_freq = resample_freq(df, freq)

    # Date window
    min_d, max_d = df_freq["date"].min().date(), df_freq["date"].max().date()
    date_range = st.date_input(
        "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, date_range

    # Filter period for counts
    mask = (df_freq["date"].dt.date >= start_date) & (df_freq["date"].dt.date <= end_date)
    df_win = df_freq.loc[mask].copy()

    # Brand selection (defaults: top-N by raw counts over window)
    defaults = default_top_brands(df_win, DEFAULT_TOPN) if not df_win.empty else []
    selected = st.multiselect("Brands", options=all_brands, default=defaults)

    # Smoothing
    do_smooth = st.checkbox("Show rolling average", value=True)
    win_daily = st.number_input("Rolling window (days)", min_value=2, max_value=30, value=7) if (do_smooth and freq == "D") else 7
    win_weekly = st.number_input("Rolling window (weeks)", min_value=2, max_value=12, value=4) if (do_smooth and freq == "W") else 4

# Guards
if df_win.empty:
    st.info("No data in the selected date range.")
    st.stop()
if not selected:
    st.info("Pick at least one brand to plot from the sidebar.")
    st.stop()

# Build long frame of raw counts for selected brands
long_counts = to_long(df_win, selected, value_col="mentions")

# Compute the display metric per normalization mode
if norm_mode == "Raw counts":
    long_metric = long_counts.rename(columns={"mentions": "value"}).copy()
    y_label = "Mentions"

else:  # Share of Voice (SoV)
    # SoV over ALL brands (not just selected) to remove platform drift
    df_win_all = df_win.copy()
    long_all = to_long(df_win_all, [c for c in df_win_all.columns if c != "date"], value_col="mentions")
    sov_all = compute_sov(long_all)
    long_metric = sov_all[sov_all["brand"].isin(selected)].copy()
    y_label = "Share of Voice (%)"

# Plot
if do_smooth:
    win = win_daily if freq == "D" else win_weekly
    long_smooth = apply_rolling(long_metric, win=win)
    fig = px.line(
        long_smooth,
        x="date", y="smoothed", color="brand",
        title=f"Brand mentions — {norm_mode} ({freq_label})",
        labels={"smoothed": y_label, "date": "Date"},
    )
    # faint raw line beneath
    fig_raw = px.line(long_metric, x="date", y="value", color="brand")
    for tr in fig_raw.data:
        tr.update(opacity=0.25, line={"width": 1}, showlegend=False)
        fig.add_trace(tr)
else:
    fig = px.line(
        long_metric,
        x="date", y="value", color="brand",
        title=f"Brand mentions — {norm_mode} ({freq_label})",
        labels={"value": y_label, "date": "Date"},
    )

fig.update_layout(hovermode="x unified", legend_title_text="Brand")
st.plotly_chart(fig, use_container_width=True)

# Summary (current window) — by current normalization
st.subheader(f"Summary (current window) — {norm_mode}")
if norm_mode == "Raw counts":
    summary = (
        long_metric.groupby("brand", as_index=False)["value"].sum()
                  .sort_values("value", ascending=False)
                  .rename(columns={"value": "Mentions"})
    )
else:
    # True SoV over the window: ratio of sums across all brands
    sov_window = compute_window_sov(df_win, selected)
    summary = sov_window.sort_values("value", ascending=False).rename(columns={"value": "SoV over window (%)"})
    summary["SoV over window (%)"] = summary["SoV over window (%)"].round(2)

st.dataframe(summary, use_container_width=True)

# Top 10 brands (entire dataset) — by current normalization
st.subheader(f"Top 10 brands — Entire dataset — {norm_mode}")
if norm_mode == "Raw counts":
    full_counts = resample_freq(load_pivot(), freq)
    top_totals = full_counts.drop(columns=["date"]).sum(axis=0).sort_values(ascending=False)
    top10 = top_totals.head(10).reset_index().rename(columns={"index": "brand", 0: "value"})
    top10.columns = ["brand", "value"]
    y_top = "Mentions"
else:  # SoV over full dataset: ratio of sums
    full_counts = resample_freq(load_pivot(), freq)
    totals_by_date_full = full_counts.drop(columns=["date"]).sum(axis=1).rename("total")
    totals_df_full = pd.DataFrame({"date": full_counts["date"], "total": totals_by_date_full})
    long_full = full_counts.melt("date", var_name="brand", value_name="mentions")
    long_full = long_full.merge(totals_df_full, on="date", how="left")
    sums_full = long_full.groupby("brand", as_index=False).agg({"mentions": "sum", "total": "sum"})
    sums_full["value"] = (sums_full["mentions"] / sums_full["total"]).fillna(0) * 100.0
    top10 = sums_full.sort_values("value", ascending=False).head(10)
    y_top = "SoV over dataset (%)"

fig_top = px.bar(top10, x="brand", y="value", labels={"value": y_top, "brand": "Brand"})
if norm_mode != "Raw counts":
    top10["value"] = top10["value"].round(2)
st.plotly_chart(fig_top, use_container_width=True)
st.dataframe(top10, use_container_width=True)

# Top subreddits & top posts (raw drill-down; optional but handy)
raw_df = load_raw_matches_or_none()
with st.expander("Top subreddits for selected brands"):
    if raw_df is None:
        st.info(f"{RAW_MATCHES} not found.")
    else:
        rmask = (raw_df["date"].dt.date >= start_date) & (raw_df["date"].dt.date <= end_date)
        rsub = raw_df.loc[rmask].copy()
        rsub = rsub[rsub["keyword"].isin(selected)]
        if rsub.empty:
            st.info("No raw matches for the selected brands/date range.")
        else:
            top_subs = (rsub.groupby(["keyword", "subreddit"]).size()
                            .reset_index(name="posts"))
            for b in selected:
                sub_b = (top_subs[top_subs["keyword"] == b]
                            .sort_values("posts", ascending=False).head(10))
                if not sub_b.empty:
                    st.markdown(f"**{b}**")
                    st.dataframe(sub_b[["subreddit", "posts"]], use_container_width=True)

with st.expander("Show top posts (raw)"):
    if raw_df is None:
        st.info(f"{RAW_MATCHES} not found.")
    else:
        rmask = (raw_df["date"].dt.date >= start_date) & (raw_df["date"].dt.date <= end_date)
        rsub = raw_df.loc[rmask].copy()
        rsub = rsub[rsub["keyword"].isin(selected)]
        cols = [c for c in ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments"] if c in rsub.columns]
        top_posts = rsub.sort_values("score", ascending=False)[cols].head(50)
        st.dataframe(top_posts, use_container_width=True)

st.caption("Normalization modes: Raw counts; Share of Voice (brand / total %).")