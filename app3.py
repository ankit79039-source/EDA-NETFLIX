# filename: netflix_frontend_13.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Netflix Dashboard (13-Graphs)", layout="wide", page_icon="🎬")

# ----------------------------
# HELPERS
# ----------------------------
@st.cache_data(show_spinner=False)
def load_df(default_path="netflix.csv"):
    try:
        return pd.read_csv(default_path)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return pd.DataFrame()

def split_and_count(series: pd.Series, sep=","):
    s = series.dropna().astype(str).str.split(sep).explode().str.strip()
    s = s[s.ne("")]
    return s.value_counts()

def add_helpers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # parse date_added
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_month_added"] = df["date_added"].dt.to_period("M").astype(str)
        df["month_added"] = df["date_added"].dt.month_name()
        df["year_added"] = df["date_added"].dt.year
    # duration_num
    if "duration" in df.columns:
        df["duration_num"] = (
            df["duration"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        )
    return df

def kpi_block(df_f: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Titles (filtered)", f"{len(df_f):,}")
    with c2:
        st.metric("Movies", f"{(df_f['type']=='Movie').sum():,}" if "type" in df_f else "—")
    with c3:
        st.metric("TV Shows", f"{(df_f['type']=='TV Show').sum():,}" if "type" in df_f else "—")
    with c4:
        st.metric("Unique Countries", f"{split_and_count(df_f.get('country', pd.Series())).shape[0]:,}" if "country" in df_f else "—")
    with c5:
        st.metric("Unique Genres", f"{split_and_count(df_f.get('listed_in', pd.Series())).shape[0]:,}" if "listed_in" in df_f else "—")

# ----------------------------
# SIDEBAR (LEFT PANEL)
# ----------------------------
st.sidebar.title("📌 Controls")

# File uploader (optional) — defaults to colocated netflix.csv
up = st.sidebar.file_uploader("Upload netflix.csv", type=["csv"], help="Leave empty to use the default file")
if up:
    df = pd.read_csv(up)
else:
    df = load_df("netflix.csv")  # uses your uploaded dataset

if df.empty:
    st.stop()

# Ensure expected columns exist (common Kaggle Netflix titles schema)
expected_like = ["type","title","director","cast","country","date_added","release_year","rating","duration","listed_in","description"]
missing = [c for c in expected_like if c not in df.columns]
if missing:
    st.warning(f"Columns not found (the app still works where possible): {missing}")

# Prepare helperss
df = add_helpers(df)

# Dataset view selector
view = st.sidebar.radio("Dataset scope", ["Main","Movies","TV Shows"], index=0)

if view == "Movies" and "type" in df:
    base = df[df["type"]=="Movie"].copy()
elif view == "TV Shows" and "type" in df:
    base = df[df["type"]=="TV Show"].copy()
else:
    base = df.copy()

# Filters
# Year range
if "release_year" in base.columns:
    yr_min, yr_max = int(base["release_year"].min()), int(base["release_year"].max())
    ysel = st.sidebar.slider("Release Year", yr_min, yr_max, (yr_min, yr_max))
else:
    ysel = None

# Type multiselect
types_available = base["type"].dropna().unique().tolist() if "type" in base.columns else []
t_sel = st.sidebar.multiselect("Type", types_available, default=types_available) if types_available else []

# Rating filter
ratings = sorted(base["rating"].dropna().unique().tolist()) if "rating" in base.columns else []
r_sel = st.sidebar.multiselect("Rating", ratings)

# Country filter (top 20 tokens)
countries_top = split_and_count(base["country"]).head(20).index.tolist() if "country" in base.columns else []
c_sel = st.sidebar.multiselect("Countries (Top 20)", countries_top)

# Apply filters
df_f = base.copy()
if ysel and "release_year" in df_f.columns:
    df_f = df_f[(df_f["release_year"] >= ysel[0]) & (df_f["release_year"] <= ysel[1])].copy()

if t_sel and "type" in df_f.columns:
    df_f = df_f[df_f["type"].isin(t_sel)].copy()

if r_sel and "rating" in df_f.columns:
    df_f = df_f[df_f["rating"].isin(r_sel)].copy()

if c_sel and "country" in df_f.columns:
    pat = "|".join([rf"\\b{c}\\b" for c in c_sel])
    df_f = df_f[df_f["country"].fillna("").str.contains(pat, case=False, na=False)].copy()

# 13 distinct graphs/buttons
GRAPH_OPTIONS = [
    "1) Count by Type",
    "2) Titles Per Year",
    "3) Top Countries",
    "4) Rating Distribution",
    "5) Movie Duration Histogram",
    "6) TV Seasons Histogram",
    "7) Top Genres",
    "8) Month-wise Additions",
    "9) Word Cloud (Titles)",
    "10) Top Directors",
    "11) Top Cast",
    "12) Avg Duration vs Year",
    "13) Type vs Rating Heatmap",
]

choice = st.sidebar.radio("Choose a chart (13 buttons):", GRAPH_OPTIONS, index=0)

# ----------------------------
# MAIN CONTENT
# ----------------------------
st.title("🎬 Netflix Dashboard")
sub = f"Scope: **{view}**"
if ysel and "release_year" in df_f.columns:
    sub += f" · Years: **{ysel[0]}–{ysel[1]}**"
sub += f" · Rows: **{len(df_f):,}**"
st.caption(sub)

kpi_block(df_f)
st.divider()

# ----------------------------
# CHART RENDERERS
# ----------------------------
def chart_count_by_type():
    if "type" not in df_f:
        st.info("No 'type' column.")
        return
    s = df_f["type"].fillna("Unknown").value_counts().reset_index()
    s.columns = ["type","count"]
    fig = px.bar(s, x="type", y="count", title="Count by Type")
    st.plotly_chart(fig, use_container_width=True)

def chart_titles_per_year():
    if "release_year" not in df_f:
        st.info("No 'release_year' column.")
        return
    s = df_f.groupby("release_year").size().reset_index(name="titles")
    fig = px.line(s, x="release_year", y="titles", markers=True, title="Titles Per Year")
    st.plotly_chart(fig, use_container_width=True)

def chart_top_countries():
    if "country" not in df_f:
        st.info("No 'country' column.")
        return
    cc = split_and_count(df_f["country"]).head(15).reset_index()
    cc.columns = ["country","count"]
    fig = px.bar(cc, x="country", y="count", title="Top Countries (by mentions)")
    st.plotly_chart(fig, use_container_width=True)

def chart_rating_distribution():
    if "rating" not in df_f:
        st.info("No 'rating' column.")
        return
    s = df_f["rating"].fillna("Unknown").value_counts().reset_index()
    s.columns = ["rating","count"]
    fig = px.bar(s, x="rating", y="count", title="Rating Distribution",
                 labels={"rating":"Rating","count":"# Titles"})
    st.plotly_chart(fig, use_container_width=True)

def chart_movie_duration_hist():
    if not {"type","duration_num"}.issubset(df_f.columns):
        st.info("Need 'type' and 'duration' columns.")
        return
    mov = df_f[df_f["type"]=="Movie"].dropna(subset=["duration_num"])
    if mov.empty:
        st.info("No movie duration data in current filter.")
        return
    fig = px.histogram(mov, x="duration_num", nbins=30,
                       labels={"duration_num":"Duration (minutes)"},
                       title="Movie Duration Histogram")
    st.plotly_chart(fig, use_container_width=True)

def chart_tv_seasons_hist():
    if not {"type","duration_num"}.issubset(df_f.columns):
        st.info("Need 'type' and 'duration' columns.")
        return
    tv = df_f[df_f["type"]=="TV Show"].dropna(subset=["duration_num"])
    if tv.empty:
        st.info("No TV seasons data in current filter.")
        return
    fig = px.histogram(tv, x="duration_num", nbins=15,
                       labels={"duration_num":"# Seasons"},
                       title="TV Show Seasons Histogram")
    st.plotly_chart(fig, use_container_width=True)

def chart_top_genres():
    if "listed_in" not in df_f:
        st.info("No 'listed_in' (genres) column.")
        return
    g = split_and_count(df_f["listed_in"]).head(25).reset_index()
    g.columns = ["genre","count"]
    fig = px.bar(g, x="genre", y="count", title="Top Genres")
    st.plotly_chart(fig, use_container_width=True)

def chart_month_additions():
    if "year_month_added" not in df_f:
        st.info("No 'date_added' information.")
        return
    m = df_f.dropna(subset=["year_month_added"]).groupby("year_month_added").size().reset_index(name="count")
    m = m.sort_values("year_month_added")
    fig = px.bar(m, x="year_month_added", y="count",
                 labels={"year_month_added":"YYYY-MM","count":"# Added"},
                 title="Month-wise Additions (by date_added)")
    st.plotly_chart(fig, use_container_width=True)

def chart_word_cloud():
    if "title" not in df_f:
        st.info("No 'title' column.")
        return
    text = " ".join(df_f["title"].dropna().astype(str))
    if not text.strip():
        st.info("No titles available for word cloud.")
        return
    fig_wc = plt.figure(figsize=(10, 4))
    wc = WordCloud(width=1200, height=500, background_color="white").generate(text)
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig_wc, clear_figure=True)

def chart_top_directors():
    if "director" not in df_f:
        st.info("No 'director' column.")
        return
    d = split_and_count(df_f["director"]).head(20).reset_index()
    d.columns = ["director","count"]
    fig = px.bar(d, x="director", y="count", title="Top Directors")
    st.plotly_chart(fig, use_container_width=True)

def chart_top_cast():
    if "cast" not in df_f:
        st.info("No 'cast' column.")
        return
    c = split_and_count(df_f["cast"]).head(20).reset_index()
    c.columns = ["cast","count"]
    fig = px.bar(c, x="cast", y="count", title="Top Cast")
    st.plotly_chart(fig, use_container_width=True)

def chart_avg_duration_year():
    if not {"release_year","duration_num"}.issubset(df_f.columns):
        st.info("Need 'release_year' and 'duration'.")
        return
    dff = df_f.dropna(subset=["duration_num"]).copy()
    if dff.empty:
        st.info("No duration data in current filter.")
        return
    # If type exists, color by it; else single color
    color_series = dff["type"] if "type" in dff.columns else None
    fig = px.scatter(dff, x="release_year", y="duration_num", color=color_series,
                     labels={"release_year":"Release Year","duration_num":"Duration/Seasons"},
                     title="Average Duration vs Year (raw points)")
    st.plotly_chart(fig, use_container_width=True)

def chart_type_rating_heatmap():
    if not {"type","rating"}.issubset(df_f.columns):
        st.info("Need 'type' and 'rating'.")
        return
    pivot = pd.crosstab(df_f["type"], df_f["rating"])
    if pivot.size == 0:
        st.info("No data for heatmap.")
        return
    fig = px.imshow(pivot, text_auto=True, aspect="auto",
                    labels=dict(x="Rating", y="Type", color="# Titles"),
                    title="Type vs Rating Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# ROUTER
# ----------------------------
if choice.startswith("1"):
    chart_count_by_type()
elif choice.startswith("2"):
    chart_titles_per_year()
elif choice.startswith("3"):
    chart_top_countries()
elif choice.startswith("4"):
    chart_rating_distribution()
elif choice.startswith("5"):
    chart_movie_duration_hist()
elif choice.startswith("6"):
    chart_tv_seasons_hist()
elif choice.startswith("7"):
    chart_top_genres()
elif choice.startswith("8"):
    chart_month_additions()
elif choice.startswith("9"):
    chart_word_cloud()
elif choice.startswith("10"):
    chart_top_directors()
elif choice.startswith("11"):
    chart_top_cast()
elif choice.startswith("12"):
    chart_avg_duration_year()
elif choice.startswith("13"):
    chart_type_rating_heatmap()

# ----------------------------
# RAW DATA (optional)
# ----------------------------
with st.expander("🔎 Preview filtered data"):
    st.dataframe(df_f.reset_index(drop=True))
