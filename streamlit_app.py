"""
AI Work Sentiment Dashboard
Visualisasi Hasil Analisis Sentimen Publik terhadap Artificial Intelligence
dalam Dunia Kerja Berbasis Pengetahuan di Indonesia
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import base64
import os
import re
from io import StringIO
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Work Sentiment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & FALLBACK DATA
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "bab4_alur_analisis_sentimen.html"
ASSETS_DIR = BASE_DIR / "extracted_assets" / "figures"

MODELS = {
    "TF-IDF + SVM tanpa stemming": 0.823177,
    "TF-IDF + SVM dengan stemming": 0.817183,
    "IndoBERT": 0.849151,
    "IndoBERTweet": 0.833167,
    "IndoBERTweet tanpa normalisasi": 0.831169,
}

MODEL_COLORS = {
    "TF-IDF + SVM tanpa stemming": "#4E79A7",
    "TF-IDF + SVM dengan stemming": "#59A14F",
    "IndoBERT": "#E15759",
    "IndoBERTweet": "#F28E2B",
    "IndoBERTweet tanpa normalisasi": "#B07AA1",
}

LABEL_COLORS = {
    "ancaman": "#E15759",
    "peluang": "#59A14F",
    "netral": "#4E79A7",
}

INDOBERT_TRAINING = [
    {"Epoch": 1, "Training Loss": 0.619223, "Validation Loss": 0.445345, "Val Accuracy": 0.832},
    {"Epoch": 2, "Training Loss": 0.390263, "Validation Loss": 0.430128, "Val Accuracy": 0.866},
    {"Epoch": 3, "Training Loss": 0.267085, "Validation Loss": 0.479817, "Val Accuracy": 0.858},
]

INDOBERTWEET_TRAINING = [
    {"Epoch": 1, "Training Loss": 0.698857, "Validation Loss": 0.556220, "Val Accuracy": 0.780},
    {"Epoch": 2, "Training Loss": 0.422145, "Validation Loss": 0.472993, "Val Accuracy": 0.842},
    {"Epoch": 3, "Training Loss": 0.298713, "Validation Loss": 0.499404, "Val Accuracy": 0.844},
]

INDOBERTWEET_RAW_TRAINING = [
    {"Epoch": 1, "Training Loss": 0.691702, "Validation Loss": 0.533993, "Val Accuracy": 0.794},
    {"Epoch": 2, "Training Loss": 0.405936, "Validation Loss": 0.464697, "Val Accuracy": 0.852},
    {"Epoch": 3, "Training Loss": 0.284906, "Validation Loss": 0.491821, "Val Accuracy": 0.856},
]

CR_SVM_NO_STEM = """              precision    recall  f1-score   support

     ancaman       0.83      0.87      0.85       345
      netral       0.84      0.78      0.81       414
     peluang       0.79      0.83      0.81       242

    accuracy                           0.82      1001
   macro avg       0.82      0.83      0.82      1001
weighted avg       0.82      0.82      0.82      1001"""

CR_SVM_STEM = """              precision    recall  f1-score   support

     ancaman       0.82      0.88      0.85       345
      netral       0.85      0.76      0.80       414
     peluang       0.77      0.83      0.80       242

    accuracy                           0.82      1001
   macro avg       0.81      0.82      0.82      1001
weighted avg       0.82      0.82      0.82      1001"""

CR_INDOBERT = """              precision    recall  f1-score   support

     ancaman       0.82      0.93      0.87       345
      netral       0.87      0.81      0.84       414
     peluang       0.86      0.81      0.83       242

    accuracy                           0.85      1001
   macro avg       0.85      0.85      0.85      1001
weighted avg       0.85      0.85      0.85      1001"""

CR_INDOBERTWEET = """              precision    recall  f1-score   support

     ancaman       0.84      0.88      0.86       345
      netral       0.86      0.78      0.82       414
     peluang       0.79      0.86      0.82       242

    accuracy                           0.83      1001
   macro avg       0.83      0.84      0.83      1001
weighted avg       0.83      0.83      0.83      1001"""

CR_INDOBERTWEET_RAW = """              precision    recall  f1-score   support

     ancaman       0.83      0.89      0.86       345
      netral       0.87      0.77      0.81       414
     peluang       0.78      0.86      0.82       242

    accuracy                           0.83      1001
   macro avg       0.83      0.84      0.83      1001
weighted avg       0.83      0.83      0.83      1001"""

# ─────────────────────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root variables */
:root {
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card-hover: #1c2128;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-red: #E15759;
    --accent-blue: #4E79A7;
    --accent-green: #59A14F;
    --accent-orange: #F28E2B;
    --accent-purple: #B07AA1;
    --gold: #F0C040;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-color, #4E79A7);
    border-radius: 12px 12px 0 0;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #484f58;
}

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    line-height: 1.2;
    color: #e6edf3;
}

.kpi-sub {
    font-size: 0.7rem;
    color: #8b949e;
    margin-top: 0.25rem;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #30363d;
}

.section-badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #e6edf3;
    margin: 0;
}

/* Info cards */
.info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.info-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.6rem;
}

/* Best model highlight */
.best-model-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #1c2030 100%);
    border: 1px solid #E15759;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
}

.best-badge {
    display: inline-block;
    background: #E15759;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

/* Code / monospace output */
.code-output {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #adbac7;
    overflow-x: auto;
    white-space: pre;
}

/* Model tags in overview */
.model-tag {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
    border: 1px solid;
}

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #1a2233 0%, #182032 100%);
    border-left: 3px solid #4E79A7;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c9d1d9;
}

.insight-box.warning {
    border-left-color: #F0C040;
    background: linear-gradient(135deg, #1f1e0d 0%, #1e1b00 100%);
}

.insight-box.success {
    border-left-color: #59A14F;
    background: linear-gradient(135deg, #0f1f0d 0%, #0a1a08 100%);
}

/* Page title */
.dashboard-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e6edf3;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.dashboard-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    font-weight: 400;
    line-height: 1.5;
}

/* Sidebar nav */
.sidebar-nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    padding: 0.5rem 0 0.25rem;
}

/* Table styling override */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* Image border */
img {
    border-radius: 8px;
}

/* Metric styling */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
}

/* Expander */
details {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* Divider */
hr {
    border-color: #30363d !important;
    margin: 1.5rem 0;
}

/* Fix low-contrast Streamlit default text on dark background */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #e6edf3 !important;
    opacity: 1 !important;
}

.stMarkdown p, .stMarkdown li {
    color: #c9d1d9 !important;
}

[data-testid="stMarkdownContainer"] {
    color: #c9d1d9 !important;
}

button[data-baseweb="tab"] p {
    color: #8b949e !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: #ff4b4b !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] label p {
    color: #c9d1d9 !important;
    font-size: 0.9rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.15rem 0 !important;
}

.stAlert {
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_html():
    """Load and parse the HTML file once."""
    try:
        from bs4 import BeautifulSoup

        if not HTML_FILE.exists():
            available_files = [p.name for p in BASE_DIR.iterdir()]
            st.error(f"File HTML tidak ditemukan: {HTML_FILE}")
            st.write("Isi folder saat deploy:", available_files)
            return None, None

        raw = HTML_FILE.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "html.parser")
        return raw, soup

    except Exception as e:
        st.error("Gagal membaca file HTML.")
        st.exception(e)
        return None, None


@st.cache_data(show_spinner=False)
def extract_images():
    """Extract embedded base64 images from HTML and save to disk."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _, soup = load_html()
    if soup is None:
        return []
    images = []
    imgs = soup.find_all("img")
    for i, img in enumerate(imgs):
        src = img.get("src", "")
        if src.startswith("data:image"):
            try:
                header, data = src.split(",", 1)
                ext = "png" if "png" in header else "jpg"
                fpath = ASSETS_DIR / f"figure_{i:02d}.{ext}"
                with open(fpath, "wb") as f:
                    f.write(base64.b64decode(data))
                images.append(str(fpath))
            except Exception:
                pass
    return images


@st.cache_data(show_spinner=False)
def get_tables():
    """Extract all HTML tables as DataFrames.

    More robust than parsing table-by-table because Jupyter HTML tables
    can contain complex attributes that make per-table parsing fail.
    """
    raw, soup = load_html()
    if raw is None:
        return {}

    # Primary method: read all tables directly from the complete HTML.
    try:
        dfs = pd.read_html(StringIO(raw), flavor="lxml")
        return {i: df for i, df in enumerate(dfs)}
    except Exception:
        pass

    # Fallback method: parse each <table> tag individually.
    result = {}
    if soup is not None:
        tables = soup.find_all("table")
        for i, t in enumerate(tables):
            try:
                df = pd.read_html(StringIO(str(t)), flavor="lxml")[0]
                result[i] = df
            except Exception:
                try:
                    df = pd.read_html(StringIO(str(t)))[0]
                    result[i] = df
                except Exception:
                    pass
    return result


def section_header(badge, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="section-badge">{badge}</span>
        <h2 class="section-title">{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def insight(text, style=""):
    st.markdown(f'<div class="insight-box {style}">💡 {text}</div>', unsafe_allow_html=True)


def training_chart(data, title):
    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Epoch"], y=df["Training Loss"],
        mode="lines+markers", name="Training Loss",
        line=dict(color="#4E79A7", width=2.5),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df["Epoch"], y=df["Validation Loss"],
        mode="lines+markers", name="Validation Loss",
        line=dict(color="#F28E2B", width=2.5, dash="dash"),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df["Epoch"], y=df["Val Accuracy"],
        mode="lines+markers", name="Val Accuracy",
        line=dict(color="#59A14F", width=2.5),
        marker=dict(size=8),
        yaxis="y2"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e6edf3")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#e6edf3")
        ),
        xaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
            tickmode="array",
            tickvals=[1, 2, 3],
            title=dict(text="Epoch", font=dict(color="#8b949e"))
        ),
        yaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
            title=dict(text="Loss", font=dict(color="#8b949e"))
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            title=dict(text="Accuracy", font=dict(color="#59A14F")),
            gridcolor="rgba(0,0,0,0)",
            tickformat=".0%",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def cr_metrics_bars(cr_text, model_name):
    """Parse classification report text and show bar chart."""
    lines = [l for l in cr_text.strip().split("\n") if l.strip()]
    labels, precisions, recalls, f1s = [], [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ["ancaman", "netral", "peluang"]:
            labels.append(parts[0].capitalize())
            precisions.append(float(parts[1]))
            recalls.append(float(parts[2]))
            f1s.append(float(parts[3]))
    if not labels:
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=labels, y=precisions,
                         marker_color="#4E79A7"))
    fig.add_trace(go.Bar(name="Recall", x=labels, y=recalls,
                         marker_color="#F28E2B"))
    fig.add_trace(go.Bar(name="F1-Score", x=labels, y=f1s,
                         marker_color="#59A14F"))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
                    borderwidth=1, font=dict(color="#e6edf3")),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d",
                   range=[0, 1], tickformat=".0%"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=270,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("Memuat data notebook..."):
    raw_html, soup = load_html()
    html_images = extract_images()
    all_tables = get_tables()

html_ok = raw_html is not None

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:2rem;">🤖</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#e6edf3;">AI Work Sentiment</div>
        <div style="font-size:0.7rem;color:#8b949e;">Dashboard Skripsi — BAB IV</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-label">Navigasi Halaman</div>', unsafe_allow_html=True)

    pages = [
        ("🏠", "Overview"),
        ("📊", "EDA"),
        ("⚙️", "Preprocessing"),
        ("✂️", "Data Split"),
        ("📐", "Baseline TF-IDF + SVM"),
        ("🔵", "IndoBERT"),
        ("🟠", "IndoBERTweet"),
        ("🔮", "IndoBERTweet Tanpa Normalisasi"),
        ("🏆", "Model Comparison"),
        ("📄", "HTML Viewer"),
    ]

    page_labels = [f"{icon} {name}" for icon, name in pages]
    selected = st.radio("", page_labels, label_visibility="collapsed")
    selected_name = selected.split(" ", 1)[1]

    st.markdown("---")
    if html_ok:
        st.markdown(
            '<div style="font-size:0.72rem;color:#59A14F;">✅ File HTML berhasil dimuat</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-size:0.72rem;color:#8b949e;">{len(html_images)} gambar diekstrak</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-size:0.72rem;color:#E15759;">⚠️ File HTML tidak ditemukan</div>',
            unsafe_allow_html=True
        )
        st.caption(f"Pastikan `{HTML_FILE}` berada satu folder dengan `app.py`")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.68rem;color:#484f58;text-align:center;line-height:1.6;">
        Dashboard Hasil Penelitian<br/>
        Analisis Sentimen AI & Dunia Kerja<br/>
        <span style="color:#30363d;">——————————————</span><br/>
        Bab IV · Read-Only
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ERROR STATE
# ─────────────────────────────────────────────────────────────
if not html_ok:
    st.error(f"**File tidak ditemukan:** `{HTML_FILE}`")
    st.info(
    f"Pastikan file `{HTML_FILE.name}` berada **satu folder** dengan `{Path(__file__).name}`, "
    "kemudian jalankan ulang dashboard."
)
    st.stop()


# ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────
if selected_name == "Overview":
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div class="dashboard-title">AI Work Sentiment Dashboard</div>
        <div class="dashboard-subtitle">
            Visualisasi Hasil Analisis Sentimen Publik terhadap Artificial Intelligence<br/>
            dalam Dunia Kerja Berbasis Pengetahuan di Indonesia
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Ringkasan Penelitian")

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Tentang Dashboard</div>
        <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;margin:0;">
            Dashboard ini menampilkan seluruh hasil eksperimen dari <strong>Bab IV Skripsi</strong> secara interaktif.
            Data dan visualisasi bersumber langsung dari file notebook <code>bab4_alur_analisis_sentimen.html</code>.
            Dashboard bersifat <em>read-only</em> — tidak melakukan prediksi, training, atau inference baru.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    st.markdown("#### Key Performance Indicators")

    kpi_cols = st.columns(5)
    kpi_data = [
        ("Total Data", "5.001", "Tweet", "#4E79A7"),
        ("Kategori Label", "3", "Ancaman · Peluang · Netral", "#F28E2B"),
        ("Skenario Model", "5", "Baseline + Fine-tuning", "#B07AA1"),
        ("Best Model", "IndoBERT", "Accuracy tertinggi", "#E15759"),
        ("Best Accuracy", "84.92%", "IndoBERT (test set)", "#59A14F"),
    ]

    for col, (label, value, sub, color) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-color:{color};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model overview & quick accuracy chart
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        section_header("02", "Model yang Dibandingkan")
        model_info = [
            ("TF-IDF + SVM tanpa stemming", "#4E79A7", "Baseline"),
            ("TF-IDF + SVM dengan stemming", "#59A14F", "Baseline"),
            ("IndoBERT", "#E15759", "🏆 Best Model"),
            ("IndoBERTweet", "#F28E2B", "Fine-tuning"),
            ("IndoBERTweet tanpa normalisasi", "#B07AA1", "Pembanding"),
        ]
        for name, color, tag in model_info:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.75rem;
                        padding:0.7rem 1rem;margin-bottom:0.5rem;
                        background:#161b22;border:1px solid #30363d;
                        border-radius:8px;border-left:3px solid {color};">
                <div style="flex:1;font-size:0.85rem;color:#e6edf3;">{name}</div>
                <div style="font-size:0.7rem;color:#8b949e;">{tag}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        section_header("03", "Performa Model (Accuracy)")
        names = list(MODELS.keys())
        accs = [MODELS[n] for n in names]
        colors_list = [MODEL_COLORS[n] for n in names]
        short_names = [
            "SVM\n(no stem)", "SVM\n(stem)",
            "IndoBERT", "IndoBERTweet", "IBTweet\n(raw)"
        ]

        fig = go.Figure(go.Bar(
            x=accs,
            y=short_names,
            orientation="h",
            marker=dict(
                color=colors_list,
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            text=[f"{a:.4f}" for a in accs],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", size=11),
            xaxis=dict(
                gridcolor="#21262d", linecolor="#30363d",
                range=[0.78, 0.89], tickformat=".0%",
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="#30363d"),
            margin=dict(l=10, r=80, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    insight(
        "IndoBERT meraih accuracy tertinggi sebesar <strong>84.92%</strong> pada data test, "
        "diikuti IndoBERTweet (83.32%) dan TF-IDF + SVM tanpa stemming (82.32%).",
        "success"
    )


# ─────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────
elif selected_name == "EDA":
    section_header("04", "Exploratory Data Analysis (EDA)")
    insight(
        "EDA digunakan untuk memahami karakteristik dataset dan distribusi sentimen "
        "sebelum proses preprocessing dimulai.",
    )

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "📊 Distribusi Label", "🔤 Top Words & Visualisasi"])

    with tab1:
        st.markdown("#### 4.1 · Gambaran Umum Dataset")
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Sumber Data</div>
            <p style="color:#c9d1d9;font-size:0.88rem;line-height:1.7;margin:0;">
                Dataset terdiri dari <strong>5.001 tweet</strong> berbahasa Indonesia yang membahas
                Artificial Intelligence dalam konteks dunia kerja. Pengumpulan data dilakukan melalui
                Twitter API dan pelabelan manual oleh tim peneliti dengan 3 kategori sentimen:
                <strong>Ancaman</strong>, <strong>Peluang</strong>, dan <strong>Netral</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Table 0: dataset preview
        if 0 in all_tables:
            df = all_tables[0]
            # show only relevant columns
            cols_show = [c for c in df.columns if c not in ["Unnamed: 0"] or True]
            st.dataframe(df.head(5), use_container_width=True)
        else:
            st.info("Tabel preview dataset tidak tersedia dari HTML.")

        # Table 1: kolom
        st.markdown("#### Struktur Kolom Dataset")
        if 1 in all_tables:
            df1 = all_tables[1]
            st.dataframe(df1, use_container_width=True)

        # Table 2: data cleansing
        st.markdown("#### 4.3 · Hasil Data Cleansing")
        if 2 in all_tables:
            df2 = all_tables[2]
            st.dataframe(df2, use_container_width=True)

    with tab2:
        st.markdown("#### 4.9.1 · Distribusi Label")
        col_tbl, col_chart = st.columns([1, 2])

        label_data = {"Label": ["Netral", "Ancaman", "Peluang"], "Jumlah": [2070, 1723, 1208]}

        # Try from HTML table
        if 10 in all_tables:
            df_lbl = all_tables[10].copy()
            try:
                df_lbl.columns = df_lbl.columns.str.lower()
                label_data = {
                    "Label": df_lbl["label"].tolist(),
                    "Jumlah": df_lbl["jumlah"].astype(int).tolist(),
                }
            except Exception:
                pass

        with col_tbl:
            st.dataframe(pd.DataFrame(label_data), use_container_width=True, hide_index=True)
            total = sum(label_data["Jumlah"])
            for lbl, cnt in zip(label_data["Label"], label_data["Jumlah"]):
                pct = cnt / total * 100
                color = LABEL_COLORS.get(lbl.lower(), "#8b949e")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.3rem 0.6rem;margin-bottom:0.3rem;
                            background:#161b22;border-radius:6px;border-left:3px solid {color};">
                    <span style="font-size:0.85rem;color:#e6edf3;">{lbl}</span>
                    <span style="font-size:0.82rem;color:{color};font-weight:600;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            fig = go.Figure(go.Pie(
                labels=label_data["Label"],
                values=label_data["Jumlah"],
                hole=0.55,
                marker=dict(colors=[
                    LABEL_COLORS.get(l.lower(), "#8b949e")
                    for l in label_data["Label"]
                ]),
                textfont=dict(color="#e6edf3", size=12),
                textinfo="label+percent",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", size=11),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=270,
                annotations=[dict(
                    text="5.001<br>Tweet",
                    x=0.5, y=0.5, font_size=15,
                    font_color="#e6edf3", showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Images from HTML (EDA: first ~2 images are likely EDA related)
        if html_images:
            st.markdown("#### Visualisasi dari Notebook")
            img_cols = st.columns(min(2, len(html_images)))
            for i, img_path in enumerate(html_images[:4]):
                with img_cols[i % 2]:
                    st.image(img_path, use_container_width=True)

    with tab3:
        st.markdown("#### Top Words Sebelum Preprocessing")
        if 3 in all_tables:
            df_words = all_tables[3]
            df_words.columns = [c.lower() for c in df_words.columns]
            if "kata" in df_words.columns and "frekuensi" in df_words.columns:
                df_words["frekuensi"] = pd.to_numeric(df_words["frekuensi"], errors="coerce")
                df_words = df_words.dropna(subset=["frekuensi"]).sort_values("frekuensi", ascending=False).head(20)
                fig = go.Figure(go.Bar(
                    x=df_words["frekuensi"],
                    y=df_words["kata"],
                    orientation="h",
                    marker_color="#4E79A7",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=11),
                    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                    yaxis=dict(
                        gridcolor="rgba(0,0,0,0)",
                        linecolor="#30363d",
                        autorange="reversed"
                    ),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top Words Setelah Preprocessing")
        if 9 in all_tables:
            df_w2 = all_tables[9].copy()
            df_w2.columns = [c.lower() for c in df_w2.columns]
            if "kata" in df_w2.columns and "frekuensi" in df_w2.columns:
                df_w2["frekuensi"] = pd.to_numeric(df_w2["frekuensi"], errors="coerce")
                df_w2 = df_w2.dropna(subset=["frekuensi"]).sort_values("frekuensi", ascending=False).head(20)
                fig2 = go.Figure(go.Bar(
                    x=df_w2["frekuensi"],
                    y=df_w2["kata"],
                    orientation="h",
                    marker_color="#59A14F",
                ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=11),
                    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                    yaxis=dict(
                        gridcolor="rgba(0,0,0,0)",
                        linecolor="#30363d",
                        autorange="reversed"
                    ),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: PREPROCESSING
# ─────────────────────────────────────────────────────────────
elif selected_name == "Preprocessing":
    section_header("05", "Preprocessing")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-card" style="border-left:3px solid #4E79A7;">
            <div class="info-card-title" style="color:#4E79A7;">Jalur TF-IDF + SVM</div>
            <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                <li>Case Folding</li>
                <li>Normalisasi teks</li>
                <li>Stopword Removal</li>
                <li>Stemming (opsional)</li>
                <li>TF-IDF Feature Extraction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-card" style="border-left:3px solid #E15759;">
            <div class="info-card-title" style="color:#E15759;">Jalur IndoBERT / IndoBERTweet</div>
            <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                <li>Light Preprocessing</li>
                <li>Normalisasi ringan (tanpa stemming)</li>
                <li>Tokenisasi dengan tokenizer bawaan model</li>
                <li>Konteks kalimat tetap terjaga</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    insight(
        "Model Transformer (IndoBERT/IndoBERTweet) menggunakan <em>light preprocessing</em> "
        "agar informasi kontekstual tidak hilang. TF-IDF + SVM membutuhkan preprocessing lebih lengkap "
        "karena bergantung pada frekuensi kata.",
        "warning"
    )

    tabs = st.tabs([
        "4.4 Case Folding", "4.5 Normalisasi", "4.6 Stopword Removal",
        "4.8 Stemming", "4.11 Setelah Preprocessing"
    ])

    table_map = {
        "4.4 Case Folding": 4,
        "4.5 Normalisasi": 5,
        "4.6 Stopword Removal": 6,
        "4.8 Stemming": (7, 8),
        "4.11 Setelah Preprocessing": 9,
    }

    tab_labels = [
        "4.4 Case Folding", "4.5 Normalisasi", "4.6 Stopword Removal",
        "4.8 Stemming", "4.11 Setelah Preprocessing"
    ]

    for tab, label in zip(tabs, tab_labels):
        with tab:
            idx = table_map[label]
            if isinstance(idx, tuple):
                for i in idx:
                    if i in all_tables:
                        st.dataframe(all_tables[i].head(10), use_container_width=True)
            else:
                if idx in all_tables:
                    st.dataframe(all_tables[idx].head(10), use_container_width=True)
                else:
                    st.info(f"Tabel untuk {label} tidak tersedia dari HTML.")


# ─────────────────────────────────────────────────────────────
# PAGE: DATA SPLIT
# ─────────────────────────────────────────────────────────────
elif selected_name == "Data Split":
    section_header("06", "Data Splitting")
    insight(
        "Pembagian data dilakukan dengan rasio <strong>train : validation : test = 70% : 10% : 20%</strong>. "
        "Validation set digunakan saat training untuk memantau performa; "
        "test set digunakan untuk evaluasi akhir.",
    )

    # Try from HTML table 11
    split_data = {
        "Label": ["Ancaman", "Peluang", "Netral", "Total"],
        "Train": [1206, 845, 1449, 3500],
        "Validation": [172, 121, 207, 500],
        "Test": [345, 242, 414, 1001],
    }
    if 11 in all_tables:
        try:
            df_split = all_tables[11].copy()
            df_split.columns = df_split.iloc[0]
            df_split = df_split[1:].reset_index(drop=True)
        except Exception:
            pass

    col_tbl, col_chart = st.columns([1, 1.5])

    with col_tbl:
        st.markdown("#### Distribusi per Label")
        df_show = pd.DataFrame(split_data)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Ringkasan Split")
        for split_name, total, pct in [("Train", 3500, "70%"), ("Validation", 500, "10%"), ("Test", 1001, "20%")]:
            color = {"Train": "#59A14F", "Validation": "#F28E2B", "Test": "#E15759"}[split_name]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.5rem 0.8rem;margin-bottom:0.4rem;
                        background:#161b22;border-radius:8px;border-left:3px solid {color};">
                <span style="font-weight:600;color:#e6edf3;">{split_name}</span>
                <span style="color:{color};font-weight:700;">{total:,} tweet ({pct})</span>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### Visualisasi Distribusi")
        labels_plot = ["Ancaman", "Peluang", "Netral"]
        train_vals = [1206, 845, 1449]
        val_vals = [172, 121, 207]
        test_vals = [345, 242, 414]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train", x=labels_plot, y=train_vals,
                             marker_color="#59A14F"))
        fig.add_trace(go.Bar(name="Validation", x=labels_plot, y=val_vals,
                             marker_color="#F28E2B"))
        fig.add_trace(go.Bar(name="Test", x=labels_plot, y=test_vals,
                             marker_color="#E15759"))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", size=11),
            legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
                        borderwidth=1, font=dict(color="#e6edf3")),
            xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=310,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stacked total
        fig2 = go.Figure(go.Bar(
            x=["Train", "Validation", "Test"],
            y=[3500, 500, 1001],
            marker_color=["#59A14F", "#F28E2B", "#E15759"],
            text=[3500, 500, 1001],
            textposition="auto",
            textfont=dict(color="white", size=13, family="DM Sans"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", size=11),
            xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: BASELINE TF-IDF + SVM
# ─────────────────────────────────────────────────────────────
elif selected_name == "Baseline TF-IDF + SVM":
    section_header("07", "Baseline TF-IDF + SVM")

    # Accuracy metric cards
    col1, col2, col3 = st.columns([1, 1, 1.2])
    with col1:
        st.markdown("""
        <div class="kpi-card" style="--accent-color:#4E79A7;">
            <div class="kpi-label">Tanpa Stemming</div>
            <div class="kpi-value" style="color:#4E79A7;">82.32%</div>
            <div class="kpi-sub">Accuracy (test set)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card" style="--accent-color:#59A14F;">
            <div class="kpi-label">Dengan Stemming</div>
            <div class="kpi-value" style="color:#59A14F;">81.72%</div>
            <div class="kpi-sub">Accuracy (test set)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        insight("Skenario <strong>tanpa stemming</strong> menghasilkan accuracy lebih tinggi. "
                "Stemming pada bahasa Indonesia bisa menyebabkan loss informasi karena morfologi yang kompleks.", "warning")

    st.markdown("<br>", unsafe_allow_html=True)

    # SVM comparison table from HTML
    if 12 in all_tables:
        st.markdown("#### Tabel Hasil Evaluasi (dari HTML)")
        df_svm = all_tables[12].copy()
        st.dataframe(df_svm, use_container_width=True, hide_index=True)

    tab_ns, tab_s = st.tabs(["📋 SVM Tanpa Stemming", "📋 SVM Dengan Stemming"])

    with tab_ns:
        st.markdown("#### Classification Report")
        st.markdown(f'<div class="code-output">{CR_SVM_NO_STEM}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Visualisasi Metrik per Kelas")
        cr_metrics_bars(CR_SVM_NO_STEM, "SVM Tanpa Stemming")

    with tab_s:
        st.markdown("#### Classification Report")
        st.markdown(f'<div class="code-output">{CR_SVM_STEM}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Visualisasi Metrik per Kelas")
        cr_metrics_bars(CR_SVM_STEM, "SVM Dengan Stemming")

    # Confusion matrix images from HTML.
    # IMPORTANT: image index mapping from bab4_alur_analisis_sentimen.html:
    # 0-1 = wordcloud/top words before preprocessing
    # 2-3 = wordcloud/top words after preprocessing
    # 4   = label distribution
    # 5   = train/validation/test split distribution
    # 6   = CM TF-IDF + SVM tanpa stemming
    # 7   = CM TF-IDF + SVM dengan stemming
    # 8   = CM IndoBERT
    # 9   = CM IndoBERTweet
    # 10  = CM IndoBERTweet tanpa normalisasi
    if html_images and len(html_images) >= 8:
        st.markdown("#### Confusion Matrix (dari HTML Notebook)")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.image(html_images[6], caption="Confusion Matrix — TF-IDF + SVM Tanpa Stemming", use_container_width=True)
        with col_i2:
            st.image(html_images[7], caption="Confusion Matrix — TF-IDF + SVM Dengan Stemming", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: INDOBERT
# ─────────────────────────────────────────────────────────────
elif selected_name == "IndoBERT":
    section_header("08", "IndoBERT — Fine-Tuning & Evaluasi")

    st.markdown("""
    <div class="best-model-card">
        <div class="best-badge">🏆 Best Model</div>
        <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#E15759;">84.92%</div>
        <div style="color:#8b949e;font-size:0.85rem;margin-top:0.3rem;">
            IndoBERT · Accuracy tertinggi di antara semua model yang diuji
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Konfigurasi IndoBERT</div>
            <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                <li>Model: <code>indobenchmark/indobert-base-p2</code></li>
                <li>Input: <code>text_for_indobert</code></li>
                <li>Epoch: 3</li>
                <li>Optimizer: AdamW</li>
                <li>Preprocessing: light (tanpa stemming)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Training table
        st.markdown("#### Log Training per Epoch")
        if 14 in all_tables:
            st.dataframe(all_tables[14], use_container_width=True, hide_index=True)

    st.markdown("#### Kurva Training IndoBERT")
    training_chart(INDOBERT_TRAINING, "IndoBERT — Loss & Accuracy per Epoch")

    tab_cr, tab_img = st.tabs(["📋 Classification Report", "🖼️ Confusion Matrix"])
    with tab_cr:
        st.markdown(f'<div class="code-output">{CR_INDOBERT}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cr_metrics_bars(CR_INDOBERT, "IndoBERT")

    with tab_img:
        if html_images and len(html_images) >= 9:
            st.image(html_images[8], caption="Confusion Matrix — IndoBERT", use_container_width=True)
        else:
            st.info("Gambar confusion matrix IndoBERT tidak tersedia.")

    insight(
        "IndoBERT mengungguli semua model lain dengan accuracy <strong>84.92%</strong>. "
        "F1-score kelas <em>Ancaman</em> mencapai 0.87, yang menunjukkan model sangat baik "
        "dalam mengenali sentimen negatif tentang AI dan ketenagakerjaan.",
        "success"
    )


# ─────────────────────────────────────────────────────────────
# PAGE: INDOBERTWEET
# ─────────────────────────────────────────────────────────────
elif selected_name == "IndoBERTweet":
    section_header("09", "IndoBERTweet — Fine-Tuning & Evaluasi")

    st.markdown("""
    <div class="kpi-card" style="--accent-color:#F28E2B;max-width:300px;margin-bottom:1.5rem;">
        <div class="kpi-label">IndoBERTweet Accuracy</div>
        <div class="kpi-value" style="color:#F28E2B;">83.32%</div>
        <div class="kpi-sub">Test set · Peringkat #2</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Konfigurasi IndoBERTweet</div>
            <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                <li>Model: <code>indolem/indobertweet-base-uncased</code></li>
                <li>Input: <code>text_light_normalized</code></li>
                <li>Epoch: 3</li>
                <li>Didesain khusus untuk teks Twitter/media sosial</li>
                <li>Preprocessing: normalisasi ringan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        insight(
            "IndoBERTweet dirancang untuk teks informal media sosial Bahasa Indonesia, "
            "menjadikannya pilihan tepat untuk dataset tweet.",
        )

    with col_right:
        st.markdown("#### Log Training per Epoch")
        if 16 in all_tables:
            st.dataframe(all_tables[16], use_container_width=True, hide_index=True)

    st.markdown("#### Kurva Training IndoBERTweet")
    training_chart(INDOBERTWEET_TRAINING, "IndoBERTweet — Loss & Accuracy per Epoch")

    tab_cr, tab_img = st.tabs(["📋 Classification Report", "🖼️ Confusion Matrix"])
    with tab_cr:
        st.markdown(f'<div class="code-output">{CR_INDOBERTWEET}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cr_metrics_bars(CR_INDOBERTWEET, "IndoBERTweet")

    with tab_img:
        if html_images and len(html_images) >= 10:
            st.image(html_images[9], caption="Confusion Matrix — IndoBERTweet", use_container_width=True)
        else:
            st.info("Gambar confusion matrix IndoBERTweet tidak tersedia.")


# ─────────────────────────────────────────────────────────────
# PAGE: INDOBERTWEET TANPA NORMALISASI
# ─────────────────────────────────────────────────────────────
elif selected_name == "IndoBERTweet Tanpa Normalisasi":
    section_header("10", "IndoBERTweet — Skenario Tanpa Normalisasi")

    st.markdown("""
    <div class="kpi-card" style="--accent-color:#B07AA1;max-width:300px;margin-bottom:1.5rem;">
        <div class="kpi-label">IndoBERTweet (Raw) Accuracy</div>
        <div class="kpi-value" style="color:#B07AA1;">83.12%</div>
        <div class="kpi-sub">Test set · Skenario Pembanding</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Tujuan Skenario Pembanding</div>
            <p style="color:#c9d1d9;font-size:0.88rem;line-height:1.7;margin:0;">
                Skenario ini menggunakan teks asli (<code>full_text</code>) tanpa normalisasi apapun
                untuk membandingkan seberapa besar pengaruh normalisasi ringan terhadap performa IndoBERTweet.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Konfigurasi</div>
            <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                <li>Model: <code>indolem/indobertweet-base-uncased</code></li>
                <li>Input: <code>full_text</code> (raw)</li>
                <li>Epoch: 3</li>
                <li>Tanpa normalisasi apapun</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Log Training per Epoch")
        if 18 in all_tables:
            st.dataframe(all_tables[18], use_container_width=True, hide_index=True)
        else:
            st.info("Log training IndoBERTweet tanpa normalisasi tidak tersedia dari HTML.")

    st.markdown("#### Kurva Training IndoBERTweet (Raw)")
    training_chart(INDOBERTWEET_RAW_TRAINING, "IndoBERTweet tanpa Normalisasi — Loss & Accuracy per Epoch")

    tab_cr, tab_img = st.tabs(["📋 Classification Report", "🖼️ Confusion Matrix"])
    with tab_cr:
        st.markdown(f'<div class="code-output">{CR_INDOBERTWEET_RAW}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cr_metrics_bars(CR_INDOBERTWEET_RAW, "IndoBERTweet tanpa Normalisasi")

    with tab_img:
        if html_images and len(html_images) >= 11:
            st.image(html_images[10], caption="Confusion Matrix — IndoBERTweet tanpa Normalisasi", use_container_width=True)
        else:
            st.info("Gambar confusion matrix IndoBERTweet tanpa normalisasi tidak tersedia.")

    insight(
        "Perbedaan accuracy antara IndoBERTweet dengan normalisasi (83.32%) dan tanpa normalisasi (83.12%) "
        "sangat kecil (0.20 percentage point). Ini menunjukkan bahwa IndoBERTweet cukup robust terhadap variasi "
        "teks raw Twitter, meskipun normalisasi ringan masih memberi hasil sedikit lebih baik.",
        "warning"
    )


# ─────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
elif selected_name == "Model Comparison":
    section_header("11", "Perbandingan Semua Model")

    # Main evaluation summary for all models. This replaces the previous
    # radar chart with a simpler, presentation-friendly evaluation table.
    eval_rows = [
        {
            "Rank": 1,
            "Model": "IndoBERT",
            "Accuracy": 0.849151,
            "Macro Precision": 0.85,
            "Macro Recall": 0.85,
            "Macro F1": 0.85,
            "Weighted Precision": 0.85,
            "Weighted Recall": 0.85,
            "Weighted F1": 0.85,
        },
        {
            "Rank": 2,
            "Model": "IndoBERTweet",
            "Accuracy": 0.833167,
            "Macro Precision": 0.83,
            "Macro Recall": 0.84,
            "Macro F1": 0.83,
            "Weighted Precision": 0.83,
            "Weighted Recall": 0.83,
            "Weighted F1": 0.83,
        },
        {
            "Rank": 3,
            "Model": "IndoBERTweet tanpa normalisasi",
            "Accuracy": 0.831169,
            "Macro Precision": 0.83,
            "Macro Recall": 0.84,
            "Macro F1": 0.83,
            "Weighted Precision": 0.83,
            "Weighted Recall": 0.83,
            "Weighted F1": 0.83,
        },
        {
            "Rank": 4,
            "Model": "TF-IDF + SVM tanpa stemming",
            "Accuracy": 0.823177,
            "Macro Precision": 0.82,
            "Macro Recall": 0.83,
            "Macro F1": 0.82,
            "Weighted Precision": 0.82,
            "Weighted Recall": 0.82,
            "Weighted F1": 0.82,
        },
        {
            "Rank": 5,
            "Model": "TF-IDF + SVM dengan stemming",
            "Accuracy": 0.817183,
            "Macro Precision": 0.81,
            "Macro Recall": 0.82,
            "Macro F1": 0.82,
            "Weighted Precision": 0.82,
            "Weighted Recall": 0.82,
            "Weighted F1": 0.82,
        },
    ]
    eval_df = pd.DataFrame(eval_rows)

    detail_data = {
        "TF-IDF + SVM tanpa stemming": {
            "Accuracy": 0.823177,
            "Macro F1": 0.82,
            "Weighted F1": 0.82,
            "Insight": "TF-IDF + SVM tanpa stemming menjadi baseline terbaik. Hasil ini menunjukkan bahwa baseline klasik masih cukup kompetitif, meskipun belum melampaui model Transformer.",
            "Rows": [
                {"Kelas": "Ancaman", "Precision": 0.83, "Recall": 0.87, "F1-score": 0.85, "Support": 345},
                {"Kelas": "Netral", "Precision": 0.84, "Recall": 0.78, "F1-score": 0.81, "Support": 414},
                {"Kelas": "Peluang", "Precision": 0.79, "Recall": 0.83, "F1-score": 0.81, "Support": 242},
            ],
        },
        "TF-IDF + SVM dengan stemming": {
            "Accuracy": 0.817183,
            "Macro F1": 0.82,
            "Weighted F1": 0.82,
            "Insight": "Skenario dengan stemming tidak meningkatkan performa baseline. Pada dataset ini, stemming berpotensi menghilangkan sebagian informasi kata yang berguna untuk klasifikasi sentimen.",
            "Rows": [
                {"Kelas": "Ancaman", "Precision": 0.82, "Recall": 0.88, "F1-score": 0.85, "Support": 345},
                {"Kelas": "Netral", "Precision": 0.85, "Recall": 0.76, "F1-score": 0.80, "Support": 414},
                {"Kelas": "Peluang", "Precision": 0.77, "Recall": 0.83, "F1-score": 0.80, "Support": 242},
            ],
        },
        "IndoBERT": {
            "Accuracy": 0.849151,
            "Macro F1": 0.85,
            "Weighted F1": 0.85,
            "Insight": "IndoBERT adalah model terbaik pada penelitian ini. Model ini memperoleh accuracy tertinggi dan performa macro serta weighted F1 yang paling stabil.",
            "Rows": [
                {"Kelas": "Ancaman", "Precision": 0.82, "Recall": 0.93, "F1-score": 0.87, "Support": 345},
                {"Kelas": "Netral", "Precision": 0.87, "Recall": 0.81, "F1-score": 0.84, "Support": 414},
                {"Kelas": "Peluang", "Precision": 0.86, "Recall": 0.81, "F1-score": 0.83, "Support": 242},
            ],
        },
        "IndoBERTweet": {
            "Accuracy": 0.833167,
            "Macro F1": 0.83,
            "Weighted F1": 0.83,
            "Insight": "IndoBERTweet memiliki performa yang kompetitif dan menjadi peringkat kedua, tetapi pada dataset ini masih sedikit di bawah IndoBERT.",
            "Rows": [
                {"Kelas": "Ancaman", "Precision": 0.84, "Recall": 0.88, "F1-score": 0.86, "Support": 345},
                {"Kelas": "Netral", "Precision": 0.86, "Recall": 0.78, "F1-score": 0.82, "Support": 414},
                {"Kelas": "Peluang", "Precision": 0.79, "Recall": 0.86, "F1-score": 0.82, "Support": 242},
            ],
        },
        "IndoBERTweet tanpa normalisasi": {
            "Accuracy": 0.831169,
            "Macro F1": 0.83,
            "Weighted F1": 0.83,
            "Insight": "IndoBERTweet tanpa normalisasi masih menghasilkan performa yang tinggi, tetapi sedikit lebih rendah dibanding skenario IndoBERTweet dengan normalisasi ringan.",
            "Rows": [
                {"Kelas": "Ancaman", "Precision": 0.83, "Recall": 0.89, "F1-score": 0.86, "Support": 345},
                {"Kelas": "Netral", "Precision": 0.87, "Recall": 0.77, "F1-score": 0.81, "Support": 414},
                {"Kelas": "Peluang", "Precision": 0.78, "Recall": 0.86, "F1-score": 0.82, "Support": 242},
            ],
        },
    }

    # 1. Summary cards
    st.markdown("#### Ringkasan Hasil Evaluasi")
    card_cols = st.columns(4)
    summary_cards = [
        ("Best Overall Model", "IndoBERT", "Ranking #1", "#E15759"),
        ("Best Accuracy", "84.92%", "Test set", "#59A14F"),
        ("Best Baseline", "SVM no stem", "TF-IDF + SVM", "#4E79A7"),
        ("Accuracy Gap", "+2.60 pp", "vs baseline terbaik", "#F28E2B"),
    ]
    for col, (label, value, sub, color) in zip(card_cols, summary_cards):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-color:{color};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Horizontal accuracy ranking chart
    st.markdown("#### Ranking Accuracy Model")
    model_names = list(MODELS.keys())
    accs = [MODELS[n] for n in model_names]
    sorted_pairs = sorted(zip(accs, model_names), reverse=True)
    sorted_accs, sorted_names = zip(*sorted_pairs)

    bar_colors = [
        "#E15759" if name == "IndoBERT" else MODEL_COLORS.get(name, "#4E79A7")
        for name in sorted_names
    ]
    text_labels = [
        f"{'Best - ' if name == 'IndoBERT' else ''}{acc:.4f} ({acc*100:.2f}%)"
        for acc, name in zip(sorted_accs, sorted_names)
    ]

    fig = go.Figure(go.Bar(
        x=list(sorted_accs),
        y=list(sorted_names),
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        text=text_labels,
        textposition="outside",
        textfont=dict(color="#e6edf3", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
            range=[0.79, 0.89],
            tickformat=".1%",
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            linecolor="#30363d",
            autorange="reversed",
        ),
        margin=dict(l=10, r=160, t=10, b=10),
        height=310,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Main evaluation table
    st.markdown("#### Tabel Evaluasi Utama Semua Model")
    insight(
        "Tabel ini merangkum metrik evaluasi utama untuk seluruh model. "
        "Accuracy digunakan sebagai dasar ranking, sedangkan macro average dan weighted average "
        "digunakan untuk melihat kestabilan performa model secara keseluruhan.",
    )

    display_df = eval_df.copy()
    display_df["Model"] = display_df["Model"].apply(
        lambda x: "IndoBERT (Best)" if x == "IndoBERT" else x
    )
    metric_cols = [
        "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
        "Weighted Precision", "Weighted Recall", "Weighted F1"
    ]

    def highlight_best(row):
        if "IndoBERT" in str(row["Model"]):
            return ["background-color: rgba(225, 87, 89, 0.22); color: #ffffff; font-weight: 700;" for _ in row]
        return ["" for _ in row]

    styled_eval = (
        display_df.style
        .format({col: "{:.4f}" if col == "Accuracy" else "{:.2f}" for col in metric_cols})
        .apply(highlight_best, axis=1)
    )
    st.dataframe(styled_eval, use_container_width=True, hide_index=True)

    # 4. Interactive detail by selected model
    st.markdown("#### Detail Evaluasi per Model")
    selected_model = st.selectbox(
        "Pilih model untuk melihat detail classification report",
        list(detail_data.keys()),
        index=2,
    )
    selected_detail = detail_data[selected_model]

    detail_cols = st.columns(3)
    mini_cards = [
        ("Accuracy", f"{selected_detail['Accuracy']*100:.2f}%", "Test set", MODEL_COLORS.get(selected_model, "#E15759")),
        ("Macro F1", f"{selected_detail['Macro F1']:.2f}", "Rata-rata antar kelas", "#59A14F"),
        ("Weighted F1", f"{selected_detail['Weighted F1']:.2f}", "Memperhitungkan support", "#F28E2B"),
    ]
    for col, (label, value, sub, color) in zip(detail_cols, mini_cards):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-color:{color};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    detail_df = pd.DataFrame(selected_detail["Rows"])
    st.dataframe(
        detail_df.style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-score": "{:.2f}",
            "Support": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    insight(selected_detail["Insight"], "success" if selected_model == "IndoBERT" else "")

    # 5. Conclusion
    st.markdown("---")
    st.markdown("""
    <div class="best-model-card">
        <div class="best-badge">Kesimpulan</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#e6edf3;margin-bottom:0.5rem;">
            IndoBERT adalah Model Terbaik
        </div>
        <p style="color:#c9d1d9;font-size:0.95rem;line-height:1.7;margin:0;max-width:760px;margin:0 auto;">
            Berdasarkan evaluasi accuracy, macro average, weighted average, dan detail performa per kelas,
            IndoBERT menjadi model terbaik dalam penelitian ini. Model ini memperoleh accuracy tertinggi
            sebesar <strong style="color:#E15759;">84.92%</strong> dan menunjukkan performa yang stabil
            dibandingkan model lainnya.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: HTML VIEWER
# ─────────────────────────────────────────────────────────────
elif selected_name == "HTML Viewer":
    section_header("12", "Raw HTML Notebook Viewer")

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">📄 Tentang Halaman Ini</div>
        <p style="color:#c9d1d9;font-size:0.88rem;line-height:1.7;margin:0;">
            Halaman ini menampilkan isi asli notebook Bab IV secara penuh di dalam dashboard.
            Gunakan sebagai fallback apabila ada bagian analisis yang tidak tertangkap oleh parser otomatis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ File HTML berukuran besar (~1.8 MB). Rendering mungkin memerlukan beberapa saat."
    )

    if raw_html:
        import streamlit.components.v1 as components
        with st.expander("🔍 Tampilkan Notebook HTML (klik untuk buka)", expanded=False):
            components.html(raw_html, height=900, scrolling=True)
    else:
        st.error("File HTML tidak dapat dimuat.")
