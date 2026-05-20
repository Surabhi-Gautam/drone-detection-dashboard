"""
Drone Detection Dashboard
Source data: Project Report - Sarthak.docx (Table 4.1 & 4.2)
             + actual epoch-by-epoch training logs from Drone_YOLO11s_BiFPN_CBAM.ipynb
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io

# ─────────────────────────────────────────────────────────────────────────────
# DATA — Table 4.1  (Model × Metric comparison)
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    "YOLO11n\n(Baseline)",
    "YOLO11n +\nCBAM+BiFPN",
    "YOLO11s\n(Baseline)",
    "YOLO11s +\nCBAM+BiFPN",
]
MODELS_FLAT = [
    "YOLO11n (Baseline)",
    "YOLO11n + CBAM+BiFPN",
    "YOLO11s (Baseline)",
    "YOLO11s + CBAM+BiFPN",
]
PRECISION   = [89.7, 90.1, 92.4, 94.8]
RECALL      = [86.5, 87.3, 88.1, 93.7]
MAP50       = [90.2, 91.0, 93.6, 96.1]
MAP50_95    = [63.4, 63.7, 68.5, 71.8]
FPS         = [98,   94,   78,   75  ]

df_t41 = pd.DataFrame({
    "Model":     MODELS_FLAT,
    "Precision": PRECISION,
    "Recall":    RECALL,
    "mAP50":     MAP50,
    "mAP50-95":  MAP50_95,
    "FPS":       FPS,
})
df_t41["F1"] = (2 * df_t41["Precision"] * df_t41["Recall"]
                / (df_t41["Precision"] + df_t41["Recall"])).round(2)
df_t41["Family"] = ["Nano", "Nano", "Small", "Small"]
df_t41["Variant"] = ["Baseline", "CBAM+BiFPN", "Baseline", "CBAM+BiFPN"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA — Table 4.2  (Improvement deltas)
# ─────────────────────────────────────────────────────────────────────────────
METRICS_DELTA = ["Precision", "Recall", "mAP50", "mAP50-95"]
DELTA_N = [0.4, 0.8, 0.8, 0.3]   # YOLO11n improvements
DELTA_S = [2.4, 5.6, 2.5, 3.3]   # YOLO11s improvements

df_t42 = pd.DataFrame({
    "Metric":             METRICS_DELTA,
    "YOLO11n Δ (%)":      DELTA_N,
    "YOLO11s Δ (%)":      DELTA_S,
})

# ─────────────────────────────────────────────────────────────────────────────
# DATA — Actual epoch training logs (YOLO11s + CBAM+BiFPN, from notebook output)
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS   = list(range(1, 51))
BOX_LOSS = [1.547,1.488,1.448,1.431,1.452,1.436,1.404,1.378,1.336,1.33,
            1.313,1.276,1.274,1.22,1.242,1.243,1.22,1.176,1.194,1.178,
            1.153,1.166,1.134,1.135,1.133,1.121,1.112,1.102,1.076,1.057,
            1.048,1.04,1.041,1.056,1.039,1.02,0.9772,0.9947,1.005,0.9591,
            0.848,0.8148,0.8089,0.7947,0.7579,0.7617,0.7319,0.7473,0.7288,0.7206]
CLS_LOSS = [2.056,1.257,1.218,1.193,1.199,1.164,1.106,1.046,0.9934,1.018,
            0.9931,0.9352,0.9248,0.8735,0.8745,0.8522,0.8301,0.8104,0.8069,0.7622,
            0.7458,0.7644,0.7394,0.7335,0.747,0.7014,0.7061,0.6776,0.6727,0.6576,
            0.6613,0.6325,0.6296,0.6324,0.6178,0.599,0.5719,0.5697,0.5765,0.5663,
            0.444,0.4144,0.4124,0.3997,0.3843,0.3811,0.3608,0.3647,0.3595,0.3541]
DFL_LOSS = [1.676,1.632,1.6,1.601,1.622,1.583,1.562,1.545,1.498,1.51,
            1.479,1.441,1.433,1.42,1.44,1.424,1.41,1.386,1.398,1.383,
            1.352,1.372,1.351,1.342,1.353,1.33,1.338,1.329,1.304,1.294,
            1.293,1.293,1.284,1.297,1.28,1.28,1.242,1.241,1.245,1.23,
            1.224,1.206,1.212,1.184,1.158,1.166,1.144,1.154,1.137,1.131]
VAL_P    = [63.6,57.4,40.5,58.3,68.7,77.0,72.7,82.6,67.3,78.0,
            79.2,79.3,81.0,78.2,90.4,86.8,79.9,86.8,86.3,94.7,
            90.3,94.2,90.0,89.2,88.6,91.7,89.6,90.6,85.1,94.1,
            88.1,89.3,93.4,90.1,83.8,93.0,96.8,91.9,92.0,97.9,
            95.8,93.3,95.3,93.1,96.6,96.8,95.9,94.6,94.4,96.7]
VAL_R    = [57.2,49.9,43.6,57.7,64.5,72.9,70.8,72.1,73.2,68.6,
            74.5,70.4,83.3,77.0,81.5,81.0,82.1,81.7,82.9,79.9,
            85.3,88.2,87.0,85.9,83.5,84.0,85.9,77.8,85.3,90.5,
            84.8,81.5,88.4,85.6,81.4,86.5,90.2,88.3,90.2,89.7,
            87.1,90.5,89.7,88.6,89.2,90.6,90.2,90.5,91.5,91.1]
VAL_M50  = [57.4,51.6,32.6,55.7,64.2,75.7,71.5,79.1,70.8,76.4,
            78.5,74.2,82.9,81.4,87.4,88.0,82.8,85.3,88.4,89.3,
            88.3,92.1,91.7,89.8,86.8,90.6,90.4,87.1,89.8,93.9,
            88.9,88.6,93.7,89.7,80.6,91.6,94.4,92.3,93.7,94.2,
            93.2,94.3,94.4,93.8,94.8,94.5,95.2,93.8,94.7,94.6]
VAL_M595 = [31.0,19.2,11.3,22.3,29.8,40.6,35.6,41.1,35.7,37.8,
            41.2,40.1,44.4,43.6,49.1,45.6,45.7,43.7,50.5,52.9,
            49.9,50.4,53.0,51.1,44.6,50.8,51.6,50.1,50.6,57.0,
            52.2,50.4,53.8,51.4,45.6,53.6,54.4,51.8,54.9,54.4,
            55.2,56.6,58.7,58.2,57.6,56.7,57.3,57.7,57.7,58.0]
F1_EPOCH = [2*p*r/(p+r+1e-9) for p,r in zip(VAL_P, VAL_R)]

df_train = pd.DataFrame({
    "Epoch": EPOCHS, "box_loss": BOX_LOSS, "cls_loss": CLS_LOSS, "dfl_loss": DFL_LOSS,
    "Precision": VAL_P, "Recall": VAL_R, "mAP50": VAL_M50, "mAP50-95": VAL_M595, "F1": F1_EPOCH,
    "phase": ["Mosaic ON" if e <= 40 else "Mosaic OFF" for e in EPOCHS],
})

# ─────────────────────────────────────────────────────────────────────────────
# Colour scheme
# ─────────────────────────────────────────────────────────────────────────────
CLR = {
    "YOLO11n (Baseline)":       "#4C78A8",
    "YOLO11n + CBAM+BiFPN":     "#72B7B2",
    "YOLO11s (Baseline)":       "#F58518",
    "YOLO11s + CBAM+BiFPN":     "#E45756",
}
CLR_LIST = list(CLR.values())

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#fafafe",
    "axes.edgecolor":    "#e5e7eb",
    "axes.labelcolor":   "#4b5563",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#1e1b4b",
    "grid.color":        "#ede9fe",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
})

def mpl_buf(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def mosaic_line(fig, row=None, col=None):
    kw = dict(line_dash="dash", line_color="rgba(255,255,0,0.7)",
              annotation_text="Mosaic OFF ⟶",
              annotation_position="top left",
              annotation_font_color="yellow")
    if row is not None:
        fig.add_vline(x=40.5, row=row, col=col, **kw)
    else:
        fig.add_vline(x=40.5, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drone Detection — Model Comparison",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #1e1b4b;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #fafafe;
}
[data-testid="block-container"] {
    padding-top: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #f5f3ff 0%, #ede9fe 100%);
    border-right: 1px solid #ddd6fe;
}
[data-testid="stSidebar"] * { color: #4c1d95 !important; }
[data-testid="stSidebarNav"] { display: none; }

/* sidebar radio */
[data-testid="stSidebar"] [role="radio"] {
    border-radius: 8px;
    padding: 7px 12px;
    transition: background 0.15s;
    font-weight: 500;
}
[data-testid="stSidebar"] [role="radio"]:hover {
    background: rgba(109,40,217,0.08);
}
[data-testid="stSidebar"] [aria-checked="true"] {
    background: rgba(109,40,217,0.12) !important;
    border-left: 3px solid #7c3aed !important;
    color: #5b21b6 !important;
    font-weight: 600 !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Page title ── */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1e1b4b;
    line-height: 1.2;
    margin: 0 0 6px 0;
}
.page-title span { color: #7c3aed; }
.page-subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 8px;
    font-weight: 400;
}

/* ── Section divider ── */
.divider {
    height: 2px;
    background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 40%, transparent 100%);
    margin: 32px 0 24px 0;
    border: none;
    border-radius: 2px;
    opacity: 0.35;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #1e1b4b;
    border-left: 3px solid #7c3aed;
    padding-left: 10px;
    margin: 28px 0 14px 0;
    letter-spacing: -0.01em;
}

/* ── Chart card wrapper ── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #ede9fe;
    box-shadow: 0 2px 12px rgba(109,40,217,0.06);
    padding: 6px;
    margin-bottom: 10px;
}

/* ── Matplotlib images ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #ede9fe;
    box-shadow: 0 2px 12px rgba(109,40,217,0.06);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #ede9fe !important;
    box-shadow: 0 1px 6px rgba(109,40,217,0.05);
    overflow: hidden;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f5f3ff; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:20px 8px 24px 8px;border-bottom:1px solid #ddd6fe;margin-bottom:16px">
  <div style="font-size:22px;font-weight:700;color:#1e1b4b;letter-spacing:-.02em">
    🛸 Drone Detection
  </div>
  <div style="font-size:11px;color:#7c3aed;font-weight:600;letter-spacing:.1em;
              text-transform:uppercase;margin-top:4px">
    Research Dashboard
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "Model Comparisons",
    "Improvement Gains",
    "Metrics Heatmaps",
    "Precision-Recall & F1",
], label_visibility="collapsed")

st.sidebar.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background:rgba(109,40,217,0.06);border-radius:12px;
            padding:16px 18px;margin:0 6px;border:1px solid #ddd6fe">
  <div style="font-size:10px;color:#7c3aed;font-weight:700;letter-spacing:.12em;
              text-transform:uppercase;margin-bottom:12px">Paper Info</div>
  <div style="font-size:12px;color:#4c1d95;line-height:2.1">
    <b style="color:#1e1b4b">Paper</b><br>Attention-Guided BiFPN-YOLO11<br>
    <b style="color:#1e1b4b">Models</b><br>YOLO11n &amp; YOLO11s ± CBAM+BiFPN<br>
    <b style="color:#1e1b4b">Dataset</b><br>YOLO Drone Detection<br>
    <b style="color:#1e1b4b">Hardware</b><br>NVIDIA T4 · Google Colab · Apple MPS
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="text-align:center;font-size:10px;color:#a78bfa;letter-spacing:.05em">
  YOLO11 · CBAM · BiFPN · 2024
</div>
""", unsafe_allow_html=True)

# ── UI helpers ────────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom:28px;padding-bottom:20px;border-bottom:2px solid #ede9fe">
      <div class="page-title">{title}</div>
      <div class="page-subtitle">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MODEL COMPARISONS
# ═════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafafe",
    font=dict(family="Inter, sans-serif", color="#1e1b4b"),
)
DEF_MARGIN = dict(t=60, b=40, l=20, r=20)
AXIS_STYLE = dict(
    showgrid=True, gridcolor="#ede9fe",
    zeroline=False, tickfont=dict(size=12, color="#6b7280"),
    linecolor="#e5e7eb", linewidth=1,
)

# hex → rgba helper
def hex_rgba(h, a=0.18):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

if page == "Model Comparisons":
    page_header("Model Performance Comparison",
                "Four configurations evaluated on the Drone Detection Dataset · NVIDIA T4")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    cards = [
        ("Best mAP50",    "96.1%", "YOLO11s + CBAM+BiFPN", "#7c3aed", "#f5f3ff"),
        ("Best mAP50-95", "71.8%", "YOLO11s + CBAM+BiFPN", "#6d28d9", "#ede9fe"),
        ("Best Precision","94.8%", "YOLO11s + CBAM+BiFPN", "#5b21b6", "#f5f3ff"),
        ("Best Recall",   "93.7%", "YOLO11s + CBAM+BiFPN", "#7c3aed", "#ede9fe"),
    ]
    for col, (label, val, sub, clr, bg) in zip([k1,k2,k3,k4], cards):
        col.markdown(
            f"""<div style="background:{bg};border-left:4px solid {clr};
            border-radius:12px;padding:18px 20px;margin-bottom:8px;
            box-shadow:0 2px 10px rgba(109,40,217,0.08)">
            <div style="font-size:11px;color:#7c3aed;text-transform:uppercase;
                        letter-spacing:.1em;font-weight:600;margin-bottom:4px">{label}</div>
            <div style="font-size:30px;font-weight:700;color:#1e1b4b;line-height:1.1">{val}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:6px">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    divider()
    section("Results Table")
    tbl_metrics = ["Precision","Recall","mAP50","mAP50-95","F1","FPS"]
    tbl_vals = [
        [f"{v:.1f}%" for v in df_t41["Precision"]],
        [f"{v:.1f}%" for v in df_t41["Recall"]],
        [f"{v:.1f}%" for v in df_t41["mAP50"]],
        [f"{v:.1f}%" for v in df_t41["mAP50-95"]],
        [f"{v:.1f}%" for v in df_t41["F1"]],
        [f"{v}" for v in df_t41["FPS"]],
    ]
    row_fill = ["#fafafe", "#f5f3ff", "#fafafe", "#f5f3ff"]
    fig = go.Figure(go.Table(
        columnwidth=[2.2, 1, 1, 1, 1, 1, 0.8],
        header=dict(
            values=["<b>Model</b>"] + [f"<b>{m}</b>" for m in tbl_metrics],
            fill_color="#7c3aed",
            font=dict(color="white", size=13),
            align="center",
            height=40,
            line_color="#6d28d9",
        ),
        cells=dict(
            values=[df_t41["Model"].tolist()] + tbl_vals,
            fill_color=[row_fill]*7,
            font=dict(color=["#1e1b4b"]*7, size=12),
            align=["left"] + ["center"]*6,
            height=38,
            line_color="#ede9fe",
        ),
    ))
    fig.update_layout(**LAYOUT, height=230, margin=dict(t=10, b=10, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    divider()
    section("All Metrics — Side by Side")
    METRICS_BAR = ["Precision", "Recall", "mAP50", "mAP50-95"]
    long = df_t41.melt(id_vars="Model", value_vars=METRICS_BAR,
                       var_name="Metric", value_name="Score")
    fig = go.Figure()
    for model, clr in CLR.items():
        sub = long[long["Model"] == model]
        fig.add_trace(go.Bar(
            x=sub["Metric"], y=sub["Score"],
            name=model, marker_color=clr,
            marker_line_color="rgba(255,255,255,0.15)",
            marker_line_width=1,
            text=[f"{v:.1f}%" for v in sub["Score"]],
            textposition="outside",
            textfont=dict(size=11, color="#e0e0e0"),
        ))
    fig.update_layout(
        **LAYOUT,
        margin=DEF_MARGIN,
        barmode="group",
        yaxis=dict(**AXIS_STYLE, range=[55, 102], title="Score (%)", title_font_size=13),
        xaxis=dict(**AXIS_STYLE, title=""),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        ),
        height=480,
        bargap=0.18, bargroupgap=0.06,
    )
    st.plotly_chart(fig, use_container_width=True)

    section("Per-Metric Deep Dive")
    short_labels = ["11n Base", "11n+CB+BF", "11s Base", "11s+CB+BF"]
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"<b>{m}</b>" for m in METRICS_BAR],
        horizontal_spacing=0.06,
    )
    y_ranges = {"Precision":[84,98],"Recall":[82,97],"mAP50":[87,99],"mAP50-95":[60,75]}
    for ci, metric in enumerate(METRICS_BAR, start=1):
        vals = df_t41[metric].tolist()
        best_val = max(vals)
        for mi, (label, val, clr) in enumerate(zip(short_labels, vals, CLR_LIST)):
            is_best = val == best_val
            fig.add_trace(go.Bar(
                x=[label], y=[val],
                marker_color=clr,
                marker_line_color="white" if is_best else "rgba(0,0,0,0)",
                marker_line_width=2 if is_best else 0,
                marker_pattern_shape="" ,
                text=[f"{val:.1f}%"],
                textposition="outside",
                textfont=dict(size=11, color="white" if is_best else "#aaa"),
                showlegend=False,
                name=MODELS_FLAT[mi],
            ), row=1, col=ci)
        # best marker annotation
        fig.add_annotation(
            x=short_labels[vals.index(best_val)], y=best_val + 1.2,
            text="★", font=dict(size=14, color="#FFD700"),
            showarrow=False, row=1, col=ci,
        )
        yr = y_ranges[metric]
        fig.update_yaxes(range=yr, showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                         tickfont=dict(size=10), row=1, col=ci)
        fig.update_xaxes(tickfont=dict(size=9), row=1, col=ci)
    fig.update_layout(
        **LAYOUT,
        height=420,
        margin=dict(t=50, b=30, l=10, r=10),
        showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#c9d1d9"
        ann.font.size = 13
    st.plotly_chart(fig, use_container_width=True)

    section("Radar Profile & Score Heatmap")
    col1, col2 = st.columns([1, 1])

    with col1:
        radar_metrics = ["Precision", "Recall", "mAP50", "mAP50-95", "F1"]
        fig = go.Figure()
        for model, clr in CLR.items():
            row = df_t41[df_t41["Model"] == model].iloc[0]
            vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
            theta = radar_metrics + [radar_metrics[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=theta,
                fill="toself", name=model.replace(" + CBAM+BiFPN", "\n+CBAM+BiFPN"),
                line=dict(color=clr, width=2.5),
                fillcolor=hex_rgba(clr, 0.12),
                marker=dict(size=6, color=clr),
            ))
        fig.update_layout(
            **LAYOUT,
            margin=DEF_MARGIN,
            polar=dict(
                bgcolor="#f5f3ff",
                radialaxis=dict(
                    visible=True, range=[60, 100],
                    tickfont=dict(size=9, color="#7c3aed"),
                    gridcolor="#ddd6fe",
                    linecolor="#ddd6fe",
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#1e1b4b"),
                    gridcolor="#ddd6fe",
                    linecolor="#ddd6fe",
                ),
            ),
            legend=dict(
                orientation="h", y=-0.18, x=0.5, xanchor="center",
                font=dict(size=10), bgcolor="rgba(0,0,0,0)",
            ),
            height=440,
            title=dict(text="Model Capability Profile", font=dict(size=14), x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        heat_cols  = ["Precision", "Recall", "mAP50", "mAP50-95", "F1"]
        heat_data  = df_t41[heat_cols].values
        heat_labels= ["YOLO11n (Base)", "YOLO11n +CB+BF", "YOLO11s (Base)", "YOLO11s +CB+BF"]
        fig_h, ax  = plt.subplots(figsize=(6, 3.6))
        fig_h.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        sns.heatmap(
            heat_data, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=heat_cols, yticklabels=heat_labels,
            vmin=63, vmax=97, linewidths=1, linecolor="#ffffff",
            ax=ax, cbar_kws={"label": "Score (%)", "shrink": 0.85},
            annot_kws={"size": 11, "weight": "bold", "color": "#1e1b4b"},
        )
        ax.set_title("Score Heatmap", color="#1e1b4b", fontsize=13, pad=10)
        ax.tick_params(colors="#6b7280", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color="#6b7280")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#6b7280")
        cbar.set_label("Score (%)", color="#6b7280")
        plt.tight_layout()
        st.image(mpl_buf(fig_h, dpi=130), use_column_width=True)

    divider()
    section("Baseline → Enhanced Lift  (Nano vs Small)")
    fig = go.Figure()
    x_labels = ["Precision", "Recall", "mAP50", "mAP50-95"]

    nano_pairs  = [(CLR["YOLO11n (Baseline)"],  CLR["YOLO11n + CBAM+BiFPN"],  "Nano")]
    small_pairs = [(CLR["YOLO11s (Baseline)"],  CLR["YOLO11s + CBAM+BiFPN"],  "Small")]

    for (base_clr, enh_clr, family), (base_idx, enh_idx) in zip(
            nano_pairs + small_pairs, [(0,1),(2,3)]):
        base_vals = [df_t41.iloc[base_idx][m] for m in x_labels]
        enh_vals  = [df_t41.iloc[enh_idx][m]  for m in x_labels]
        dash_style = "dot" if family == "Nano" else "solid"
        # baseline line
        fig.add_trace(go.Scatter(
            x=x_labels, y=base_vals, mode="lines+markers",
            name=f"YOLO11{family[0].lower()} Baseline",
            line=dict(color=base_clr, width=2, dash=dash_style),
            marker=dict(size=10, color=base_clr,
                        line=dict(width=2, color="white")),
        ))
        # enhanced line
        fig.add_trace(go.Scatter(
            x=x_labels, y=enh_vals, mode="lines+markers",
            name=f"YOLO11{family[0].lower()} + CBAM+BiFPN",
            line=dict(color=enh_clr, width=2.5, dash=dash_style),
            marker=dict(size=10, symbol="diamond", color=enh_clr,
                        line=dict(width=2, color="white")),
        ))
        # shaded lift region
        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=enh_vals + base_vals[::-1],
            fill="toself",
            fillcolor=hex_rgba(enh_clr, 0.10),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # delta annotations on enhanced points
        for x, bv, ev in zip(x_labels, base_vals, enh_vals):
            fig.add_annotation(
                x=x, y=ev + 0.5,
                text=f"+{ev-bv:.1f}%",
                showarrow=False,
                font=dict(size=10, color=enh_clr),
                yshift=8,
            )

    fig.update_layout(
        **LAYOUT,
        margin=DEF_MARGIN,
        yaxis=dict(**AXIS_STYLE, range=[82, 100], title="Score (%)", title_font_size=13),
        xaxis=dict(**AXIS_STYLE, title=""),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TABLE 4.2
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Improvement Gains":
    page_header("Improvement Gains",
                "How much each metric improves after integrating CBAM and BiFPN (Baseline → Enhanced)")

    section("Raw Improvement Table")
    st.dataframe(df_t42, use_container_width=True, hide_index=True)
    divider()

    section("Improvement per Metric")
    long42 = df_t42.melt(id_vars="Metric", var_name="Model", value_name="Δ%")
    fig = px.bar(long42, x="Metric", y="Δ%", color="Model", barmode="group",
                 color_discrete_map={"YOLO11n Δ (%)":"#4C78A8","YOLO11s Δ (%)":"#E45756"},
                 text="Δ%")
    fig.update_traces(texttemplate="+%{text:.1f}%", textposition="outside", textfont_size=13)
    fig.update_layout(yaxis_title="Improvement (%)", yaxis_range=[0, 7],
                      legend_title="Model", height=420)
    st.plotly_chart(fig, use_container_width=True)

    section("Improvement Lollipop — Nano vs Small")
    fig = go.Figure()
    for i, (metric, delta_n, delta_s) in enumerate(zip(METRICS_DELTA, DELTA_N, DELTA_S)):
        # Nano
        fig.add_trace(go.Scatter(x=[0, delta_n], y=[f"{metric} (Nano)"]*2,
                                 mode="lines", line=dict(color="#4C78A8", width=2),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[delta_n], y=[f"{metric} (Nano)"],
                                 mode="markers+text", marker=dict(size=14, color="#4C78A8"),
                                 text=[f"+{delta_n}%"], textposition="middle right",
                                 name="YOLO11n", showlegend=(i==0)))
        # Small
        fig.add_trace(go.Scatter(x=[0, delta_s], y=[f"{metric} (Small)"]*2,
                                 mode="lines", line=dict(color="#E45756", width=2),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[delta_s], y=[f"{metric} (Small)"],
                                 mode="markers+text", marker=dict(size=14, color="#E45756"),
                                 text=[f"+{delta_s}%"], textposition="middle right",
                                 name="YOLO11s", showlegend=(i==0)))
    fig.add_vline(x=0, line_color="white", line_width=0.8)
    fig.update_layout(xaxis_title="Improvement (%)", xaxis_range=[-0.3, 7],
                      height=430, legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, use_container_width=True)

    section("Improvement Heatmap")
    heat42 = np.array([DELTA_N, DELTA_S])
    fig_h, ax = plt.subplots(figsize=(7, 2.5))
    sns.heatmap(heat42, annot=True, fmt=".1f",
                cmap="YlOrRd",
                xticklabels=METRICS_DELTA,
                yticklabels=["YOLO11n Δ%","YOLO11s Δ%"],
                vmin=0, vmax=6, linewidths=0.5, ax=ax,
                cbar_kws={"label":"Improvement (%)"},
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("Table 4.2 — Improvement Heatmap (larger = better gain)", pad=10, fontsize=12)
    plt.tight_layout()
    st.image(mpl_buf(fig_h), use_column_width=False)

    divider()
    section("Before vs After — Absolute Values")
    col1, col2 = st.columns(2)
    for col, (fam, base_idx, enh_idx, clr_base, clr_enh) in zip(
            [col1, col2],
            [("Nano",  0, 1, "#4C78A8", "#72B7B2"),
             ("Small", 2, 3, "#F58518", "#E45756")]):
        with col:
            base = df_t41.iloc[base_idx]
            enh  = df_t41.iloc[enh_idx]
            fig = go.Figure()
            for metric in ["Precision","Recall","mAP50","mAP50-95"]:
                fig.add_trace(go.Bar(
                    name=f"Baseline ({base['Model'].split('(')[0].strip()})",
                    x=[metric], y=[base[metric]],
                    marker_color=clr_base,
                    text=[f"{base[metric]:.1f}%"], textposition="inside",
                    showlegend=(metric=="Precision"),
                ))
                fig.add_trace(go.Bar(
                    name="+ CBAM+BiFPN",
                    x=[metric], y=[enh[metric]],
                    marker_color=clr_enh,
                    text=[f"{enh[metric]:.1f}%"], textposition="inside",
                    showlegend=(metric=="Precision"),
                ))
            fig.update_layout(
                title=f"YOLO11{fam[0].lower()} — Before vs After",
                barmode="group", yaxis_range=[55, 100],
                legend=dict(orientation="h", y=1.1), height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

    section("Radar — Improvement Magnitude")
    fig = go.Figure()
    closed = METRICS_DELTA + [METRICS_DELTA[0]]
    fig.add_trace(go.Scatterpolar(
        r=DELTA_N + [DELTA_N[0]],
        theta=closed, fill="toself", name="YOLO11n Δ",
        line=dict(color="#4C78A8", width=2),
    ))
    fig.add_trace(go.Scatterpolar(
        r=DELTA_S + [DELTA_S[0]],
        theta=closed, fill="toself", name="YOLO11s Δ",
        line=dict(color="#E45756", width=2),
    ))
    fig.update_layout(
        **LAYOUT,
        margin=DEF_MARGIN,
        polar=dict(
            bgcolor="#f5f3ff",
            radialaxis=dict(visible=True, range=[0, 6.5],
                            gridcolor="#ddd6fe", linecolor="#ddd6fe",
                            tickfont=dict(color="#7c3aed", size=9)),
            angularaxis=dict(gridcolor="#ddd6fe", linecolor="#ddd6fe",
                             tickfont=dict(color="#1e1b4b", size=12)),
        ),
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        height=440,
        title=dict(text="Radar — Which metric improves most?",
                   font=dict(color="#1e1b4b", size=14), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)



# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — METRICS HEATMAPS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Metrics Heatmaps":
    page_header("Metrics Heatmaps",
                "Visual breakdown of scores and training dynamics across all models and epochs")

    BEST_E = int(df_train.loc[df_train["mAP50"].idxmax(), "Epoch"])

    section("Model Score Heatmap — All Metrics")
    heat_data = df_t41[["Precision","Recall","mAP50","mAP50-95","F1"]].values
    fig_h, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=["Precision","Recall","mAP50","mAP50-95","F1"],
                yticklabels=["YOLO11n (Base)","YOLO11n+CB+BF",
                              "YOLO11s (Base)","YOLO11s+CB+BF"],
                vmin=60, vmax=98, linewidths=0.6, ax=ax,
                cbar_kws={"label": "Score (%)"},
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Table 4.1 — All Models, All Metrics", pad=10, fontsize=13)
    plt.tight_layout()
    st.image(mpl_buf(fig_h), use_column_width=True)

    divider()
    section("Improvement Gain Heatmap (Δ%)")
    heat42 = np.array([DELTA_N, DELTA_S])
    fig_h, ax = plt.subplots(figsize=(8, 2.5))
    sns.heatmap(heat42, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=METRICS_DELTA,
                yticklabels=["YOLO11n Δ%","YOLO11s Δ%"],
                vmin=0, vmax=6.5, linewidths=0.6, ax=ax,
                cbar_kws={"label":"Improvement (%)"},
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Table 4.2 — Performance Gain Heatmap (CBAM+BiFPN integration)", pad=10, fontsize=13)
    plt.tight_layout()
    st.image(mpl_buf(fig_h), use_column_width=False)

    divider()
    section("Training Dynamics — All Metrics × 50 Epochs")
    metrics_all = ["mAP50","mAP50-95","Precision","Recall","F1",
                   "box_loss","cls_loss","dfl_loss"]
    hm_df = df_train[["Epoch"] + metrics_all].set_index("Epoch")
    hm_norm = (hm_df - hm_df.min()) / (hm_df.max() - hm_df.min())
    fig_h, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(hm_norm.T, cmap="RdYlGn", ax=ax,
                xticklabels=[e if e % 5 == 0 else "" for e in EPOCHS],
                yticklabels=True,
                cbar_kws={"label":"Normalized (per-metric min-max)"},
                linewidths=0.1, linecolor="gray")
    ax.axvline(x=BEST_E - 0.5, color="cyan", lw=2, ls="--")
    ax.text(BEST_E - 0.5, -0.7, f"Best E{BEST_E}", color="cyan", ha="center", fontsize=9)
    ax.axvline(x=40.5, color="yellow", lw=2, ls=":")
    ax.text(40.5, -0.7, "Mosaic\nOFF", color="yellow", ha="center", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_title("All Training Metrics — Normalized Heatmap (green = better per metric)", pad=10)
    plt.tight_layout()
    st.image(mpl_buf(fig_h, dpi=130), use_column_width=True)

    divider()
    section("Metric Correlation Heatmap")
    corr = df_train[metrics_all].corr()
    fig_h, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink":0.8},
                annot_kws={"size": 10})
    ax.set_title("Pearson Correlation — Training Metrics (YOLO11s+CBAM+BiFPN)", pad=10, fontsize=12)
    plt.tight_layout()
    st.image(mpl_buf(fig_h), use_column_width=False)

    divider()
    section("5-Epoch Rolling Average Heatmap")
    def smooth(arr, w=5):
        return pd.Series(arr).rolling(w, min_periods=1).mean().tolist()
    roll_data = np.array([
        smooth(VAL_M50), smooth(VAL_M595),
        smooth(VAL_P), smooth(VAL_R), smooth(F1_EPOCH)
    ])
    fig_h, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(roll_data, cmap="YlOrRd",
                yticklabels=["mAP50","mAP50-95","Precision","Recall","F1"],
                xticklabels=[str(e) if e % 5 == 0 else "" for e in EPOCHS],
                vmin=30, vmax=100, linewidths=0.1, linecolor="gray",
                cbar_kws={"label":"Score (%)"}, ax=ax)
    ax.axvline(x=BEST_E - 0.5, color="cyan", lw=2, ls="--")
    ax.axvline(x=40.5, color="yellow", lw=2, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_title("5-Epoch Rolling Average — Validation Metrics", pad=8)
    plt.tight_layout()
    st.image(mpl_buf(fig_h, dpi=130), use_column_width=True)

    divider()
    section("Phase Comparison — Mosaic ON vs OFF")
    phase_on  = df_train[df_train["phase"]=="Mosaic ON"][metrics_all].mean()
    phase_off = df_train[df_train["phase"]=="Mosaic OFF"][metrics_all].mean()
    ph_df = pd.DataFrame([phase_on, phase_off],
                          index=["Mosaic ON (E1-40)","Mosaic OFF (E41-50)"])
    fig_h, ax = plt.subplots(figsize=(12, 2.5))
    sns.heatmap(ph_df, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax,
                cbar_kws={"label":"Value"})
    ax.set_title("Phase Mean Values — Mosaic ON vs OFF", pad=8, fontsize=12)
    plt.tight_layout()
    st.image(mpl_buf(fig_h), use_column_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — P-R & F1
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Precision-Recall & F1":
    page_header("Precision-Recall & F1 Analysis",
                "Detection quality decomposed — how precisely and completely the model finds drones")

    BEST_E = int(df_train.loc[df_train["mAP50"].idxmax(), "Epoch"])
    BEST_F1_E = int(df_train.loc[df_train["F1"].idxmax(), "Epoch"])

    section("Precision vs Recall — All Models (with iso-F1 contours)")
    fig = go.Figure()
    for model, clr in CLR.items():
        row = df_t41[df_t41["Model"]==model].iloc[0]
        fig.add_trace(go.Scatter(
            x=[row["Recall"]], y=[row["Precision"]],
            mode="markers+text",
            marker=dict(size=18, color=clr, line=dict(width=2, color="white")),
            text=[model.replace(" + CBAM+BiFPN","<br>+CBAM+BiFPN")],
            textposition="top center",
            name=model,
        ))
    # iso-F1 contour lines
    r_arr = np.linspace(0.1, 1.0, 100)
    for f1_val in [0.80, 0.85, 0.90, 0.95]:
        p_arr = f1_val * r_arr / (2 * r_arr - f1_val + 1e-9)
        valid = (p_arr > 0) & (p_arr <= 1)
        fig.add_trace(go.Scatter(
            x=r_arr[valid]*100, y=p_arr[valid]*100,
            mode="lines", line=dict(color="gray", dash="dash", width=1),
            name=f"F1={f1_val:.2f}", opacity=0.5,
        ))
        fig.add_annotation(x=r_arr[valid][-1]*100, y=p_arr[valid][-1]*100,
                            text=f"F1={f1_val:.2f}", showarrow=False,
                            font=dict(color="gray", size=9))
    fig.update_layout(xaxis_title="Recall (%)", yaxis_title="Precision (%)",
                      xaxis_range=[80, 100], yaxis_range=[85, 100],
                      showlegend=True, legend=dict(orientation="h", y=-0.2), height=460)
    st.plotly_chart(fig, use_container_width=True)

    divider()
    section("P-R Trajectory Over 50 Epochs  (YOLO11s + CBAM+BiFPN)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=VAL_R, y=VAL_P, mode="markers+lines",
        marker=dict(size=7, color=EPOCHS, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Epoch")),
        line=dict(color="gray", width=1),
        text=[f"E{e}" for e in EPOCHS],
        hovertemplate="Epoch %{text}<br>Recall=%{x:.1f}%<br>Precision=%{y:.1f}%",
        showlegend=False,
    ))
    for e_idx, label, sym, clr in [
        (0, "Start (E1)", "star", "red"),
        (BEST_E-1, f"Best mAP50 (E{BEST_E})", "diamond", "gold"),
        (49, "End (E50)", "star", "lime"),
    ]:
        fig.add_trace(go.Scatter(
            x=[VAL_R[e_idx]], y=[VAL_P[e_idx]],
            mode="markers", marker=dict(size=16, symbol=sym, color=clr),
            name=label,
        ))
    # mark the final model vs table 4.1 val
    fig.add_trace(go.Scatter(
        x=[93.7], y=[94.8], mode="markers",
        marker=dict(size=20, symbol="star", color="white",
                    line=dict(color="gold", width=2)),
        name="Table 4.1 YOLO11s+CB+BF",
    ))
    fig.update_layout(xaxis_title="Recall (%)", yaxis_title="Precision (%)",
                      height=440, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    divider()
    section("F1 Score Over Training Epochs")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=EPOCHS, y=F1_EPOCH, mode="lines+markers",
                             marker_size=5, fill="tozeroy",
                             line=dict(color="#FF97FF", width=2.5),
                             fillcolor="rgba(255,151,255,0.12)", name="F1"))
    fig.add_trace(go.Scatter(x=EPOCHS, y=VAL_P, mode="lines",
                             line=dict(color="#FFA15A", width=1.5, dash="dash"), name="Precision"))
    fig.add_trace(go.Scatter(x=EPOCHS, y=VAL_R, mode="lines",
                             line=dict(color="#19D3F3", width=1.5, dash="dash"), name="Recall"))
    fig.add_vline(x=BEST_F1_E, line_dash="dot", line_color="white",
                  annotation_text=f"Best F1 (E{BEST_F1_E}={df_train['F1'].max():.1f}%)")
    mosaic_line(fig)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Score (%)",
                      yaxis_range=[30, 105],
                      legend=dict(orientation="h", y=1.08), height=380)
    st.plotly_chart(fig, use_container_width=True)

    divider()
    section("Precision / Recall / F1 Heatmap — 50 Epochs")
    fig_h, ax = plt.subplots(figsize=(14, 2.8))
    pr_mat = np.array([VAL_P, VAL_R, F1_EPOCH])
    sns.heatmap(pr_mat, cmap="YlOrRd",
                yticklabels=["Precision","Recall","F1"],
                xticklabels=[str(e) if e % 5 == 0 else "" for e in EPOCHS],
                vmin=30, vmax=100, ax=ax, linewidths=0.1, linecolor="gray",
                cbar_kws={"label":"Score (%)"})
    ax.axvline(x=BEST_E - 0.5, color="cyan", lw=2, ls="--")
    ax.axvline(x=40.5, color="yellow", lw=2, ls=":")
    ax.set_title("Precision / Recall / F1 Heatmap — 50 Training Epochs", pad=8)
    plt.tight_layout()
    st.image(mpl_buf(fig_h, dpi=130), use_column_width=True)

    divider()
    section("Top-10 Epochs by mAP50")
    top10 = (df_train.nlargest(10, "mAP50")
             [["Epoch","mAP50","mAP50-95","Precision","Recall","F1","box_loss","phase"]]
             .reset_index(drop=True))
    top10.index += 1
    for c in ["mAP50","mAP50-95","Precision","Recall","F1"]:
        top10[c] = top10[c].map(lambda x: f"{x:.1f}%")
    st.dataframe(top10, use_container_width=True)