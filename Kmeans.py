# kmeans_streamlit_lines.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# ---------------- Page config & minimal dark styling ----------------
st.set_page_config(page_title="K-Means — Animated with Lines", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071023 0%, #06131b 60%);
             color: #e6eef8; font-family: Inter, Roboto, 'Segoe UI'; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03),
                                        rgba(255,255,255,0.01));
             padding:12px; border-radius:12px; }
    .muted { color:#9fb2d7; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Helpers ----------------
def generate_points(n, seed=None):
    rs = np.random.RandomState(seed) if seed else np.random.RandomState()
    pts = rs.rand(n, 2)
    return pd.DataFrame(pts, columns=["x", "y"])

def init_centroids(points, k, seed=None):
    rs = np.random.RandomState(seed) if seed else np.random.RandomState()
    n = points.shape[0]
    idx = rs.choice(n, size=min(k, n), replace=False)
    return points[idx].astype(float)

def assign_labels(points, centroids):
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)

def update_centroids(points, labels, k, seed=None):
    rs = np.random.RandomState(seed) if seed else np.random.RandomState()
    new = np.zeros((k, 2))
    for i in range(k):
        members = points[labels == i]
        new[i] = members.mean(axis=0) if len(members) else points[rs.randint(0, points.shape[0])]
    return new

# utilities
PALETTE = px.colors.qualitative.Plotly
PALETTE_RGB = [tuple(int(h[i:i+2],16) for i in (0,2,4)) for h in [c.lstrip('#') for c in PALETTE]]
GREY = (200, 200, 200)
def rgba(rgb, a=1.0): r,g,b=rgb; return f"rgba({int(r)},{int(g)},{int(b)},{a})"
def lerp(a,b,t): return a+(b-a)*t
def lerp_rgb(c1,c2,t): return (lerp(c1[0],c2[0],t),lerp(c1[1],c2[1],t),lerp(c1[2],c2[2],t))

# -------------- Figure builder with assignment lines --------------
def build_figure(df, centroids, point_colors, title, labels=None):
    fig = go.Figure()
    # lines from points to centroids
    if labels is not None and centroids is not None:
        pts = df.values
        for i in range(centroids.shape[0]):
            mask = labels == i
            if not np.any(mask): continue
            cx, cy = centroids[i]
            xs, ys = [], []
            for (px, py) in pts[mask]:
                xs += [px, cx, None]
                ys += [py, cy, None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=0.6, color="rgba(255,255,255,0.15)"),
                hoverinfo="skip", showlegend=False))

    # points
    fig.add_trace(go.Scattergl(
        x=df["x"], y=df["y"], mode="markers",
        marker=dict(size=8, color=point_colors,
                    line=dict(width=0.6, color="rgba(0,0,0,0.6)")),
        hoverinfo="skip", showlegend=False))

    # centroids
    if centroids is not None:
        for i,(cx,cy) in enumerate(centroids):
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(size=20, symbol="x",
                            color=PALETTE[i % len(PALETTE)],
                            line=dict(width=2, color="white")),
                hoverinfo="skip", showlegend=False))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        margin=dict(l=6, r=6, t=44, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,20,0.55)",
        xaxis=dict(visible=False, range=[-0.03,1.03]),
        yaxis=dict(visible=False, range=[-0.03,1.03], scaleanchor="x"),
    )
    return fig

# -------------- Session state setup --------------
if "points" not in st.session_state:
    st.session_state.points = generate_points(100)
    st.session_state.k = 4
    st.session_state.centroids = init_centroids(st.session_state.points.values, st.session_state.k, seed=1)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

# -------------- UI ----------------
st.markdown("<div class='card'><h2 style='margin:0'>K-Means — Step-by-Step (with Lines)</h2>"
            "<div class='muted'>Manual steps • smooth 1s animations • dark gradient theme</div></div>",
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1: N = st.slider("N", 20, 800, len(st.session_state.points), 10)
with col2: K = st.slider("K", 2, 12, st.session_state.k, 1)
with col3: new_clicked = st.button("New dataset")
with col4: restart_clicked = st.button("Restart centroids")
step_clicked = st.button("Step ▶")

plot_slot = st.empty()

# --- handle controls ---
if N != len(st.session_state.points):
    st.session_state.points = generate_points(N)
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=1)
    st.session_state.k = K; st.session_state.labels=None; st.session_state.phase="assign"; st.session_state.iter=0
if K != st.session_state.k:
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=1)
    st.session_state.k = K; st.session_state.labels=None; st.session_state.phase="assign"; st.session_state.iter=0
if new_clicked:
    st.session_state.points = generate_points(N)
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=int(time.time())%1000)
    st.session_state.k = K; st.session_state.labels=None; st.session_state.phase="assign"; st.session_state.iter=0
if restart_clicked:
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=int(time.time())%1000)
    st.session_state.labels=None; st.session_state.phase="assign"; st.session_state.iter=0

# animation settings
FRAMES = 20
FRAME_SLEEP = 1.0 / FRAMES

pts_np = st.session_state.points.values
centroids = st.session_state.centroids
labels = st.session_state.labels
npts = pts_np.shape[0]
base_colors = [rgba(GREY,0.45)]*npts

# if no labels yet, render neutral
if labels is None:
    fig = build_figure(st.session_state.points, centroids, base_colors,
                       f"phase: {st.session_state.phase.upper()} • iter: {st.session_state.iter}")
    plot_slot.plotly_chart(fig, use_container_width=True)

# ---- step button pressed ----
if step_clicked:
    if st.session_state.phase == "assign":
        new_labels = assign_labels(pts_np, centroids)
        target_rgb = [PALETTE_RGB[int(l)%len(PALETTE_RGB)] for l in new_labels]
        for f in range(1, FRAMES+1):
            t = f/FRAMES
            colors = [rgba(lerp_rgb(GREY,target_rgb[i],t),0.85) for i in range(npts)]
            fig = build_figure(st.session_state.points, centroids, colors,
                               "phase: ASSIGN (animating)", labels=new_labels)
            plot_slot.plotly_chart(fig, use_container_width=True)
            time.sleep(FRAME_SLEEP)
        st.session_state.labels = new_labels
        st.session_state.phase = "update"

    else:
        old = st.session_state.centroids.copy()
        new = update_centroids(pts_np, st.session_state.labels, st.session_state.k)
        for f in range(1, FRAMES+1):
            t = f/FRAMES
            inter = np.array([lerp(old[i],new[i],t) for i in range(st.session_state.k)])
            rgb = [PALETTE_RGB[int(l)%len(PALETTE_RGB)] for l in st.session_state.labels]
            colors = [rgba(rgb[i],0.85) for i in range(npts)]
            fig = build_figure(st.session_state.points, inter, colors,
                               "phase: UPDATE (animating)", labels=st.session_state.labels)
            plot_slot.plotly_chart(fig, use_container_width=True)
            time.sleep(FRAME_SLEEP)
        st.session_state.centroids = new
        st.session_state.iter += 1
        st.session_state.phase = "assign"

# ---- render static if not animating ----
if not step_clicked:
    if st.session_state.labels is None:
        colors = base_colors
    else:
        rgb = [PALETTE_RGB[int(l)%len(PALETTE_RGB)] for l in st.session_state.labels]
        colors = [rgba(rgb[i],0.85) for i in range(npts)]
    fig = build_figure(st.session_state.points, st.session_state.centroids, colors,
                       f"phase: {st.session_state.phase.upper()} • iter: {st.session_state.iter}",
                       labels=st.session_state.labels)
    plot_slot.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='muted' style='margin-top:8px'>Each step animates for ~1 s. "
            "Faint grey lines show assignments between points and centroids.</div>",
            unsafe_allow_html=True)
