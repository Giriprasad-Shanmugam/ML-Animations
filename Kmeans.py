# kmeans_streamlit_d3like.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# ---------------- Page config & minimal dark styling ----------------
st.set_page_config(page_title="K-Means — D3-like (Animated)", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071023 0%, #06131b 60%); color: #e6eef8; font-family: Inter, Roboto, 'Segoe UI'; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:12px; border-radius:12px; }
    .muted { color:#9fb2d7; font-size:0.95rem; }
    .controls { display:flex; gap:8px; align-items:center; }
    .btn { padding:8px 12px; border-radius:10px; border:none; cursor:pointer; font-weight:600; }
    .accent { background: linear-gradient(90deg,#7c3aed,#06b6d4); color:white; }
    .ghost { background:transparent; border:1px solid rgba(255,255,255,0.08); color:#cfe7ff; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Helpers: data + kmeans steps + animation utilities ----------------
def generate_points(n: int, seed: int | None = None) -> pd.DataFrame:
    rs = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    pts = rs.rand(n, 2)
    return pd.DataFrame(pts, columns=["x", "y"])

def init_centroids(points: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    rs = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    n = points.shape[0]
    if k >= n:
        idx = np.arange(n)
    else:
        idx = rs.choice(n, size=k, replace=False)
    return points[idx].astype(float)

def assign_labels(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)

def update_centroids(points: np.ndarray, labels: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    rs = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    new = np.zeros((k, 2), dtype=float)
    for i in range(k):
        members = points[labels == i]
        if len(members) == 0:
            new[i] = points[rs.randint(0, points.shape[0])]
        else:
            new[i] = members.mean(axis=0)
    return new

# palette and utilities to create RGBA strings for animation interpolation
PALETTE = px.colors.qualitative.Plotly
GREY = (200, 200, 200)

def rgba_str(rgb_tuple, alpha=1.0):
    r, g, b = rgb_tuple
    return f"rgba({int(r)},{int(g)},{int(b)},{alpha})"

def hex_to_rgb(hexcol):
    hexcol = hexcol.lstrip('#')
    return tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))

PALETTE_RGB = [hex_to_rgb(h) for h in PALETTE]

# build figure with marker color array and centroid markers
def build_figure(df: pd.DataFrame, centroids: np.ndarray, point_rgba: list, title: str):
    fig = go.Figure()
    # single points trace (all points in one trace for performance)
    fig.add_trace(go.Scattergl(
        x=df["x"], y=df["y"],
        mode="markers",
        marker=dict(size=8, color=point_rgba, line=dict(width=0.6, color="rgba(0,0,0,0.6)")),
        hoverinfo="skip",
        showlegend=False,
    ))
    # centroids as big X markers
    if centroids is not None:
        for i, (cx, cy) in enumerate(centroids):
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(size=20, symbol="x", color=PALETTE[i % len(PALETTE)], line=dict(width=2, color="white")),
                hoverinfo="skip",
                showlegend=False,
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        margin=dict(l=6, r=6, t=44, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,20,0.55)",
        xaxis=dict(visible=False, range=[-0.03, 1.03]),
        yaxis=dict(visible=False, range=[-0.03, 1.03], scaleanchor="x"),
        transition=dict(duration=0),  # we handle per-frame timing
    )
    return fig

# linear interpolation helpers
def lerp(a, b, t):
    return a + (b - a) * t

def lerp_rgb(c1, c2, t):
    return (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))

# ---------------- Session state initialization ----------------
if "points" not in st.session_state:
    st.session_state.points = generate_points(100)       # default N=100
    st.session_state.k = 5
    st.session_state.centroids = init_centroids(st.session_state.points.values, st.session_state.k, seed=1)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

# ---------------- Top UI and controls ----------------
st.markdown("<div class='card'><h2 style='margin:0'>Visualizing K-Means (D3-like)</h2>"
            "<div class='muted'>Manual steps • smooth 1s animations • random uniform points</div></div>",
            unsafe_allow_html=True)

ctrl_cols = st.columns([1, 1, 1, 1])
with ctrl_cols[0]:
    N = st.slider("N (points)", min_value=20, max_value=1000, value=len(st.session_state.points), step=10)
with ctrl_cols[1]:
    K = st.slider("K (clusters)", min_value=2, max_value=12, value=st.session_state.k, step=1)
with ctrl_cols[2]:
    new_clicked = st.button("New dataset", key="new", help="Generate new random uniform points")
with ctrl_cols[3]:
    restart_clicked = st.button("Restart centroids", key="restart", help="Randomize centroids only")

step_clicked = st.button("Step ▶", key="step", help="Perform one step (assign → update → assign …)")

# container to hold the animated chart so we can update it in-place
plot_slot = st.empty()

# ---------------- Handlers for N / K / New / Restart ----------------
if N != len(st.session_state.points):
    st.session_state.points = generate_points(N)
    st.session_state.k = K
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=1)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

if K != st.session_state.k:
    # reinit centroids for new K, keep points
    st.session_state.k = K
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=1)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

if new_clicked:
    st.session_state.points = generate_points(N)
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=1)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

if restart_clicked:
    st.session_state.centroids = init_centroids(st.session_state.points.values, K, seed=int(time.time()) % 10000)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

# ---------------- Animation step logic ----------------
# Animation parameters
FRAMES = 20             # number of frames per 1s animation -> 1s total
FRAME_SLEEP = 1.0 / FRAMES

points_np = st.session_state.points.values
centroids = st.session_state.centroids
labels = st.session_state.labels

# build initial neutral colors for points (grey)
npts = points_np.shape[0]
initial_point_rgba = [rgba_str(GREY, 0.45)] * npts

# If first render (no labels) show neutral plot
if labels is None:
    fig0 = build_figure(st.session_state.points, centroids, initial_point_rgba,
                        title=f"phase: {st.session_state.phase.upper()} • iter: {st.session_state.iter}")
    plot_slot.plotly_chart(fig0, use_container_width=True)

# On Step click perform either assignment (color transition) or update (centroid movement)
if step_clicked:
    if st.session_state.phase == "assign":
        # compute labels
        new_labels = assign_labels(points_np, centroids)
        # prepare target colors array per point based on new_labels
        target_colors_rgb = [PALETTE_RGB[int(l) % len(PALETTE_RGB)] for l in new_labels]

        # animate color transition from grey to palette over FRAMES frames (~1 sec)
        for f in range(1, FRAMES + 1):
            t = f / FRAMES
            frame_colors = [rgba_str(lerp_rgb(GREY, target_colors_rgb[i], t), alpha=0.85) for i in range(npts)]
            fig = build_figure(st.session_state.points, centroids, frame_colors,
                               title=f"phase: ASSIGN (animating) • iter: {st.session_state.iter}")
            plot_slot.plotly_chart(fig, use_container_width=True)
            time.sleep(FRAME_SLEEP)
        # finalize assignment
        st.session_state.labels = new_labels
        st.session_state.phase = "update"

    else:
        # update step: compute new centroids, then animate movement from old to new over FRAMES frames
        if st.session_state.labels is None:
            # safety: if no labels, do an assign first
            st.session_state.labels = assign_labels(points_np, centroids)
            st.session_state.phase = "update"
        else:
            old_centroids = st.session_state.centroids.copy()
            new_centroids = update_centroids(points_np, st.session_state.labels, st.session_state.k)
            # animate over FRAMES frames (1 second)
            for f in range(1, FRAMES + 1):
                t = f / FRAMES
                interp_centroids = np.array([lerp(old_centroids[i], new_centroids[i], t) for i in range(st.session_state.k)])
                # keep point colors at assigned cluster colors during centroid move
                assigned_colors_rgb = [PALETTE_RGB[int(l) % len(PALETTE_RGB)] for l in st.session_state.labels]
                frame_colors = [rgba_str(assigned_colors_rgb[i], alpha=0.85) for i in range(npts)]
                fig = build_figure(st.session_state.points, interp_centroids, frame_colors,
                                   title=f"phase: UPDATE (animating) • iter: {st.session_state.iter}")
                plot_slot.plotly_chart(fig, use_container_width=True)
                time.sleep(FRAME_SLEEP)
            # finalize update
            st.session_state.centroids = new_centroids
            st.session_state.iter += 1
            st.session_state.phase = "assign"

# After any non-step change, re-render the current state (ensures UI updates)
if not step_clicked:
    # show current state (centroids and either neutral or assigned colors)
    if st.session_state.labels is None:
        colors = initial_point_rgba
    else:
        assigned_colors_rgb = [PALETTE_RGB[int(l) % len(PALETTE_RGB)] for l in st.session_state.labels]
        colors = [rgba_str(assigned_colors_rgb[i], alpha=0.85) for i in range(npts)]
    fig = build_figure(st.session_state.points, st.session_state.centroids, colors,
                       title=f"phase: {st.session_state.phase.upper()} • iter: {st.session_state.iter}")
    plot_slot.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("<div style='margin-top:8px' class='muted'>Click <b>Step</b> to see <i>assign</i> (colors) animate over 1s, then click again to see <i>update</i> (centroids) animate over 1s. "
            "Use <b>Restart centroids</b> to randomize centroids or <b>New dataset</b> for fresh points.</div>",
            unsafe_allow_html=True)
