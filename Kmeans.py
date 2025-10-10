# kmeans_streamlit_clean.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------- Page config & styling ----------------
st.set_page_config(page_title="K-Means — Interactive (Dark)", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    /* page background gradient */
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #071023 45%, #06131b 100%);
        color: #e6eef8;
        font-family: Inter, "Segoe UI", Roboto, system-ui, -apple-system, "Helvetica Neue", Arial;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 6px 24px rgba(2,6,23,0.6);
        color: #e6eef8;
    }
    .controls { display:flex; gap:10px; align-items:center; }
    .big-btn { padding:8px 14px; border-radius:10px; border:none; cursor:pointer; }
    .btn-accent { background: linear-gradient(90deg,#7c3aed,#06b6d4); color:white; }
    .btn-ghost { background:transparent; border:1px solid rgba(255,255,255,0.08); color:#cfe7ff; }
    .muted { color: #9fb2d7; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Minimal helpers ----------------
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

# Plot builder with transition enabled (Plotly will animate between renders)
def build_figure(df: pd.DataFrame, centroids: np.ndarray, labels: np.ndarray | None, title: str) -> go.Figure:
    palette = px.colors.qualitative.Vivid
    fig = go.Figure()

    # Points
    if labels is None:
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(size=8, color="rgba(200,200,200,0.45)", line=dict(width=0.6, color="rgba(0,0,0,0.6)")),
            hoverinfo="skip",
            name="points"
        ))
    else:
        k = centroids.shape[0]
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=df["x"].values[mask], y=df["y"].values[mask],
                mode="markers",
                marker=dict(size=8, color=palette[i % len(palette)], line=dict(width=0.6, color="rgba(0,0,0,0.5)")),
                hoverinfo="skip",
                name=f"cluster {i}"
            ))

    # Centroids (large X marks)
    if centroids is not None:
        for i, (cx, cy) in enumerate(centroids):
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(size=20, symbol="x", color=palette[i % len(palette)], line=dict(width=2, color="white")),
                hoverinfo="skip",
                showlegend=False,
                name=f"centroid {i}"
            ))

    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,20,0.6)",
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.05], scaleanchor="x"),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=title, font=dict(color="white", size=18)),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    return fig

# ---------------- Session state minimal initialization ----------------
if "points" not in st.session_state:
    st.session_state.points = generate_points(100)          # default N=100
    st.session_state.k = 4
    st.session_state.centroids = init_centroids(st.session_state.points.values, st.session_state.k)
    st.session_state.labels = None
    st.session_state.phase = "assign"   # next Step will assign
    st.session_state.iter = 0

# ---------------- Top UI ----------------
st.markdown("<div class='card'><h2 style='margin:0 0 6px 0'>K-Means — Step-by-Step</h2>"
            "<div class='muted'>Manual steps • 2D points • modern dark gradient</div></div>",
            unsafe_allow_html=True)

# Controls row
cols = st.columns([1, 2, 1])
with cols[0]:
    # Minimal sliders + buttons
    N = st.slider("N (points)", min_value=20, max_value=800, value=100, step=10)
    K = st.slider("K (clusters)", min_value=2, max_value=12, value=4, step=1)

with cols[1]:
    st.write("")  # spacer
    # Buttons laid out horizontally
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        new_clicked = st.button("New dataset", key="new", help="Generate a fresh random scatter (uniform).")
    with c2:
        restart_clicked = st.button("Restart centroids", key="restart", help="Keep points; randomize centroids.")
    with c3:
        step_clicked = st.button("Step ▶", key="step", help="Advance a single step (assign → update → ...).")

with cols[2]:
    st.markdown(f"<div class='card' style='text-align:center'><div style='font-weight:600'>{st.session_state['phase'].upper()}</div>"
                f"<div class='muted' style='margin-top:6px'>Iteration: {st.session_state['iter']}</div></div>",
                unsafe_allow_html=True)

# ---------------- Handlers (very small & deterministic) ----------------
# If N or K changed by slider, regenerate/reinit
if N != len(st.session_state.points):
    st.session_state.points = generate_points(N)
    st.session_state.k = K
    st.session_state.centroids = init_centroids(st.session_state.points.values, K)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

if K != st.session_state.k:
    # reinitialize centroids for new K, keep points
    st.session_state.k = K
    st.session_state.centroids = init_centroids(st.session_state.points.values, K)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

if new_clicked:
    st.session_state.points = generate_points(N)
    st.session_state.centroids = init_centroids(st.session_state.points.values, K)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

if restart_clicked:
    st.session_state.centroids = init_centroids(st.session_state.points.values, K)
    st.session_state.labels = None
    st.session_state.phase = "assign"
    st.session_state.iter = 0

# Step logic: one user click performs one phase (assign OR update)
if step_clicked:
    pts_np = st.session_state.points.values
    if st.session_state.phase == "assign":
        st.session_state.labels = assign_labels(pts_np, st.session_state.centroids)
        st.session_state.phase = "update"
    else:
        # update centroids and increment iteration
        labels = st.session_state.labels
        if labels is None:
            st.session_state.labels = assign_labels(pts_np, st.session_state.centroids)
            st.session_state.phase = "update"
        else:
            prev_cent = st.session_state.centroids.copy()
            st.session_state.centroids = update_centroids(pts_np, labels, st.session_state.k)
            st.session_state.iter += 1
            st.session_state.phase = "assign"

# ---------------- Visualization (animated via Plotly transitions) ----------------
title = f"K-Means — phase: {st.session_state['phase'].upper()} • iter: {st.session_state['iter']}"
fig = build_figure(st.session_state.points, st.session_state.centroids, st.session_state.labels, title)

# Render centered
st.plotly_chart(fig, use_container_width=True)

# small footer tip
st.markdown("<div style='margin-top:6px' class='muted'>Click 'Step' repeatedly to see assignment → centroid update cycles. "
            "Use 'Restart centroids' to re-randomize just the centroids or 'New dataset' to get fresh points.</div>",
            unsafe_allow_html=True)
