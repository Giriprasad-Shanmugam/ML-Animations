# ---------------- Segment 1/3 ----------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from typing import Tuple

# Page config + minimal global styling for a professional look
st.set_page_config(page_title="K-Means Interactive — Step-by-Step", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f8fafc 0%, #ffffff 60%); }
    .header {font-family: 'Segoe UI', Roboto, Helvetica, Arial; margin: 0; padding: 0.2rem 0;}
    .card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 6px 18px rgba(38,50,56,0.06); }
    .muted { color: #6b7280; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers: data + KMeans step functions ----------
def generate_points(n: int, seed: int) -> pd.DataFrame:
    """
    Generate n 2D points like the original HTML demo: uniform distribution in [0,1]^2.
    Returns a DataFrame with columns ['x','y'].
    """
    rs = np.random.RandomState(seed)
    pts = rs.rand(n, 2)
    df = pd.DataFrame(pts, columns=["x", "y"])
    return df

def init_centroids(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Initialize centroids by sampling k points from the data (same style as the HTML).
    """
    rs = np.random.RandomState(seed)
    n = points.shape[0]
    if k >= n:
        # if k >= n, just take unique points (fallback)
        indices = np.arange(n)
    else:
        indices = rs.choice(n, size=k, replace=False)
    return points[indices].astype(float)

def assign_labels(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assignment step: assign each point to the nearest centroid.
    """
    # distances shape: (n_points, k)
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return labels

def update_centroids(points: np.ndarray, labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Update step: recompute centroids as mean of assigned points.
    If a cluster has no points, reinitialize that centroid by picking a random point.
    """
    rs = np.random.RandomState(seed)
    new_centroids = np.zeros((k, 2), dtype=float)
    for i in range(k):
        members = points[labels == i]
        if len(members) == 0:
            # reinitialize to a random point from data
            new_centroids[i] = points[rs.randint(0, points.shape[0])]
        else:
            new_centroids[i] = members.mean(axis=0)
    return new_centroids

def compute_inertia(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Sum of squared distances (inertia) for diagnostics.
    """
    assigned_centroids = centroids[labels]
    ssd = np.sum((points - assigned_centroids) ** 2)
    return float(ssd)

def build_figure(points_df: pd.DataFrame,
                 centroids: np.ndarray,
                 labels: np.ndarray,
                 phase: str,
                 iteration: int,
                 show_lines: bool = False) -> go.Figure:
    """
    Build an aesthetic Plotly figure showing points and centroids.
    - `phase` is used purely for title/annotation (e.g., "assignment" / "update").
    - `show_lines`: optionally draw lines from points to their centroids (only for small N).
    """
    colors = px.colors.qualitative.Plotly  # pleasant color cycle
    n = points_df.shape[0]
    k = centroids.shape[0] if centroids is not None else 0

    fig = go.Figure()
    # Points: group by label for color
    if labels is None:
        # unassigned: grey
        fig.add_trace(go.Scatter(
            x=points_df["x"], y=points_df["y"],
            mode="markers",
            marker=dict(size=8, color="lightgray", line=dict(width=0.5, color="#333")),
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            name="points"
        ))
    else:
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            lab_mask = labels == lab
            color = colors[int(lab) % len(colors)]
            fig.add_trace(go.Scatter(
                x=points_df["x"].values[lab_mask],
                y=points_df["y"].values[lab_mask],
                mode="markers",
                marker=dict(size=8, color=color, line=dict(width=0.5, color="#222")),
                hovertemplate="cluster: {}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>".format(int(lab)),
                name=f"cluster {int(lab)}",
                showlegend=True
            ))

    # Optionally draw assignment lines (only if N small to avoid clutter)
    if show_lines and labels is not None and n <= 300:
        pts = points_df.values
        for idx, lab in enumerate(labels):
            cx, cy = centroids[lab]
            fig.add_trace(go.Scatter(
                x=[pts[idx, 0], cx], y=[pts[idx, 1], cy],
                mode="lines",
                line=dict(width=0.6, color="rgba(100,100,100,0.15)"),
                hoverinfo="none",
                showlegend=False
            ))

    # Centroids: large symbols
    if centroids is not None:
        for i, (cx, cy) in enumerate(centroids):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(size=20, symbol="x", color=color, line=dict(width=2, color="#111")),
                name=f"centroid {i}",
                hovertemplate="centroid: {}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>".format(i),
            ))

    fig.update_layout(
        title=f"K-Means — Iter: {iteration} · phase: {phase}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        width=860,
        height=640,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        legend=dict(itemsizing="constant")
    )
    return fig
# ---------------- end of Segment 1/3 ----------------
# ---------------- Segment 2/3 ----------------
# Sidebar: controls
st.sidebar.header("K-Means controls")
N = st.sidebar.slider("Number of points (N)", min_value=10, max_value=1000, value=200, step=10)
K = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=15, value=4, step=1)
seed = st.sidebar.number_input("Random seed", value=42, step=1, format="%d")
max_iter = st.sidebar.slider("Max iterations", min_value=1, max_value=200, value=50, step=1)
tol = st.sidebar.number_input("Convergence tol (max centroid shift)", value=1e-4, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions**")
st.sidebar.markdown("""
- Click **New dataset** to generate points (random uniform as in your HTML demo).  
- Click **Step** to advance one step (assignment → update → assignment → ...).  
- Click **Restart centroids** to randomize centroids while keeping the same points.  
- The simulation stops automatically when centroids converge or max iterations reached.
""")

# Initialize session state (persistent between reruns)
if "points_df" not in st.session_state:
    st.session_state["points_df"] = generate_points(N, seed)
    st.session_state["centroids"] = init_centroids(st.session_state["points_df"].values, K, seed + 1)
    st.session_state["labels"] = None
    st.session_state["iteration"] = 0
    st.session_state["phase"] = "assign"  # next Step will perform assignment
    st.session_state["history"] = []
    st.session_state["converged"] = False
    st.session_state["prev_N"] = N
    st.session_state["prev_K"] = K
    st.session_state["prev_seed"] = seed
    st.session_state["max_iter"] = max_iter
    st.session_state["tol"] = tol

# If N or seed changed -> new dataset automatically
if (st.session_state.get("prev_N") != N) or (st.session_state.get("prev_seed") != seed):
    st.session_state["points_df"] = generate_points(N, seed)
    st.session_state["centroids"] = init_centroids(st.session_state["points_df"].values, K, seed + 1)
    st.session_state["labels"] = None
    st.session_state["iteration"] = 0
    st.session_state["phase"] = "assign"
    st.session_state["history"] = []
    st.session_state["converged"] = False
    st.session_state["prev_N"] = N
    st.session_state["prev_seed"] = seed

# If K changed -> reinitialize centroids (keep points)
if st.session_state.get("prev_K") != K:
    st.session_state["centroids"] = init_centroids(st.session_state["points_df"].values, K, seed + 1)
    st.session_state["labels"] = None
    st.session_state["iteration"] = 0
    st.session_state["phase"] = "assign"
    st.session_state["history"] = []
    st.session_state["converged"] = False
    st.session_state["prev_K"] = K

# If max_iter or tol changes store them
st.session_state["max_iter"] = max_iter
st.session_state["tol"] = tol

# Top: title + short description
st.markdown("<div class='card header'><h1 style='margin-bottom:4px;'>K-Means — Step by Step</h1>"
            "<div class='muted'>A clean interactive version of your D3 demo — 2D points, step execution.</div></div>",
            unsafe_allow_html=True)

# Layout: left control column, center plot, right stats
left_col, center_col, right_col = st.columns([1, 2, 1])
with left_col:
    st.markdown("### Controls")
    # Buttons
    new_clicked = st.button("🔁 New dataset", key="new")
    restart_clicked = st.button("🧭 Restart centroids", key="restart")
    step_clicked = st.button("▶ Step", key="step")

    # quick toggles
    show_assignment_lines = st.checkbox("Show assignment lines (small N)", value=False)
    show_inertia = st.checkbox("Show inertia & counts", value=True)

with right_col:
    st.markdown("### Status")
    st.write(f"Iteration: **{st.session_state['iteration']}**")
    st.write(f"Phase: **{st.session_state['phase']}**")
    st.write(f"Converged: **{st.session_state['converged']}**")
    st.write(f"Max iter: **{st.session_state['max_iter']}**")
    st.write(f"tol: **{st.session_state['tol']}**")
# ---------------- end of Segment 2/3 ----------------
# ---------------- Segment 3/3 ----------------
# Helper for performing one step (assignment/update) and updating session_state
def perform_step():
    pts_df = st.session_state["points_df"]
    pts = pts_df.values
    centroids = st.session_state["centroids"]
    k = K
    seed_local = seed + 100 + st.session_state["iteration"]  # for deterministic reinit if needed

    if st.session_state["converged"]:
        return

    if st.session_state["phase"] == "assign":
        # Assignment step: compute labels (visual changes only)
        labels = assign_labels(pts, centroids)
        st.session_state["labels"] = labels
        st.session_state["phase"] = "update"
    elif st.session_state["phase"] == "update":
        # Update step: compute new centroids
        labels = st.session_state.get("labels")
        if labels is None:
            labels = assign_labels(pts, centroids)
            st.session_state["labels"] = labels

        prev_centroids = centroids.copy()
        new_centroids = update_centroids(pts, labels, k, seed_local)
        st.session_state["centroids"] = new_centroids
        st.session_state["iteration"] += 1
        st.session_state["phase"] = "assign"  # next step will reassign with updated centroids

        # check convergence
        max_shift = np.max(np.linalg.norm(new_centroids - prev_centroids, axis=1))
        st.session_state["history"].append({
            "iteration": st.session_state["iteration"],
            "max_shift": float(max_shift),
            "inertia": compute_inertia(pts, labels, new_centroids)
        })
        if (max_shift <= st.session_state["tol"]) or (st.session_state["iteration"] >= st.session_state["max_iter"]):
            st.session_state["converged"] = True
            st.session_state["phase"] = "converged"

# Button handlers:
if new_clicked:
    # create brand new dataset (uniform points, like HTML)
    st.session_state["points_df"] = generate_points(N, seed)
    st.session_state["centroids"] = init_centroids(st.session_state["points_df"].values, K, seed + 1)
    st.session_state["labels"] = None
    st.session_state["iteration"] = 0
    st.session_state["phase"] = "assign"
    st.session_state["history"] = []
    st.session_state["converged"] = False

if restart_clicked and (st.session_state.get("points_df") is not None):
    # reinit centroids but keep points
    st.session_state["centroids"] = init_centroids(st.session_state["points_df"].values, K, seed + 1)
    st.session_state["labels"] = None
    st.session_state["iteration"] = 0
    st.session_state["phase"] = "assign"
    st.session_state["history"] = []
    st.session_state["converged"] = False

if step_clicked:
    perform_step()

# Build visualization
pts_df = st.session_state["points_df"]
centroids = st.session_state["centroids"]
labels = st.session_state["labels"]

fig = build_figure(pts_df, centroids, labels, st.session_state["phase"], st.session_state["iteration"],
                   show_lines=show_assignment_lines)

with center_col:
    st.plotly_chart(fig, use_container_width=True)

# Right column: stats & small diagnostics
with right_col:
    st.markdown("### Diagnostics")
    if st.session_state.get("labels") is not None:
        labels_arr = st.session_state["labels"]
        counts = pd.Series(labels_arr).value_counts().sort_index()
        counts_df = counts.rename_axis("cluster").reset_index(name="count")
        st.table(counts_df)

        if show_inertia:
            inertia_val = compute_inertia(pts_df.values, labels_arr, centroids)
            st.write(f"**Inertia (SSD):** {inertia_val:.4f}")

    else:
        st.write("No assignment yet — click **Step** to assign points to initial centroids.")

    st.markdown("---")
    st.markdown("### History (recent)")
    history = st.session_state.get("history", [])
    if history:
        hist_df = pd.DataFrame(history).sort_values("iteration", ascending=False).head(6)
        st.dataframe(hist_df.reset_index(drop=True))
    else:
        st.write("No updates recorded yet.")

# Footer / small help
st.markdown("<div class='muted' style='margin-top:10px;'>Tip: press Step repeatedly to see assignment → centroid update cycles. "
            "Restart centroids will re-randomize centroids while keeping your points.</div>",
            unsafe_allow_html=True)
# ---------------- end of Segment 3/3 ----------------
