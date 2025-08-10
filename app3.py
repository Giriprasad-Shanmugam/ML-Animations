import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Generate sample dataset ---
np.random.seed(42)
X = np.linspace(0, 10, 20)
y_true = 3 * X + 5 + np.random.randn(20) * 2  # slope=3, intercept=5

# --- Title ---
st.title("Interactive Linear Regression")

# --- Sliders for m and c ---
if "m_val" not in st.session_state:
    st.session_state.m_val = 1.0
if "c_val" not in st.session_state:
    st.session_state.c_val = 0.0

m = st.slider("Slope (m)", -10.0, 10.0, st.session_state.m_val, 0.1)
c = st.slider("Intercept (c)", -10.0, 10.0, st.session_state.c_val, 0.1)

# --- Button to compute best fit ---
if st.button("Best Fit Line"):
    # Closed-form solution for simple linear regression
    X_mean = np.mean(X)
    y_mean = np.mean(y_true)
    m_best = np.sum((X - X_mean) * (y_true - y_mean)) / np.sum((X - X_mean) ** 2)
    c_best = y_mean - m_best * X_mean

    st.session_state.m_val = m_best
    st.session_state.c_val = c_best

    # Force sliders to update
    m = m_best
    c = c_best

# --- Compute predictions ---
y_pred = m * X + c
SSE = np.sum((y_true - y_pred) ** 2)

# --- Plot ---
fig, ax = plt.subplots()
ax.scatter(X, y_true, color="blue", label="Data Points")
ax.plot(X, y_pred, color="red", label=f"y = {m:.2f}x + {c:.2f}")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Linear Regression Fit")
ax.legend()

st.pyplot(fig)

# --- Show SSE ---
st.write(f"**Sum of Squared Errors (SSE):** {SSE:.2f}")
