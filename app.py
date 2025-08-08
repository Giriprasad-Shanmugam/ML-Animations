import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Set page title
st.set_page_config(page_title="Interactive Linear Regression", layout="centered")

st.title("📈 Interactive Linear Regression with Error Lines")

# Generate data
np.random.seed(42)
x_data = np.random.uniform(0, 15, 20)
y_data = np.random.uniform(0, 15, 20)
x_vals = np.array([0, 15])

# Sidebar controls
st.sidebar.header("Adjust Regression Line")

# Initialize or update state
if 'm' not in st.session_state:
    st.session_state['m'] = 1.0
if 'c' not in st.session_state:
    st.session_state['c'] = 2.0

# Sliders
m = st.sidebar.slider("Slope (m)", -2.0, 4.0, st.session_state['m'], step=0.1)
c = st.sidebar.slider("Intercept (c)", 0.0, 12.0, st.session_state['c'], step=0.1)

# Best fit button
if st.sidebar.button("📌 Best Fit Line"):
    best_m, best_c = np.polyfit(x_data, y_data, 1)
    st.session_state['m'] = best_m
    st.session_state['c'] = best_c
    m = best_m
    c = best_c

# Prediction
y_pred = m * x_data + c
sse = np.sum((y_data - y_pred) ** 2)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xticks(range(0, 16))
ax.set_yticks(range(0, 16))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, linestyle='--', color='lightgray')
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_aspect('equal')

# Scatter points
ax.scatter(x_data, y_data, color='darkred', s=60)

# Regression line
y_line = m * x_vals + c
ax.plot(x_vals, y_line, color='blue', linewidth=2)

# Error lines
for xi, yi, y_hat in zip(x_data, y_data, y_pred):
    ax.plot([xi, xi], [yi, y_hat], color='orange', linestyle='--')

# SSE text
st.markdown(f"### Sum of Squared Errors (SSE): `{sse:.2f}`")

# Show plot
st.pyplot(fig)
