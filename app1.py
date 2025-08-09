import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
x_data = np.random.uniform(0, 15, 20)
y_data = np.random.uniform(0, 15, 20)

# Sliders for slope (m) and intercept (c)
st.sidebar.header("Regression Line Controls")
m = st.sidebar.slider("Slope (m)", -8.0, 8.0, 1.0, 0.1)
c = st.sidebar.slider("Intercept (c)", 0.0, 12.0, 2.0, 0.1)

# Calculate regression line
y_pred = m * x_data + c

# Calculate Sum of Squared Errors (SSE)
sse = np.sum((y_data - y_pred) ** 2)

# Create DataFrame
df = pd.DataFrame({"X values": x_data, "Y values": y_data})

# Plot
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
ax.scatter(x_data, y_data, color='darkred', s=30)

# Regression line
x_vals = np.array([0, 15])
y_vals = m * x_vals + c
ax.plot(x_vals, y_vals, color='blue', linewidth=2)

# Error lines
for x, y in zip(x_data, y_data):
    ax.plot([x, x], [y, m * x + c], color='orange', linestyle='--')

# Show SSE on plot
ax.text(0.5, 14.5, f"SSE: {sse:.2f}", fontsize=12, color='black')

# Streamlit outputs
st.title("Interactive Simple Linear Regression")
st.dataframe(df)
st.pyplot(fig)
