# app.py
import streamlit as st
import numpy as np
import pandas as pd

# Force a non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")   # safe for server environments
import matplotlib.pyplot as plt

st.title("Scatter + Table (Matplotlib in Streamlit)")

# Sample data (your real data here)
np.random.seed(42)
x_data = np.random.uniform(0, 15, 20)
y_data = np.random.uniform(0, 15, 20)

# Display table first
df = pd.DataFrame({"X": x_data, "Y": y_data})
st.dataframe(df.round(2))

# Create Matplotlib figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xticks(range(0, 16))
ax.set_yticks(range(0, 16))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, linestyle="--", color="lightgray")
ax.axhline(0, color="black")
ax.axvline(0, color="black")
ax.set_aspect("equal")
ax.scatter(x_data, y_data, color="darkred", s=50)

# Render figure in Streamlit
st.pyplot(fig)
