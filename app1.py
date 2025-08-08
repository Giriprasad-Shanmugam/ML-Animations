# app_plotly.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("Scatter + Table (Plotly)")

np.random.seed(42)
x_data = np.random.uniform(0, 15, 20)
y_data = np.random.uniform(0, 15, 20)
df = pd.DataFrame({"X": x_data, "Y": y_data})
st.dataframe(df.round(2))

fig = px.scatter(df, x="X", y="Y")
fig.update_layout(xaxis=dict(range=[0,15]), yaxis=dict(range=[0,15]), width=600, height=600)
st.plotly_chart(fig, use_container_width=False)
