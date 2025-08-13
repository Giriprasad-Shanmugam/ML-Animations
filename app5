import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Sidebar: File uploader ----------
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 2:
        st.error("CSV must have at least two columns (X and Y values).")
        st.stop()
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ---------- Session state ----------
if "slider_changes" not in st.session_state:
    st.session_state.slider_changes = 0
if "last_m" not in st.session_state:
    st.session_state.last_m = 0.0
if "last_c" not in st.session_state:
    st.session_state.last_c = 0.0

# ---------- Sliders ----------
st.sidebar.header("Regression Line Controls")
m = st.sidebar.slider("Slope (m)", -8.0, 8.0, st.session_state.last_m, 0.1)
c = st.sidebar.slider("Intercept (c)", 0.0, 12.0, st.session_state.last_c, 0.1)

# Track slider changes
if m != st.session_state.last_m or c != st.session_state.last_c:
    st.session_state.slider_changes += 1
    st.session_state.last_m = m
    st.session_state.last_c = c

# ---------- Best fit button ----------
button_enabled = st.session_state.slider_changes >= 10
find_best_fit = st.sidebar.button(
    "Find Best Fit Line",
    disabled=not button_enabled
)

# If button clicked → compute best fit line
if find_best_fit:
    m, c = np.polyfit(x_data, y_data, 1)

# ---------- SSE & Best fit ----------
y_pred = m * x_data + c
sse = np.sum((y_data - y_pred) ** 2)
m_best, c_best = np.polyfit(x_data, y_data, 1)

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
ax.set_ylim(min(y_data) - 1, max(y_data) + 1)
ax.grid(True, linestyle='--', color='lightgray')
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_aspect('equal')

# Axis titles
ax.set_xlabel("Independent Variable")
ax.set_ylabel("Dependent Variable")

# Scatter points
ax.scatter(x_data, y_data, color='darkred', s=30)

# Regression line
x_vals = np.array([min(x_data), max(x_data)])
y_vals = m * x_vals + c
ax.plot(x_vals, y_vals, color='blue', linewidth=2)

# Error lines (only if m and c are not both zero)
if not (m == 0.0 and c == 0.0):
    for x, y in zip(x_data, y_data):
        ax.plot([x, x], [y, m * x + c], color='orange', linestyle='--')

# SSE & Best fit text
ax.text(max(x_data) - 5, max(y_data), f"SSE: {sse:.2f}", fontsize=12, color='black')
ax.text(max(x_data) - 5, max(y_data) - 1, f"Best Fit: y = {m_best:.2f}x + {c_best:.2f}", fontsize=12, color='black')

# ---------- Display outputs ----------
st.title("Interactive Simple Linear Regression")
st.dataframe(df)
st.pyplot(fig)

# ---------- Right Pane Instructions ----------
st.markdown("""
## 📌 Instructions for Using the Simulation

### 1. Prepare Your CSV File
- Create a CSV file with **two columns**:
  1. **First column:** Independent variable (X values)
  2. **Second column:** Dependent variable (Y values)
- Do not include headers unless you handle them in preprocessing (this code works with or without headers).

### 2. Upload Your CSV
 - In the left sidebar, click "Browse files" in the Upload CSV file section.
 - Select your prepared CSV file.
 - Once uploaded, the scatter plot will be displayed.


### 2. Upload Your CSV
- Use the **Upload CSV file** option in the **left sidebar**.
- Select your CSV file.
- The scatter plot will appear.

### 3. Adjust the Regression Line
- Use the **Slope (m)** slider to change line steepness.
- Use the **Intercept (c)** slider to move the line up/down.
- Initially, both sliders are `0.0` (flat line, no error lines).

### 4. SSE (Sum of Squared Errors)
- Shows how far predicted points are from actual data points.
- **Lower SSE → better fit**.
- Updates live when sliders move.

### 5. Find Best Fit Line
- Make **10 slider changes** to enable the button.
- Click to calculate **best slope & intercept** using least squares.

### 6. Best Fit Equation
- Always displays the mathematically optimal line: `y = mx + c`.

### 7. Error Lines
- Orange dashed lines show the difference between actual and predicted values.
- Hidden initially when m = 0 and c = 0.

### 8. Axes
- **X-axis:** Independent Variable
- **Y-axis:** Dependent Variable
""")

# Display attempts left message
if not button_enabled:
    st.sidebar.write(f"Make {10 - st.session_state.slider_changes} more changes to enable the button.")

