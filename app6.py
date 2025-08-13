import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Instructions Text ---
instructions_text = """
### Instructions for Using the Simulation
1. **Upload your dataset (CSV)**:  
   - The file must have two columns:  
     - First column: **Independent variable (X)**  
     - Second column: **Dependent variable (Y)**
   - No headers are required in the CSV.

2. **Adjust the sliders**:  
   - Change the slope (**m**) and intercept (**c**) to see how the regression line fits the data.
   - Each change counts towards enabling the "Find Best Fit Line" button.

3. **Finding the best fit line**:  
   - After **10 slider changes**, the "Find Best Fit Line" button will be enabled.
   - Clicking it will calculate the optimal slope and intercept using the least squares method.

4. **Viewing results**:  
   - The plot will show your data points, the regression line, and the SSE (Sum of Squared Errors).
   - Once the best fit line is found, its equation will be displayed on the plot.
"""

# --- Sidebar Controls ---
st.sidebar.header("Controls")

show_instructions = st.sidebar.button("Instructions")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Session state to track slider changes
if "slider_changes" not in st.session_state:
    st.session_state.slider_changes = 0
if "last_m" not in st.session_state:
    st.session_state.last_m = 0.0
if "last_c" not in st.session_state:
    st.session_state.last_c = 0.0

# Sliders
m = st.sidebar.slider("Slope (m)", -8.0, 8.0, 0.0, 0.1)
c = st.sidebar.slider("Intercept (c)", 0.0, 12.0, 0.0, 0.1)

# Track slider changes
if m != st.session_state.last_m or c != st.session_state.last_c:
    st.session_state.slider_changes += 1
    st.session_state.last_m = m
    st.session_state.last_c = c

# Enable Find Best Fit button only after 10 changes
button_enabled = st.session_state.slider_changes >= 10
find_best_fit = st.sidebar.button(
    "Find Best Fit Line",
    disabled=not button_enabled
)

# If instructions button is clicked
if show_instructions:
    st.markdown(instructions_text)
    st.stop()

# If CSV file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values
else:
    # Default random data
    np.random.seed(42)
    x_data = np.random.uniform(0, 15, 20)
    y_data = np.random.uniform(0, 15, 20)

# If Find Best Fit Line is clicked
if find_best_fit:
    m, c = np.polyfit(x_data, y_data, 1)  # slope, intercept

# Calculate predicted values
y_pred = m * x_data + c

# Calculate SSE
sse = np.sum((y_data - y_pred) ** 2)

# Create DataFrame
df = pd.DataFrame({"X values": x_data, "Y values": y_data})

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xticks(range(0, 16))
ax.set_yticks(range(0, 16))
ax.grid(True, linestyle='--', color='lightgray')
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_aspect('equal')
ax.set_xlabel("Independent Variable")

# Scatter points
ax.scatter(x_data, y_data, color='darkred', s=30)

# Draw regression line
x_vals = np.array([0, 15])
y_vals = m * x_vals + c
ax.plot(x_vals, y_vals, color='blue', linewidth=2)

# Show error lines only if m and c are not zero
if not (m == 0 and c == 0):
    for x, y in zip(x_data, y_data):
        ax.plot([x, x], [y, m * x + c], color='orange', linestyle='--')

# Show SSE
ax.text(10.5, 14.5, f"SSE: {sse:.2f}", fontsize=12, color='black')

# Show line equation only if Find Best Fit is clicked
if find_best_fit:
    ax.text(10.5, 13.5, f"y = {m:.2f}x + {c:.2f}", fontsize=12, color='green')

# Streamlit outputs
st.title("Interactive Simple Linear Regression")
st.dataframe(df)
st.pyplot(fig)

# Display attempts left
if not button_enabled:
    st.sidebar.write(f"Make {10 - st.session_state.slider_changes} more changes to enable the button.")
