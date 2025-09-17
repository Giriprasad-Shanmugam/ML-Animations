import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Instructions text ---
instructions_text = """
### Instructions for Using the Simulation
1. Upload a CSV file with two columns (no headers):
   - First column: Independent variable (X)
   - Second column: Dependent variable (Y)
2. Adjust the slope and intercept sliders to see how the line fits the data.
3. After at least 10 adjustments, you can click 'Find Best Fit Line' to automatically fit the data.
4. SSE (Sum of Squared Errors) is shown to help you understand the fit quality.
"""

# --- Sidebar controls ---
st.sidebar.header("Controls")
show_instructions = st.sidebar.button("Instructions")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# --- Show instructions ---
if show_instructions:
    st.markdown(instructions_text)

# --- Initialize session state ---
if "slider_changes" not in st.session_state:
    st.session_state.slider_changes = 0
if "last_m" not in st.session_state:
    st.session_state.last_m = 0.0
if "last_c" not in st.session_state:
    st.session_state.last_c = 0.0
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# --- Load data ---
def load_data():
    try:
        data = pd.read_csv(uploaded_file, header=None)
        x_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values
        return x_data, y_data
    except Exception as e:
        st.error("Error reading the CSV file. Please ensure it has two numeric columns without headers.")
        return None, None

if uploaded_file is not None:
    x_data, y_data = load_data()
    if x_data is not None and y_data is not None:
        st.session_state.data_loaded = True
else:
    # Default random data
    np.random.seed(42)
    x_data = np.random.uniform(0, 15, 20)
    y_data = np.random.uniform(0, 15, 20)
    st.session_state.data_loaded = True

# --- Sliders for m and c ---
if st.session_state.data_loaded:
    st.sidebar.header("Regression Line Controls")
    m = st.sidebar.slider("Slope (m)", -8.0, 8.0, 0.0, 0.1)
    c = st.sidebar.slider("Intercept (c)", 0.0, 12.0, 0.0, 0.1)

    # Track slider changes
    if m != st.session_state.last_m or c != st.session_state.last_c:
        st.session_state.slider_changes += 1
        st.session_state.last_m = m
        st.session_state.last_c = c

    # Enable button only after 10 changes
    button_enabled = st.session_state.slider_changes >= 10
    find_best_fit = st.sidebar.button(
        "Find Best Fit Line",
        disabled=not button_enabled
    )

    # Apply best fit if button clicked
    if find_best_fit:
        m_fit, c_fit = np.polyfit(x_data, y_data, 1)
        m = float(m_fit)
        c = float(c_fit)
        st.session_state.last_m = m
        st.session_state.last_c = c

    # Compute predictions and SSE
    y_pred = m * x_data + c
    sse = np.sum((y_data - y_pred) ** 2)

    # Create DataFrame for display
    df = pd.DataFrame({"X values": x_data, "Y values": y_data})

    # --- Plot ---
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
    ax.set_ylabel("Dependent Variable")

    # Scatter points
    ax.scatter(x_data, y_data, color='darkred', s=30)

    # Regression line
    x_vals = np.array([0, 15])
    y_vals = m * x_vals + c
    ax.plot(x_vals, y_vals, color='blue', linewidth=2)

    # Error lines if m and c are not zero
    if not (m == 0 and c == 0):
        for x, y in zip(x_data, y_data):
            ax.plot([x, x], [y, m * x + c], color='orange', linestyle='--')

    # Show SSE
    ax.text(10.5, 14.5, f"SSE: {sse:.2f}", fontsize=12, color='black')

    # Show equation if button clicked
    if find_best_fit:
        ax.text(10.5, 13.5, f"y = {m:.2f}x + {c:.2f}", fontsize=12, color='green')

    # --- Streamlit outputs ---
    st.title("Simple Linear Regression")
    st.dataframe(df)
    st.pyplot(fig)

    # Attempts left
    if not button_enabled:
        st.sidebar.write(f"Make {10 - st.session_state.slider_changes} more changes to enable the button.")
