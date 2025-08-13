import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------- Sidebar: CSV Upload --------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 2:
        st.error("CSV must have at least two columns (X and Y).")
        st.stop()
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# -------------------- Sidebar: Sliders --------------------
if "slider_changes" not in st.session_state:
    st.session_state.slider_changes = 0
if "prev_m" not in st.session_state:
    st.session_state.prev_m = 0.0
if "prev_c" not in st.session_state:
    st.session_state.prev_c = 0.0

m = st.sidebar.slider("Slope (m)", -10.0, 10.0, 0.0, step=0.1)
c = st.sidebar.slider("Intercept (c)", -50.0, 50.0, 0.0, step=0.1)

if m != st.session_state.prev_m or c != st.session_state.prev_c:
    st.session_state.slider_changes += 1
    st.session_state.prev_m = m
    st.session_state.prev_c = c

# -------------------- Compute Regression --------------------
y_pred = m * x_data + c
sse = np.sum((y_data - y_pred) ** 2)

# -------------------- Matplotlib Plot --------------------
fig, ax = plt.subplots()
ax.scatter(x_data, y_data, color='blue', label='Data Points')
ax.plot(x_data, y_pred, color='red', label=f"y = {m:.2f}x + {c:.2f}")
ax.set_xlabel("Independent Variable")
ax.set_ylabel("Dependent Variable")

# Show error lines only if m or c != 0
if not (m == 0.0 and c == 0.0):
    for xi, yi, ypi in zip(x_data, y_data, y_pred):
        ax.plot([xi, xi], [yi, ypi], color='orange', linestyle='--', linewidth=1)

ax.legend()

# -------------------- Best Fit Button --------------------
if st.session_state.slider_changes >= 10:
    if st.sidebar.button("Find Best Fit Line"):
        m_best, c_best = np.polyfit(x_data, y_data, 1)
        m = m_best
        c = c_best
        y_pred = m * x_data + c
        sse = np.sum((y_data - y_pred) ** 2)
        ax.clear()
        ax.scatter(x_data, y_data, color='blue', label='Data Points')
        ax.plot(x_data, y_pred, color='red', label=f"Best Fit: y = {m:.2f}x + {c:.2f}")
        ax.set_xlabel("Independent Variable")
        ax.set_ylabel("Dependent Variable")
        for xi, yi, ypi in zip(x_data, y_data, y_pred):
            ax.plot([xi, xi], [yi, ypi], color='orange', linestyle='--', linewidth=1)
        ax.legend()

# -------------------- Layout: Left (Data+Plot) | Right (Instructions) --------------------
st.title("Interactive Simple Linear Regression")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dataset")
    st.dataframe(df)
    st.subheader("Regression Plot")
    st.pyplot(fig)
    st.markdown(f"**SSE (Sum of Squared Errors):** {sse:.2f}")

with col2:
    st.subheader("📌 Instructions")
    st.markdown("""
    **1. Prepare Your CSV File**
    - CSV must have two columns:  
      1. Independent variable (X)  
      2. Dependent variable (Y)  

    **2. Upload Your CSV**
    - Use the **Upload CSV file** option in the **left sidebar**.

    **3. Adjust the Regression Line**
    - Use sliders for slope (m) and intercept (c).  
    - Start values are `0.0` (flat line, no error lines).  

    **4. SSE (Sum of Squared Errors)**
    - Smaller SSE → better fit.  

    **5. Find Best Fit Line**
    - Make **10 slider changes** to enable the button.  
    - Click to show least-squares best fit.  

    **6. Best Fit Equation**
    - Always shows the optimal line: `y = mx + c`.  

    **7. Error Lines**
    - Orange dashed lines = vertical distance from data to regression line.  
    - Hidden initially.  

    **8. Axes**
    - X-axis = Independent Variable  
    - Y-axis = Dependent Variable  
    """)

