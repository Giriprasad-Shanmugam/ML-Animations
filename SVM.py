import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import time

st.set_page_config(page_title="Enhanced SVM Simulator", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔍 Enhanced SVM Simulation")

# Create tabs
tab1, tab2 = st.tabs(["📘 Instructions", "📊 SVM Simulator"])

with tab1:
    st.header("How to Use This Simulator")
    st.markdown("""
    **Welcome! Explore how SVM works with this interactive tool.**
    
    **Steps:**
    1. Upload a CSV dataset with at least two features and one label column.
    2. Select the kernel: Linear, Polynomial, or Sigmoid.
    3. Adjust parameters:
       - **C (Soft Margin)** controls error tolerance.
       - **Degree (for Polynomial)** controls curve complexity.
    4. Click "Reset" to clear data and restart the app.
    5. View decision boundaries and margins with animations.
    
    **Note:** Polynomial kernel is activated with adjustable degree for exploring non-linear separation.
    """)

with tab2:
    st.header("SVM Simulator")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None

    # Dataset upload
    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if data.shape[1] < 3:
            st.error("Dataset must have at least 2 feature columns and 1 label column.")
            st.stop()
        st.session_state.data = data

    if st.session_state.data is None:
        st.info("Please upload a CSV file to begin.")
        st.stop()

    data = st.session_state.data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Sidebar parameters
    st.sidebar.header("Model Configuration")
    kernel = st.sidebar.selectbox("Kernel Type", ["linear", "poly", "sigmoid"])
    C = st.sidebar.slider("Soft Margin (C)", 0.01, 10.0, 1.0, 0.01)
    degree = 3
    if kernel == "poly":
        degree = st.sidebar.slider("Polynomial Degree", 2, 5, 3)

    # Reset functionality
    if st.sidebar.button("🔄 Reset App"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

    # Display animation loading
    with st.spinner('Training the model...'):
        time.sleep(1)

    # Train model
    model = SVC(kernel=kernel, C=C, degree=degree, gamma='scale', coef0=1)
    model.fit(X, y)

    # Plotting
    def plot_decision_regions(X, y, model, kernel):
        cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

        if kernel == 'linear':
            w = model.coef_[0]
            b = model.intercept_[0]
            x_plot = np.linspace(x_min, x_max, 500)
            y_decision = -(w[0] * x_plot + b) / w[1]
            margin = 1 / np.linalg.norm(w)
            y_margin_pos = y_decision + margin
            y_margin_neg = y_decision - margin
            plt.plot(x_plot, y_decision, 'k-', label="Decision Boundary")
            plt.plot(x_plot, y_margin_pos, 'k--', label="Margin +1")
            plt.plot(x_plot, y_margin_neg, 'k--', label="Margin -1")

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50, alpha=0.8)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"SVM with {kernel.capitalize()} Kernel")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.close()

    plot_decision_regions(X, y, model, kernel)
