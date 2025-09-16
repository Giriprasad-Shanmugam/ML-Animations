import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="SVM Simulator", layout="wide")

st.title("SVM Simulator with Kernel Options")

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if data.shape[1] < 3:
        st.error("Dataset must have at least 2 features and 1 label column.")
        st.stop()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
else:
    st.info("Upload a CSV file with features and labels.")
    st.stop()

# Tabs for kernel selection
kernel = st.sidebar.selectbox("Select Kernel", ["linear", "poly", "sigmoid"])

# Sliders for C and learning rate
C = st.sidebar.slider("Soft Margin (C)", 0.01, 10.0, 1.0, 0.01)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, 0.001)

# Button to reset
if st.sidebar.button("Reset"):
    st.experimental_rerun()

# Train SVM
model = SVC(kernel=kernel, C=C, gamma='scale', coef0=1)
model.fit(X, y)

# Plotting
def plot_decision_regions(X, y, model, kernel):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"SVM with {kernel.capitalize()} Kernel")
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

plot_decision_regions(X, y, model, kernel)
