import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

st.title("Diabetes Risk Predictor (Simulated)")

# Define feature details similar to the HTML version
features = [
    {"key": "sex", "label": "Sex (0=female,1=male)", "min": 0, "max": 1, "value": 1, "step": 1},
    {"key": "bp", "label": "Resting Blood Pressure (94 - 200 mmHg)", "min": 94, "max": 200, "value": 105},
    {"key": "bmi", "label": "BMI (18 - 45)", "min": 18, "max": 45, "value": 28},
    {"key": "s1", "label": "Serum Total Cholesterol (100 - 300)", "min": 100, "max": 300, "value": 180},
    {"key": "s2", "label": "Low-Density Lipoproteins (50 - 200)", "min": 50, "max": 200, "value": 110},
    {"key": "s3", "label": "High-Density Lipoproteins (20 - 80)", "min": 20, "max": 80, "value": 40},
    {"key": "s4", "label": "Triglycerides Level (50 - 300)", "min": 50, "max": 300, "value": 140},
    {"key": "s5", "label": "Serum Glucose Level (70 - 200)", "min": 70, "max": 200, "value": 100},
    {"key": "s6", "label": "Diabetes Pedigree (0.0 - 2.5)", "min": 0.0, "max": 2.5, "value": 0.5},
    {"key": "age", "label": "Age (20 - 80)", "min": 20, "max": 80, "value": 50}
]

# Simulated weights for logistic regression
weights = {
    "sex": 2.0,
    "bp": 3.0,
    "bmi": 4.0,
    "s1": 2.0,
    "s2": 1.5,
    "s3": -1.5,
    "s4": 2.5,
    "s5": 5.0,
    "s6": 2.5,
    "age": 3.5
}

# Sidebar: CSV upload or manual input
st.sidebar.header("Input Options")
option = st.sidebar.radio("Select input method", ["Manual Input", "Upload CSV"])

if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        # Apply prediction for each row
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val)
        
        def predict(row):
            z = -5  # intercept
            for f in features:
                key = f["key"]
                norm = normalize(row[key], f["min"], f["max"])
                z += norm * weights[key]
            prob = sigmoid(z)
            return round(prob * 100, 2)
        
        df["Risk (%)"] = df.apply(predict, axis=1)
        st.write("Predictions:")
        st.dataframe(df)
else:
    st.sidebar.header("Manual Input Features")
    input_data = {}
    for f in features:
        if f["step"] == 1:
            input_data[f["key"]] = st.sidebar.slider(f["label"], int(f["min"]), int(f["max"]), int(f["value"]))
        else:
            input_data[f["key"]] = st.sidebar.slider(f["label"], float(f["min"]), float(f["max"]), float(f["value"]), step=0.1)

    # Prediction logic
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    z = -5  # intercept
    for f in features:
        key = f["key"]
        norm = normalize(input_data[key], f["min"], f["max"])
        z += norm * weights[key]
    prob = sigmoid(z)
    percent = round(prob * 100, 2)

    st.subheader("Prediction")
    st.progress(int(percent))
    st.write(f"**Risk Score:** {percent}%")

    st.subheader("Input Values")
    st.json(input_data)
