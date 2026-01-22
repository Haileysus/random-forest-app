import streamlit as st
import pandas as pd
import joblib
import os
import requests

# ----------------------------
# Model download settings
# ----------------------------
pkl_path = "rf_defect_pipeline.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1z3dYkabmBgpWO8ip6wNJYPhw-h_6zMPf"

# Download the model if it does not exist
if not os.path.exists(pkl_path):
    st.info("Downloading model, please wait...")
    r = requests.get(MODEL_URL)
    with open(pkl_path, "wb") as f:
        f.write(r.content)
    st.success("Model downloaded successfully!")

# Load pipeline (model + scaler + features)
pipeline, features = joblib.load(pkl_path)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🛠 Software Defect Predictor")
st.write("Enter the software metrics below to predict if the module is defect-prone:")

# Manual 21 feature inputs
values = [
    st.number_input("Lines of Code (loc)", 0.0),
    st.number_input("Cyclomatic Complexity v(g)", 0.0),
    st.number_input("Essential Complexity ev(g)", 0.0),
    st.number_input("Design Complexity iv(g)", 0.0),
    st.number_input("Halstead Length (n)", 0.0),
    st.number_input("Halstead Volume (v)", 0.0),
    st.number_input("Halstead Level (l)", 0.0),
    st.number_input("Halstead Difficulty (d)", 0.0),
    st.number_input("Halstead Intelligence (i)", 0.0),
    st.number_input("Halstead Effort (e)", 0.0),
    st.number_input("Estimated Bugs (b)", 0.0),
    st.number_input("Halstead Time (t)", 0.0),
    st.number_input("Logical LOC", 0.0),
    st.number_input("Logical Comments", 0.0),
    st.number_input("Logical Blank Lines", 0.0),
    st.number_input("LOC Code + Comment", 0.0),
    st.number_input("Unique Operators", 0.0),
    st.number_input("Unique Operands", 0.0),
    st.number_input("Total Operators", 0.0),
    st.number_input("Total Operands", 0.0),
    st.number_input("Branch Count", 0.0),
]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    # Create DataFrame with proper feature order
    df = pd.DataFrame([values], columns=features)

    # Make prediction
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Defect-Prone Module (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Non-Defective Module (Probability: {1 - prob:.2f})")
