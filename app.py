import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load model + preprocessor
# -------------------------------
with open("insurance_model.pkl", "rb") as f:
    preprocessor, model = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💊 Medical Insurance Cost Predictor")
st.subheader("By Swayam Sodha")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "southeast", "northwest", "southwest"])

if st.button("Predict"):
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }])
    sample_prepared = preprocessor.transform(sample)
    prediction = model.predict(sample_prepared)[0]
    st.success(f"Estimated Insurance Cost: ${prediction:,.2f}")

