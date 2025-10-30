import streamlit as st
import pandas as pd
import joblib

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🏢 Employee Attrition Predictor")
st.write("A simple ML web app to predict whether an employee will leave the company.")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load("xgb_attrition_pipeline.joblib")
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model()

# ---------------------- USER INPUT ----------------------
st.header("🧍 Employee Details")
age = st.number_input("Age", 18, 65, 30)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
satisfaction = st.selectbox("Job Satisfaction (1–4)", [1, 2, 3, 4], index=2)
overtime = st.selectbox("OverTime", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

if st.button("🔮 Predict"):
    if pipeline is None:
        st.error("Model not found! Ensure 'xgb_attrition_pipeline.joblib' is in the same folder.")
    else:
        try:
            preprocessor, model = pipeline

            data = pd.DataFrame([{
                "Age": age,
                "MonthlyIncome": income,
                "JobSatisfaction": satisfaction,
                "OverTime": overtime,
                "Gender": gender,
                "MaritalStatus": marital
            }])

            X = preprocessor.transform(data)
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[:, 1][0]

            st.subheader("📊 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ Employee likely to leave! (Probability: {prob*100:.2f}%)")
            else:
                st.success(f"✅ Employee likely to stay. (Probability: {(1-prob)*100:.2f}%)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("© 2025 Mohammed Tailor — Built with ❤️ using Streamlit & XGBoost")
