import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🏢 Employee Attrition Predictor")
st.write("A simple ML web app to predict whether an employee will leave the company.")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model_path = "xgb_attrition_pipeline.joblib"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        pipeline = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
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

# ---------------------- PREDICT ----------------------
if st.button("🔮 Predict"):
    if pipeline is None:
        st.error("Model not found! Ensure 'xgb_attrition_pipeline.joblib' is in this folder.")
    else:
        try:
            # If saved as (preprocessor, model)
            if isinstance(pipeline, tuple):
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
            
            # If saved as sklearn Pipeline
            else:
                data = pd.DataFrame([{
                    "Age": age,
                    "MonthlyIncome": income,
                    "JobSatisfaction": satisfaction,
                    "OverTime": overtime,
                    "Gender": gender,
                    "MaritalStatus": marital
                }])
                pred = pipeline.predict(data)[0]
                prob = pipeline.predict_proba(data)[:, 1][0]

            # ---------------------- RESULT ----------------------
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
