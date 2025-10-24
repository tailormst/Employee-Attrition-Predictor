import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------- APP CONFIGURATION ----------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

DEV_NAME = "Mohammed Tailor"
COLLEGE = "RCOEM, Nagpur"
BRANCH = "AIML"
SECTION = "B"
ROLL_NO = "27"
LINKEDIN = "https://www.linkedin.com/in/mohammed-tailor-002968288/"
EMAIL = "mohammedtailor5253@gmail.com"
PHONE = "+91-9067718254"
PROJECT_TITLE = "Employee Attrition Predictor"
PROJECT_DESC = "Predict whether an employee is likely to leave the organization using an XGBoost model with preprocessing pipeline."

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_pipeline():
    model_path = "xgb_attrition_pipeline.joblib"

    st.sidebar.write("### Debug Info")
    st.sidebar.write(f"Current directory: {os.getcwd()}")
    st.sidebar.write(f"Files: {os.listdir('.')}")

    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Model file not found: {model_path}")
        return None

    try:
        pipeline = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return pipeline
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.sidebar.info("Ensure scikit-learn and XGBoost versions match the training environment.")
        return None

pipeline = load_pipeline()

# ---------------------- NAVIGATION ----------------------
tabs = st.tabs(["🏠 Home", "ℹ️ About", "📧 Contact"])

# ---------------------- HOME TAB ----------------------
with tabs[0]:
    st.title(PROJECT_TITLE)
    st.write(PROJECT_DESC)
    st.markdown("---")

    if pipeline is None:
        st.error("❌ Model pipeline not loaded! Please ensure 'xgb_attrition_pipeline.joblib' is in this folder.")
    else:
        st.success("✅ Model loaded and ready for prediction!")

    st.header("🔍 Predict Employee Attrition")

    with st.form("attrition_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Age", 18, 65, 30)
            DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
            DistanceFromHome = st.number_input("Distance From Home (km)", 0, 50, 5)
            EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1–4)", [1, 2, 3, 4], index=2)
            JobInvolvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4], index=2)
            JobSatisfaction = st.selectbox("Job Satisfaction (1–4)", [1, 2, 3, 4], index=2)
            MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            OverTime = st.selectbox("OverTime", ["No", "Yes"])

        with col2:
            NumCompaniesWorked = st.number_input("Num Companies Worked", 0, 20, 3)
            TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 8)
            YearsAtCompany = st.number_input("Years At Company", 0, 40, 5)
            WorkLifeBalance = st.selectbox("Work Life Balance (1–4)", [1, 2, 3, 4], index=2)
            MonthlyRate = st.number_input("Monthly Rate", 1000, 20000, 12000)
            PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 50, 15)
            PerformanceRating = st.selectbox("Performance Rating (1–4)", [1, 2, 3, 4], index=3)
            YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20, 2)

        with col3:
            MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
            HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
            RelationshipSatisfaction = st.selectbox("Relationship Satisfaction (1–4)", [1, 2, 3, 4], index=2)
            YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20, 3)
            TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 3)
            StockOptionLevel = st.selectbox("Stock Option Level (0–3)", [0, 1, 2, 3], index=1)
            Gender = st.selectbox("Gender", ["Male", "Female"])

        col4, col5, col6 = st.columns(3)
        with col4:
            Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        with col5:
            BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        with col6:
            JobRole = st.selectbox(
                "Job Role", [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manager", "Human Resources", "Manufacturing Director",
                    "Sales Representative", "Healthcare Representative", "Research Director"
                ]
            )

        EducationField = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])
        Education = st.selectbox("Education (1–5)", [1, 2, 3, 4, 5], index=2)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        if pipeline is None:
            st.error("❌ Pipeline not found! Make sure 'xgb_attrition_pipeline.joblib' is in the directory.")
        else:
            try:
                preprocessor, xgb_model = pipeline  # unpack tuple

                input_data = pd.DataFrame([{
                    "Age": Age,
                    "DailyRate": DailyRate,
                    "DistanceFromHome": DistanceFromHome,
                    "Education": Education,
                    "EnvironmentSatisfaction": EnvironmentSatisfaction,
                    "JobInvolvement": JobInvolvement,
                    "JobSatisfaction": JobSatisfaction,
                    "MaritalStatus": MaritalStatus,
                    "OverTime": OverTime,
                    "NumCompaniesWorked": NumCompaniesWorked,
                    "TotalWorkingYears": TotalWorkingYears,
                    "YearsAtCompany": YearsAtCompany,
                    "WorkLifeBalance": WorkLifeBalance,
                    "MonthlyRate": MonthlyRate,
                    "PercentSalaryHike": PercentSalaryHike,
                    "PerformanceRating": PerformanceRating,
                    "YearsSinceLastPromotion": YearsSinceLastPromotion,
                    "MonthlyIncome": MonthlyIncome,
                    "HourlyRate": HourlyRate,
                    "RelationshipSatisfaction": RelationshipSatisfaction,
                    "YearsInCurrentRole": YearsInCurrentRole,
                    "TrainingTimesLastYear": TrainingTimesLastYear,
                    "StockOptionLevel": StockOptionLevel,
                    "Gender": Gender,
                    "Department": Department,
                    "BusinessTravel": BusinessTravel,
                    "JobRole": JobRole,
                    "EducationField": EducationField
                }])

                # Preprocess and predict
                transformed = preprocessor.transform(input_data)
                pred = xgb_model.predict(transformed)[0]
                prob = xgb_model.predict_proba(transformed)[:, 1][0]

                st.markdown("---")
                st.subheader("📈 Prediction Result")

                if pred == 1:
                    st.error(f"⚠️ Employee likely to leave! (Probability: {prob*100:.2f}%)")
                    if prob > 0.7:
                        st.warning("🚨 High Risk: Immediate retention action recommended!")
                    elif prob > 0.4:
                        st.info("💡 Suggest improving satisfaction, workload, or engagement.")
                else:
                    st.success(f"✅ Employee likely to stay. (Probability: {(1-prob)*100:.2f}%)")
                    if prob > 0.3:
                        st.info("💡 Some moderate risk factors identified — Monitor employee satisfaction.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Ensure feature names match model's training schema.")

# ---------------------- ABOUT TAB ----------------------
with tabs[1]:
    st.header("📘 About the Project")
    st.write(PROJECT_DESC)
    st.markdown("""### Features:
    - **Demographics:** Age, Gender, Marital Status  
    - **Job Details:** Role, Department, Satisfaction Metrics  
    - **Compensation:** Income, Salary Hike, Stock Options  
    - **Work Environment:** Work-Life Balance, Overtime  
    - **Career Path:** Promotions, Training, Experience
    """)

    st.markdown("### 👨‍💻 Developer Info")
    st.write(f"**Name:** {DEV_NAME}")
    st.write(f"**College:** {COLLEGE}")
    st.write(f"**Branch:** {BRANCH}")
    st.write(f"**Section:** {SECTION}")
    st.write(f"**Roll No:** {ROLL_NO}")
    st.write(f"**LinkedIn:** [LinkedIn Profile]({LINKEDIN})")
    st.write(f"📧 Email: {EMAIL}")

# ---------------------- CONTACT TAB ----------------------
with tabs[2]:
    st.header("📞 Contact Information")
    st.write(f"📧 Email: {EMAIL}")
    st.write(f"🔗 LinkedIn: {LINKEDIN}")
    st.write(f"📱 Phone: {PHONE}")
    st.markdown("---")
    st.write("This app helps organizations predict employee attrition using machine learning.")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption(f"© 2025 {DEV_NAME} — Developed with ❤️ using Streamlit and XGBoost")