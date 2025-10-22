import streamlit as st
import pandas as pd
import joblib
import os
import sys

# ---------------------- CONFIG ----------------------
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
PROJECT_DESC = "Predict whether an employee is likely to leave the organization using an XGBoost model."

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model_path = "preprocessor_xgb_pipeline.joblib"
    
    # Debug information
    st.sidebar.write("### Debug Info")
    st.sidebar.write(f"Current directory: {os.getcwd()}")
    st.sidebar.write(f"Files in directory: {os.listdir('.')}")
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File '{model_path}' not found!")
        st.sidebar.info(f"Looking for: {os.path.abspath(model_path)}")
        return None
    
    st.sidebar.success(f"✅ File '{model_path}' found!")
    
    try:
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.sidebar.info("Try: conda install scikit-learn=1.3.2")
        return None

model = load_model()

# ---------------------- NAVIGATION ----------------------
tabs = st.tabs(["🏠 Home", "ℹ️ About", "📧 Contact"])

# ---------------------- HOME TAB ----------------------
with tabs[0]:
    st.title(PROJECT_TITLE)
    st.write(PROJECT_DESC)
    st.markdown("---")

    # Show model status
    if model is None:
        st.error("❌ Model not loaded! Please ensure 'preprocessor_xgb_pipeline.joblib' is in the same folder.")
        st.info(f"Current directory: {os.getcwd()}")
    else:
        st.success("✅ Model loaded successfully!")

    st.header("🔍 Predict Employee Attrition")

    with st.form("attrition_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Age", 18, 65, 30)
            DistanceFromHome = st.number_input("Distance From Home (km)", 0, 50, 5)
            Education = st.selectbox("Education (1–5)", [1, 2, 3, 4, 5], index=2)
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
            DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
            MonthlyRate = st.number_input("Monthly Rate", 1000, 20000, 10000)
            PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 50, 15)
            PerformanceRating = st.selectbox("Performance Rating (1–4)", [1, 2, 3, 4], index=3)
            
        with col3:
            # ADD THE MISSING FEATURES
            YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20, 2)
            MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
            HourlyRate = st.number_input("Hourly Rate", 30, 100, 50)
            RelationshipSatisfaction = st.selectbox("Relationship Satisfaction (1–4)", [1, 2, 3, 4], index=2)
            YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20, 3)
            TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 3)
            StockOptionLevel = st.selectbox("Stock Option Level (0–3)", [0, 1, 2, 3], index=0)
            Gender = st.selectbox("Gender", ["Male", "Female"])
        
        # ADD THESE AT THE BOTTOM (they take more space)
        col4, col5, col6 = st.columns(3)
        with col4:
            Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        with col5:
            BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        with col6:
            JobRole = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
        
        EducationField = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
        ])

        submitted = st.form_submit_button("🔮 Predict")

    # Prediction
    if submitted:
        if model is None:
            st.error("❌ Model not found! Please place 'preprocessor_xgb_pipeline.joblib' in the same folder.")
        else:
            input_data = pd.DataFrame([{
                # Your existing features
                "Age": Age,
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
                "DailyRate": DailyRate,
                "MonthlyRate": MonthlyRate,
                "PercentSalaryHike": PercentSalaryHike,
                "PerformanceRating": PerformanceRating,
                
                # ADD THE MISSING FEATURES
                "YearsSinceLastPromotion": YearsSinceLastPromotion,
                "Department": Department,
                "MonthlyIncome": MonthlyIncome,
                "BusinessTravel": BusinessTravel,
                "HourlyRate": HourlyRate,
                "RelationshipSatisfaction": RelationshipSatisfaction,
                "JobRole": JobRole,
                "YearsInCurrentRole": YearsInCurrentRole,
                "TrainingTimesLastYear": TrainingTimesLastYear,
                "StockOptionLevel": StockOptionLevel,
                "Gender": Gender,
                "EducationField": EducationField
            }])

            try:
                # Debug: Check what's in the tuple
                st.sidebar.write("Tuple contents:")
                for i, item in enumerate(model):
                    st.sidebar.write(f"Item {i}: {type(item)}")
                    if hasattr(item, 'predict'):
                        st.sidebar.write(f"  → Has predict method!")
                    if hasattr(item, 'transform'):
                        st.sidebar.write(f"  → Has transform method!")
            
                # Extract both preprocessor and model from the tuple
                if isinstance(model, tuple) and len(model) >= 2:
                    preprocessor = model[0]  # ColumnTransformer
                    xgb_model = model[1]    # XGBoost model
                
                    st.sidebar.success("✅ Extracted preprocessor and model from tuple")
                
                    # Transform the input data using preprocessor
                    transformed_data = preprocessor.transform(input_data)
                    st.sidebar.write(f"Transformed data shape: {transformed_data.shape}")
                
                    # Make predictions using the XGBoost model
                    pred = xgb_model.predict(transformed_data)[0]
                    prob = xgb_model.predict_proba(transformed_data)[:, 1][0]

                    st.markdown("---")
                    st.subheader("📊 Prediction Result")

                    if pred == 1:
                        st.error(f"⚠️ Employee likely to leave! (Probability: {prob*100:.2f}%)")
                        if prob > 0.7:
                            st.warning("🚨 High Risk: Immediate intervention recommended!")
                        elif prob > 0.4:
                            st.info("💡 Suggestion: Offer better work-life balance, rewards, or career growth opportunities.")
                    else:
                        st.success(f"✅ Employee likely to stay. (Probability: {(1-prob)*100:.2f}%)")
                        if prob > 0.3:
                            st.info("💡 Monitor: Employee shows some risk factors")

                else:
                    st.error("❌ Invalid model format. Expected tuple with preprocessor and model.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("This might be due to feature mismatch between your model and the input data.")

# ---------------------- ABOUT TAB ----------------------
with tabs[1]:
    st.header("📘 About the Project")
    st.markdown(f"**Project:** {PROJECT_TITLE}")
    st.write(PROJECT_DESC)
    st.markdown("""
    ### 📊 Features Used in Prediction
    - **Demographic**: Age, Gender, Marital Status
    - **Job Details**: Department, Job Role, Job Satisfaction
    - **Compensation**: Monthly Income, Daily Rate, Salary Hike
    - **Work Environment**: Business Travel, OverTime, Work-Life Balance
    - **Career Growth**: Years at Company, Years Since Promotion, Training
    - **Satisfaction Metrics**: Environment, Job, Relationship Satisfaction
    """)
    st.markdown("### 👨‍💻 Developer Info")
    st.write(f"- **Name:** {DEV_NAME}")
    st.write(f"- **College:** {COLLEGE}")
    st.write(f"- **Branch:** {BRANCH}")
    st.write(f"- **Section:** {SECTION}")
    st.write(f"- **Roll No:** {ROLL_NO}")
    st.write(f"- [LinkedIn Profile]({LINKEDIN})")

# ---------------------- CONTACT TAB ----------------------
with tabs[2]:
    st.header("📞 Contact")
    st.write(f"📧 Email: {EMAIL}")
    st.write(f"🔗 LinkedIn: {LINKEDIN}")
    st.write(f"📱 Phone: {PHONE}")
    
    st.markdown("---")
    st.subheader("💼 Project Repository")
    st.write("This project uses Machine Learning to predict employee attrition and help organizations with retention strategies.")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption(f"© 2025 {DEV_NAME} — Built with ❤️ using Streamlit")