# 🧠 Employee Attrition Predictor

A machine learning web app built with **Streamlit** and **XGBoost** to predict whether an employee is likely to leave an organization.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28?style=flat&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📌 Overview
This project predicts employee attrition using a trained **XGBoost model** integrated with a **Streamlit** web app for real-time predictions.  
It is designed for HR analytics teams to detect potential attrition risks and make data-driven decisions.

---

## ⚙️ Features
- ⚡ Real-time employee attrition prediction  
- 🧮 Preprocessing pipeline with **scaling**, **encoding**, and **balancing**  
- 📊 Visualization and data exploration (EDA)  
- 🤖 ML pipeline: **XGBoost + Scikit-learn ColumnTransformer**  
- 🎨 Interactive Streamlit interface  

---

## 🧩 Tech Stack
- **Python 3.10+** (recommended)
- **Streamlit**, **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, **Joblib**, **Matplotlib**, **Seaborn**

---

## 🧹 Data Preprocessing
The preprocessing pipeline ensures clean, standardized input before model training:

| Step | Description |
|------|--------------|
| **1. Handling Missing Values** | Cleaned and filled missing data where necessary |
| **2. Encoding** | Used `OneHotEncoder` for categorical features |
| **3. Scaling** | Applied both `StandardScaler` and `MinMaxScaler` for numeric normalization |
| **4. Balancing** | Addressed class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique) |
| **5. Feature Selection** | Retained only the most important predictors based on correlation and XGBoost feature importance |

---

## 📊 Exploratory Data Analysis (EDA)
Comprehensive EDA was performed to understand data patterns and relationships:

- Distribution plots of numeric variables  
- Categorical frequency and attrition ratio plots  
- Correlation heatmap of numeric features  
- Boxplots and violin plots to detect outliers  
- Visualization tools used: **Matplotlib**, **Seaborn**

---

## 🧠 Model Info

| Aspect | Details |
|--------|----------|
| **Algorithm** | XGBoost Classifier |
| **Preprocessing** | ColumnTransformer (Scaling + OneHotEncoding) |
| **Balancing** | SMOTE applied on minority class |
| **Evaluation Metrics** | Precision, Recall, F1-score, ROC-AUC |
| **Model Serialization** | Joblib |
| **Accuracy** | ~90% on test data |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ML_Mini_Project_27.git
cd ML_Mini_Project_27
```
### 2. Create a virtual environment
```bash
py -3.10 -m venv ml_env
ml_env\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
streamlit run app.py
```

#### 📁 Project Structure
```bash
ML_Mini_Project_27/
├── app.py                          # Streamlit frontend app
├── xgb_model.joblib                # Trained XGBoost model
├── requirements.txt                # Python dependencies
├── data/                           # (optional) Raw and processed datasets
├── notebooks/                      # Jupyter notebooks for EDA & model training
└── README.md                       # Documentation               
```

### 📈 Visualization Examples
🔹 Attrition distribution by department and job role

🔹 Correlation heatmap of numeric features

🔹 Feature importance chart from XGBoost

🔹 ROC-AUC curve visualization

(These were created during the model development phase using Matplotlib and Seaborn.)

### 🛡️ License
```bash
Licensed under the MIT License.
```

## ⭐ If you found this project helpful, please give it a star on GitHub!
