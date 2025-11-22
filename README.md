# 🧠 Employee Attrition Predictor

A Streamlit web app that uses a trained XGBoost pipeline to predict whether an employee is likely to leave an organisation. This repository contains a production-ready Streamlit app (app.py), a serialized model pipeline (xgb_attrition_pipeline.pkl), notebooks and resources used during development.

Badges
- Python (runtime specified in runtime.txt)
- Streamlit
- XGBoost
- Scikit-learn
- Pandas

---

## 📌 Overview

This project demonstrates a complete end-to-end ML workflow for predicting employee attrition:
- Data exploration and preprocessing (notebooks/)
- Model training and evaluation
- A deployed/publishable Streamlit web app (app.py) that loads a serialized pipeline and outputs real-time predictions

The app is designed for HR analytics prototyping to surface employees at higher risk of leaving so teams can investigate and take action.

---

## Features

- Real-time attrition prediction via Streamlit
- Preprocessing pipeline embedded with the model (scaling, encoding, etc.)
- Trained XGBoost classifier serialized as a pipeline
- Notebooks for EDA and training (notebooks/)
- Render configuration included for quick deployment (render.yaml)

---

## Tech stack

- Python 3.10 (see runtime.txt)
- Streamlit
- XGBoost
- Scikit-learn
- Pandas, NumPy
- Joblib / pickle for model serialization
- Matplotlib / Seaborn for visualizations in notebooks

## Quickstart (local)

1. Clone the repository
```bash
git clone https://github.com/tailormst/Employee-Attrition-Predictor.git
cd Employee-Attrition-Predictor
```

2. Create and activate a virtual environment (example using python3.10)
```bash
python3.10 -m venv venv
source venv/bin/activate   # macOS / Linux
# or
venv\Scripts\activate      # Windows (PowerShell/CMD)
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in your browser (default http://localhost:8501). The Streamlit `app.py` loads `xgb_attrition_pipeline.pkl` to perform predictions — be sure that file remains in the repository root.

---

## Files & Project structure

Use this as the authoritative structure for this repo:

```
Employee-Attrition-Predictor/
├── app.py                          # Streamlit frontend application
├── xgb_attrition_pipeline.pkl      # Trained XGBoost pipeline used by the app
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Runtime used (e.g., python-3.10)
├── render.yaml                     # Render deployment configuration
├── data/                           # (optional) Place datasets here (not versioned)
├── notebooks/                      # EDA and training notebooks
├── Mini project report B27.docx    # Project report / write-up
└── README.md                       # This documentation
```

File descriptions:
- app.py: the Streamlit UI and inference code. It expects preprocessed input fields and feeds data to the serialized pipeline.
- xgb_attrition_pipeline.pkl: the trained pipeline (preprocessing + XGBoost). Do not commit sensitive data to the repo.
- requirements.txt: pinned or minimal dependency versions for reproducible installs.
- render.yaml: simple configuration for deploying to Render.com (if you use it).
- notebooks/: analysis, feature engineering and model training notebooks. Use them to re-train or to explore feature importance and metrics.

---

## Model & data notes

- Model: XGBoost classifier embedded in a scikit-learn Pipeline (serialized as `xgb_attrition_pipeline.pkl`).
- Preprocessing (applies inside the pipeline): scaling, encoding, and any feature transformations required by the model.
- Evaluation metrics used during development: precision, recall, F1-score, ROC-AUC.
- The dataset used during development is not included here. If you intend to re-train, add your dataset to `data/` (or point the notebooks to a local path). Ensure any employee data is handled securely and complies with privacy rules.

---

## Deployment

- Local: `streamlit run app.py`
- Render: `render.yaml` is included; you can deploy the repository to Render and it will use `runtime.txt` and `requirements.txt` to build. If you deploy, ensure the model file `xgb_attrition_pipeline.pkl` is present or update the deployment to retrieve the model from secure storage.

---

## Recommendations & Next steps

1. Add a LICENSE file (README currently mentions MIT — add an explicit LICENSE file).
2. Add a small CONTRIBUTING.md and CODE_OF_CONDUCT.md if you want external contributors.
3. Add tests (unit tests for data transforms and a smoke test for the app).
4. Add a GitHub Actions workflow to run tests and linting on push/PR.
5. If model size is large, consider storing the model in an external artifact store (Git LFS, S3, or an artifacts registry) and loading it at runtime to keep the repo lean.
6. Add example screenshots or a short demo GIF in README to show the app UI and sample outputs.
7. Document expected input ranges and missing-value behavior in the README or in a separate docs file.
8. If the dataset contains PII, add notes about anonymization and data storage.

---

## Troubleshooting

- "Module not found" errors: ensure your virtual environment is activated and you've installed the exact packages from `requirements.txt`.
- Port conflicts when running Streamlit: run `streamlit run app.py --server.port <PORT>` to change the port.
- Model load errors: verify `xgb_attrition_pipeline.pkl` exists and was created with compatible versions of scikit-learn and XGBoost. If not, re-run the training notebook and re-serialize the pipeline.

---

## License

Please add a LICENSE file to the repository. The previous README referenced the MIT license — if you want MIT, create a `LICENSE` file containing the MIT text.

---

## Contact

Maintainer: tailormst
