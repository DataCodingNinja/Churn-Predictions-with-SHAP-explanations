# Churn Prediction with SHAP explanations

What: Tabular churn prediction pipeline with LightGBM/RandomForest, Streamlit UI, and SHAP-based explainability.

Why: Demonstrates classification, preprocessing pipelines, model evaluation, and interpretable explanations — valuable for data scientist roles.

Files:
- requirements.txt
- generate_churn_data.py — creates sample_churn.csv
- sample_churn.csv — generated after running generator or uploading your own
- churn_pipeline.py — train & evaluate pipeline
- app.py — Streamlit UI: generate/upload data, train model, view metrics and SHAP plots
- churn_model.joblib — saved model after training

Quick start:
1. Create venv and install:
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt

2. Generate sample:
   python generate_churn_data.py

3. Run Streamlit:
   streamlit run app.py

Notes and tips:
- SHAP plotting can be slow; app limits to first 200 rows for global summary.
- If SHAP plotting raises errors due to version mismatches, remove local waterfall/legacy call and use shap.plots.signature or shap.plots.bar for local explanations.
- All files kept in one folder for easy GitHub upload.
