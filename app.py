# app.py
import streamlit as st
import pandas as pd
import numpy as np
import churn_pipeline as cp
import generate_churn_data as gen
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Churn Prediction Demo", layout="centered")
st.title("Churn Prediction with SHAP Explanations")

st.sidebar.header("Data")
if st.sidebar.button("Generate sample_churn.csv (2000 rows)"):
    df = gen.generate_churn_like(2000)
    df.to_csv("sample_churn.csv", index=False)
    st.sidebar.success("sample_churn.csv generated")

uploaded = st.file_uploader("Or upload sample_churn.csv", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.to_csv("sample_churn.csv", index=False)
    st.success("Uploaded sample_churn.csv")

if not os.path.exists("sample_churn.csv"):
    st.info("No data found. Generate or upload sample_churn.csv.")
else:
    df = pd.read_csv("sample_churn.csv")
    st.subheader("Data preview")
    st.dataframe(df.head(10))

st.sidebar.header("Model")
model_choice = st.sidebar.selectbox("Model", ["lgbm","rf"])
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)
run = st.sidebar.button("Train & Explain")

if run:
    if not os.path.exists("sample_churn.csv"):
        st.error("No data file. Generate or upload first.")
    else:
        df = cp.load_data("sample_churn.csv")
        with st.spinner("Training model..."):
            model, metrics, X_test, y_test = cp.train_and_eval(df, test_size=test_size, model_name=model_choice)
            cp.save_model(model, "churn_model.joblib")
        st.success("Model trained.")
        st.write("Metrics:")
        st.json(metrics)
        # SHAP explanations: use TreeExplainer for tree models
        st.subheader("Global SHAP summary (first 200 rows for speed)")
        explainer = shap.TreeExplainer(model.named_steps["clf"])
        # need transformed feature names; get preprocessor output
        pre = model.named_steps["pre"]
        X_sample = X_test.head(200)
        X_trans = pre.transform(X_sample)
        # build feature names
        num_feats = ["age","tenure","monthly_charges"]
        cat_feats = ["contract","internet_service","senior_citizen"]
        # get onehot feature names
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        try:
            ohe_names = ohe.get_feature_names_out(cat_feats).tolist()
        except:
            ohe_names = []
        feature_names = num_feats + ohe_names
        X_df = pd.DataFrame(X_trans, columns=feature_names)
        shap_values = explainer.shap_values(X_df)
        # For binary, shap_values is list [neg,pos] for some versions
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        fig = shap.summary_plot(sv, X_df, show=False, plot_type="bar")
        st.pyplot(bbox_inches="tight")
        st.subheader("Local explanation for a sample user")
        idx = st.number_input("Index in test set (0..{})".format(len(X_test)-1), min_value=0, max_value=max(0, len(X_test)-1), value=0)
        X_row = X_test.iloc[[idx]]
        X_row_tr = pre.transform(X_row)
        X_row_df = pd.DataFrame(X_row_tr, columns=feature_names)
        shap_vals_row = explainer.shap_values(X_row_df)
        if isinstance(shap_vals_row, list):
            sv_row = shap_vals_row[1]
        else:
            sv_row = shap_vals_row
        fig2 = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value, sv_row[0], feature_names=feature_names, show=False)
        st.pyplot(bbox_inches="tight")
        st.markdown("- Model saved to churn_model.joblib")
