# churn_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib

def load_data(path="sample_churn.csv"):
    df = pd.read_csv(path)
    return df

def build_pipeline(model_name="lgbm"):
    num_feats = ["age","tenure","monthly_charges"]
    cat_feats = ["contract","internet_service","senior_citizen"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_feats),
        ("cat", categorical_transformer, cat_feats)
    ])
    if model_name == "lgbm":
        clf = LGBMClassifier(n_estimators=100, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    return pipe

def train_and_eval(df, target="churn", test_size=0.2, random_state=42, model_name="lgbm"):
    X = df.drop(columns=[target,"customer_id"], errors="ignore")
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    pipe = build_pipeline(model_name=model_name)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }
    return pipe, metrics, X_test, y_test

def save_model(pipeline, path="churn_model.joblib"):
    joblib.dump(pipeline, path)
    return path
