# model_utils.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import streamlit as st

@st.cache_resource
def train_models(data, model_type="random_forest"):
    x_train_pca, y_train = data
    models = {}

    if model_type == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError("Invalid model_type")

    model.fit(x_train_pca, y_train)
    models["classification"] = model
    return models
