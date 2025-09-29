import pandas as pd
import streamlit as st

@st.cache_data
def loaded_data(path=r"C:\Users\bhusa\Downloads\Deposit_Subscription_Prediction\data\Deposit_Subscription_Prediction.csv"):
    # read csv 
    df = pd.read_csv(path)

    # Drop ID if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"], axis=1) 

    # Ensure target 'y' is binary (0/1)
    if df["y"].dtype == "object":
        df["y"] = df["y"].map({"no": 0, "yes": 1})

    # Apply one-hot encoding on categorical features
    df_n = pd.get_dummies(df, columns=['marital', "job", "education", "default",
                                       "housing", "loan", "contact", "month", "poutcome"])

    return df_n
