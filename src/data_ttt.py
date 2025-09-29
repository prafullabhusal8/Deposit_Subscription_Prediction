# data_ttt.py
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.decomposition import PCA

@st.cache_data
def data_train_test_split(df_n):
    x = df_n.drop("y", axis=1)
    y = df_n["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pca = PCA(n_components=35)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    return (x_train_pca, y_train), (x_test_pca, y_test), pca
