# predict_utils.py
import pandas as pd

def preprocess_user_input(user_data, df_n, pca, n_components=35):
    """
    Preprocess a single-row user dataframe (encoding + PCA).
    """
    # Convert dict to DataFrame
    user_df = pd.DataFrame(user_data)

    # One-hot encode same as training
    user_df = pd.get_dummies(user_df).reindex(columns=df_n.drop("y", axis=1).columns, fill_value=0)

    # Apply trained PCA
    user_pca = pca.transform(user_df)
    return user_pca
