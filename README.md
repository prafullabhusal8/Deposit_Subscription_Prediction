# Deposit Subscription Prediction using Machine Learning and Streamlit

## Overview
This project is a **web-based machine learning application** that predicts whether a bank customer will subscribe to a term deposit. The system uses customer demographic and financial information to provide accurate predictions.  

### Key Features
- Preprocesses data using **one-hot encoding** and **PCA** for dimensionality reduction.
- Trains and evaluates **Decision Tree, Random Forest, and XGBoost** models.
- Displays key evaluation metrics: **Accuracy, Precision, Recall, F1 Score, Confusion Matrix**.
- Provides an **interactive Streamlit form** for real-time predictions (Yes/No).

## Project Structure
Deposit_Subscription_Prediction/
│
├─ data/
│ └─ Deposit_Subscription_Prediction.csv # Raw dataset
│
├─ src_n/
│ ├─ app.py # Main Streamlit app
│ ├─ load_data.py # Load and preprocess dataset
│ ├─ data_ttt.py # Train-test split and PCA
│ ├─ model_utils.py # Train ML models
│ ├─ evaluate_model.py # Model evaluation metrics
│ ├─ predict_utils.py # Preprocess user input for prediction
│ └─ ui_utils.py # Custom CSS for Streamlit UI
│
└─ requirements.txt # Python dependencies


Create a virtual environment (optional but recommended):

code
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
    
Install dependencies:

code
pip install -r requirements.txt

How to Run
Start the Streamlit app:

code
streamlit run src_n/app.py
Use the sidebar to select a model type (Decision Tree, Random Forest, XGBoost).

Check model performance metrics in the sidebar.

Fill the customer details form and get a real-time subscription prediction.

Technologies Used
     Python – core programming
     Scikit-learn – ML models and metrics
     XGBoost – Gradient boosting classifier
     Pandas – Data manipulation
     Streamlit – Web application interface
     PCA – Dimensionality reduction for faster training
