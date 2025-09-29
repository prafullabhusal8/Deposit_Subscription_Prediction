# evaluate_model.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_models(data, models):
    x_test_pca, y_test = data
    results = {}

    for task, model in models.items():
        y_pred = model.predict(x_test_pca)
        results[task] = calculate_metrics(y_test, y_pred)

    return results

def calculate_metrics(y_true, y_pred):
    return {
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1),
        "Recall": recall_score(y_true, y_pred, pos_label=1),
        "F1 Score": f1_score(y_true, y_pred, pos_label=1),
    }
