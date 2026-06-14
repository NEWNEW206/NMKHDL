import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve

def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def plot_confusion_matrix(model, X_test, y_test, model_name, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        cmap=plt.cm.Blues,
        normalize='true' 
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()

def plot_learning_curve(model, X, y, model_name, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='f1_weighted'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("F1-Score (Weighted)")
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_learning_curve.png"), dpi=150)
    plt.close()
