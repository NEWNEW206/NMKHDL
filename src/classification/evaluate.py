import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
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
        model, X, y, cv=5, n_jobs=1, 
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

def plot_classification_report_heatmap(y_true, y_pred, model_name, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    
    if 'accuracy' in df_report.index:
        df_report = df_report.drop('accuracy')
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report[['precision', 'recall', 'f1-score']], annot=True, cmap="Blues", fmt=".3f", linewidths=.5)
    plt.title(f"Classification Report: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_classification_report.png"), dpi=150)
    plt.close()

def plot_roc_curve_multiclass(model, X_test, y_test, model_name, output_dir="outputs/figures"):
    if not hasattr(model, "predict_proba"):
        return
        
    os.makedirs(output_dir, exist_ok=True)
    y_score = model.predict_proba(X_test)
    
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    if n_classes == 1:
        n_classes = 2
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
        
    plt.figure(figsize=(8, 6))
    colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'Class {classes[i]} (AUC = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (OvR): {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"), dpi=150)
    plt.close()
