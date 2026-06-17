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

def analyze_decision_tree_overfitting(X_train, X_test, y_train, y_test):
    """
    Compares Train and Test Accuracy of DecisionTreeClassifier with and without depth limit.
    Prints the results and returns a summary of the overfitting analysis.
    """
    from sklearn.tree import DecisionTreeClassifier
    
    # Case 1: No limit (max_depth=None)
    dt_unlimited = DecisionTreeClassifier(random_state=42)
    dt_unlimited.fit(X_train, y_train)
    train_acc_unlim = accuracy_score(y_train, dt_unlimited.predict(X_train))
    test_acc_unlim = accuracy_score(y_test, dt_unlimited.predict(X_test))
    
    # Case 2: Limited depth (max_depth=5)
    dt_limited = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_limited.fit(X_train, y_train)
    train_acc_lim = accuracy_score(y_train, dt_limited.predict(X_train))
    test_acc_lim = accuracy_score(y_test, dt_limited.predict(X_test))
    
    print("\n" + "="*60)
    print("  PHÂN TÍCH OVERFITTING - DECISION TREE CLASSIFIER")
    print("="*60)
    print(f"Trường hợp 1: Không giới hạn độ sâu (max_depth=None)")
    print(f"  - Độ chính xác tập Train: {train_acc_unlim*100:.2f}%")
    print(f"  - Độ chính xác tập Test:  {test_acc_unlim*100:.2f}%")
    print(f"  - Khoảng cách (Gap):      {(train_acc_unlim - test_acc_unlim)*100:.2f}%")
    print(f"Trường hợp 2: Giới hạn độ sâu (max_depth=5)")
    print(f"  - Độ chính xác tập Train: {train_acc_lim*100:.2f}%")
    print(f"  - Độ chính xác tập Test:  {test_acc_lim*100:.2f}%")
    print(f"  - Khoảng cách (Gap):      {(train_acc_lim - test_acc_lim)*100:.2f}%")
    
    conclusion = (
        "\n[KẾT LUẬN THÀNH VIÊN 6]:\n"
        f"Khi không giới hạn độ sâu (max_depth=None), Decision Tree đạt độ chính xác tập Train là {train_acc_unlim*100:.2f}% "
        f"(gần như tuyệt đối) nhưng độ chính xác tập Test chỉ đạt {test_acc_unlim*100:.2f}%. Khoảng lệch lớn này ({(train_acc_unlim - test_acc_unlim)*100:.2f}%) "
        "chứng minh rõ ràng mô hình bị 'học vẹt' (Overfitting).\n"
        f"Khi giới hạn độ sâu (max_depth=5), độ chính xác tập Train giảm xuống còn {train_acc_lim*100:.2f}%, nhưng độ chính xác "
        f"tập Test tăng lên (hoặc giữ ở mức ổn định) {test_acc_lim*100:.2f}%. Khoảng cách sai lệch thu hẹp đáng kể ({(train_acc_lim - test_acc_lim)*100:.2f}%), "
        "giúp mô hình tổng quát hóa tốt hơn nhiều trên dữ liệu mới."
    )
    print(conclusion)
    print("="*60 + "\n")
    return {
        'unlimited': {'train': train_acc_unlim, 'test': test_acc_unlim},
        'limited': {'train': train_acc_lim, 'test': test_acc_lim},
        'conclusion': conclusion
    }

