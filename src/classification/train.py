import os
import joblib
import pandas as pd
from src.classification.models import get_baseline_models
from src.classification.optimizer import ModelOptimizer
from src.classification.evaluate import (
    calculate_metrics, 
    plot_confusion_matrix, 
    plot_learning_curve,
    plot_classification_report_heatmap,
    plot_roc_curve_multiclass
)

def train_and_evaluate_classification(X_train, X_test, y_train, y_test):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(outputs_dir, "figures", "classification")
    models_dir = os.path.join(outputs_dir, "classification")

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    models = get_baseline_models()
    optimizer = ModelOptimizer(X_train, y_train)

    models['RandomForestClassifier'] = optimizer.optimize_random_forest(n_trials=25)
    models['SVC'] = optimizer.optimize_svc(n_trials=25)
    
    #Huấn luyện, đánh giá và vẽ đồ thị cho tất cả 5 mô hình
    results = []
    print("\n")
    print("="*50)
    print("     CHI TIẾT THAM SỐ VÀ KẾT QUẢ TỪNG MÔ HÌNH ")
    print("="*50)

    for name, model in models.items():
        # Vẽ Learning Curve (dành cho toàn bộ mô hình) để phân tích Overfitting
        plot_learning_curve(model, X_train, y_train, name, output_dir=figures_dir)
        
        # Huấn luyện mô hình trên tập Train
        model.fit(X_train, y_train)
        
        # Dự đoán trên tập Test
        y_pred = model.predict(X_test)
        
        # Tính toán các chỉ số
        metrics = calculate_metrics(y_test, y_pred)
        metrics['Model'] = name
        results.append(metrics)
        
        # Vẽ Confusion Matrix
        plot_confusion_matrix(model, X_test, y_test, name, output_dir=figures_dir)
        
        # Vẽ Heatmap Report và ROC Curve
        plot_classification_report_heatmap(y_test, y_pred, name, output_dir=figures_dir)
        plot_roc_curve_multiclass(model, X_test, y_test, name, output_dir=figures_dir)
        
        # In ra tham số và kết quả
        print(f"\n MÔ HÌNH: {name}")
        print("Tham số đang sử dụng:")
        print(model.get_params())
        print(f"Kết quả trên tập test : F1-Score={metrics['F1-Score']:.4f} | Accuracy={metrics['Accuracy']:.4f}")
        
        # Lưu lại toàn bộ các mô hình vào thư mục outputs/classification
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
            
    df_results = pd.DataFrame(results)

    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_results = df_results[cols]
    
    results_path = os.path.join(outputs_dir, "classification_results.csv")
    df_results.to_csv(results_path, index=False)
    print("\n")
    print("="*50)
    print("      BẢNG SO SÁNH CÁC MÔ HÌNH PHÂN LOẠI")
    print("="*50)
    print(df_results.to_string(index=False))
    print(f"\n Bảng kết quả lưu tại: {results_path}")
    print(f" Đã lưu toàn bộ 5 file mô hình (.pkl) tại thư mục 'outputs/classification/'")
    

    return models, df_results
