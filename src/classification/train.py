import os
import joblib
import pandas as pd
from src.classification.models import get_baseline_models
from src.classification.optimizer import ModelOptimizer
from src.classification.evaluate import calculate_metrics, plot_confusion_matrix, plot_learning_curve, analyze_decision_tree_overfitting

def train_and_evaluate_classification(X_train, X_test, y_train, y_test):
    # Sử dụng đường dẫn tuyệt đối để tránh bị lưu nhầm ra ngoài thư mục gốc
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(outputs_dir, "figures")
    models_dir = os.path.join(outputs_dir, "classification") # Lưu models vào outputs/classification

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    models = get_baseline_models()
    optimizer = ModelOptimizer(X_train, y_train)

    models['RandomForestClassifier'] = optimizer.optimize_random_forest(n_trials=20)
    models['SVC'] = optimizer.optimize_svc(n_trials=20)
    
    # Tiến hành huấn luyện, đánh giá và vẽ đồ thị cho tất cả 5 mô hình
    results = []
    
    print("\n" + "="*50)
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
        
        # In ra tham số và kết quả
        print(f"\n---> MÔ HÌNH: {name}")
        print("Tham số đang sử dụng:")
        print(model.get_params())
        print(f"Kết quả (Test): F1-Score={metrics['F1-Score']:.4f} | Accuracy={metrics['Accuracy']:.4f}")
        
        # Lưu lại toàn bộ các mô hình vào thư mục outputs/classification
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
            
    df_results = pd.DataFrame(results)

    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_results = df_results[cols]
    
    results_path = os.path.join(outputs_dir, "classification_results.csv")
    df_results.to_csv(results_path, index=False)
    
    print("\n" + "="*50)
    print("         BẢNG TỔNG HỢP SO SÁNH 5 MÔ HÌNH     ")
    print("="*50)
    print(df_results.to_string(index=False))
    print(f"\n Đã lưu bảng kết quả tại: {results_path}")
    print(f" Đã lưu toàn bộ 5 file mô hình (.pkl) tại thư mục 'outputs/classification/'")
    
    # 1. Thực hiện phân tích Overfitting của DecisionTreeClassifier (Thành viên 6)
    analyze_decision_tree_overfitting(X_train, X_test, y_train, y_test)
    
    # 2. Tìm và lưu mô hình tốt nhất (Best Model) dựa trên F1-Score vào thư mục models/ ở gốc (Thành viên 6)
    root_models_dir = os.path.join(base_dir, "models")
    os.makedirs(root_models_dir, exist_ok=True)
    best_idx = df_results['F1-Score'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    best_model = models[best_model_name]
    best_model_path = os.path.join(root_models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f" Đã tìm thấy mô hình xuất sắc nhất: {best_model_name} (F1-Score={df_results.loc[best_idx, 'F1-Score']:.4f})")
    print(f" Đã lưu mô hình xuất sắc nhất này vào: {best_model_path}")
    
    return models, df_results
