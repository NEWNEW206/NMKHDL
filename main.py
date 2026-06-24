import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_preprocessing import clean_data, encode_stress_level, split_and_scale_data
from src.feature_engineering import engineer_features
from src.classification.train import train_and_evaluate_classification
from src.classification.importance_score import calculate_importance_score, barplot_feature_importance
from src.eda_visualizations import generate_eda_plots

# Thử import module hồi quy (nếu đã hoàn thành)
regression_available = False
try:
    from src.model_training import train_and_evaluate_regression
    regression_available = True
except ImportError:
    pass

def main():
    filepath = os.path.join(BASE_DIR, "data", "raw", "student_lifestyle_dataset.csv")
    if not os.path.exists(filepath):
        print(f"Error: Không tìm thấy file dữ liệu tại {filepath}")
        return
    df = pd.read_csv(filepath)
    df = clean_data(df)
    
    print("Generating EDA plots...")
    generate_eda_plots(df, output_dir=os.path.join(BASE_DIR, "outputs", "figures"))
    
    df = encode_stress_level(df)
    df = engineer_features(df)
    
    # Lưu lại dữ liệu đã qua tiền xử lý
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_filepath = os.path.join(processed_dir, "student_lifestyle_processed.csv")
    df.to_csv(processed_filepath, index=False)
    print(f"Saved processed data to {processed_filepath}")
    
    # Đảm bảo không còn giá trị khuyết thiếu trong cột Stress_Level
    df = df.dropna(subset=['Stress_Level'])

    print("Splitting and scaling data...")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_and_scale_data(df, target_col='Stress_Level')

    print("Running classification pipeline...")
    models_cls, results_cls = train_and_evaluate_classification(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

    # Tính & vẽ feature importance cho Stress_Level (dùng model RandomForest đã train ở trên)
    if 'RandomForestClassifier' in models_cls:
        print("Calculating feature importance for Stress_Level...")
        importance_df_cls = calculate_importance_score(
            models_cls['RandomForestClassifier'], list(X_train_cls.columns)
        )
        barplot_feature_importance(importance_df_cls, model_name="StressLevel_RandomForestClassifier")
    
    if regression_available:
        # Đảm bảo không còn giá trị khuyết thiếu trong cột GPA
        df_reg = df.dropna(subset=['GPA'])
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_and_scale_data(df_reg, target_col='GPA')
        
        models_reg, results_reg = train_and_evaluate_regression(X_train_reg, X_test_reg, y_train_reg, y_test_reg)

        # Tính & vẽ feature importance cho GPA (dùng model RandomForest đã train ở trên)
        if 'RandomForestRegressor' in models_reg:
            print("Calculating feature importance for GPA...")
            importance_df_reg = calculate_importance_score(
                models_reg['RandomForestRegressor'], list(X_train_reg.columns)
            )
            barplot_feature_importance(importance_df_reg, model_name="GPA_RandomForestRegressor")
    else:
        print("  [Thông báo] Module hồi quy ('src/model_training.py') chưa hoàn thành hoặc đang được viết.")
        print("  Tạm thời bỏ qua bước Hồi quy GPA cho đến khi module được cập nhật.")
        
if __name__ == "__main__":
    main()
