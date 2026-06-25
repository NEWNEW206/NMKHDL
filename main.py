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

from src.regression.model_training import train_and_evaluate_regression
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

    # ==========================================
    # PIPELINE 1: PHÂN LOẠI (CLASSIFICATION)
    # ==========================================
    print("\n CLASSIFICATION ")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_and_scale_data(df, target_col='Stress_Level')

    models_cls, results_cls = train_and_evaluate_classification(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

    # Tính & vẽ feature importance cho Stress_Level
    if 'RandomForestClassifier' in models_cls:
        importance_df_cls = calculate_importance_score(
            models_cls['RandomForestClassifier'], list(X_train_cls.columns),
            output_dir=os.path.join(BASE_DIR, "outputs", "classification")
        )
        barplot_feature_importance(
            importance_df_cls, 
            model_name="StressLevel_RandomForestClassifier",
            output_dir=os.path.join(BASE_DIR, "outputs", "figures", "classification")
        )
    
    # ==========================================
    # PIPELINE 2: HỒI QUY (REGRESSION) 
    # ==========================================
    print("\n REGRESSION ")
    df_reg = df.dropna(subset=['GPA'])
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_and_scale_data(df_reg, target_col='GPA')
    
    models_reg, results_reg = train_and_evaluate_regression(X_train_reg, X_test_reg, y_train_reg, y_test_reg)

    # Tính & vẽ feature importance cho GPA
    if 'RandomForestRegressor' in models_reg:
        importance_df_reg = calculate_importance_score(
            models_reg['RandomForestRegressor'], list(X_train_reg.columns),
            output_dir=os.path.join(BASE_DIR, "outputs", "regression")
        )
        barplot_feature_importance(
            importance_df_reg, 
            model_name="GPA_RandomForestRegressor",
            output_dir=os.path.join(BASE_DIR, "outputs", "figures", "regression")
        )

if __name__ == "__main__":
    main()
