import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_preprocessing import clean_data, encode_stress_level, split_and_scale_data
from src.feature_engineering import engineer_features
from src.eda import perform_eda
from src.classification.train import train_and_evaluate_classification

# Thử import module hồi quy của Thành viên 5 (nếu đã hoàn thành)
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
        
    print("=========================================================")
    print(" BẮT ĐẦU CHẠY PIPELINE MÁY HỌC TỰ ĐỘNG (INTEGRATOR - TV1)")
    print("=========================================================\n")
    
    print("[Bước 1] Đọc dữ liệu...")
    df = pd.read_csv(filepath)
    
    print("[Bước 2] Làm sạch dữ liệu (Thành viên 3)...")
    df = clean_data(df)
    
    print("[Bước 3] Thực hiện Khám phá dữ liệu EDA (Thành viên 2)...")
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    perform_eda(df, figures_dir)
    print("  -> Đã lưu các biểu đồ EDA tại 'outputs/figures/'")
    
    print("\n[Bước 4] Mã hóa thuộc tính Stress_Level (Thành viên 3)...")
    df = encode_stress_level(df)
    
    print("[Bước 5] Trích xuất đặc trưng mới (Thành viên 4)...")
    df = engineer_features(df)
    
    # Đảm bảo không còn giá trị khuyết thiếu trong cột Stress_Level
    df = df.dropna(subset=['Stress_Level'])
    
    print("\n[Bước 6] Chạy Pipeline Phân loại Stress Level (Thành viên 6)...")
    print("  Chia dữ liệu và chuẩn hóa đặc trưng (target: Stress_Level)...")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_and_scale_data(df, target_col='Stress_Level')
    
    print("  Đang tối ưu hóa siêu tham số (Optuna cv=5) và huấn luyện 5 mô hình...")
    models_cls, results_cls = train_and_evaluate_classification(X_train_cls, X_test_cls, y_train_cls, y_test_cls)
    print("  -> Hoàn thành Pipeline Phân loại!")
    
    print("\n[Bước 7] Chạy Pipeline Hồi quy GPA (Thành viên 5)...")
    if regression_available:
        # Đảm bảo không còn giá trị khuyết thiếu trong cột GPA
        df_reg = df.dropna(subset=['GPA'])
        print("  Chia dữ liệu và chuẩn hóa đặc trưng (target: GPA)...")
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_and_scale_data(df_reg, target_col='GPA')
        
        print("  Huấn luyện 5 mô hình hồi quy và đánh giá...")
        models_reg, results_reg = train_and_evaluate_regression(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
        print("  -> Hoàn thành Pipeline Hồi quy!")
    else:
        print("  [Thông báo] Module hồi quy ('src/model_training.py') chưa hoàn thành hoặc đang được viết.")
        print("  Tạm thời bỏ qua bước Hồi quy GPA cho đến khi Thành viên 5 cập nhật file.")
        
    print("\n=========================================================")
    print(" HOÀN TẤT PIPELINE CHẠY TỰ ĐỘNG!")
    print("=========================================================")

if __name__ == "__main__":
    main()

