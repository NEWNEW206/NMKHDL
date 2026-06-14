import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_preprocessing import clean_data, encode_stress_level, split_and_scale_data
from src.feature_engineering import engineer_features
from src.classification.train import train_and_evaluate_classification

def main():
    filepath = os.path.join(BASE_DIR, "data", "raw", "student_lifestyle_dataset.csv")
    if not os.path.exists(filepath):
        print("Not found")
        return
        
    df = pd.read_csv(filepath)
    
    df = clean_data(df)

    df = encode_stress_level(df)
    
    df = engineer_features(df)
    
    df = df.dropna(subset=['Stress_Level'])
    
    print("Splitting and scaling data...")
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(df, target_col='Stress_Level')
    
    print("Running classification pipeline...")
    models, results_df = train_and_evaluate_classification(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("Biểu đồ CÁC CLASSFICATION MODEL đã được lưu tại 'outputs/figures' VÀ BẢNG TỔNG HỢP TẠI 'outputs/classification_results.csv'.")

if __name__ == "__main__":
    main()
