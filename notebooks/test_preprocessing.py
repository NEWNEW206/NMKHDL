import sys
import os
import pandas as pd

# Add the src directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_data, encode_stress_level, split_and_scale_data

def test_pipeline():
    # Path to the raw dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/student_lifestyle_dataset.csv'))
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("Loading raw dataset...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Null values count:")
    print(df.isnull().sum())
    
    print("\n--- Running clean_data ---")
    df_clean = clean_data(df)
    print(f"Shape after clean_data: {df_clean.shape}")
    print("Columns after clean_data:", list(df_clean.columns))
    print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
    print(f"Duplicates remaining: {df_clean.duplicated().sum()}")
    
    print("\n--- Running encode_stress_level ---")
    df_encoded = encode_stress_level(df_clean)
    print("Stress_Level unique values:")
    print(df_encoded['Stress_Level'].value_counts())
    
    print("\n--- Running split_and_scale_data (Target: Stress_Level) ---")
    X_train, X_test, y_train, y_test = split_and_scale_data(df_encoded, target_col='Stress_Level')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    print("\nSample of X_train (scaled):")
    print(X_train.head())
    
    print("\nSample of y_train:")
    print(y_train.head())
    
    print("\nAll preprocessing functions verified successfully!")

if __name__ == "__main__":
    test_pipeline()
