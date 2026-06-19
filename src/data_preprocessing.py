import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    """
    Cleans the raw dataframe by dropping identifier columns, duplicates, 
    and optionally dropping rows with missing target values.
    
    Parameters:
    df (pd.DataFrame): Raw dataframe
    target_cols (list): Optional list of target columns. Rows with missing values 
                        in these columns will be dropped.
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()
    
    # 1. Drop identifier column Student_ID if it exists
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        
    # 2. Check and remove duplicate rows
    df = df.drop_duplicates()
    
    # 3. Drop rows with missing target values instead of imputing them
    if target_cols:
        existing_targets = [col for col in target_cols if col in df.columns]
        df = df.dropna(subset=existing_targets)
                
    return df

def encode_stress_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps categorical Stress_Level ('Low', 'Moderate', 'High') to numeric (0, 1, 2).
    
    Parameters:
    df (pd.DataFrame): Dataframe with categorical Stress_Level
    
    Returns:
    pd.DataFrame: Dataframe with encoded Stress_Level
    """
    df = df.copy()
    
    if 'Stress_Level' in df.columns:
        # Check if it's already numeric to avoid mapping errors if run multiple times
        if not pd.api.types.is_numeric_dtype(df['Stress_Level']):
            stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
            df['Stress_Level'] = df['Stress_Level'].map(stress_map)
        
    return df

def split_and_scale_data(df: pd.DataFrame, target_col: str):
    """
    Splits the dataframe into Train/Test sets (80/20), imputes missing feature
    values using training set statistics (preventing data leakage), and scales 
    continuous numerical features.
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    target_col (str): The name of the target column (e.g. 'GPA' or 'Stress_Level')
    
    Returns:
    X_train_scaled (pd.DataFrame): Processed features for training
    X_test_scaled (pd.DataFrame): Processed features for testing
    y_train (pd.Series): Target labels/values for training
    y_test (pd.Series): Target labels/values for testing
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split train/test (80/20) with random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # 1. Feature Imputation after split (Avoiding Data Leakage)
    for col in X_train.columns:
        if X_train[col].isnull().any() or X_test[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train[col]):
                fill_value = X_train[col].median()
            else:
                fill_value = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else np.nan
            
            X_train_scaled[col] = X_train_scaled[col].fillna(fill_value)
            X_test_scaled[col] = X_test_scaled[col].fillna(fill_value)
            
    # 2. Scaling (Only scale continuous features, exclude binary features like 'Is_Overworked')
    all_numeric_cols = X_train_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude columns that are binary (only containing 0, 1, or their float equivalents)
    continuous_cols = [
        col for col in all_numeric_cols 
        if not set(X_train_scaled[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]
    
    if len(continuous_cols) > 0:
        scaler = StandardScaler()
        X_train_scaled[continuous_cols] = scaler.fit_transform(X_train_scaled[continuous_cols])
        X_test_scaled[continuous_cols] = scaler.transform(X_test_scaled[continuous_cols])
        
    return X_train_scaled, X_test_scaled, y_train, y_test
