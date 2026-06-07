import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe by dropping identifier columns, duplicates, 
    and filling missing values with the median of each column.
    
    Parameters:
    df (pd.DataFrame): Raw dataframe
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()
    
    # 1. Drop identifier column Student_ID if it exists
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        
    # 2. Check and remove duplicate rows
    df = df.drop_duplicates()
    
    # 3. Fill missing values with median of each column
    for col in df.columns:
        if df[col].isnull().any():
            # For columns that are numerical, fill with median
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                # If a column is categorical (e.g., Stress_Level before encoding), 
                # we fill with mode as median is not defined.
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                df[col] = df[col].fillna(mode_val)
                
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
        stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        df['Stress_Level'] = df['Stress_Level'].map(stress_map)
        
    return df

def split_and_scale_data(df: pd.DataFrame, target_col: str):
    """
    Splits the dataframe into Train/Test sets (80/20) and scales the numerical features.
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    target_col (str): The name of the target column (e.g. 'GPA' or 'Stress_Level')
    
    Returns:
    X_train_scaled (pd.DataFrame or np.ndarray): Scaled features for training
    X_test_scaled (pd.DataFrame or np.ndarray): Scaled features for testing
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
    
    # Scale numeric feature columns
    # Identifying numeric columns in X
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        # Fit and transform on training features, transform on test features to prevent data leakage
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
    return X_train_scaled, X_test_scaled, y_train, y_test
