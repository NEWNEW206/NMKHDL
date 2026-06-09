import pandas as pd
import numpy as np

def add_total_productive_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the Total_Productive_Hours feature.
    Formula: Study_Hours + Physical_Activity_Hours + Extracurricular_Hours
    """
    df = df.copy()
    required_cols = ['Study_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'Extracurricular_Hours_Per_Day']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in input data.")
            
    df['Total_Productive_Hours'] = (
        df['Study_Hours_Per_Day'] + 
        df['Physical_Activity_Hours_Per_Day'] + 
        df['Extracurricular_Hours_Per_Day']
    )
    return df

def add_free_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the Free_Time feature and the supplementary Is_Overworked feature.
    Formula: Free_Time = 24 - (Total hours of all activities)
    Use .clip(lower=0) to avoid negative values due to input errors.
    """
    df = df.copy()
    required_cols = [
        'Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
        'Physical_Activity_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
        'Social_Hours_Per_Day'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in input data.")
            
    total_activity_hours = (
        df['Study_Hours_Per_Day'] +
        df['Extracurricular_Hours_Per_Day'] +
        df['Physical_Activity_Hours_Per_Day'] +
        df['Sleep_Hours_Per_Day'] +
        df['Social_Hours_Per_Day']
    )
    
    # Create a supplementary feature Is_Overworked to signal time overload (> 24 hours)
    df['Is_Overworked'] = (total_activity_hours > 24).astype(int)
    
    # Calculate free time and set lower bound to 0
    df['Free_Time'] = 24.0 - total_activity_hours
    df['Free_Time'] = df['Free_Time'].clip(lower=0.0)
    
    return df

def add_study_productive_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the Study_Productive_Ratio feature.
    Formula: Study Hours / (Total Productive Hours + 1e-6)
    """
    df = df.copy()
    
    # Ensure Total_Productive_Hours exists 
    if 'Total_Productive_Hours' not in df.columns:
        df = add_total_productive_hours(df)
        
    df['Study_Productive_Ratio'] = df['Study_Hours_Per_Day'] / (df['Total_Productive_Hours'] + 1e-6)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline to extract all new features.
    Execution order:
    1. add_total_productive_hours
    2. add_free_time
    3. add_study_productive_ratio
    """
    df = add_total_productive_hours(df)
    df = add_free_time(df)
    df = add_study_productive_ratio(df)
        
    # Safety check: Ensure no unexpected missing values are generated
    cols_to_check = ['Total_Productive_Hours', 'Free_Time', 'Study_Productive_Ratio', 'Is_Overworked']
        
    for col in cols_to_check:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
            
    return df
