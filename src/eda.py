import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda(df: pd.DataFrame, output_dir: str):
    """
    Performs Exploratory Data Analysis (EDA) on the dataset.
    Generates and saves three figures:
    1. GPA distribution (histplot)
    2. Stress Level count (countplot)
    3. Correlation matrix (heatmap)
    
    Parameters:
    df (pd.DataFrame): The cleaned dataframe
    output_dir (str): Directory where figures will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a clean, modern aesthetic style
    sns.set_theme(style="whitegrid")
    
    # Biểu đồ 1: Vẽ phân phối điểm GPA bằng sns.histplot
    plt.figure(figsize=(8, 5))
    sns.histplot(df['GPA'], kde=True, color='royalblue', bins=20, edgecolor='black', alpha=0.7)
    plt.title('Phân phối điểm GPA của Sinh viên', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('GPA', fontsize=12)
    plt.ylabel('Số lượng sinh viên', fontsize=12)
    plt.tight_layout()
    gpa_path = os.path.join(output_dir, 'gpa_distribution.png')
    plt.savefig(gpa_path, dpi=150)
    plt.close()
    print(f"  [EDA] Đã lưu biểu đồ phân phối GPA tại: {gpa_path}")
    
    # Biểu đồ 2: Vẽ phân phối mức độ Stress Level (Low, Moderate, High) bằng sns.countplot
    plt.figure(figsize=(8, 5))
    stress_order = ['Low', 'Moderate', 'High']
    
    # Check if Stress_Level column values are still string objects
    # If they are already numeric, map them back for readable visualization labels
    temp_df = df.copy()
    if temp_df['Stress_Level'].dtype != object:
        reverse_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
        temp_df['Stress_Level'] = temp_df['Stress_Level'].map(reverse_map)
        
    sns.countplot(x='Stress_Level', data=temp_df, order=stress_order, palette='Set2', edgecolor='black', alpha=0.8)
    plt.title('Phân phối mức độ Stress Level của Sinh viên', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Mức độ Stress', fontsize=12)
    plt.ylabel('Số lượng sinh viên', fontsize=12)
    plt.tight_layout()
    stress_path = os.path.join(output_dir, 'stress_level_distribution.png')
    plt.savefig(stress_path, dpi=150)
    plt.close()
    print(f"  [EDA] Đã lưu biểu đồ phân phối Stress Level tại: {stress_path}")
    
    # Biểu đồ 3: Vẽ ma trận tương quan nhiệt bằng sns.heatmap
    plt.figure(figsize=(10, 8))
    # Select numeric columns only for correlation analysis
    numeric_df = df.select_dtypes(include=['number'])
    # Drop Student_ID if it accidentally exists, but clean_data should have dropped it
    if 'Student_ID' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Student_ID'])
        
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
    plt.title('Ma trận tương quan giữa các đặc trưng lối sống và GPA', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"  [EDA] Đã lưu ma trận tương quan nhiệt tại: {corr_path}")
