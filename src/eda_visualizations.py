import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_plots(df: pd.DataFrame, output_dir: str = "outputs/figures"):
    """
    Generates and saves the 4 EDA plots:
    1. GPA Distribution
    2. Stress Level Distribution
    3. Correlation Matrix of Numerical Features
    4. Study vs Sleep Optimal Zone Analysis
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe (before Stress_Level is encoded).
    output_dir (str): Directory where the plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. GPA Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="GPA", kde=True, color='blue')
    plt.title("GPA Distribution")
    plt.xlabel("GPA")
    plt.ylabel("Amount of Students")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpa_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Stress Level Distribution
    plt.figure(figsize=(10, 6))
    # Order categories to ensure consistent plotting
    order = ["Low", "Moderate", "High"]
    # Only use order if those categories exist
    existing_categories = [cat for cat in order if cat in df["Stress_Level"].unique()]
    if not existing_categories:
        existing_categories = None
    sns.countplot(data=df, x="Stress_Level", order=existing_categories, color='blue')
    plt.title("Stress Level Distribution")
    plt.xlabel("Stress Level")
    plt.ylabel("Amount of Students")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stress_level_distribution.png"), dpi=150)
    plt.close()
    
    # 3. Correlation Matrix of Numerical Features
    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if 'Student_ID' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Student_ID'])
        
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close()
    
    # 4. Study vs Sleep Optimal Zone Analysis
    df_copy = df.copy()
    color_map = {"Low": "#4daf4a", "Moderate": "#ff7f00", "High": "#e41a1c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Study vs Sleep colored by Stress_Level ---
    ax = axes[0]
    for level, color in color_map.items():
        subset = df_copy[df_copy["Stress_Level"] == level]
        ax.scatter(
            subset["Study_Hours_Per_Day"], subset["Sleep_Hours_Per_Day"],
            c=color, label=level, alpha=0.5, s=25, edgecolors="none"
        )

    # Thresholds: Study <= 8.05 & Sleep > 5.95
    ax.axvline(8.05, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(5.95, color="black", linestyle="--", linewidth=1.2)
    ax.fill_between([0, 8.05], 5.95, 12, color="green", alpha=0.07)
    ax.text(1, 11.3, "Vùng an toàn\n(Study ≤ 8.05 & Sleep > 5.95)",
            fontsize=9, color="green", fontweight="bold")

    ax.set_xlabel("Study Hours / ngày")
    ax.set_ylabel("Sleep Hours / ngày")
    ax.set_title("Study vs Sleep theo mức độ Stress")
    ax.legend(title="Stress_Level")

    # --- Plot 2: GPA by Study_Hours (bin) ---
    ax2 = axes[1]
    df_copy["Study_bin"] = pd.cut(df_copy["Study_Hours_Per_Day"], bins=[4, 5, 6, 7, 8, 9, 10, 11])
    sns.boxplot(data=df_copy, x="Study_bin", y="GPA", ax=ax2, color="#4C72B0")
    ax2.set_xlabel("Study Hours / ngày (khoảng)")
    ax2.set_ylabel("GPA")
    ax2.set_title("Phân phối GPA theo số giờ tự học")
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    figure_path = os.path.join(output_dir, "optimal_zone_analysis.png")
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)
