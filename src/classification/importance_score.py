import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def calculate_importance_score(model, feature_names, output_dir=None):
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    importance_df = pd.DataFrame({
        'Feature': sorted_features,
        'Importance': sorted_importances
    })

    model_name = type(model).__name__
    file_path = os.path.join(output_dir, f"{model_name}_importance.csv")

    importance_df.to_csv(file_path, index=False)

    print(f"Bảng kết quả lưu tại: {file_path}")
    return importance_df

def barplot_feature_importance(importance_df, model_name, output_dir=None):
  if output_dir is None:
      base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
      output_dir = os.path.join(base_dir, "outputs", "figures")

  os.makedirs(output_dir, exist_ok=True)

  plt.figure(figsize=(9, 6))
  ax = sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        palette='viridis',
        hue='Feature',
        legend=False
    )

  for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f'{width:.4f}',
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )

  plt.title(f"Mức độ ảnh hưởng của thói quen ({model_name})", fontsize=14, fontweight='bold', pad=15)
  plt.xlabel("Importance Score", fontsize=11, labelpad=10)

  plt.tight_layout()
  figure_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
  plt.savefig(figure_path, dpi=150)
  plt.close()