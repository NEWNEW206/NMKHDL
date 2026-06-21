import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def calculate_importance_score(model, feature_names):
    """
    Hàm tổng quát để lấy feature_importances_.

    Parameters:
        model: Đối tượng Random Forest đã huấn luyện (có thể là Regressor hoặc Classifier)
        feature_names (list): Danh sách tên các thói quen ['Study', 'Extracurricular', 'Sleep', 'Social', 'Physical_Activity']

    Returns:
        importance_df (DataFrame): Bảng kết quả bao gồm tên thói quen và mức độ ảnh hưởng đến điểm số, stress tương ứng
    """
    # --- Thiết lập đường dẫn lưu trữ ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outputs_dir = os.path.join(base_dir, "outputs")

    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Trích xuất thuộc tính .feature_importances_ từ mô hình truyền vào
    importances = model.feature_importances_

    # 2. Sắp xếp chỉ số từ cao đến thấp
    indices = np.argsort(importances)[::-1]

    # 3. Gom tên thói quen và điểm số tương ứng sau khi sắp xếp
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    importance_df = pd.DataFrame({
        'Feature': sorted_features,
        'Importance': sorted_importances
    })

    # --- Xuất ra file CSV ---
    model_name = type(model).__name__
    file_path = os.path.join(outputs_dir, f"{model_name}_importance.csv")

    importance_df.to_csv(file_path, index=False)

    print(f"Đã lưu bảng kết quả tại: {file_path}")
    return importance_df

def barplot_feature_importance(importance_df, model_name):
  """
  Vẽ biểu đồ cột nằm ngang (barplot) từ DataFrame mức độ quan trọng từng feature

  Parameters:
  importance_df: Tên DataFrame đầu vào (Ví dụ: RandomForestRegressor_importance_df)
  model_name (str): Tên model, dùng để đặt tên file ảnh xuất ra

  """
  # --- Thiết lập đường dẫn đồng bộ với dự án ---
  base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  outputs_dir = os.path.join(base_dir, "outputs")
  figures_dir = os.path.join(outputs_dir, "figures")

  os.makedirs(figures_dir, exist_ok=True)

  # --- Vẽ biểu đồ Barplot
  plt.figure(figsize=(9, 6))

  ax = sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        palette='viridis',
        hue='Feature',
        legend=False
    )

  # --- Hiển thị giá trị ở đầu mỗi cột
  for p in ax.patches:
        width = p.get_width()
        # Thêm text cách đầu cột một khoảng nhỏ
        ax.text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f'{width:.3f}',
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )

  # --- Thêm title ---
  plt.title(f"Mức độ ảnh hưởng của thói quen ({model_name})", fontsize=14, fontweight='bold', pad=15)
  plt.xlabel("Importance Score", fontsize=11, labelpad=10)

  plt.tight_layout()

  # --- Lưu hình ảnh vào outputs/figures
  figure_path = os.path.join(figures_dir, f"{model_name}_feature_importance.png")
  plt.savefig(figure_path, dpi=150)
  plt.close()