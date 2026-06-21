import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# --- Đọc dữ liệu ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "raw", "student_lifestyle_dataset.csv")
df = pd.read_csv(data_path)

feature_cols = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]
X = df[feature_cols]
y = df["Stress_Level"]

# ============================================================
# Kiểm chứng Stress_Level mang tính rule-based:
# Một cây quyết định nông (max_depth=3) có đạt accuracy = 1.0 không?
# Nếu có -> nhãn được sinh ra theo ngưỡng cố định trên input,
# không phải quan hệ học từ dữ liệu thực tế phức tạp.
# ============================================================
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

train_acc = tree.score(X, y)
print(f"Train accuracy (max_depth=3): {train_acc}")
print()
print(export_text(tree, feature_names=feature_cols))
