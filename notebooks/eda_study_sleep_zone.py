import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Đọc dữ liệu ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "raw", "student_lifestyle_dataset.csv")
figures_dir = os.path.join(base_dir, "outputs", "figures")
os.makedirs(figures_dir, exist_ok=True)

df = pd.read_csv(data_path)

color_map = {"Low": "#4daf4a", "Moderate": "#ff7f00", "High": "#e41a1c"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Biểu đồ 1: Study vs Sleep, tô màu theo Stress_Level ---
ax = axes[0]
for level, color in color_map.items():
    subset = df[df["Stress_Level"] == level]
    ax.scatter(
        subset["Study_Hours_Per_Day"], subset["Sleep_Hours_Per_Day"],
        c=color, label=level, alpha=0.5, s=25, edgecolors="none"
    )

# Vùng an toàn theo ngưỡng cây quyết định: Study <= 8.05 & Sleep > 5.95
ax.axvline(8.05, color="black", linestyle="--", linewidth=1.2)
ax.axhline(5.95, color="black", linestyle="--", linewidth=1.2)
ax.fill_between([0, 8.05], 5.95, 12, color="green", alpha=0.07)
ax.text(1, 11.3, "Vùng an toàn\n(Study ≤ 8.05 & Sleep > 5.95)",
        fontsize=9, color="green", fontweight="bold")

ax.set_xlabel("Study Hours / ngày")
ax.set_ylabel("Sleep Hours / ngày")
ax.set_title("Study vs Sleep theo mức độ Stress")
ax.legend(title="Stress_Level")

# --- Biểu đồ 2: GPA theo nhóm Study_Hours (bin) ---
ax2 = axes[1]
df["Study_bin"] = pd.cut(df["Study_Hours_Per_Day"], bins=[4, 5, 6, 7, 8, 9, 10, 11])
sns.boxplot(data=df, x="Study_bin", y="GPA", ax=ax2, color="#4C72B0")
ax2.set_xlabel("Study Hours / ngày (khoảng)")
ax2.set_ylabel("GPA")
ax2.set_title("Phân phối GPA theo số giờ tự học")
ax2.tick_params(axis='x', rotation=30)

plt.tight_layout()

figure_path = os.path.join(figures_dir, "optimal_zone_analysis.png")
plt.savefig(figure_path, dpi=150)
plt.close()

print(f"Đã lưu biểu đồ tại: {figure_path}")
