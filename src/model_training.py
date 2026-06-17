"""
Module: model_training.py
Mục tiêu: Dự báo điểm số liên tục (GPA) bằng 5 mô hình hồi quy,
           đánh giá bằng R², MAE, RMSE, vẽ biểu đồ phân tán (Scatter Plot)
           so sánh GPA thực tế và GPA dự đoán, xuất bảng so sánh kết quả.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không hiển thị (dành cho server / script)
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================================
# HÀM PHỤ TRỢ
# ============================================================================

def _get_regression_models():
    """
    Khởi tạo 5 mô hình hồi quy cơ bản.

    Returns:
        dict: Từ điển {tên_mô_hình: đối_tượng_mô_hình}
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }


def _calculate_regression_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số đánh giá mô hình hồi quy.

    Parameters:
        y_true: Giá trị GPA thực tế
        y_pred: Giá trị GPA dự đoán

    Returns:
        dict: Từ điển chứa R², MAE, RMSE
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    }


def _plot_scatter(y_true, y_pred, model_name, output_dir):
    """
    Vẽ biểu đồ phân tán (Scatter Plot) so sánh GPA thực tế và GPA dự đoán.
    Thêm đường chéo y = x (màu đỏ) làm mốc chuẩn.

    Parameters:
        y_true: Giá trị GPA thực tế
        y_pred: Giá trị GPA dự đoán
        model_name (str): Tên mô hình
        output_dir (str): Thư mục lưu biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Vẽ các điểm dữ liệu
    ax.scatter(
        y_true, y_pred,
        alpha=0.6, edgecolors='k', linewidth=0.5,
        color='#4C72B0', s=60, label='Dự đoán'
    )

    # Vẽ đường y = x (đường chuẩn hoàn hảo)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.05
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, 'r--', linewidth=2, label='y = x (Lý tưởng)')

    # Cấu hình tiêu đề và nhãn
    ax.set_title(f"Scatter Plot: {model_name}\nGPA Thực tế vs GPA Dự đoán", fontsize=13, fontweight='bold')
    ax.set_xlabel("GPA Thực tế", fontsize=11)
    ax.set_ylabel("GPA Dự đoán", fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{model_name}_scatter_plot.png")
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


# ============================================================================
# HÀM CHÍNH – HUẤN LUYỆN & ĐÁNH GIÁ HỒI QUY
# ============================================================================

def train_and_evaluate_regression(X_train, X_test, y_train, y_test):
    """
    Huấn luyện 5 mô hình hồi quy dự đoán GPA, đánh giá bằng R², MAE, RMSE,
    vẽ biểu đồ Scatter Plot và xuất bảng kết quả ra file CSV.

    Parameters:
        X_train: Tập đặc trưng huấn luyện (đã chuẩn hóa)
        X_test: Tập đặc trưng kiểm thử (đã chuẩn hóa)
        y_train: Nhãn GPA tập huấn luyện
        y_test: Nhãn GPA tập kiểm thử

    Returns:
        tuple: (dict các mô hình đã huấn luyện, DataFrame bảng kết quả)
    """
    # --- Thiết lập đường dẫn lưu trữ ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(outputs_dir, "figures")

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # --- Khởi tạo 5 mô hình hồi quy ---
    models = _get_regression_models()

    results = []

    for name, model in models.items():
        # 1. Huấn luyện mô hình trên tập Train
        model.fit(X_train, y_train)

        # 2. Dự đoán trên tập Test
        y_pred = model.predict(X_test)

        # 3. Tính toán chỉ số đánh giá
        metrics = _calculate_regression_metrics(y_test, y_pred)
        metrics["Model"] = name
        results.append(metrics)

        # 4. Vẽ biểu đồ Scatter Plot
        _plot_scatter(y_test, y_pred, name, figures_dir)

    # --- Tổng hợp kết quả vào DataFrame ---
    df_results = pd.DataFrame(results)
    cols = ["Model", "R2", "MAE", "RMSE"]
    df_results = df_results[cols]

    # Sắp xếp theo R² giảm dần (mô hình tốt nhất ở trên)
    df_results = df_results.sort_values(by="R2", ascending=False).reset_index(drop=True)

    # --- Xuất ra file CSV ---
    results_path = os.path.join(outputs_dir, "regression_results.csv")
    df_results.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("        BẢNG SO SÁNH CÁC MÔ HÌNH HỒI QUY (GPA)")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print(f"\nĐã lưu bảng kết quả tại: {results_path}")
    print(f"Đã lưu biểu đồ Scatter Plot tại thư mục: {figures_dir}")
    return models, df_results

# HÀM TIỆN ÍCH – CHUẨN BỊ DỮ LIỆU CHO HỒI QUY
def prepare_regression_data(df, target_col='GPA'):
    """
    Tách đặc trưng và nhãn cho bài toán hồi quy, chia Train/Test,
    và chuẩn hóa dữ liệu bằng StandardScaler.

    Parameters:
        df (pd.DataFrame): DataFrame đã qua tiền xử lý và feature engineering
        target_col (str): Tên cột mục tiêu (mặc định: 'GPA')

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    if target_col not in df.columns:
        raise KeyError(f"Cột mục tiêu '{target_col}' không tồn tại trong DataFrame.")

    # Chỉ lấy các cột số để làm đặc trưng (loại bỏ cột mục tiêu)
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]

    X = df[feature_cols]
    y = df[target_col]

    # Chia tập Train / Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Chuẩn hóa đặc trưng bằng StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nChuẩn bị dữ liệu hồi quy:")
    print(f"   - Số đặc trưng: {len(feature_cols)}")
    print(f"   - Tập Train: {X_train_scaled.shape[0]} mẫu")
    print(f"   - Tập Test : {X_test_scaled.shape[0]} mẫu")
    print(f"   - Đặc trưng: {feature_cols}")

    return X_train_scaled, X_test_scaled, y_train, y_test
