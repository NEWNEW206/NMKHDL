# Dự án Khoa học Dữ liệu: Lối sống & Kết quả học tập của Sinh viên
**Môn học:** Nhập môn Khoa học Dữ liệu (Mini-Project)  

## 1. Giới thiệu Pipeline (Workflow)
Dự án được xây dựng theo chuẩn Machine Learning Pipeline với các bước sau:
1. **Tiền xử lý (Preprocessing)**: Đọc dữ liệu `student_lifestyle_dataset.csv`, loại bỏ các cột định danh, xóa trùng lặp và xử lý dữ liệu khuyết thiếu. Mã hóa nhãn Stress Level thành dạng số.
2. **Kỹ thuật đặc trưng (Feature Engineering)**: Khởi tạo các đặc trưng mới từ dữ liệu thô như Tổng số giờ hoạt động hiệu quả (`Total_Productive_Hours`), Thời gian rảnh (`Free_Time`) và Tỷ lệ ưu tiên học tập (`Study_Productive_Ratio`).
3. **Phân chia & Chuẩn hóa (Split & Scale)**: Tách tập dữ liệu thành tập huấn luyện và kiểm thử, áp dụng Standardization để đưa đặc trưng về cùng thang đo.
4. **Phân loại (Classification Pipeline)**: 
   - Khởi tạo 5 mô hình cơ bản (Logistic Regression, KNN, Decision Tree, Random Forest, SVC).
   - Ứng dụng **Optuna** và 5-Fold Cross Validation để tối ưu siêu tham số.
   - Vẽ **Learning Curve** và **Confusion Matrix** để phân tích hiện tượng Overfitting.
   - Lưu báo cáo điểm số (Accuracy, F1) và lưu file model xuất sắc nhất ra đĩa.
5. **Hồi quy (Regression Pipeline)**:
   - Khởi tạo 5 mô hình hồi quy cơ bản (**Linear Regression, Ridge, Support Vector Regression (SVR), Decision Tree Regressor, Random Forest Regressor**).
   - Sử dụng **GPA** làm biến mục tiêu (**Target**) để dự báo điểm số.
   - Vẽ biểu đồ **Scatter Plot** (GPA Thực tế vs. GPA Dự đoán) để đánh giá khả năng khái quát hóa.
   - Lưu 5 mô hình (.pkl) vào `outputs/regression/` và báo cáo kết quả (**MSE, RMSE, MAE, R² Score**) vào tệp `outputs/regression_results.csv`.

## 2. Cấu trúc thư mục

```text
NMKHDL/
│
├── data/                      # Dữ liệu dự án
│   ├── raw/                   # Dữ liệu gốc (student_lifestyle_dataset.csv)
│   └── processed/             # Dữ liệu đã qua xử lý (student_lifestyle_processed.csv)
│
├── outputs/                   # Thư mục lưu kết quả sau khi chạy Pipeline
│   ├── classification/        # Chứa 5 file mô hình (.pkl) phân loại
│   ├── regression/            # Chứa 5 file mô hình (.pkl) hồi quy
│   ├── figures/               # Biểu đồ phân tích trực quan
│   │   ├── classification/    # ROC, Heatmap, Confusion Matrix, Learning Curve
│   │   └── regression/        # Scatter Plot, Feature Importance
│   ├── classification_results.csv # Bảng điểm số Classification
│   └── regression_results.csv     # Bảng điểm số Regression
│
├── notebooks/                 # Script kiểm thử và phân tích
│   ├── EDA.ipynb              # Phân tích khám phá dữ liệu trực quan (Jupyter)
│   └── test_preprocessing.py  # Script chạy thử nghiệm module tiền xử lý
│
├── src/                       # Mã nguồn phát triển dạng module
│   ├── classification/        # Module tối ưu thuật toán Phân loại
│   ├── regression/            # Module tối ưu thuật toán Hồi quy
│   ├── data_preprocessing.py  # Làm sạch và chuẩn hóa thang đo
│   ├── feature_engineering.py # Tạo lập đặc trưng mới
│   └── eda_visualizations.py  # Trực quan hóa dữ liệu EDA
│
├── report.ipynb               # Báo cáo chính thức (Jupyter Notebook)
├── report.md                  # Bản draft báo cáo (Markdown)
├── main.py                    # Script chính kích hoạt toàn bộ luồng Pipeline
├── run_pipeline.bat           # File thực thi chạy tự động (Windows)
├── run_pipeline.sh            # File thực thi chạy tự động (Linux/Mac)
├── Dockerfile                 # Cấu hình đóng gói hệ thống Docker
├── requirements.txt           # Danh sách các thư viện hỗ trợ
└── README.md                  # Hướng dẫn chạy dự án
```

## 3. Hướng dẫn Chạy Dự án
Bạn có thể chạy toàn bộ dự án từ đầu đến cuối để ra kết quả mô hình và biểu đồ bằng một trong ba cách sau:

### Cách 1: Chạy bằng file Script
- **Trên Windows:** Chỉ cần click đúp chuột vào file `run_pipeline.bat` hoặc mở Command Prompt gõ:
  ```cmd
  .\run_pipeline.bat
  ```
- **Trên Linux/Mac/GitBash:** Mở Terminal và gõ:
  ```bash
  bash run_pipeline.sh
  ```

### Cách 2: Chạy thủ công bằng Python
Yêu cầu đã cài đặt Python 3.9+.
```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 2. Chạy file main
python main.py
```

### Cách 3: Chạy bằng hệ thống ảo hóa Docker
Đảm bảo máy bạn đã cài sẵn Docker Desktop.
```bash
docker build -t student_lifestyle_ml .

# Chạy Docker và đồng bộ thư mục outputs ra ngoài máy thật để xem biểu đồ
docker run -v "%cd%/outputs:/app/outputs" student_lifestyle_ml   # Dành cho Windows CMD
# Hoặc: docker run -v "$(pwd)/outputs:/app/outputs" student_lifestyle_ml   # Dành cho Mac/Linux

```
