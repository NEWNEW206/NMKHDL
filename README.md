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

## 2. Cấu trúc thư mục thực tế

```text
NMKHDL/
│
├── data/                      # Dữ liệu dự án
│   └── raw/                   # Dữ liệu gốc (student_lifestyle_dataset.csv)
│
├── outputs/                   # Sản phẩm kết quả đầu ra
│   ├── figures/               # Biểu đồ phân tích (Learning Curve, Ma trận nhầm lẫn)
│   └── classification_results.csv # Bảng kết quả mô hình
│
├── models/                    # Lưu trữ các mô hình đã huấn luyện (best_model.pkl)
│
├── src/                       # Mã nguồn phát triển chính dạng module
│   ├── data_preprocessing.py  # Xử lý làm sạch và chuẩn hóa thang đo
│   ├── feature_engineering.py # Tạo lập đặc trưng mới
│   └── classification/        # Module phân loại chính (OOP)
│       ├── models.py          # Các mô hình cơ sở
│       ├── optimizer.py       # Tối ưu siêu tham số bằng Optuna
│       ├── evaluate.py        # Vẽ Learning Curve, Confusion Matrix
│       └── train.py           # Quản lý luồng huấn luyện phân loại
│
├── main.py                    # Chương trình chính điều phối pipeline chạy tự động
├── run_pipeline.bat           # File script tự động cài đặt và chạy (Windows)
├── run_pipeline.sh            # File script tự động cài đặt và chạy (Linux/Mac)
├── Dockerfile                 # File cấu hình môi trường Docker
├── requirements.txt           # Danh sách các thư viện hỗ trợ
├── README.md                  # Phân công công việc nhóm và hướng dẫn chi tiết
└── TeamMemberRoles.md         # Hướng dẫn chạy dự án (Running Guide)
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
docker run student_lifestyle_ml
```
