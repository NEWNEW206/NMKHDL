Kế hoạch Dự án Khoa học Dữ liệu: Lối sống & Kết quả học tập của Sinh viên
**Môn học:** Nhập môn Khoa học Dữ liệu (Mini-Project)  


---

## 1. Phân chia vai trò 
| STT | Thành viên | Vai trò đảm nhiệm | Chi tiết nội dung công việc | Đóng góp giá trị |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Thành viên 1** | **Project Manager & Integrator** | - Thiết kế kiến trúc dự án DS chuẩn.<br>- Điều phối luồng và tích hợp mã nguồn chính.<br>- Viết tài liệu kiểm thử cuối cùng. | **10%** |
| **2** | **Thành viên 2** | **EDA Engineer** | - Khảo sát dữ liệu thô bằng thống kê mô tả.<br>- Trực quan hóa phân phối GPA và Stress Level.<br>- Phân tích tương quan tuyến tính giữa các biến số. | **15%** |
| **3** | **Thành viên 3** | **Preprocessing Engineer** | - Làm sạch dữ liệu trùng lặp và khuyết thiếu.<br>- Mã hóa biến phân loại Stress Level thành dạng số.<br>- Chuẩn hóa thang đo các đặc trưng (Scaling). | **20%** |
| **4** | **Thành viên 4** | **Feature Engineer** | - Thiết kế các chỉ số tổng hợp thời gian hoạt động.<br>- Trích xuất 2 đặc trưng mới:<br>  * `Total_Productive_Hours`<br>  * `Free_Time` | **15%** |
| **5** | **Thành viên 5** | **Regression Engineer** | - Xây dựng mô hình hồi quy tuyến tính & phi tuyến.<br>- Dự đoán biến số liên tục (điểm **GPA**).<br>- Đánh giá bằng các chỉ số: $R^2$, MAE, RMSE. | **15%** |
| **6** | **Thành viên 6** | **Classification Engineer** | - Xây dựng mô hình phân loại đa lớp (Stress Level).<br>- Phân tích hiện tượng quá khớp (Overfitting).<br>- Đánh giá bằng các chỉ số: Accuracy, Recall, F1. | **15%** |
| **7** | **Thành viên 7** | **Data Analyst & Reporter** | - Trích xuất mức độ quan trọng đặc trưng (Feature Importance).<br>- Đúc kết ý nghĩa khoa học từ kết quả mô hình.<br>- Biên soạn báo cáo tổng hợp kết luận. | **10%** |
| | | **TỔNG CỘNG** | **Dự án Khoa học Dữ liệu hoàn chỉnh** | **100%** |

---

## 2. Cấu trúc thư mục dự kiến hoàn thành

```text
student_lifestyle_ml_project/
│
├── data/                      # Dữ liệu dự án
│   ├── raw/                   # Dữ liệu gốc (student_lifestyle_dataset.csv)
│   └── processed/             # Dữ liệu sạch sau khi tiền xử lý
│
├── notebooks/                 # Thử nghiệm phân tích và chạy nháp
│
├── outputs/                   # Sản phẩm kết quả đầu ra
│   ├── figures/               # Biểu đồ phân tích và ma trận nhầm lẫn
│   ├── regression_results.csv # Bảng kết quả so sánh dự đoán GPA
│   └── classification_results.csv # Bảng kết quả so sánh dự đoán Stress Level
│
├── src/                       # Mã nguồn phát triển chính dạng module
│   ├── __init__.py
│   ├── data_preprocessing.py  # Xử lý làm sạch và chuẩn hóa thang đo
│   ├── feature_engineering.py # Tạo lập đặc trưng mới
│   └── model_training.py      # Huấn luyện mô hình hồi quy và phân loại
│
├── requirements.txt           # Danh sách các thư viện hỗ trợ
├── main.py                    # Chương trình chính điều phối pipeline chạy tự động
└── README.md                  # Hướng dẫn chạy & Phân công công việc nhóm
```

---

## 3. Hướng dẫn Thực hiện Chi tiết cho Từng Thành viên

Để hoàn thành Mini-Project này một cách chỉnh chu nhất, từng thành viên hãy thực hiện các bước viết mã nguồn và phân tích cụ thể sau đây:

### Thành viên 1: Project Manager & Integrator
*   **Mục tiêu**: Xây dựng bộ khung dự án và kết nối các module của các thành viên khác thành một chương trình chạy hoàn chỉnh.
*   **Các bước làm**:
    1. Thiết lập các thư mục trên máy tính của cả nhóm: `data/raw/`, `data/processed/`, `notebooks/`, `outputs/figures/`, `src/`.
    2. Viết file `requirements.txt` khai báo các thư viện hỗ trợ: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
    3. Tạo file `main.py` ở thư mục gốc. Khi các thành viên khác hoàn thành module của họ trong thư mục `src/`, hãy thực hiện import các hàm preprocess, engineering, và training vào `main.py` để chạy liên mạch từ đầu đến cuối.
    4. Kiểm tra xem luồng chạy có phát sinh lỗi hay không, kiểm thử tính đúng đắn của dữ liệu đầu ra và thực hiện đóng gói dự án.

### Thành viên 2: EDA Engineer (Khám phá dữ liệu)
*   **Mục tiêu**: Phân tích đặc điểm phân phối của dữ liệu và mối tương quan giữa các biến lối sống.
*   **Các bước làm**:
    1. Viết hàm `perform_eda(df, output_dir)` để phân tích dữ liệu.
    2. Sử dụng thư viện `seaborn` và `matplotlib` để trực quan hóa:
        *   **Biểu đồ 1**: Vẽ phân phối điểm GPA bằng `sns.histplot(df['GPA'], kde=True)` để xem phân phối có bị lệch hay không.
        *   **Biểu đồ 2**: Vẽ phân phối mức độ Stress Level (Low, Moderate, High) bằng biểu đồ cột `sns.countplot` để xem các lớp có bị mất cân bằng nhãn không.
        *   **Biểu đồ 3**: Vẽ ma trận tương quan nhiệt bằng `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')` để phát hiện những thói quen nào tương quan thuận/nghịch mạnh nhất với GPA.
    3. Lưu toàn bộ các biểu đồ dưới dạng ảnh `.png` vào thư mục `outputs/figures/` với tham số độ nét `dpi=150`.

### Thành viên 3: Preprocessing Engineer (Tiền xử lý dữ liệu)
*   **Mục tiêu**: Làm sạch dữ liệu thô thành dữ liệu dạng số, chuẩn hóa thang đo chuẩn bị cho máy học.
*   **Các bước làm**:
    1. Viết các hàm xử lý trong file `src/data_preprocessing.py`.
    2. Viết hàm `clean_data(df)`: Loại bỏ cột định danh `Student_ID`. Kiểm tra và loại bỏ các dòng bị trùng lặp bằng `df.drop_duplicates()`. Nếu có giá trị khuyết thiếu, điền bằng trung vị (`median`) của thuộc tính đó.
    3. Viết hàm `encode_stress_level(df)`: Chuyển đổi thuộc tính `Stress_Level` từ dạng chữ (`Low`, `Moderate`, `High`) sang dạng số (`0`, `1`, `2`) bằng hàm `.map()` của pandas.
    4. Viết hàm `split_and_scale_data(df, target_col)`: Chia dữ liệu thành tập huấn luyện (Train) và tập kiểm thử (Test) theo tỷ lệ 80/20 bằng `train_test_split` của `scikit-learn`.
    5. Sử dụng `StandardScaler` để chuẩn hóa các cột đặc trưng số. **Chú ý quan trọng**: Chỉ gọi hàm `.fit_transform()` trên tập Train, và gọi `.transform()` trên tập Test để tránh hiện tượng rò rỉ thông tin tập kiểm thử (Data Leakage).

### Thành viên 4: Feature Engineer (Trích xuất đặc trưng)
*   **Mục tiêu**: Sáng tạo thêm các đặc trưng mới từ các biến số thời gian thô nhằm cải thiện độ chính xác cho mô hình.
*   **Các bước làm**:
    1. Viết các hàm xử lý trong file `src/feature_engineering.py`.
    2. Thiết lập đặc trưng mới **`Total_Productive_Hours`** (Tổng giờ hoạt động hiệu quả trong ngày):
       $$\text{Total Productive Hours} = \text{Study Hours Per Day} + \text{Physical Activity Hours Per Day} + \text{Extracurricular Hours Per Day}$$
    3. Thiết lập đặc trưng mới **`Free_Time`** (Thời gian rảnh rỗi thực sự còn lại trong ngày):
       $$\text{Free Time} = 24 - (\text{Tổng số giờ của tất cả các hoạt động học tập, ngoại khóa, thể thao, ngủ và giao lưu})$$
    4. Sử dụng hàm `.clip(lower=0)` của pandas đối với cột `Free_Time` để ngăn ngừa các giá trị âm phát sinh do sai lệch nhỏ trong quá trình sinh viên tự ghi nhận khảo sát thô.

### Thành viên 5: Regression Engineer (Mô hình Hồi quy dự đoán GPA)
*   **Mục tiêu**: Dự báo điểm số liên tục (GPA) và so sánh chất lượng của các mô hình hồi quy.
*   **Các bước làm**:
    1. Viết mã nguồn trong `src/model_training.py` (hàm `train_and_evaluate_regression`).
    2. Huấn luyện song song **5 mô hình hồi quy**:
        *   `LinearRegression` (Hồi quy tuyến tính)
        *   `Ridge` (Hồi quy tuyến tính có hiệu chỉnh L2)
        *   `DecisionTreeRegressor` (Cây quyết định hồi quy)
        *   `RandomForestRegressor` (Rừng ngẫu nhiên hồi quy)
        *   `SVR` (Máy vector hỗ trợ hồi quy)
    3. Dự đoán trên tập Test và đánh giá chất lượng mô hình bằng các chỉ số:
        *   $R^2$ (Hệ số xác định - đánh giá độ giải thích của mô hình).
        *   `MAE` (Sai số tuyệt đối trung bình).
        *   `RMSE` (Căn bậc hai sai số bình phương trung bình).
    4. Vẽ biểu đồ phân tán (Scatter Plot) so sánh giữa **GPA Thực tế** và **GPA Dự đoán** của từng mô hình, vẽ thêm một đường chéo $y=x$ màu đỏ làm mốc chuẩn. Lưu ảnh vào `outputs/figures/`.
    5. Xuất bảng so sánh chỉ số của 5 mô hình ra file `outputs/regression_results.csv`.

### Thành viên 6: Classification Engineer (Mô hình Phân loại dự đoán Stress Level)
*   **Mục tiêu**: Dự đoán mức độ căng thẳng của sinh viên thành 3 nhóm (0, 1, 2) và phân tích lỗi.
*   **Các bước làm**:
    1. Viết mã nguồn trong `src/model_training.py` (hàm `train_and_evaluate_classification`).
    2. Huấn luyện **5 mô hình phân loại**:
        *   `LogisticRegression` (Hồi quy Logistic)
        *   `KNeighborsClassifier` (KNN)
        *   `DecisionTreeClassifier` (Cây quyết định phân loại)
        *   `RandomForestClassifier` (Rừng ngẫu nhiên phân loại)
        *   `SVC` (Máy vector hỗ trợ phân loại)
    3. **Phân tích Overfitting**: So sánh độ chính xác (Accuracy) trên tập Train và tập Test của `DecisionTreeClassifier` trong 2 trường hợp: giới hạn độ sâu (`max_depth=5`) và không giới hạn độ sâu. Rút ra kết luận về việc cây quyết định bị "học vẹt".
    4. Đánh giá chất lượng mô hình bằng các chỉ số: `Accuracy`, `Precision`, `Recall`, `F1-Score` (tất cả tính theo trung bình `weighted` do phân loại 3 lớp).
    5. Vẽ ma trận nhầm lẫn bằng `ConfusionMatrixDisplay` của `scikit-learn` để xem mô hình thường phân loại nhầm lớp nào nhiều nhất. Lưu ảnh vào `outputs/figures/`.
    6. Xuất bảng so sánh chỉ số của 5 mô hình ra file `outputs/classification_results.csv`.

### Thành viên 7: Data Analyst & Reporter (Phân tích & Viết báo cáo)
*   **Mục tiêu**: Trích xuất các phát hiện khoa học từ kết quả mô hình máy học và viết báo cáo tổng hợp.
*   **Các bước làm**:
    1. Sử dụng thuộc tính `.feature_importances_` từ mô hình Random Forest Regressor & Classifier để xem thói quen nào (Học, chơi, ngủ hay thể thao) có ảnh hưởng lớn nhất đến GPA và mức độ Stress của sinh viên.
    2. Trực quan hóa mức độ quan trọng bằng biểu đồ cột nằm ngang `sns.barplot` và lưu hình ảnh vào `outputs/figures/`.
    3. Đọc dữ liệu từ hai file kết quả `outputs/regression_results.csv` và `outputs/classification_results.csv`. So sánh và rút ra nhận xét: Mô hình nào hoạt động tốt nhất cho bài toán dự đoán GPA? Mô hình nào hoạt động tốt nhất cho bài toán dự đoán Stress Level? Vì sao?
    4. Biên soạn file báo cáo kết quả tổng hợp `report.md` gửi giảng viên, đúc kết khuyến nghị thực tiễn: "Một sinh viên nên cân đối bao nhiêu tiếng tự học, ngủ và chơi mỗi ngày để đạt GPA tối ưu mà không bị Stress mức độ High?".
