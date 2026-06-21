# Báo cáo So sánh Mô hình: Dự đoán GPA và Mức độ Stress

## 1. Mục tiêu

Lối sống của sinh viên — thời gian học tập, ngủ nghỉ, hoạt động ngoại khóa, giao tiếp xã hội và rèn luyện thể chất — vốn được cho là có ảnh hưởng trực tiếp đến kết quả học tập và trạng thái tâm lý. Báo cáo này nhằm trả lời câu hỏi: **thói quen sinh hoạt nào thực sự tác động mạnh nhất đến GPA và việc xác định mức độ Stress của sinh viên, và mức độ tác động đó lớn đến đâu?**

Để trả lời, nhóm xây dựng và so sánh nhiều mô hình học máy trên cùng một bộ dữ liệu thói quen sinh hoạt, nhằm:
- Xác định mô hình dự đoán GPA và mô hình phân loại mức độ Stress hiệu quả nhất.
- Dựa trên mô hình đó, định lượng mức độ ảnh hưởng của từng thói quen (Học, Ngoại khóa, Ngủ, Giao tiếp, Vận động) thông qua chỉ số `feature_importances_`.

Kết quả kỳ vọng sẽ cung cấp cơ sở định lượng giúp sinh viên điều chỉnh lối sống hợp lý hơn để cải thiện thành tích học tập và giảm áp lực tâm lý.

---

## 2. Kết quả Regression (dự đoán GPA)

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **LinearRegression** | **0.5494** | **0.1639** | **0.2051** |
| Ridge | 0.5493 | 0.1639 | 0.2051 |
| DecisionTreeRegressor | 0.5342 | 0.1671 | 0.2085 |
| SVR | 0.5005 | 0.1713 | 0.2160 |
| RandomForestRegressor | 0.4623 | 0.1792 | 0.2240 |

### Nhận xét

- **Mô hình tốt nhất: `LinearRegression`** — R² cao nhất, MAE và RMSE thấp nhất trong nhóm.
- `Ridge` cho kết quả gần như giống `LinearRegression` (do chỉ thêm regularization nhẹ), xác nhận quan hệ giữa các biến thói quen và GPA mang tính **tuyến tính**.
- Các mô hình phức tạp hơn (`DecisionTreeRegressor`, `RandomForestRegressor`) cho kết quả **kém hơn** mô hình tuyến tính — dấu hiệu overfitting trên nhiễu khi quan hệ thực chất giữa input và GPA đơn giản, không cần mô hình phi tuyến phức tạp.
- R² toàn bộ chỉ đạt khoảng 0.50–0.55, cho thấy 5 biến thói quen hiện tại chỉ giải thích được khoảng một nửa phương sai của GPA — phần còn lại có thể do nhiễu hoặc thiếu các yếu tố quan trọng khác (năng lực nền, phương pháp học, môn học...).

---

## 3. Kết quả Classification (phân loại mức độ Stress)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| LogisticRegression | 0.855 | 0.854 | 0.855 | 0.854 |
| KNeighborsClassifier | 0.880 | 0.879 | 0.880 | 0.879 |
| DecisionTreeClassifier | **1.000** | **1.000** | **1.000** | **1.000** |
| RandomForestClassifier | **1.000** | **1.000** | **1.000** | **1.000** |
| SVC | 0.970 | 0.970 | 0.970 | 0.970 |

### Nhận xét

- Theo số liệu thuần túy, `DecisionTreeClassifier` và `RandomForestClassifier` đạt **accuracy tuyệt đối 1.0** — cao nhất trong bảng.
- **Tuy nhiên, kết quả này cần được xem xét cẩn trọng** trước khi kết luận là "tốt nhất". Độ chính xác 100% trên một bài toán dự đoán hành vi con người (stress) từ vài biến sinh hoạt là **bất thường**, thường là dấu hiệu của:
  - **Data leakage**: nhãn `Stress_Level` có thể được gán dựa trực tiếp theo công thức/ngưỡng cố định của chính các biến input (ví dụ: Study > X & Sleep < Y → "High"), khiến cây quyết định "học vẹt" được đúng luật sinh dữ liệu thay vì học một mối quan hệ tổng quát.
  - Tập test quá nhỏ hoặc không độc lập với tập train.
- Nếu loại trừ 2 mô hình có kết quả đáng nghi này, **`SVC` (97%)** là lựa chọn đáng tin cậy nhất — hiệu suất cao nhưng vẫn nằm trong phạm vi hợp lý, không tuyệt đối.

### Kiểm tra Data Leakage trong code

Nhóm đã rà soát `data_preprocessing.py` và `feature_engineering.py`: pipeline tiền xử lý **không phải nguyên nhân** gây ra kết quả này — việc chia train/test diễn ra trước khi scale/impute (đúng kỹ thuật, không leak qua code), và các feature mới đều được tính từ dữ liệu thô, không sử dụng target. Nguyên nhân nằm ở **cách bộ dữ liệu được sinh ra** (rule-based), không phải lỗi xử lý — xem phát hiện dưới đây.

### Phát hiện: `Stress_Level` mang tính rule-based

Để kiểm chứng nghi vấn trên, nhóm đã thử giới hạn một cây quyết định chỉ được rẽ nhánh tối đa **3 lần** (`max_depth=3`) và huấn luyện trên toàn bộ dữ liệu. Mô hình tự học ra một luật rất đơn giản, đạt **accuracy = 1.0** ngay với luật này:

```
Study_Hours <= 8.05:
    Sleep_Hours <= 5.95   → High
    Sleep_Hours > 5.95:
        Study_Hours <= 5.95 → Low
        Study_Hours > 5.95  → Moderate
Study_Hours > 8.05 → High
```

Điều này cho thấy `Stress_Level` trong bộ dữ liệu được **xác định gần như tuyệt đối chỉ qua 2 ngưỡng đơn giản** trên `Study_Hours_Per_Day` và `Sleep_Hours_Per_Day` — đặc trưng của một bộ dữ liệu **tổng hợp (synthetic)**, trong đó cột `Stress_Level` nhiều khả năng được người tạo dữ liệu **tính ra từ công thức ngưỡng** trên các cột giờ sinh hoạt, chứ không phải nhãn độc lập thu thập từ khảo sát thực tế. Vì vậy nó không phản ánh đầy đủ độ phức tạp của stress trong đời thực (vốn còn chịu ảnh hưởng bởi nhiều yếu tố tâm lý, xã hội khác không có trong dữ liệu này).

**Hệ quả:** Kết quả accuracy = 1.0 của `DecisionTreeClassifier`/`RandomForestClassifier` phản ánh đúng cấu trúc của dữ liệu này, nhưng **không nên ngoại suy** rằng mô hình sẽ đạt hiệu suất tương tự trên dữ liệu thực tế ngoài tập này.

---

## 4. Kết luận tổng quan về mô hình

| Bài toán | Mô hình tốt nhất | Lý do |
|---|---|---|
| Dự đoán GPA | `LinearRegression` | Quan hệ dữ liệu mang tính tuyến tính; mô hình đơn giản khái quát tốt hơn mô hình phức tạp |
| Phân loại mức độ Stress | `SVC` *(đáng tin cậy nhất)* | Hiệu suất cao (97%) và hợp lý về thống kê; kết quả 100% của Decision Tree/Random Forest là do `Stress_Level` mang tính rule-based (đã kiểm chứng ở mục 3), không phản ánh khả năng tổng quát hóa thực sự |

---

## 5. Mức độ ảnh hưởng của từng thói quen (`feature_importances_`)

### 5.1. Đối với GPA

| Feature | Importance |
|---|---|
| Study | 0.599 |
| Sleep | 0.105 |
| Physical_Activity | 0.101 |
| Social | 0.099 |
| Extracurricular | 0.096 |

![Feature importance đối với GPA](https://github.com/thienphuc0206/image-storage/blob/main/RandomForestRegressor_feature_importance.png?raw=true)

### 5.2. Đối với Stress_Level

| Feature | Importance |
|---|---|
| Study | 0.679 |
| Sleep | 0.231 |
| Physical_Activity | 0.050 |
| Social | 0.020 |
| Extracurricular | 0.010 |

![Feature importance đối với Stress_Level](https://github.com/thienphuc0206/image-storage/blob/main/RandomForestClassifier_feature_importance.png?raw=true)

### Nhận xét

- Cả 2 bài toán đều cho thấy `Study` là yếu tố ảnh hưởng vượt trội (~0.60 với GPA, ~0.68 với Stress).
- Với GPA, 4 yếu tố còn lại có importance khá đồng đều (~0.10), không có yếu tố thứ 2 nổi bật rõ.
- Với Stress, `Sleep` nổi lên là yếu tố quan trọng thứ 2 (0.231), còn 3 yếu tố Physical_Activity, Social, Extracurricular đều rất thấp (≤ 0.05).
- Kết hợp với phát hiện ở mục 3 (Decision Tree `max_depth=3` chỉ cần `Study` và `Sleep` để đạt accuracy = 1.0), có thể khẳng định: **`Study` và `Sleep` là 2 biến cần ưu tiên phân tích sâu** để trả lời câu hỏi cân đối thời gian, 3 biến còn lại đóng vai trò phụ.

---

## 6. Cơ sở để trả lời câu hỏi: "Nên cân đối bao nhiêu giờ tự học, ngủ, chơi mỗi ngày để GPA tối ưu mà không bị Stress mức độ High?"

Câu hỏi này cần một **con số/khoảng giá trị cụ thể**, nên không thể chỉ dựa vào `feature_importances_` (mục 5) — chỉ số đó cho biết mức độ ảnh hưởng tương đối, không cho biết nên chọn giá trị bao nhiêu. Để trả lời, nhóm kết hợp 3 cơ sở:

| Cơ sở | Trả lời được gì | Hạn chế |
|---|---|---|
| **`feature_importances_`** (mục 5) | Xác định *biến nào* cần ưu tiên phân tích sâu (Study, Sleep nổi bật nhất) | Không cho biết chiều hướng tác động hay giá trị cụ thể |
| **Luật ngưỡng từ Decision Tree** (`max_depth=3`, mục 3) | Đưa ra *ngưỡng số cụ thể* phân biệt Stress High/Moderate/Low (Study ≤ 8.05, Sleep > 5.95) | Ngưỡng phản ánh cách dữ liệu được tạo ra, không chắc đúng với thực tế |
| **Phân tích nhóm điều kiện** (lọc sinh viên GPA cao **và** không Stress cao, xem phân vị 25–75%) | Đưa ra *khoảng giá trị thực tế* thỏa đồng thời cả 2 mục tiêu | Chỉ mô tả tương quan trong dữ liệu hiện có, không chứng minh quan hệ nhân quả |

### Minh chứng trực quan

*Biểu đồ dưới đây chỉ tập trung vào `Study_Hours` và `Sleep_Hours` — không phải bỏ qua 3 biến còn lại, mà vì cả `feature_importances_` (mục 5) và luật của Decision Tree (mục 3) đều cho thấy đây là 2 biến duy nhất cần thiết để phân biệt mức Stress trong dữ liệu này.*

![Phân tích vùng tối ưu Study/Sleep và GPA](https://github.com/thienphuc0206/image-storage/blob/main/optimal_zone_analysis.png?raw=true)

**Biểu đồ trái** — phân bố Study vs Sleep, tô màu theo `Stress_Level`: vùng xanh (Study ≤ 8.05 & Sleep > 5.95) tập trung phần lớn điểm Low/Moderate (xanh lá/cam); ngoài vùng này chủ yếu là High (đỏ) — xác nhận trực quan ngưỡng mà Decision Tree đã suy ra.

**Biểu đồ phải** — GPA theo khoảng số giờ tự học: GPA **tăng dần đều** khi học nhiều hơn, kể cả vượt 8 giờ/ngày. Đây chính là điểm mâu thuẫn cốt lõi của câu hỏi: nhóm học 8–10h/ngày có GPA cao nhất nhưng đồng thời cũng là nhóm rơi vào Stress "High" ở biểu đồ trái — chứng minh **GPA tối ưu** và **tránh Stress cao** là hai mục tiêu kéo theo hai hướng ngược nhau, nên khuyến nghị phải chọn điểm cân bằng (~7–7.8h) chứ không phải học tối đa.

### Vậy 3 biến còn lại (Extracurricular, Social, Physical_Activity) nên phân bổ thế nào?

Vì importance thấp, không có ngưỡng "nguy hiểm" để suy ra như Study/Sleep — biến đổi trong biên độ hiện có của 3 hoạt động này không tạo khác biệt rõ rệt lên GPA hay Stress. Khuyến nghị cho nhóm này dựa trên 2 căn cứ:

1. **Khoảng giá trị thực tế của nhóm "thành công"**: quan sát trực tiếp 3 biến này trong nhóm sinh viên đạt cả 2 mục tiêu (xem mục 7), thay vì tìm quan hệ nhân quả.
2. **Ràng buộc quỹ thời gian**: tổng thời gian trong ngày cố định ở 24 giờ. Sau khi dành ~7–7.8h cho Study và ~7–8.5h cho Sleep (tổng ~14.5–16.3h), quỹ thời gian còn lại (~7.7–9.5h) mới là giới hạn thực tế để chia cho Extracurricular + Social + Physical_Activity. Vì 3 hoạt động này có importance thấp tương đương nhau, sinh viên có thể **đánh đổi linh hoạt giữa chúng** mà không ảnh hưởng đáng kể đến GPA hoặc Stress — miễn tổng không vượt quỹ thời gian còn lại.

---

## 7. Khuyến nghị thực tiễn: Nên cân đối thời gian sinh hoạt thế nào?

Dựa trên nhóm sinh viên đạt **GPA thuộc top 25% cao nhất** (≥ 3.30) **và không rơi vào mức Stress "High"** trong dữ liệu (70/2000 sinh viên), khoảng thời gian sinh hoạt phổ biến (theo phân vị 25%–75%) như sau:

| Hoạt động | Khoảng khuyến nghị (giờ/ngày) |
|---|---|
| **Tự học** | 7.1 – 7.8 |
| **Ngủ** | 6.8 – 8.5 |
| **Giao tiếp xã hội** | 1.4 – 4.2 |
| **Ngoại khóa** | 0.9 – 2.8 |
| **Vận động thể chất** | 2.2 – 6.0 |

**Hai ngưỡng quan trọng nhất** (dựa trên luật Decision Tree ở mục 3 và minh chứng trực quan ở mục 6):
- **Study_Hours ≤ 8.05 giờ/ngày** — học vượt ngưỡng này gắn liền với nhóm Stress "High", bất kể ngủ bao nhiêu.
- **Sleep_Hours > 5.95 giờ/ngày** (thực tế nên ≥ 6.8–7 giờ để an toàn) — ngủ dưới ngưỡng này, dù học ít, vẫn rơi vào nhóm Stress "High".

### Tóm tắt khuyến nghị

> Một sinh viên nên **tự học khoảng 7–7.8 giờ/ngày**, **ngủ đủ 7–8.5 giờ/ngày**, dành khoảng **1–4 giờ giao tiếp xã hội**, **1–3 giờ hoạt động ngoại khóa** và **2–6 giờ vận động thể chất** mỗi ngày. Đây là vùng cân đối mà dữ liệu cho thấy sinh viên vừa đạt GPA cao (≈ 3.4) vừa không rơi vào mức Stress cao — quan trọng nhất là **không học quá 8 giờ/ngày và không ngủ dưới 6 giờ/ngày**, vì đây là hai yếu tố quyết định trực tiếp đến việc bị xếp vào nhóm Stress "High". Ba yếu tố còn lại, do mức độ ảnh hưởng thấp (theo `feature_importances_` ở mục 5), có thể điều chỉnh linh hoạt theo sở thích cá nhân, miễn không vượt quỹ thời gian còn lại trong ngày.

*Lưu ý: khuyến nghị này rút ra từ một bộ dữ liệu mang tính tổng hợp (xem mục 3), nên mang giá trị tham khảo về xu hướng hơn là một quy tắc tuyệt đối áp dụng cho mọi sinh viên trong thực tế.*

---

## Phụ lục: Tệp mã nguồn liên quan

| Phần report | File code tương ứng |
|---|---|
| Tính `feature_importances_` & vẽ `barplot` (mục 5) | `src/classification/importance_score.py` (hàm `calculate_importance_score`, `barplot_feature_importance`) → xuất ra `outputs/*_importance.csv`, `outputs/figures/*_feature_importance.png` |
| Kiểm chứng rule-based & vùng tối ưu Study/Sleep (mục 3, 6) | `notebooks/verify_stress_rule_based.py`, `notebooks/eda_study_sleep_zone.py` → xuất ra `outputs/figures/optimal_zone_analysis.png` |
