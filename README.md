# Phishing URL Detection

Dự án phân loại URL thành `Legitimate` (0) hoặc `Phishing` (1) từ 24 đặc trưng lexical/structural. Ứng dụng chỉ phân tích chuỗi URL: không truy cập website đích, không gửi HTTP request, không tải nội dung website và không tra cứu DNS.

## Ba thuật toán chính thức

- **Random Forest**: mô hình hiện có và mặc định của ứng dụng, dùng artifact `models/model_v0.joblib`.
- **Logistic Regression**: thuật toán chính thức, có pipeline và trainer riêng (thành viên 2).
- **K-Nearest Neighbors (KNN)**: thuật toán chính thức, có pipeline và trainer riêng (thành viên 3); số láng giềng có thể đặt bằng `--neighbors`.

Cả ba pipeline dùng chung `URLFeatureTransformer` và cùng 24 đặc trưng URL. Logistic Regression và KNN chuẩn hóa các đặc trưng bằng `StandardScaler` trong pipeline. Preprocessing kiểm tra URL/nhãn, chuẩn hóa nhãn PhiUSIIL và chia dữ liệu theo registered domain để hạn chế domain leakage. Việc trích xuất đặc trưng và chia domain chỉ xử lý dữ liệu cục bộ, không gọi dịch vụ mạng.

`models/model_v0.joblib` là artifact Random Forest hiện có và vẫn là mặc định của ứng dụng. Các trainer tạo artifact ứng viên riêng, từ chối ghi đè artifact đã tồn tại và đánh giá trên Validation. Chưa có huấn luyện mới hoặc so sánh mới của ba mô hình trong lần cập nhật tài liệu này; các báo cáo cũ được lưu trong `reports/` và `reports/archive/`.

## Cài đặt môi trường

Các lệnh dưới đây giả định đang ở project root `Phishing-URL-Detection` và môi trường ảo `.venv` nằm tại repository root:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy ứng dụng Streamlit

Từ project root `Phishing-URL-Detection`:

```powershell
..\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Ứng dụng mặc định nạp `models/model_v0.joblib`. Trường model path có thể để trống để dùng artifact mặc định.

## Dataset và preprocessing

Dự án dùng **PhiUSIIL Phishing URL Dataset** từ UCI Machine Learning Repository:

- Trang chính thức: <https://archive.ics.uci.edu/dataset/967/phiusil+phishing+url+dataset>
- Dataset ID: `967`
- Số bản ghi: `235,795`
- License: `CC BY 4.0`

### Tải và cài đặt dataset

1. Tải dataset từ trang UCI chính thức ở trên.
2. Giải nén archive nếu cần.
3. Đặt CSV tại `Phishing-URL-Detection/data/raw/`.
4. Dùng tên file `PhiUSIIL_Phishing_URL_Dataset.csv`.
5. Từ project root, trainer mặc định đọc `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`. Kiểm tra tên cột phải khớp `src/preprocessing.py`: URL là `URL`, nhãn là `label`.
6. `src/preprocessing.py` ánh xạ nhãn nguồn PhiUSIIL `0` thành `Phishing` (1), `1` thành `Legitimate` (0); kiểm tra mapping trước khi dùng dataset khác.

CSV dữ liệu không được tải lên GitHub; các CSV trong `data/raw/` và `data/external/` bị loại khỏi Git bằng `.gitignore`.

## Huấn luyện và so sánh

Các lệnh chạy từ project root `Phishing-URL-Detection`. Mỗi trainer cần dataset ở vị trí mặc định ở trên; có thể xem `-h` để biết các tham số tùy chọn.

```powershell
..\.venv\Scripts\python.exe -m src.train.train_random_forest
..\.venv\Scripts\python.exe -m src.train.train_logistic_regression
..\.venv\Scripts\python.exe -m src.train.train_knn --neighbors 5
```

Random Forest trainer mặc định tạo `models/random_forest_candidate.joblib`; Logistic Regression và KNN tạo `models/logistic_regression.joblib` và `models/knn.joblib`. Các lệnh tạo artifact mới và không ghi đè tệp đã có. KNN `--neighbors` mặc định là `5`.

Sau khi artifacts LR/KNN đã được tạo, lệnh dưới đây so sánh chúng với kết quả Validation Random Forest từ `reports/baseline_metrics.json`; nó không dùng Test để chọn mô hình và tạo `reports/model_comparison.json` nếu tệp đích chưa tồn tại:

```powershell
..\.venv\Scripts\python.exe -m src.train.compare_models
```

Chưa chạy các lệnh huấn luyện hay so sánh trong lần hoàn thiện README này. Báo cáo lịch sử của các thử nghiệm đã nghỉ hưu, nếu có, chỉ là tài liệu lưu trữ, không phải thuật toán chính thức hiện tại.

## Kiểm thử

Từ project root `Phishing-URL-Detection`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

## 24 đặc trưng URL

Tất cả đặc trưng được tính từ URL string: `url_length`, `hostname_length`, `count_dots`, `count_hyphens`, `count_underscores`, `count_question_marks`, `count_equal_signs`, `count_at_signs`, `count_ampersands`, `count_percent_signs`, `count_slashes`, `count_digits`, `digit_ratio`, `special_char_count`, `has_https`, `is_ip_hostname`, `has_www`, `path_length`, `path_segment_count`, `query_length`, `query_parameter_count`, `subdomain_count`, `tld_length`, `character_entropy`.

## Cấu trúc mã nguồn

```text
Phishing-URL-Detection/
├── app/app.py
├── data/{raw,processed,external}/
├── models/model_v0.joblib
├── notebooks/
├── reports/                 # gồm báo cáo hiện tại và archive lịch sử
├── scripts/                 # tiện ích audit và phân tích dữ liệu
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   ├── predict.py
│   ├── models/{random_forest,logistic_regression,knn}.py
│   └── train/{train_random_forest,train_logistic_regression,train_knn,compare_models}.py
├── tests/
├── requirements.txt
└── requirements-lock.txt
```

## Phân công nhóm

- **Thành viên 1 (trưởng nhóm)**: Random Forest.
- **Thành viên 2**: Logistic Regression.
- **Thành viên 3**: K-Nearest Neighbors (KNN).