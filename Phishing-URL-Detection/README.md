# Phishing URL Detection

Hệ thống phân loại URL thành `Phishing` hoặc `Legitimate` bằng Machine Learning. Dự án chỉ phân tích chuỗi URL, **không truy cập website đích**, không tải HTML và không thực hiện HTTP request tới URL người dùng nhập.

## 1. Yêu cầu
- Python 3.10+
- pip

## 2. Cài đặt
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu
Đặt file CSV vào `data/raw/`. Mặc định các lệnh huấn luyện tìm `data/raw/dataset.csv`.

CSV cần có:
- Một cột URL (mặc định `URL`)
- Một cột nhãn (mặc định `label`)

Có thể đổi tên cột bằng tham số CLI. Nhãn được hỗ trợ: `1/phishing/malicious/bad` và `0/legitimate/benign/good`. **Kiểm tra nhãn gốc của dataset trước khi huấn luyện**; không mặc định rằng số 1 trong mọi dataset đều là phishing.

Ví dụ:
```bash
python -m src.train_baseline --data data/raw/dataset.csv --url-col URL --label-col label
python -m src.train_tfidf --data data/raw/dataset.csv --url-col URL --label-col label
```

Nếu dataset PhiUSIIL có tên cột hoặc mã nhãn khác, truyền đúng tên cột và điều chỉnh ánh xạ trong `src/preprocessing.py` sau khi kiểm tra tài liệu dataset. Không đưa các cột đặc trưng được tính sẵn của dataset vào mô hình URL-only nếu mục tiêu là tự trích xuất đặc trưng từ URL.

## 4. Chạy ứng dụng
```bash
streamlit run app/app.py
```
Ứng dụng dùng `models/final_model.joblib` nếu có, nếu không sẽ dùng `models/model_v0.joblib`. Hãy huấn luyện ít nhất một mô hình trước.

## 5. Cấu trúc
```text
Phishing-URL-Detection/
├── data/raw/                 # Dataset gốc (không commit dữ liệu lớn)
├── data/processed/           # Dữ liệu đã xử lý
├── notebooks/                # EDA và thử nghiệm
├── src/
│   ├── preprocessing.py      # Đọc CSV, chuẩn hóa nhãn và chia dữ liệu
│   ├── feature_extraction.py  # 24 đặc trưng URL
│   ├── train_baseline.py      # Logistic Regression + Random Forest + HistGradientBoosting
│   ├── train_tfidf.py         # TF-IDF char n-grams + Linear SVM
│   ├── evaluation.py          # Metrics và confusion matrix
│   └── predict.py             # Nạp model và dự đoán
├── models/                    # Model đã huấn luyện
├── app/app.py                 # Streamlit UI
├── tests/                     # Unit tests
├── reports/figures/           # Biểu đồ báo cáo
└── requirements.txt
```

## 6. 24 đặc trưng URL
Các đặc trưng được tính từ chuỗi URL, không truy cập đích: độ dài URL, độ dài hostname, số dấu chấm, dấu gạch ngang, gạch dưới, dấu hỏi, dấu bằng, ký hiệu @, &, %, số slash, số chữ số, tỷ lệ chữ số, số ký tự đặc biệt, HTTPS, IP literal, hostname có `www`, độ dài path, số segment path, độ dài query, số tham số query, số subdomain, độ dài TLD và entropy ký tự.

Đây là bộ đặc trưng khởi đầu phục vụ đồ án; cần mô tả rõ trong báo cáo và có thể điều chỉnh sau EDA.

## 7. Lưu ý đánh giá khoa học
- Tách train/test trước khi fit vectorizer hoặc scaler; pipeline đảm bảo TF-IDF chỉ fit trên train.
- Dữ liệu URL có thể trùng hoặc cùng domain ở nhiều tập. Với kết quả báo cáo cuối, nên kiểm tra trùng lặp và cân nhắc chia theo registered domain để giảm leakage.
- Accuracy không đủ; báo cáo Precision, Recall, F1, ROC-AUC (nếu có score), confusion matrix và thời gian dự đoán.
- Điểm xác suất không đồng nghĩa với xác suất rủi ro đã được hiệu chuẩn.
- Mô hình chỉ hỗ trợ sàng lọc, không bảo đảm phát hiện mọi URL lừa đảo.

## 8. Phân công gợi ý
- Thành viên 1 (trưởng nhóm): TF-IDF, mô hình nâng cao, tích hợp và quản lý phiên bản model.
- Thành viên 2: kiểm tra dataset, EDA, 24 đặc trưng, baseline và đánh giá.
- Thành viên 3: Streamlit, kiểm thử giao diện, demo và README.

## 9. Lệnh kiểm thử
```bash
pytest -q
```
