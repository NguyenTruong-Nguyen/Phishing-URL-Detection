# Báo cáo dọn dẹp và tái cấu trúc

Ngày thực hiện: 2026-11-22

## 1. Kết quả

Luồng code và ứng dụng hiện dùng đúng ba thuật toán: Random Forest, Logistic Regression và K-Nearest Neighbors (KNN). Streamlit mặc định nạp `models/random_forest.joblib`. Không thay nhãn, dataset, chiến lược chia theo registered domain hoặc `src/feature_extraction.py`. Không cài package, không commit/push.

`models/model_v0.joblib` được bảo toàn nguyên trạng. Artifact chuẩn `models/random_forest.joblib` là bản sao byte-for-byte của model_v0; cả hai có cùng SHA-256.

## 2. File được giữ lại

- Nền tảng dùng chung: `src/preprocessing.py`, `src/feature_extraction.py`, `src/evaluation.py`, `src/predict.py`.
- Dữ liệu, nhãn và cách chia: `data/` và implementation hiện hữu trong `src/preprocessing.py`.
- Ứng dụng, test nền tảng, cấu hình, requirements và notebooks.
- Các báo cáo/biểu đồ thực nghiệm có trước được giữ nguyên. Chúng có thể nhắc tên thuật toán E1/E2, HistGradientBoosting hoặc TF-IDF/SVM như dữ liệu lịch sử; không còn trainer, artifact hay tham chiếu trong luồng ứng dụng/huấn luyện chính.

## 3. File tạo mới

- `src/models/__init__.py`
- `src/models/random_forest.py`
- `src/models/logistic_regression.py`
- `src/models/knn.py`
- `src/train/__init__.py`
- `src/train/_common.py`
- `src/train/train_random_forest.py`
- `src/train/train_logistic_regression.py`
- `src/train/train_knn.py`
- `src/train/compare_models.py`
- `tests/test_model_pipelines.py`
- `reports/model_comparison.json`
- `reports/cleanup_report.md` (báo cáo này)
- `models/random_forest.joblib` (bản sao model_v0)
- `models/logistic_regression.joblib`
- `models/knn.joblib`

## 4. File chỉnh sửa

- `README.md`: cập nhật cấu trúc, ba thuật toán, lệnh chạy và cách chia dữ liệu.
- `Phishing-URL-Detection/app/app.py`: đặt `random_forest.joblib` làm mặc định.
- `Phishing-URL-Detection/src/predict.py`: ưu tiên Random Forest, fallback chỉ còn model_v0, Logistic Regression và KNN.

`src/feature_extraction.py`, preprocessing, nhãn, dataset và các report thực nghiệm có trước không bị chỉnh sửa trong công việc này. `tests/test_predict.py` và `.gitignore` đã ở trạng thái modified trước khi bắt đầu tác vụ này; không chỉnh sửa chúng trong cleanup này.

## 5. File đã xóa

- `src/train_advanced.py` (HistGradientBoosting và TF-IDF Logistic/SVM, đồng thời chứa luồng train model cũ).
- `src/train_tfidf.py` (TF-IDF + Linear SVM).
- `src/train_baseline.py` (trainer cũ có thể ghi đè `model_v0.joblib`; chức năng builder được tách sang module mô hình mới).
- `src/robust_features.py`, `src/hostname_features.py`.
- `scripts/train_robust.py`, `scripts/train_hostname_only.py`, `scripts/diagnose_feature_differences.py`.
- `tests/test_robust_features.py`, `tests/test_hostname_features.py`.
- `models/model_v2_hostname_only.joblib` (artifact chỉ phục vụ E2; 211,841,432 bytes).

Đã quét tham chiếu trước khi xóa. Các báo cáo cũ có nhắc tới những file/artifact này được giữ làm lịch sử; không còn import hay dependency hoạt động trong `src`, `app`, `tests`, `scripts` hoặc README.

## 6. Artifact model hiện có

| Artifact | Dung lượng | SHA-256 | Ghi chú |
|---|---:|---|---|
| `models/model_v0.joblib` | 35,860,726 bytes | `099DB72DDD27948460E69DC66B35D60F5D242A963893A66EC40D688AC439C8EA` | Model gốc, không thay đổi |
| `models/random_forest.joblib` | 35,860,726 bytes | `099DB72DDD27948460E69DC66B35D60F5D242A963893A66EC40D688AC439C8EA` | Bản sao byte-for-byte của model_v0 |
| `models/logistic_regression.joblib` | 4,633 bytes | `1A1B09E01A32FC9076A5AF7F006F1675A1FD887D7D960DC095EB4BAC9D173EEB` | Fit trên Fit, reload thành công |
| `models/knn.joblib` | 32,006,165 bytes | `A45D1F1A79FFD99E2C924B51997D3A1643E6F1BA62F60D8F9E6AA99CD7064062` | Fit trên Fit, reload thành công |

Model_v0 trước/sau đều 35,860,726 bytes và SHA-256 giữ nguyên. Dataset PhiUSIIL giữ nguyên: 56,854,345 bytes, SHA-256 `A236549CD369CD80BD478FF8E1779CBF44C58D5C3F79F7A51A1ADBED7D06D1C6`.

## 7. Chia dữ liệu và so sánh

Các trainer mới dùng cùng quy tắc chia hiện tại: outer registered-domain train/test, seed 42; inner registered-domain Fit/Validation, seed 43. Kích thước lần chạy: Fit 160,009, Validation 28,203, Test 47,158. Giao registered domain giữa các partition bằng 0. Trainer chỉ fit trên Fit; model comparison chỉ dùng Validation, không dùng Test để lựa chọn.

Validation F1 trong `reports/model_comparison.json`:

| Mô hình | Precision | Recall | F1 | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.999587 | 0.995149 | 0.997363 | 5 | 59 |
| Logistic Regression | 0.999917 | 0.992847 | 0.996370 | 1 | 87 |
| KNN | 0.999752 | 0.994327 | 0.997032 | 3 | 69 |

Random Forest validation metric được lấy từ `reports/baseline_metrics.json` (RF fit-only cùng split); không huấn luyện lại RF. LR fit khoảng 6.77 giây. KNN fit khoảng 5.35 giây; dự đoán toàn Validation trung bình khoảng 0.421 ms/URL khi đánh giá, và benchmark riêng 1,000 URL validation khoảng 0.262 ms/URL. Đây là số đo môi trường hiện tại, không phải cam kết hiệu năng trên máy khác.

## 8. Kiểm tra

- Import thành công toàn bộ 14 module dưới `src`.
- Nạp cả ba artifact mới: mỗi pipeline có đúng 24 `FEATURE_COLUMNS`, lớp `[0, 1]`, và classifier tương ứng RF/LR/KNN.
- `src.predict.find_model_path()` mặc định trả về `models/random_forest.joblib`; kiểm tra inference local thành công.
- Test mới kiểm tra ba pipeline dùng chung schema và save/load round-trip.
- Pytest (final run): `22 passed in 4.81s`.
- Quét `src`, `app`, `tests`, `scripts`, README: không còn import/tham chiếu hoạt động tới HistGradientBoosting, TF-IDF/SVM, E1/E2 trainer/transformer hoặc artifact E1/E2.
- `git diff --check`: không có whitespace errors; Git chỉ hiển thị cảnh báo line ending LF/CRLF ở file Windows.

## 9. Vấn đề còn lại

- Kiểm tra inference local với `https://example.com` vẫn cho Random Forest `Phishing`, `probability_phishing=1.0`. Đây là hiện tượng đã quan sát trước cleanup và vẫn chưa được giải quyết; cleanup không hiệu chỉnh hoặc thay model_v0.
- Các report lịch sử được giữ nguyên nên nội dung của chúng còn đề cập các thuật toán/experiment đã gỡ khỏi luồng chạy.
- Artifact KNN lớn hơn LR đáng kể và dự đoán chậm hơn hai mô hình tuyến tính/RF trong số đo validation hiện tại.

## 10. Xác nhận model_v0

**Được bảo toàn.** Không ghi đè, không xóa và không huấn luyện lại `models/model_v0.joblib`. SHA-256 trước/sau: `099DB72DDD27948460E69DC66B35D60F5D242A963893A66EC40D688AC439C8EA`; dung lượng trước/sau: 35,860,726 bytes.
