# Phase 4A — E1 diagnosis và E2 hostname-only

## Tóm tắt kết quả

E1 được tái tạo trên đúng X_fit của split domain, cùng RobustURLFeatureTransformer, SimpleImputer median và Random Forest 250 cây. F1 validation tái tạo là 0.8095216254643857, khớp kết quả Phase 4 trước đó trong sai số 1e-12. E2 được fit chỉ trên X_fit, lưu tại models/model_v2_hostname_only.joblib và reload/predict thành công.

Trên cùng PhiUSIIL validation (28,203 mẫu), E0 đạt F1 0.998023, E1 0.809522 và E2 0.749201. E2 không thay thế được E0. E2 có ít false positive hơn E1/E0 trên bộ manual 25 URL, nhưng manual set quá nhỏ và được tuyển chọn nên không phải benchmark độc lập.

## E1 khác E0 thế nào?

- E0 dùng 24 feature từ URL gốc; E1 có 22 cột và bỏ has_https, has_www.
- 22 tên cột còn chung, nhưng chỉ 20 được xem là cùng định nghĩa để so sánh trực tiếp. subdomain_count và tld_length được báo riêng vì E0 dùng cách tách label đơn giản, còn E1 dùng PSL-aware extraction.
- E1 loại scheme, một tiền tố www. và slash gốc ở URL chỉ có hostname. Vì vậy các phép đếm, độ dài, special_char_count và character_entropy được tính trên chuỗi đã chuẩn hóa, khác E0.
- Trên 16 ví dụ tĩnh, mỗi URL có từ 0 đến 7 giá trị khác nhau trong 20 feature so sánh trực tiếp. Với https://www.example.com, khác biệt gồm url_length, hostname_length, count_dots, count_slashes, special_char_count và character_entropy. Với example.com không scheme/www không có khác biệt ở nhóm 20 feature đó.
- Ở daa.uit.edu.vn, E0 heuristic đếm subdomain theo số label trừ hai; E1 dùng suffix PSL nên định nghĩa khác. Chênh lệch feature values chứng minh hai pipeline tạo đầu vào khác nhau, nhưng chưa chứng minh riêng thay đổi nào gây giảm F1.

## E2 architecture

E2 tạo 12 feature chỉ từ hostname đã parse bằng urllib.parse.urlsplit: hostname_length, count_dots, count_hyphens, count_underscores, count_digits, digit_ratio, special_char_count, character_entropy, subdomain_count, tld_length, has_ip_address và has_punycode.

Hostname được lowercase, chuyển IDN sang ASCII/IDNA, bỏ một dấu chấm cuối và một tiền tố www. Scheme, path, query và fragment không tham gia vector. PSL extractor dùng suffix_list_urls=() nên không tải suffix list. Input rỗng tạo vector 0; URL malformed không bị diễn giải lại âm thầm và sẽ raise ValueError.

Pipeline là HostnameFeatureTransformer → SimpleImputer(median) → RandomForestClassifier. RF dùng cùng cấu hình E0: 250 cây, sqrt, max_depth=None, min_samples_leaf=1, min_samples_split=2, bootstrap, balanced_subsample, random_state=42.

## So sánh trên validation

| Model | Accuracy | Precision phishing | Recall phishing | F1 phishing | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.998298 | 1.000000 | 0.996054 | 0.998023 | 16040 | 0 | 48 | 12115 |
| E1 | 0.852746 | 0.915456 | 0.725561 | 0.809522 | 15225 | 815 | 3338 | 8825 |
| E2 | 0.810800 | 0.874575 | 0.655266 | 0.749201 | 14897 | 1143 | 4193 | 7970 |

Split được tái tạo bằng registered_domain: outer test_size=0.2, seed=42; validation test_size=0.15, seed=43. Fit/validation/test registered-domain overlap đều bằng 0. Test không được dùng để fit, lựa chọn hoặc báo cáo metric trong Phase 4A.

## Path/query subgroup analysis

Các nhóm dưới đây loại trừ lẫn nhau: hostname_only (path rỗng hoặc /, không query), path_only, query_only và path_and_query. Fragment không làm đổi nhóm. Trong validation này, ba nhóm có path/query đều chỉ có nhãn Phishing; vì vậy các confusion matrix chỉ có support ở lớp 1 nên không đo được hành vi false positive trên Legitimate trong các nhóm này.

| Model | Nhóm | n | F1 | Recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| E0 | hostname_only | 24,051 | 0.996995 | 0.994008 | 0 | 48 |
| E1 | hostname_only | 24,051 | 0.692348 | 0.583323 | 815 | 3,338 |
| E2 | hostname_only | 24,051 | 0.688820 | 0.600300 | 1,143 | 3,202 |
| E0 | path_only | 3,118 | 1.000000 | 1.000000 | 0 | 0 |
| E1 | path_only | 3,118 | 1.000000 | 1.000000 | 0 | 0 |
| E2 | path_only | 3,118 | 0.862874 | 0.758820 | 0 | 752 |
| E0 | query_only | 237 | 1.000000 | 1.000000 | 0 | 0 |
| E1 | query_only | 237 | 1.000000 | 1.000000 | 0 | 0 |
| E2 | query_only | 237 | 0.960526 | 0.924051 | 0 | 18 |
| E0 | path_and_query | 797 | 1.000000 | 1.000000 | 0 | 0 |
| E1 | path_and_query | 797 | 1.000000 | 1.000000 | 0 | 0 |
| E2 | path_and_query | 797 | 0.839039 | 0.722710 | 0 | 221 |

Kết quả phù hợp với việc hostname-only mất thông tin phân biệt nằm trong path/query trên split này. Nhưng validation path/query groups đều chỉ chứa Phishing, vì vậy không thể suy ra hiệu quả trên phân phối URL thực tế hoặc nhân quả tổng quát.

## Manual check tham khảo

Manual check: 25 URL, 20 legitimate (labels do người dùng cung cấp, chưa kiểm chứng độc lập) và 5 phishing mô phỏng. Toàn bộ E0/E1/E2 đều có 0 false negative; false positive lần lượt là 20, 12, 7. E2 đạt 0.72 accuracy trong bộ này. Đây không phải căn cứ để tuyên bố E2 tổng quát hơn. Có 5 URL thuộc registered domain thấy trong PhiUSIIL và 20 URL thuộc domain chưa thấy; báo cáo JSON tách riêng các nhóm.

## Consistency checks

Ba biến thể https://daa.uit.edu.vn/, daa.uit.edu.vn và https://www.daa.uit.edu.vn/ có vector hostname giống nhau, cùng dự đoán Legitimate và phishing probability 0.29045856881432464. URL https://www.google.com/ dự đoán Legitimate với phishing probability 0.42946943195554554. Đây chỉ là kiểm tra nhất quán normalization, không xác minh ground truth.

## Khuyến nghị

Giữ E0 làm mô hình demo/chính hiện tại; không tích hợp E2 vào Streamlit ở Phase 4A. E2 nên được giữ như ablation/đối chứng hostname-only, vì validation F1 và recall thấp hơn E0 và E1, đặc biệt ở nhóm có path/query. E1 cũng suy giảm mạnh so với E0; diagnostic cho thấy nhiều feature thay đổi đồng thời, nên chưa thể quy nguyên nhân cho một feature riêng.

Bước tiếp theo hợp lý là mở rộng đánh giá trên nguồn ngoài có nhãn và domain-level split riêng (E3), đồng thời điều tra shortcut/distribution shift của PhiUSIIL. Không dùng test để chọn mô hình và không coi manual 25 URL là bằng chứng tổng quát.
