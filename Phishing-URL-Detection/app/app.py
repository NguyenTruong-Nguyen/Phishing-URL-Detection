from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.predict import predict_url

st.set_page_config(page_title="Phishing URL Detection", page_icon="🛡️", layout="centered")
st.title("🛡️ Phishing URL Detection")
st.write("Nhập một URL để hệ thống phân loại dựa trên đặc trưng của chuỗi URL.")
st.info("Ứng dụng không truy cập website, không tải nội dung trang và không kiểm tra URL bằng HTTP.")

url = st.text_input("URL cần kiểm tra", placeholder="https://example.com/login")
model_path = st.text_input("Đường dẫn model (để trống dùng model mặc định)", value="")

if st.button("Phân tích URL", type="primary", use_container_width=True):
    if not url.strip():
        st.warning("Vui lòng nhập URL.")
    else:
        try:
            result = predict_url(url.strip(), model_path.strip() or None)
            if result["prediction"] == 1:
                st.error("Kết quả: PHISHING – URL có dấu hiệu đáng ngờ.")
            else:
                st.success("Kết quả: LEGITIMATE – mô hình không phát hiện dấu hiệu phishing rõ ràng.")
            if "probability_phishing" in result:
                st.metric("Điểm mô hình cho lớp Phishing", f"{result['probability_phishing'] * 100:.2f}%")
                st.caption("Đây là điểm xác suất do mô hình ước lượng, không phải bảo đảm an toàn hay xác suất rủi ro đã được hiệu chuẩn.")
            st.caption("Kết quả chỉ mang tính hỗ trợ sàng lọc; không có mô hình nào bảo đảm phát hiện tất cả URL lừa đảo.")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)
