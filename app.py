import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
import numpy as np

# --- TẢI MODEL VÀ VECTORIZER ĐÃ LƯU ---
try:
    with open('saved_model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('saved_model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy tệp model.pkl hoặc vectorizer.pkl. Vui lòng chạy notebook Model_Training.ipynb trước.")
    st.stop()


# --- HÀM TIỀN XỬ LÝ (giống lúc huấn luyện) ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- HÀM TẠO GIẢI THÍCH VỚI LIME ---
# LIME cần một hàm dự đoán trả về xác suất cho cả 2 lớp
def predictor(texts):
    processed_texts = [preprocess_text(text) for text in texts]
    feature_vectors = vectorizer.transform(processed_texts)
    # model.decision_function trả về điểm số, ta cần chuyển sang xác suất
    # Ở đây dùng một xấp xỉ đơn giản, thực tế có thể dùng CalibratedClassifierCV
    # Nhưng với PAC và LIME, decision_function thường đủ tốt
    scores = model.decision_function(feature_vectors)
    # Chuyển điểm số thành "xác suất" giả định cho 2 lớp [P(Fake), P(True)]
    probs = np.array([[1 - 1 / (1 + np.exp(-s)), 1 / (1 + np.exp(-s))] for s in scores])
    return probs

# Khởi tạo LIME explainer
explainer = LimeTextExplainer(class_names=['Tin giả', 'Tin thật'])


# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Trình phát hiện Tin giả", layout="wide")
st.title("🔎 AI Phát Hiện Tin Tức Giả")
st.markdown("Nhập một đoạn văn bản tin tức (tiếng Anh) vào ô dưới đây để AI dự đoán độ tin cậy.")

# Ô nhập liệu
input_text = st.text_area("Nhập nội dung tin tức tại đây:", height=250)

if st.button("Kiểm tra"):
    if input_text.strip() == "":
        st.warning("Vui lòng nhập nội dung tin tức.")
    else:
        # 1. Dự đoán
        prediction = model.predict(vectorizer.transform([preprocess_text(input_text)]))
        
        # Hiển thị kết quả
        st.subheader("Kết quả Dự đoán")
        if prediction[0] == 1:
            st.success("✅ Tin này có vẻ là **TIN THẬT**.")
        else:
            st.error("❌ Tin này có vẻ là **TIN GIẢ**.")
            
        # 2. Giải thích lý do với LIME
        st.subheader("Giải thích của AI")
        with st.spinner("Đang phân tích lý do..."):
            explanation = explainer.explain_instance(
                preprocess_text(input_text),
                predictor,
                num_features=10, # Số lượng từ khóa muốn giải thích
                labels=(1,) # Chỉ giải thích cho lớp "Tin thật"
            )
            
            # Hiển thị giải thích dưới dạng HTML
            st.components.v1.html(explanation.as_html(), height=800, scrolling=True)