import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Custom CSS to inject
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
}
.stSelectbox {
    margin-bottom: 1rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 2rem;
    border-radius: 1rem;
    margin: 1rem 0;
    text-align: center;
}
.title-box {
    background-color: #FF4B4B;
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

class ApartmentPricePredictor:
    def __init__(self, model_path):
        path = os.path.dirname(__file__)
        self.model = joblib.load(path + '/' + model_path)

    def predict_for_sale(self, data):
        return self.model.predict(data)

def main():
    # Title section with custom styling
    st.markdown("""
    <div class="title-box">
        <h1>🏢 Dự đoán giá căn hộ tại TP.HCM</h1>
        <p style='font-size: 1.2rem;'>Hệ thống dự đoán giá bán và cho thuê căn hộ dựa trên Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # About section
    with st.expander("ℹ️ Thông tin về dự án"):
        st.markdown("""
        ### Nhóm 8
        - 21120112 - Bùi Kim Phúc
        
        ### Hướng dẫn sử dụng
        1. Chọn loại giao dịch (Mua bán)
        2. Chọn quận/huyện
        3. Điều chỉnh diện tích và số phòng ngủ
        4. Nhấn nút dự đoán để xem kết quả
        """)

    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Thông tin căn hộ")
        type = st.selectbox("**Loại giao dịch**", ('For sale',))

        if type == 'For sale':
            district_list = ['District 1',
                        'District 10', 'District 11', 'District 12', 'District 2', 'District 3',
                        'District 4', 'District 5', 'District 6', 'District 7', 'District 8',
                        'District 9', 'Bình Chánh District', 'Bình Thạnh District', 'Bình Tân District', 
                        'Gò Vấp District', 'Hóc Môn District','Nhà Bè District', 'Phú Nhuận District', 
                        'Thủ Đức District', 'Tân Bình District', 'Tân Phú District']

        selected_district = st.selectbox("**Quận/Huyện**", tuple(district_list))
        
        st.markdown("### 📐 Thông số căn hộ")
        area = st.slider('**Diện tích (m²)**', 20, 250, value=50, help="Kéo thanh trượt để chọn diện tích căn hộ")
        bedroom = st.slider('**Số phòng ngủ**', 1, 6, value=2, help="Kéo thanh trượt để chọn số phòng ngủ")
        floor = st.slider('**Số tầng**', 1, 5, value=2, help="Kéo thanh trượt để chọn số tầng")

    with col2:
        st.markdown("### 📊 Thống kê")
        st.metric(label="Diện tích đã chọn", value=f"{area} m²")
        st.metric(label="Số phòng ngủ đã chọn", value=f"{bedroom} phòng")
        st.metric(label="Số tầng đã chọn", value=f"{floor} tầng")

    input_data = {
        'Area': [area],
        'Bedroom': [bedroom],
        'Floor':[floor]
    }

    for district in district_list:
        input_data[district] = [1 if district == selected_district else 0]
    df = pd.DataFrame(input_data)

    # Prediction section
    st.markdown("### 🎯 Kết quả dự đoán")
    if st.button('**Dự đoán giá**'):
        with st.spinner('Đang tính toán...'):
            if type == 'For sale':
                predictor = ApartmentPricePredictor('decision_tree_model_for_sale.sav')
                price = predictor.predict_for_sale(df.values)

                if price[0] < 0:
                    st.error("❌ Không thể dự đoán giá cho căn hộ này")
                else:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style='color: #FF4B4B;'>Giá dự đoán</h2>
                        <h1 style='font-size: 2.5rem;'>{format(int(np.round(price[0], 2)*1000000), ',d')} VNĐ</h1>
                        <p>tại {selected_district}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.info("💡 Lưu ý: Đây là giá dự đoán dựa trên dữ liệu thị trường, giá thực tế có thể dao động tùy theo nhiều yếu tố khác.")

if __name__ == "__main__":
    main()
