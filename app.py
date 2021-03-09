import cv2
import numpy as np

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from tensorflow.keras.models import load_model


# 한번만 로드
@st.cache(allow_output_mutation=True)
def load():
    return load_model('super_mnist.h5')


model = load()

# 사이드바
st.sidebar.title("Mechathon")
select = st.sidebar.selectbox(
    'Share', ['Mechathon Introduction', 'MNIST'], key='1')


if select == 'Mechathon Introduction':
    st.write("# Hi it's Mechathon project server")

if select == 'MNIST':
    # 메인
    st.title("MNIST 판독기")

    # 그림그리기

    col1, col2 = st.beta_columns(2)

    with col1:
        canvas = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=192,
            height=192,
            drawing_mode='freedraw',
            key='canvas'
        )

    # 이미지 데이터가 있으면
    if canvas.image_data is not None:
        img = canvas.image_data.astype(np.uint8)
        # 학습할 이미지
        img = cv2.resize(img, dsize=(28, 28))
        # 보여줄 이미지
        preview_img = cv2.resize(img, dsize=(
            192, 192), interpolation=cv2.INTER_NEAREST)

        col2.image(preview_img)

        # 학습
        x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = x.reshape((-1, 28, 28, 1))
        y = model.predict(x).squeeze()

        st.write(f'## RESULT : {np.argmax(y)}')
        st.bar_chart(y)
