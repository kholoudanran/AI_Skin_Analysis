import streamlit as st
from deepface import DeepFace
import json

st.title("🎭 Face Age Prediction - NTI")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img_path = uploaded_file.name

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        analysis = DeepFace.analyze(img_path, actions=['age'])
        
        important_info = {
            "Face Age": analysis[0]["age"]
        }

        st.subheader("🌈 Analysis Results:")
        for key, value in important_info.items():
            color = {
                "Age": "black"
            }.get(key, "black")
            st.markdown(f"<span style='color:{color}; font-size: 20px;'><strong>{key}:</strong> {value}</span>", unsafe_allow_html=True)

        # Display the uploaded image with a smaller size
        st.image(uploaded_file, caption='Uploaded Image', width=300)  # Adjust the width as needed

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("This application analyzes facial age.")
st.markdown("🌐 Developed by [Kholoud Khaled.]")