import streamlit as st
from PIL import Image

from src.predict import load_model, predict_image


st.set_page_config(
    page_title="CIFAR-10 Explainable AI",
    page_icon="🔍"
)


@st.cache_resource
def get_model():
    return load_model()


st.title("🔍 CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    model = get_model()

    result = predict_image(
        image,
        model
    )

    st.success(
        f"Prediction: {result['class']}"
    )

    st.write(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )
