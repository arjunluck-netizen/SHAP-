import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from src.predict import load_model,predict_image
from src.explain import create_background,explain_image

st.set_page_config(page_title="Explainable AI - CIFAR-10",page_icon="🔍",layout="wide")
st.title("🔍 Explainable AI Image Classification")
st.subheader("CNN + DeepSHAP on CIFAR-10")

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@st.cache_resource
def get_model(): return load_model()

@st.cache_resource
def get_background(): return create_background()

uploaded = st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload a JPG or PNG image to begin.")
    st.caption("This model is trained specifically on the ten CIFAR-10 classes.")
else:
    from PIL import Image
    image = Image.open(uploaded).convert("RGB")
    c1,c2 = st.columns(2)
    with c1: st.image(image, caption="Uploaded image", width="stretch")
    result = predict_image(image,get_model())
    with c2:
        st.metric("Prediction",result["class_name"])
        st.metric("Confidence",f"{result['confidence']:.2%}")
        for name,p in zip(CLASS_NAMES,result["probabilities"]):
            st.write(f"**{name}**: {p:.2%}")

    st.divider()
    st.subheader("DeepSHAP Explanation")
    try:
        with st.spinner("Calculating DeepSHAP explanation..."):
            values = explain_image(result["input_array"],get_background(),get_model())
        if isinstance(values,list): values = values[result["class_index"]]
        values = np.asarray(values)
        while values.ndim > 4: values = values[0]
        if values.ndim == 4: values = values[0]
        attribution = np.abs(values).sum(axis=-1)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.imshow(attribution,cmap="hot"); ax.axis("off")
        ax.set_title(f"DeepSHAP attribution: {result['class_name']}")
        st.pyplot(fig); plt.close(fig)
    except Exception as exc:
        st.error("DeepSHAP could not be generated in this environment.")
        st.code(str(exc))
