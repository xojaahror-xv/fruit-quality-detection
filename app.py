import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(page_title="Fruit Quality Detection", page_icon="\U0001F34E")

CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 900px;}
.hero {background: linear-gradient(135deg, #2E8B57 0%, #1B5E3A 100%); border-radius: 18px; padding: 28px 32px; margin-bottom: 24px;}
.hero h1 {color: #FFFFFF; margin: 0 0 6px 0; font-size: 2rem;}
.hero p {color: #E6F4EC; margin: 0; font-size: 1.05rem;}
.result-card {border-radius: 16px; padding: 22px 24px; margin-bottom: 14px; color: #1B1F2A;}
.result-card.fresh {background: #E8F6EE; border-left: 8px solid #2E8B57;}
.result-card.rotten {background: #FBEAE8; border-left: 8px solid #C0392B;}
.result-card .label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #5A6270;}
.result-card .fruit {font-size: 1.9rem; font-weight: 700; margin: 4px 0;}
.result-card .quality {font-size: 1.3rem; font-weight: 600;}
.result-card.fresh .quality {color: #1E7A45;}
.result-card.rotten .quality {color: #B03226;}
.conf-wrap {background: #D5DAE0; border-radius: 10px; height: 14px; overflow: hidden; margin-top: 8px;}
.conf-bar {height: 100%; border-radius: 10px;}
.conf-text {font-size: 0.95rem; margin-top: 14px;}
.tips {border-radius: 14px; padding: 18px 22px; background: rgba(46,139,87,0.10); border: 1px dashed #2E8B57;}
</style>
"""

HERO = """
<div class="hero">
<h1>\U0001F34E Fruit Quality Detection</h1>
<p>Upload a photo of an apple, banana or orange. The model identifies the fruit and tells you whether it is fresh or rotten.</p>
</div>
"""

FRUITS = {"apples": "Apple", "banana": "Banana", "oranges": "Orange"}


@st.cache_resource
def load():
    model = keras.models.load_model("fruit_model.keras")
    with open("class_names.json", encoding="utf-8") as f:
        names = json.load(f)
    return model, names


def parse(label):
    for prefix, quality in (("fresh", "Fresh"), ("rotten", "Rotten")):
        if label.startswith(prefix):
            return FRUITS[label[len(prefix):]], quality
    return label, "Unknown"


model, class_names = load()
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("A deep learning model (MobileNetV2, transfer learning) trained on 10,901 images of fresh and rotten fruit.")
    st.metric("Test accuracy", "98.63%")
    st.subheader("Supported fruits")
    st.write("Apple, Banana, Orange")
    st.subheader("Tips for best results")
    st.markdown("- One fruit per photo\n- Good lighting\n- Plain background")
    st.caption("Built by XV")

st.markdown(HERO, unsafe_allow_html=True)

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if not file:
    st.markdown(
        '<div class="tips"><b>How it works</b><br>'
        "1. Upload a photo or take one with your phone camera.<br>"
        "2. The model analyzes it in a few seconds.<br>"
        "3. You get the fruit type, its quality and a confidence score.</div>",
        unsafe_allow_html=True,
    )
else:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")

    with st.spinner("Analyzing image..."):
        x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
        probs = model.predict(x, verbose=0)[0]

    best = int(np.argmax(probs))
    fruit, quality = parse(class_names[best])
    conf = float(probs[best]) * 100

    if quality == "Fresh":
        css_class, color, badge = "fresh", "#2E8B57", "\u2705 Fresh"
    else:
        css_class, color, badge = "rotten", "#C0392B", "\u274C Rotten"

    card = (
        f'<div class="result-card {css_class}">'
        f'<div class="label">Result</div>'
        f'<div class="fruit">{fruit}</div>'
        f'<div class="quality">{badge}</div>'
        f'<div class="conf-text">Confidence: <b>{conf:.1f}%</b></div>'
        f'<div class="conf-wrap"><div class="conf-bar" style="width:{conf:.1f}%; background:{color};"></div></div>'
        f'</div>'
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Your photo")
    with col2:
        st.markdown(card, unsafe_allow_html=True)

    if conf < 70:
        st.warning(
            "The model is not very confident about this image. "
            "Try a clearer photo of a single apple, banana or orange."
        )

    st.subheader("Probability for each class")
    labels = []
    for n in class_names:
        f_name, q_name = parse(n)
        labels.append(f"{q_name} {f_name.lower()}")
    st.bar_chart(pd.DataFrame({"Probability": probs}, index=labels))
