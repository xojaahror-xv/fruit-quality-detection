import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="Meva sifatini aniqlash", page_icon="🍎")

FRUITS = {"apples": "Olma", "banana": "Banan", "oranges": "Apelsin"}
QUALITY = {"fresh": "Yangi ✅", "rotten": "Chirigan ❌"}

@st.cache_resource
def load():
    model = keras.models.load_model("fruit_model.keras")
    names = json.load(open("class_names.json", encoding="utf-8"))
    return model, names

model, class_names = load()

st.title("🍎 Meva turi va sifatini aniqlash")
st.write("Olma, banan yoki apelsin rasmini yuklang. Model uning turini va yangi yoki chirigan ekanini aniqlaydi.")

file = st.file_uploader("Rasm tanlang", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Yuklangan rasm", use_container_width=True)

    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))
    label = class_names[best]

    quality = next(k for k in QUALITY if label.startswith(k))
    fruit = FRUITS[label[len(quality):]]

    st.subheader(f"Natija: {fruit} — {QUALITY[quality]}")
    st.metric("Ishonch darajasi", f"{probs[best] * 100:.1f}%")

    if probs[best] < 0.70:
        st.warning("Model bu rasmda ishonchsiz. Rasm aniqroq bo'lsa yoki meva bu 3 turdan biri bo'lsa, natija yaxshiroq bo'ladi.")

    st.write("Barcha sinflar bo'yicha ehtimollik:")
    st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})
