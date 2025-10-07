import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# MODEL CONFIGURATION
# ==========================
MODEL_PATH = "inceptionv3_best.h5"  # change model here

# Auto-select input size based on model name
if "inception" in MODEL_PATH.lower():
    target_size = (299, 299)
else:
    target_size = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    'animal fish','animal fish bass','fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream','fish sea_food hourse_mackerel',
    'fish sea_food red_mullet','fish sea_food red_sea_bream',
    'fish sea_food sea_bass','fish sea_food shrimp',
    'fish sea_food striped_red_mullet','fish sea_food trout'
]

# ==========================
# Streamlit UI
# ==========================
st.title("🐟 Fish Classification Web App")
st.write("Upload an image and let the selected model predict its class!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize correctly
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.write(f"### 🏷️ Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")

    fig, ax = plt.subplots()
    ax.bar(class_labels, preds[0])
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
