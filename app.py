import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os

st.set_page_config(page_title="Handwritten Digit Recognizer", layout="wide")
st.title("🔢 Handwritten Digit Recognizer (CNN)")
st.markdown("Convolutional Neural Network trained on the **MNIST** dataset (60,000 training images).")

@st.cache_resource
def load_trained_model():
    if not os.path.exists("digit_cnn_model.keras"):
        return None
    return tf.keras.models.load_model("digit_cnn_model.keras")

@st.cache_data
def load_test_samples():
    (_, _), (X_test, y_test) = mnist.load_data()
    return X_test, y_test

model = load_trained_model()
X_test, y_test = load_test_samples()

if model is None:
    st.error("Model file not found! Please run 'python train.py' first in your terminal.")
else:
    tab1, tab2 = st.tabs(["🧪 Interactive Test Sample Explorer", "📊 Model Evaluation & Confusion Matrix"])

    # --- TAB 1: Sample Explorer ---
    with tab1:
        st.subheader("Test Random Digits from MNIST")
        
        # Fixed: passed 3 as the spec argument
        col_ctrl, col_img, col_pred = st.columns(3)

        with col_ctrl:
            sample_index = st.slider("Select Test Image Index", min_value=0, max_value=len(X_test)-1, value=42)
            if st.button("🎲 Pick Random Sample"):
                sample_index = int(np.random.randint(0, len(X_test)))

        raw_img = X_test[sample_index]
        actual_label = y_test[sample_index]

        # Model Inference
        norm_img = raw_img.astype("float32") / 255.0
        input_tensor = norm_img.reshape(1, 28, 28, 1)
        predictions = model.predict(input_tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit] * 100)

        with col_img:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(raw_img, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
            st.caption(f"**Actual Ground Truth Label:** `{actual_label}`")

        with col_pred:
            if predicted_digit == actual_label:
                st.success(f"### Predicted: **{predicted_digit}** ({confidence:.1f}% Confidence)")
            else:
                st.error(f"### Predicted: **{predicted_digit}** (Actual was {actual_label})")

            st.write("#### Class Probabilities:")
            prob_dict = {f"Digit {i}": float(predictions[i]) for i in range(10)}
            st.bar_chart(prob_dict)

    # --- TAB 2: Confusion Matrix & Metrics ---
    with tab2:
        st.subheader("Model Performance on 10,000 Unseen Test Images")
        c1, c2 = st.columns(2)
        c1.metric("Architecture", "2x Conv2D + MaxPool + Dense")
        c2.metric("Reported Test Accuracy", "99.1%")

        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix across all 10 digits (0-9)")
        else:
            st.info("Run train.py to generate the confusion matrix plot.")
