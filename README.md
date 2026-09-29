# Digit Recognizer

A compact machine learning project that trains a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset and serves the model through a sleek Streamlit web app.

Live demo: https://digit-recognizer-kkc4a6qufkhq46kastrdwx.streamlit.app/

## ✨ Highlights

- CNN-based digit classification using TensorFlow/Keras
- Trained on the classic MNIST dataset (10 classes: digits 0-9)
- Interactive visualization of prediction confidence
- Test sample explorer for individual digits
- Confusion matrix evaluation and performance summary
- Streamlit app for easy browser-based interaction

## 🧠 Model Overview

This project uses a convolutional architecture with:

- 2 convolutional layers
- Max-pooling layers
- Flattening and dense classification head
- Softmax output for 10 digit classes

The training script saves the model as `digit_cnn_model.keras`, and the app loads it to make real-time predictions on digit images.

## 📁 Repository Structure

- `train.py` — loads the MNIST dataset, trains the CNN, and saves the model
- `app.py` — Streamlit interface for exploring predictions and model metrics
- `digit_cnn_model.keras` — trained model artifact
- `confusion_matrix.png` — evaluation visualization
- `requirements.txt` — Python dependencies

## 🚀 Run Locally

1. Clone the repository:

```bash
git clone https://github.com/RV6730/digit-recognizer.git
cd digit-recognizer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train.py
```

4. Launch the app:

```bash
streamlit run app.py
```

## 📊 Performance

The model is trained and evaluated on the standard MNIST test set, with results summarized in the Streamlit dashboard. The confusion matrix gives insight into which digits are most often confused.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Scikit-learn

## 🔗 Useful Links

- Live App: https://digit-recognizer-kkc4a6qufkhq46kastrdwx.streamlit.app/
- GitHub Repository: https://github.com/RV6730/digit-recognizer

## 🧪 Example Use

Open the app, select any handwritten digit sample from the MNIST test set, and see the model's prediction along with class confidence scores.

---

Built to make handwritten digit recognition both understandable and interactive.
