import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow version:", tf.__version__)

# ==========================================
# STEP 1: Data Loading
# ==========================================
print("\n[Step 1] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ==========================================
# STEP 2: Image Processing
# ==========================================
print("\n[Step 2] Processing images...")
# 1. Reshape to include the color channel dimension (samples, 28, 28, 1)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# 2. Normalize pixel values from 0-255 down to 0.0-1.0
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. One-hot encode the target labels (0-9)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# ==========================================
# STEP 3: Define CNN Architecture
# ==========================================
print("\n[Step 3] Defining CNN architecture...")
model = Sequential([
    # Convolutional Base
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Classification Head
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.25),
    Dense(10, activation="softmax")  # 10 output classes with softmax probability
])

model.summary()

# ==========================================
# STEP 4: Compilation and Training
# ==========================================
print("\n[Step 4] Compiling and training the model...")
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train for 5 epochs
history = model.fit(
    X_train, y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Save the trained model
model.save("digit_cnn_model.keras")
print("Model saved to digit_cnn_model.keras")

# ==========================================
# STEP 5: Evaluation & Confusion Matrix
# ==========================================
print("\n[Step 5] Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

# Generate Predictions & Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

# Plot and save confusion matrix
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=range(10), yticklabels=range(10))
plt.title(f"MNIST Test Confusion Matrix (Accuracy: {test_acc*100:.2f}%)", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual True Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix visualization to confusion_matrix.png")
