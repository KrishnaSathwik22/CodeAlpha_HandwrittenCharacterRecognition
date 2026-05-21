import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import zipfile
import gzip

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def build_cnn_model(num_classes):
    """
    Standard academic CNN architecture as requested:
    - Input Layer (28x28x1)
    - Conv2D (32 filters, 3x3, ReLU)
    - MaxPooling2D (2x2)
    - Conv2D (64 filters, 3x3, ReLU)
    - MaxPooling2D (2x2)
    - Flatten
    - Dense (128, ReLU)
    - Dropout (0.5)
    - Output Layer (Softmax)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history, title, save_path):
    """Plots accuracy and loss history curves and saves them to disk."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title(f'{title} - Accuracy', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'{title} - Loss', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Saved training plot to {save_path}")

def train_mnist():
    print("\n" + "="*50)
    print("         1. TRAINING MNIST DIGIT MODEL")
    print("="*50)
    
    # Load dataset
    print("[Info] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocessing
    print("[Info] Preprocessing data...")
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)
    
    # Build model
    model = build_cnn_model(num_classes=10)
    model.summary()
    
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    print("[Info] Training MNIST model for 5 epochs...")
    history = model.fit(
        x_train, y_train, 
        epochs=5, 
        batch_size=128, 
        validation_split=0.1, 
        verbose=1
    )
    
    # Evaluate
    print("[Info] Evaluating MNIST model...")
    test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"--> MNIST Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
    
    # Save Model
    model_path = os.path.join("models", "mnist_model.keras")
    model.save(model_path)
    print(f"[Info] Saved MNIST model to {model_path}")
    
    # Plot curves
    plot_history(history, "MNIST Digits", os.path.join("data", "mnist_training.png"))
    
    # Classification Report
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    print("\n--- MNIST Classification Report ---")
    print(classification_report(y_test, y_pred, digits=4))
    
def load_emnist_letters_from_zip(zip_path):
    """
    Robust, NumPy 2.x-compliant loader to read EMNIST Letters from gzip-compressed 
    IDX binary files inside the cached emnist.zip.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"EMNIST zip not found at {zip_path}. Please run download_emnist.py first."
        )
        
    print(f"[Loader] Extracting EMNIST binary streams from: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Load train images
        with z.open('gzip/emnist-letters-train-images-idx3-ubyte.gz') as f_gz:
            with gzip.open(f_gz, 'rb') as f:
                header = np.frombuffer(f.read(16), dtype='>u4')
                num_images, rows, cols = int(header[1]), int(header[2]), int(header[3])
                x_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
                
        # Load train labels
        with z.open('gzip/emnist-letters-train-labels-idx1-ubyte.gz') as f_gz:
            with gzip.open(f_gz, 'rb') as f:
                header = np.frombuffer(f.read(8), dtype='>u4')
                num_labels = int(header[1])
                y_train = np.frombuffer(f.read(), dtype=np.uint8)
                
        # Load test images
        with z.open('gzip/emnist-letters-test-images-idx3-ubyte.gz') as f_gz:
            with gzip.open(f_gz, 'rb') as f:
                header = np.frombuffer(f.read(16), dtype='>u4')
                num_images, rows, cols = int(header[1]), int(header[2]), int(header[3])
                x_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
                
        # Load test labels
        with z.open('gzip/emnist-letters-test-labels-idx1-ubyte.gz') as f_gz:
            with gzip.open(f_gz, 'rb') as f:
                header = np.frombuffer(f.read(8), dtype='>u4')
                num_labels = int(header[1])
                y_test = np.frombuffer(f.read(), dtype=np.uint8)
                
    return x_train, y_train, x_test, y_test

def train_emnist():
    print("\n" + "="*50)
    print("         2. TRAINING EMNIST LETTERS MODEL")
    print("="*50)
    
    # Load dataset using custom robust zip loader to avoid NumPy 2.x scalar conversion bugs in emnist library
    print("[Info] Loading EMNIST Letters dataset from local cache...")
    user_home = os.path.expanduser("~")
    zip_path = os.path.join(user_home, ".cache", "emnist", "emnist.zip")
    
    x_train, y_train, x_test, y_test = load_emnist_letters_from_zip(zip_path)
    
    print(f"[Info] Raw EMNIST training shape: {x_train.shape}")
    
    # Fix the orientation (transpose images due to historical format)
    print("[Info] Correcting EMNIST image transpositions (swapping rows/cols)...")
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    
    # Preprocessing
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Labels are 1-26 in EMNIST. We convert them to 0-25 by subtracting 1
    y_train = y_train - 1
    y_test_shifted = y_test - 1
    
    # One-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, 26)
    y_test_categorical = keras.utils.to_categorical(y_test_shifted, 26)
    
    # Build model
    model = build_cnn_model(num_classes=26)
    model.summary()
    
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    print("[Info] Training EMNIST model for 5 epochs...")
    history = model.fit(
        x_train, y_train_cat, 
        epochs=5, 
        batch_size=128, 
        validation_split=0.1, 
        verbose=1
    )
    
    # Evaluate
    print("[Info] Evaluating EMNIST model...")
    test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"--> EMNIST Letters Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
    
    # Save Model
    model_path = os.path.join("models", "emnist_model.keras")
    model.save(model_path)
    print(f"[Info] Saved EMNIST model to {model_path}")
    
    # Plot curves
    plot_history(history, "EMNIST Letters", os.path.join("data", "emnist_training.png"))
    
    # Classification Report
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Labels corresponding to letters (A-Z)
    target_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    print("\n--- EMNIST Classification Report ---")
    print(classification_report(y_test_shifted, y_pred, target_names=target_names, digits=4))

if __name__ == "__main__":
    train_mnist()
    train_emnist()
    print("\n[Success] Training complete! All models saved to models/ directory.")
