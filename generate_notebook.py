import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_markdown(source_lines):
        cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines]
        }
        notebook["cells"].append(cell)

    def add_code(source_lines):
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines]
        }
        notebook["cells"].append(cell)

    # Cell 1: Introduction
    add_markdown([
        "# Deep Learning Handwritten Character Recognition System",
        "### Identifying Digits (0–9) and Alphabet Letters (A–Z) using Convolutional Neural Networks (CNN)",
        "",
        "This Jupyter Notebook implements a complete, academic, and highly robust **Handwritten Character Recognition** system. It covers data loading, advanced preprocessing, convolutional network design, model training, accuracy evaluation, and inference on custom handwritten drawings.",
        "",
        "---",
        "### Project Objectives",
        "1. **Handwritten Digit Recognition**: Using the standard **MNIST** dataset (10 classes: `0`-`9`).",
        "2. **Handwritten Letter Recognition**: Using the **EMNIST Letters** dataset (26 classes: `A`-`Z`).",
        "3. **CNN Architecture Design**: Standard, highly effective convolutional layer stacking.",
        "4. **Robust Image Preprocessing**: Applying OpenCV Otsu thresholding, bounding-box cropping, and margin padding to enable predictions on custom user-provided drawings.",
        "5. **Comprehensive Evaluation**: Metric reporting (Accuracy, Precision, Recall, Confusion Matrix)."
    ])

    # Cell 2: Library Setup
    add_markdown([
        "## Step 1: Import Libraries and Setup Environment",
        "Let's import all necessary standard machine learning and computer vision libraries: `TensorFlow/Keras`, `NumPy`, `Matplotlib`, `OpenCV`, and `scikit-learn`.",
        "We also add the core custom preprocessing module `src/preprocess.py`."
    ])

    add_code([
        "import os",
        "import cv2",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import tensorflow as tf",
        "from tensorflow import keras",
        "from sklearn.metrics import classification_report, confusion_matrix",
        "import sys",
        "",
        "# Append source path to access custom preprocessing modules",
        "sys.path.append('src')",
        "from preprocess import preprocess_image",
        "",
        "print(f'TensorFlow Version: {tf.__version__}')",
        "print(f'Keras Version: {keras.__version__}')"
    ])

    # Cell 3: Datasets Explanation
    add_markdown([
        "## Step 2: Loading and Visualizing the Datasets",
        "",
        "### A) MNIST (Digits 0–9)",
        "- Contains 70,000 grayscale images (28x28 pixels).",
        "- 10 distinct classes.",
        "- Built natively into Keras.",
        "",
        "### B) EMNIST Letters (Alphabet A–Z)",
        "- Contains 145,600 handwritten character images (28x28 pixels).",
        "- 26 distinct classes (representing letters A-Z).",
        "- **Note on EMNIST Orientation**: Due to historical formatting, the images in the EMNIST dataset are saved in a transposed column-major format. To correct their orientation and make them upright, we apply a transpose operation: `images.transpose(0, 2, 1)`.",
        "- Labels in EMNIST Letters are 1-indexed (1 to 26). We subtract 1 (`labels - 1`) to make them 0-indexed (0 to 25) for categorical cross-entropy compatibility.",
        "",
        "Let's load both datasets, correct their shapes, and visualize some random samples."
    ])

    # Cell 4: Code to Load and Preprocess Data
    add_code([
        "# 1. Load MNIST Digits",
        "print('Loading MNIST dataset...')",
        "(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()",
        "",
        "# Normalize and Reshape to (28, 28, 1) channel-last format",
        "x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0",
        "x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0",
        "",
        "# 2. Load EMNIST Letters",
        "print('Loading EMNIST Letters dataset...')",
        "from emnist import extract_training_samples, extract_test_samples",
        "x_train_emnist, y_train_emnist = extract_training_samples('letters')",
        "x_test_emnist, y_test_emnist = extract_test_samples('letters')",
        "",
        "# Correct EMNIST Transposition",
        "print('Applying EMNIST image transposition (rows/cols swap) to fix orientation...')",
        "x_train_emnist = x_train_emnist.transpose(0, 2, 1)",
        "x_test_emnist = x_test_emnist.transpose(0, 2, 1)",
        "",
        "# Normalize and Reshape to (28, 28, 1)",
        "x_train_emnist = x_train_emnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0",
        "x_test_emnist = x_test_emnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0",
        "",
        "# Shift EMNIST labels to be 0-indexed (0 to 25 representing A to Z)",
        "y_train_emnist = y_train_emnist - 1",
        "y_test_emnist_shifted = y_test_emnist - 1",
        "",
        "print(f'MNIST Train Images: {x_train_mnist.shape} | Test Images: {x_test_mnist.shape}')",
        "print(f'EMNIST Train Images: {x_train_emnist.shape} | Test Images: {x_test_emnist.shape}')"
    ])

    # Cell 5: Visualize Samples
    add_markdown([
        "### Visualization of Datasets",
        "Let's write a function to plot a random grid of samples to verify that the orientation is correct and visually inspect the datasets."
    ])

    add_code([
        "def plot_sample_grid(images, labels, mode='digit', grid_size=(3, 6)):",
        "    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 6))",
        "    indices = np.random.choice(len(images), grid_size[0] * grid_size[1], replace=False)",
        "    ",
        "    for i, ax in enumerate(axes.flat):",
        "        idx = indices[i]",
        "        # Squeeze out channel dimension for plotting",
        "        img = images[idx].squeeze()",
        "        ax.imshow(img, cmap='gray')",
        "        ",
        "        if mode == 'digit':",
        "            label_text = f'Digit: {labels[idx]}'",
        "        else:",
        "            # Convert 0-25 back to character",
        "            label_text = f'Char: {chr(ord(\"A\") + labels[idx])}'",
        "            ",
        "        ax.set_title(label_text, fontsize=10)",
        "        ax.axis('off')",
        "    plt.tight_layout()",
        "    plt.show()",
        "",
        "print('MNIST DIGIT SAMPLES:')",
        "plot_sample_grid(x_train_mnist, y_train_mnist, mode='digit')",
        "",
        "print('EMNIST LETTERS SAMPLES:')",
        "plot_sample_grid(x_train_emnist, y_train_emnist, mode='letter')"
    ])

    # Cell 6: CNN Architecture Explanation
    add_markdown([
        "## Step 3: Convolutional Neural Network (CNN) Architecture",
        "",
        "A CNN consists of layers designed to extract spatial features from input image grids:",
        "- **Convolutional (Conv2D) Layer**: Passes a set of learnable kernels/filters across the image. It detects low-level features (edges, curves, lines) in early layers and high-level features in deeper layers.",
        "- **MaxPooling2D Layer**: Performs spatial downsampling (taking the maximum value in a window, e.g., $2 \\times 2$). This reduces computational cost and makes the representations translation-invariant.",
        "- **Flatten Layer**: Flattens the final 2D feature grid into a 1D vector.",
        "- **Dense Layer**: Standard fully-connected neural layer to synthesize the features and make class-level calculations.",
        "- **Dropout Layer**: Randomly sets input elements to 0 during training at a set rate (e.g. 50%). This prevents co-adaptation of activations, serving as a powerful regularizer to **prevent overfitting**.",
        "- **Softmax Output Layer**: Normalizes the network's output values (logits) into a probability distribution summing to 1. The class with the highest probability is our prediction.",
        "",
        "### CNN Design Specifications",
        "The standard architecture requested is designed as follows:",
        "```",
        "Input Layer (28x28x1)",
        "        ↓",
        "Conv2D (32 filters, 3x3, ReLU)",
        "        ↓",
        "MaxPooling2D (2x2)",
        "        ↓",
        "Conv2D (64 filters, 3x3, ReLU)",
        "        ↓",
        "MaxPooling2D (2x2)",
        "        ↓",
        "Flatten",
        "        ↓",
        "Dense (128, ReLU)",
        "        ↓",
        "Dropout (0.5)",
        "        ↓",
        "Output Layer (Softmax)",
        "```",
        "Let's write a function to construct this network."
    ])

    # Cell 7: CNN Model Definition Code
    add_code([
        "def build_cnn_model(num_classes):",
        "    model = keras.Sequential([",
        "        # Input Layer",
        "        keras.layers.Input(shape=(28, 28, 1)),",
        "        ",
        "        # First Conv Block",
        "        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),",
        "        keras.layers.MaxPooling2D(pool_size=(2, 2)),",
        "        ",
        "        # Second Conv Block",
        "        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),",
        "        keras.layers.MaxPooling2D(pool_size=(2, 2)),",
        "        ",
        "        # Classifier Head",
        "        keras.layers.Flatten(),",
        "        keras.layers.Dense(128, activation='relu'),",
        "        keras.layers.Dropout(0.5), # Regularization to prevent overfitting",
        "        keras.layers.Dense(num_classes, activation='softmax') # Multi-class probability distribution",
        "    ])",
        "    return model",
        "",
        "# Preview model summary for digit classification (10 classes)",
        "mnist_model_draft = build_cnn_model(num_classes=10)",
        "mnist_model_draft.summary()"
    ])

    # Cell 8: Model Training Markdown
    add_markdown([
        "## Step 4: Model Training and Saving",
        "Now, we will perform training on both datasets.",
        "We compile our models using the **Adam** optimizer, **Categorical Cross-entropy** loss, and track **Accuracy**.",
        "For efficiency in this demonstration, we train for **5 epochs**, which quickly converges to high accuracy (>98% on MNIST, >90% on EMNIST).",
        "We also save the trained weights to the disk under `models/` directory in standard `.keras` format."
    ])

    # Cell 9: Model Training Code
    add_code([
        "# One-hot encode the labels for categorical crossentropy loss",
        "y_train_mnist_cat = keras.utils.to_categorical(y_train_mnist, 10)",
        "y_test_mnist_cat = keras.utils.to_categorical(y_test_mnist, 10)",
        "",
        "y_train_emnist_cat = keras.utils.to_categorical(y_train_emnist, 26)",
        "y_test_emnist_cat = keras.utils.to_categorical(y_test_emnist_shifted, 26)",
        "",
        "os.makedirs('models', exist_ok=True)",
        "os.makedirs('data', exist_ok=True)",
        "",
        "# --- A) Train MNIST Model ---",
        "print('=== Training MNIST Digit Model ===')",
        "mnist_model = build_cnn_model(num_classes=10)",
        "mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])",
        "",
        "mnist_history = mnist_model.fit(",
        "    x_train_mnist, y_train_mnist_cat, ",
        "    epochs=5, ",
        "    batch_size=128, ",
        "    validation_split=0.1, ",
        "    verbose=1",
        ")",
        "",
        "# Save digit model",
        "mnist_model.save('models/mnist_model.keras')",
        "print('[Success] MNIST Model saved to models/mnist_model.keras\\n')",
        "",
        "# --- B) Train EMNIST Letters Model ---",
        "print('=== Training EMNIST Letters Model ===')",
        "emnist_model = build_cnn_model(num_classes=26)",
        "emnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])",
        "",
        "emnist_history = emnist_model.fit(",
        "    x_train_emnist, y_train_emnist_cat, ",
        "    epochs=5, ",
        "    batch_size=128, ",
        "    validation_split=0.1, ",
        "    verbose=1",
        ")",
        "",
        "# Save letter model",
        "emnist_model.save('models/emnist_model.keras')",
        "print('[Success] EMNIST Model saved to models/emnist_model.keras')"
    ])

    # Cell 10: Evaluation Explanation
    add_markdown([
        "## Step 5: Accuracy Evaluation and Metrics",
        "To measure how well the neural networks generalize to unseen data, we evaluate them on their respective test sets.",
        "We will print a detailed classification report containing:",
        "- **Precision**: Proportion of correct positive identifications out of all predicted positive classes.",
        "- **Recall**: Proportion of correct positive identifications out of all actual positive classes.",
        "- **F1-score**: Harmonic mean of Precision and Recall.",
        "We also plot the accuracy and loss curves for both training runs."
    ])

    # Cell 11: Evaluation Code
    add_code([
        "def plot_training_curves(history, title):",
        "    plt.figure(figsize=(12, 4))",
        "    ",
        "    # Accuracy",
        "    plt.subplot(1, 2, 1)",
        "    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)",
        "    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)",
        "    plt.title(f'{title} - Accuracy')",
        "    plt.xlabel('Epoch')",
        "    plt.ylabel('Accuracy')",
        "    plt.legend()",
        "    plt.grid(True, linestyle='--', alpha=0.5)",
        "    ",
        "    # Loss",
        "    plt.subplot(1, 2, 2)",
        "    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)",
        "    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)",
        "    plt.title(f'{title} - Loss')",
        "    plt.xlabel('Epoch')",
        "    plt.ylabel('Loss')",
        "    plt.legend()",
        "    plt.grid(True, linestyle='--', alpha=0.5)",
        "    ",
        "    plt.tight_layout()",
        "    plt.show()",
        "",
        "# Plot MNIST training history",
        "print('MNIST TRAINING METRICS:')",
        "plot_training_curves(mnist_history, 'MNIST Digits')",
        "",
        "# Plot EMNIST training history",
        "print('EMNIST LETTERS TRAINING METRICS:')",
        "plot_training_curves(emnist_history, 'EMNIST Letters')"
    ])

    # Cell 12: Classification Reports
    add_code([
        "# Evaluate Digit Model on Test set",
        "print('=== MNIST DIGITS EVALUATION ===')",
        "test_loss_m, test_acc_m = mnist_model.evaluate(x_test_mnist, y_test_mnist_cat, verbose=0)",
        "print(f'Test Loss: {test_loss_m:.4f} | Test Accuracy: {test_acc_m*100:.2f}%\\n')",
        "",
        "y_pred_mnist = np.argmax(mnist_model.predict(x_test_mnist, verbose=0), axis=1)",
        "print(classification_report(y_test_mnist, y_pred_mnist, digits=4))",
        "",
        "# Evaluate Letter Model on Test set",
        "print('=== EMNIST LETTERS EVALUATION ===')",
        "test_loss_e, test_acc_e = emnist_model.evaluate(x_test_emnist, y_test_emnist_cat, verbose=0)",
        "print(f'Test Loss: {test_loss_e:.4f} | Test Accuracy: {test_acc_e*100:.2f}%\\n')",
        "",
        "y_pred_emnist = np.argmax(emnist_model.predict(x_test_emnist, verbose=0), axis=1)",
        "target_names_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]",
        "print(classification_report(y_test_emnist_shifted, y_pred_emnist, target_names=target_names_letters, digits=4))"
    ])

    # Cell 13: Custom Preprocessing & Predictions Markdown
    add_markdown([
        "## Step 6: Prediction on Custom Handwritten Images",
        "",
        "When predicting real-world images (like photos taken from a mobile phone or drawings created in paint tools), they rarely match the MNIST clean dataset. Simply scaling an image down to 28x28 will distort the character or make it touch the margins, breaking CNN predictions.",
        "",
        "To make the prediction system robust, our custom OpenCV preprocessing module (`src/preprocess.py`) performs the following operations:",
        "1. **Grayscale & Smoothing**: Converts the image to grayscale and applies a Gaussian blur.",
        "2. **Adaptive Otsu Thresholding**: Segments the drawing from the background.",
        "3. **Auto-Inversion**: Identifies if the background is bright (white paper with black ink) and automatically inverts it so that it becomes white pixels on a black background (matching training conditions).",
        "4. **Contour Cropping**: Finds the bounding box of the actual character and crops out empty outer space.",
        "5. **Padding Margin**: Pads the cropped image (adding 15% margins on all sides). This prevents the character boundaries from touching the edges and centers it properly.",
        "6. **Resize & Scale**: Downsamples to 28x28 using INTER_AREA interpolation, scales pixel values to `[0.0, 1.0]`, and expands dims to channel shape `(28, 28, 1)`.",
        "",
        "Let's write an interactive prediction cell to perform inference on a custom image!"
    ])

    # Cell 14: Interactive Prediction Cell
    add_code([
        "def run_custom_inference(img_path, mode='digit'):",
        "    if not os.path.exists(img_path):",
        "        print(f'Error: Image not found at {img_path}')",
        "        return",
        "        ",
        "    # 1. Preprocess using our custom module",
        "    preprocessed_img, binary_img = preprocess_image(img_path)",
        "    ",
        "    # Add batch axis: (28, 28, 1) -> (1, 28, 28, 1)",
        "    input_batch = np.expand_dims(preprocessed_img, axis=0)",
        "    ",
        "    # 2. Select appropriate model",
        "    if mode == 'digit':",
        "        model = keras.models.load_model('models/mnist_model.keras')",
        "    else:",
        "        model = keras.models.load_model('models/emnist_model.keras')",
        "        ",
        "    # 3. Predict",
        "    predictions = model.predict(input_batch, verbose=0)[0]",
        "    top_idx = np.argmax(predictions)",
        "    confidence = predictions[top_idx] * 100",
        "    ",
        "    # Map index to string label",
        "    if mode == 'digit':",
        "        pred_char = str(top_idx)",
        "    else:",
        "        pred_char = chr(ord('A') + top_idx)",
        "        ",
        "    print(f'Prediction Mode: {mode.upper()}')",
        "    print(f'Result: Found character \"{pred_char}\" with {confidence:.2f}% confidence!')",
        "    ",
        "    # 4. Beautiful Visualization",
        "    plt.figure(figsize=(10, 4))",
        "    ",
        "    # Show original",
        "    plt.subplot(1, 2, 1)",
        "    orig = cv2.imread(img_path)",
        "    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)",
        "    plt.imshow(orig_rgb)",
        "    plt.title('Original Custom Drawing')",
        "    plt.axis('off')",
        "    ",
        "    # Show Preprocessed",
        "    plt.subplot(1, 2, 2)",
        "    plt.imshow(preprocessed_img.squeeze(), cmap='gray')",
        "    plt.title(f'CNN Input (Predicted: {pred_char})')",
        "    plt.axis('off')",
        "    ",
        "    plt.tight_layout()",
        "    plt.show()",
        "    ",
        "    # Plot top 3 bar charts",
        "    top_3_indices = np.argsort(predictions)[::-1][:3]",
        "    top_labels = []",
        "    top_confs = []",
        "    for idx in top_3_indices:",
        "        lbl = str(idx) if mode == 'digit' else chr(ord('A') + idx)",
        "        top_labels.append(lbl)",
        "        top_confs.append(predictions[idx] * 100)",
        "        ",
        "    plt.figure(figsize=(6, 3))",
        "    plt.bar(top_labels, top_confs, color='#1E90FF', edgecolor='grey')",
        "    plt.title('Top 3 Prediction Probabilities')",
        "    plt.ylabel('Confidence (%)')",
        "    plt.ylim(0, 110)",
        "    for i, conf in enumerate(top_confs):",
        "        plt.text(i, conf + 2, f'{conf:.1f}%', ha='center', fontweight='bold')",
        "    plt.tight_layout()",
        "    plt.show()",
        "",
        "print('[Info] Custom inference helper defined! Use it on any file: run_custom_inference(\"path_to_image.png\", mode=\"digit\")')"
    ])

    # Save to file
    notebook_path = "handwritten_character_recognition.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"[Success] Generated notebook saved to: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
