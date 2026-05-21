# Handwritten Character Recognition System

An academic-grade, highly robust deep learning system for recognizing handwritten digits (0–9) and alphabet letters (A–Z). This project includes a Convolutional Neural Network (CNN) model built with TensorFlow/Keras, a custom OpenCV preprocessing pipeline optimized for user-drawn images, and a clean prediction visualization reporting mechanism.

---

## 🌟 Key Features

*   **Dual Classification Modes**:
    *   **Digit Mode**: Trained on the standard **MNIST** dataset (10 classes: `0` to `9`).
    *   **Letter Mode**: Trained on the **EMNIST Letters** dataset (26 classes: `A` to `Z`).
*   **Robust OpenCV Preprocessing Pipeline**: Standard resizing often distorts hand-drawn characters or causes them to touch borders, degrading model accuracy. Our pipeline ([src/preprocess.py](file:///c:/Users/Dell/OneDrive/Documents/CodeAlpha_HandwrittenCharacterRecognition/src/preprocess.py)) performs:
    1.  *Grayscale Conversion & Smoothing*: Noise reduction using Gaussian Blur.
    2.  *Otsu's Thresholding*: Binarization to cleanly segment the stroke.
    3.  *Auto-Inversion*: Automatically flips image colors if drawn as dark ink on a light background, ensuring the network always receives white strokes on a black background (matching training sets).
    4.  *Contour Bounding-Box Cropping*: Crops tightly around the character to eliminate excessive background space.
    5.  *MNIST-Standard Padding*: Pads the cropped character with a 15% margin to prevent edge-touching and improve centering.
    6.  *Resize & Normalize*: Downsamples to $28 \times 28$ pixels using area interpolation, scales intensities to `[0.0, 1.0]`, and reshapes to channel-last format `(28, 28, 1)`.
*   **Aesthetic Visualization Reports**: Predicts characters from user-provided images and generates a side-by-side plot of the original image, preprocessed visual representation, and a horizontal bar chart displaying the top-3 prediction confidences.
*   **Comprehensive Interactive Notebook**: An automatically generated, rich Jupyter Notebook ([handwritten_character_recognition.ipynb](file:///c:/Users/Dell/OneDrive/Documents/CodeAlpha_HandwrittenCharacterRecognition/handwritten_character_recognition.ipynb)) walking through the design process, theory, data distribution, training, and custom inference.

---

## 🏗️ CNN Architecture

The model architecture is built to follow a classic, highly effective academic design:

| Layer | Type | Specifications | Details |
| :--- | :--- | :--- | :--- |
| **Input** | Input | Shape: `(28, 28, 1)` | Grayscale normalized pixels |
| **Conv 1** | Conv2D | 32 filters, $3 \times 3$ kernel, ReLU | Spatial feature extraction (edges) |
| **Pool 1** | MaxPooling2D | $2 \times 2$ pool size | Downsampling spatial grid by half |
| **Conv 2** | Conv2D | 64 filters, $3 \times 3$ kernel, ReLU | Higher-level composition detection |
| **Pool 2** | MaxPooling2D | $2 \times 2$ pool size | Downsampling spatial grid by half |
| **Flatten** | Flatten | - | Collapses 2D grid into a 1D vector |
| **Dense 1** | Dense | 128 units, ReLU activation | Fully-connected reasoning layer |
| **Dropout** | Dropout | Rate: `0.5` | Regularization (prevents overfitting) |
| **Output** | Dense | Categorical output (10 or 26), Softmax | Probability distribution |

---

## 📂 Project Directory Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── data/                            # Training curves, user test images, and prediction outputs
│   ├── custom_digit_3.png           # Sample handwritten digit drawing
│   ├── custom_letter_A.png          # Sample handwritten letter 'A'
│   ├── custom_letter_B.png          # Sample handwritten letter 'B'
│   ├── mnist_training.png           # MNIST training curves (accuracy & loss)
│   ├── emnist_training.png          # EMNIST training curves (accuracy & loss)
│   └── prediction_result.png        # Generated prediction report for custom inference
│
├── models/                          # Serialized trained weights in Keras format
│   ├── mnist_model.keras            # Trained digit classifier
│   └── emnist_model.keras           # Trained letter classifier
│
├── src/                             # Python source code
│   ├── __init__.py
│   ├── preprocess.py                # OpenCV image preprocessing routines
│   ├── train.py                     # Script to train MNIST and EMNIST models
│   └── predict.py                   # CLI tool to run inference on custom images
│
├── generate_notebook.py             # Script to programmatically generate the Jupyter Notebook
├── handwritten_character_recognition.ipynb # Comprehensive tutorial Jupyter notebook
└── README.md                        # Project documentation (this file)
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Configure Virtual Environment & Install Dependencies
From the project root directory, set up your environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install numpy opencv-python tensorflow matplotlib scikit-learn emnist
```

> [!NOTE]
> The training script uses a robust custom extraction parser to load the `emnist.zip` dataset directly from the local cache folder. This avoids conversion errors with NumPy versions during standard package loads.

---

## 💻 Usage

### 1. Run Model Training
Train both the digit and letter models and generate performance evaluation metrics:

```bash
python src/train.py
```
*   This trains both CNN architectures for **5 epochs** each.
*   Saves the trained models into `models/`.
*   Generates accuracy and loss plots in `data/`.
*   Prints detailed classification report metrics (Precision, Recall, F1-Score).

### 2. Run Inference on Custom Handwritten Drawings
Predict custom character images using the command line:

```bash
# Predict a digit (using MNIST model)
python src/predict.py --image data/custom_digit_3.png --mode digit

# Predict a letter (using EMNIST Letters model)
python src/predict.py --image data/custom_letter_A.png --mode letter
```
*   Output reports are printed directly to the terminal showing the top-3 predictions and confidence margins.
*   A premium side-by-side visualization report is saved to `data/prediction_result.png`.

### 3. Generate Jupyter Notebook
If the notebook is missing or you want to recreate it:
```bash
python generate_notebook.py
```

---

## 📊 Evaluation Visualizations

*   **Training Histories**: Accuracy and loss plots are stored in `data/mnist_training.png` and `data/emnist_training.png`.
*   **Prediction Output**: View `data/prediction_result.png` after running a prediction to see the visual panel comparing the original drawing, binary thresholded/aligned character input, and model classification probabilities.
