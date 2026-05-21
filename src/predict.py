import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

# Import the custom preprocessing utility
from preprocess import preprocess_image

def get_label_name(mode, idx):
    """Converts prediction index to human-readable digit or letter."""
    if mode == 'digit':
        return str(idx)
    else:  # 'letter' mode
        return chr(ord('A') + idx)

def predict_custom_image(image_path, mode='digit'):
    """
    Loads model, preprocesses target image, performs inference, and generates visual report.
    """
    print(f"\n[Predict] Running handwritten {mode} recognition on: {image_path}")
    
    # 1. Check if model exists
    model_name = "mnist_model.keras" if mode == 'digit' else "emnist_model.keras"
    model_path = os.path.join("models", model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Please run model training first by executing: python src/train.py"
        )
        
    # 2. Load the model
    print(f"[Predict] Loading model: {model_path}...")
    model = keras.models.load_model(model_path)
    
    # 3. Robust preprocessing using our custom OpenCV pipeline
    print("[Predict] Preprocessing image...")
    preprocessed_img, binary_img = preprocess_image(image_path)
    
    # 4. Prepare batch for Keras input (batch size = 1)
    input_batch = np.expand_dims(preprocessed_img, axis=0)  # Shape becomes (1, 28, 28, 1)
    
    # 5. Model prediction
    predictions = model.predict(input_batch, verbose=0)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]
    
    # Print clean results to console
    print("\n" + "="*40)
    print(f"      PREDICTION RESULTS ({mode.upper()})")
    print("="*40)
    for rank, idx in enumerate(top_indices, 1):
        label = get_label_name(mode, idx)
        confidence = predictions[idx] * 100
        print(f"Rank {rank}: '{label}' | Confidence: {confidence:.2f}%")
    print("="*40)
    
    # Load original image for plotting
    orig_img = cv2.imread(image_path)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # 6. Generate stunning premium visual report using Matplotlib
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: Original User Image
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img_rgb)
    plt.title("Original Custom Image", fontsize=12, fontweight='bold', pad=10)
    plt.axis('off')
    
    # Panel 2: Preprocessed Image (What the CNN sees)
    plt.subplot(1, 3, 2)
    # We display the preprocessed image by squeezing the channel dimension
    plt.imshow(preprocessed_img.squeeze(), cmap='gray')
    plt.title("Preprocessed (28x28 CNN Input)", fontsize=12, fontweight='bold', pad=10)
    plt.axis('off')
    
    # Panel 3: Prediction Probabilities
    plt.subplot(1, 3, 3)
    top_labels = [get_label_name(mode, idx) for idx in top_indices][::-1]
    top_confs = [predictions[idx] * 100 for idx in top_indices][::-1]
    
    colors = ['#B0C4DE', '#87CEFA', '#1E90FF']  # Sleek blue gradient
    bars = plt.barh(top_labels, top_confs, color=colors, edgecolor='none', height=0.55)
    
    # Add text labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1.5, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', 
            ha='left', 
            va='center', 
            fontsize=10, 
            fontweight='bold', 
            color='#333333'
        )
        
    plt.title("Top Predictions Confidence", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Probability / Confidence (%)", fontsize=10)
    plt.xlim(0, 115)  # Leave room for percentage labels
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#cccccc')
    plt.gca().spines['bottom'].set_color('#cccccc')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Save the output visualization
    output_path = os.path.join("data", "prediction_result.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()
    print(f"\n[Success] Premium visual report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten characters from custom images.")
    parser.add_argument(
        "--image", "-i", 
        type=str, 
        required=True, 
        help="Path to the custom image file."
    )
    parser.add_argument(
        "--mode", "-m", 
        type=str, 
        choices=["digit", "letter"], 
        default="digit", 
        help="Mode of prediction: 'digit' (MNIST, 0-9) or 'letter' (EMNIST, A-Z)."
    )
    
    args = parser.parse_args()
    predict_custom_image(args.image, args.mode)
