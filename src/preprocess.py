import cv2
import numpy as np

def preprocess_image(image_path, target_size=(28, 28)):
    """
    Robust preprocessing pipeline for custom hand-drawn character images.
    1. Read in grayscale.
    2. Apply Gaussian blur to smooth out noise.
    3. Apply Otsu's thresholding to get a clean binary image.
    4. Auto-invert: Detect if the background is bright and invert it so that 
       the character is white (foreground) and background is black (0).
    5. Crop to the character's bounding box using contours.
    6. Pad with standard margins to simulate MNIST/EMNIST dataset characteristics.
    7. Resize to target size (28x28) using INTER_AREA downsampling.
    8. Normalize pixel values to [0.0, 1.0] and reshape to (28, 28, 1).
    
    Parameters:
        image_path (str): Path to the input image file.
        target_size (tuple): Target dimensions (width, height) - default is (28, 28).
        
    Returns:
        np.ndarray: Preprocessed image of shape (target_size[0], target_size[1], 1).
        np.ndarray: Intermediate binary image for visualization.
    """
    # 1. Load image in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not load image at path: {image_path}")
        
    # 2. Apply Gaussian Blur to smooth edges and reduce noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # 3. Apply Otsu's thresholding to convert to binary (0 and 255)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Auto-invert: Analyze the boundary pixels to determine if background is light
    # We check the border of the thresholded image
    h, w = thresh.shape
    border_pixels = np.concatenate([
        thresh[0, :],          # Top row
        thresh[-1, :],         # Bottom row
        thresh[:, 0],          # Left col
        thresh[:, -1]          # Right col
    ])
    
    # If the majority of border pixels are white (255), invert the image
    # so we get a black background and white strokes
    if np.mean(border_pixels) > 127:
        thresh = cv2.bitwise_not(thresh)
        
    # 5. Crop to the bounding box of the character
    # Find all contours of the character
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get the largest contour by area (assumes the character is the main element)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        cropped = thresh[y:y+h_box, x:x+w_box]
        
        # 6. Pad the bounding box with a margin to simulate MNIST margins
        # MNIST/EMNIST characters are centered in a 28x28 box with spacing around the boundaries.
        # We will add padding equal to 15% of the larger dimension of the bounding box
        padding = int(max(w_box, h_box) * 0.15)
        if padding < 4:
            padding = 4  # Minimum padding of 4 pixels
            
        padded = cv2.copyMakeBorder(
            cropped, 
            padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, 
            value=0
        )
    else:
        # Fallback if no contours are found
        padded = thresh
        
    # 7. Resize to target size (28x28) using INTER_AREA (ideal for scaling down)
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    
    # 8. Normalize to [0.0, 1.0] and add channel dimension
    normalized = resized.astype('float32') / 255.0
    preprocessed_img = np.expand_dims(normalized, axis=-1)
    
    return preprocessed_img, thresh
