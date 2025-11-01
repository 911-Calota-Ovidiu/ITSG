import cv2
import numpy as np

def process_clahe(img_gray):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def process_adaptive_binary(img_gray):
    """Applies adaptive thresholding, great for varying lighting conditions."""
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

def process_clahe_sharpen(img_gray):
    """Applies CLAHE followed by a sharpening filter for maximum clarity."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_contrast = clahe.apply(img_gray)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img_contrast, -1, sharpen_kernel)

# Echipa de elita
PROCESSING_ALGORITHMS = {
    "CLAHE + Ascuțire": process_clahe_sharpen,
    "Binarizare Adaptivă": process_adaptive_binary,
    "Contrast (CLAHE)": process_clahe,
}
