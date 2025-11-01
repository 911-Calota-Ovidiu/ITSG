import cv2
import numpy as np

def process_clahe(img_gray):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def process_adaptive_binary(img_gray):
    """Applies adaptive thresholding, which is great for varying lighting conditions."""
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

def process_otsu_binary(img_gray):
    """Applies Otsu's thresholding to binarize the image."""
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binary

def process_sharpen(img_gray):
    """Applies a sharpening filter."""
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img_gray, -1, sharpen_kernel)

def process_simple_binary(img_gray):
    """Applies a simple, global binary threshold."""
    _, img_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    return img_binary

# Dicționar cu toți algoritmii pentru a itera ușor prin ei
PROCESSING_ALGORITHMS = {
    "Contrast (CLAHE)": process_clahe,
    "Binarizare Adaptivă": process_adaptive_binary,
    "Binarizare (Otsu)": process_otsu_binary,
    "Ascuțire (Sharpening)": process_sharpen,
    "Binarizare Simplă": process_simple_binary,
}
