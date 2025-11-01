import cv2
import numpy as np

def treat_print(roi_gray):
    """Tratament agresiv pentru text de tipar: Contrast + Ascuțire."""
    if roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
        return np.zeros_like(roi_gray) # Returnează o imagine goală dacă ROI e invalid
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    img_contrast = clahe.apply(roi_gray)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_contrast, -1, sharpen_kernel)
    return img_sharp

def treat_handwriting(roi_gray):
    """Tratament de finețe pentru scris de mână: Binarizare + Îngroșare."""
    if roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
        return np.zeros_like(roi_gray) # Returnează o imagine goală dacă ROI e invalid
    # Binarizare inteligentă pentru a izola scrisul
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_contrast = clahe.apply(roi_gray)
    _, img_binary = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Îngroșăm textul pentru a uni caracterele întrerupte
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_binary, kernel, iterations=1)
    return img_dilated
