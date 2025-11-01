import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import io
import re
from image_processing import treat_print, treat_handwriting

# Configurare Tesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(layout="wide")
st.title("🩺 OPERAȚIUNEA: Bisturiul (v4 - Curățenie Generală)")
st.write("Fiecare câmp este procesat de o funcție specializată.")

uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"])

# --- "Schema" pentru Traducere ---
CHAR_MAP = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'
}


def translate_chars_to_digits(text):
    for char, digit in CHAR_MAP.items():
        text = text.replace(char.upper(), digit).replace(char.lower(), digit)
    return text


# --- FUNCȚII SPECIALIZATE PENTRU FIECARE CÂMP ---

def extract_serie(image_gray):
    # Am mărit aria de căutare cu 5%
    roi_coords = (1982, 573, 269, 59)
    x, y, w, h = roi_coords
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_print(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 7').strip()
    return text, roi_coords


def extract_numar(image_gray):
    # Am mărit aria de căutare cu 5%
    roi_coords = (2382, 557, 394, 78)
    x, y, w, h = roi_coords
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_print(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng",
                                       config=r'--psm 7 -c tessedit_char_whitelist=0123456789').strip()
    return text, roi_coords


def extract_cod(image_gray):
    roi_coords = (2236, 648, 185, 105)
    x, y, w, h = roi_coords
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    raw_code = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 8').strip()

    translated_code = translate_chars_to_digits(raw_code.upper())
    try:
        num = int(re.sub(r'\D', '', translated_code))
        if 1 <= num <= 17:
            return f"{num:02d}", roi_coords, raw_code
    except (ValueError, IndexError):
        pass
    return "necunoscut", roi_coords, raw_code


def detect_checkbox_continuare(image_gray):
    roi_coords = (2340, 187, 76, 91)
    x, y, w, h = roi_coords
    roi_checkbox = image_gray[y:y + h, x:x + w]

    inner_x_start, inner_y_start = w // 4, h // 4
    inner_w, inner_h = w // 2, h // 2
    roi_center = roi_checkbox[inner_y_start:inner_y_start + inner_h, inner_x_start:inner_x_start + inner_w]

    _, thresh = cv2.threshold(roi_center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black_pixels_ratio = cv2.countNonZero(thresh) / roi_center.size

    if black_pixels_ratio > 0.15:
        return "da", roi_coords
    return "nu", roi_coords


def extract_urgenta(image_gray):
    # Am mărit aria de căutare
    roi_coords = (1580, 95, 250, 80) # (x, y, w, h) - ajustat
    x, y, w, h = roi_coords
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    # Forțăm să caute un singur cuvânt format din 2 caractere
    raw_text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 8').strip()

    translated_text = translate_chars_to_digits(raw_text.upper())
    # Păstrăm doar cifrele
    digits = re.sub(r'\D', '', translated_text)
    if len(digits) == 2:
        return digits, roi_coords, raw_text
    return "necunoscut", roi_coords, raw_text


def extract_cnp_copil(image_gray):
    # Coordonate estimative. VA TREBUI SA LE AJUSTEZI TU!
    # Am pus o zona larga, bazata pe pozitia CNP-ului adultului din harta anterioara
    roi_coords = (794, 955, 400, 100)  # (x, y, w, h) - estimativ
    x, y, w, h = roi_coords
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    # Căutăm exact 13 cifre
    raw_cnp = pytesseract.image_to_string(processed_roi, lang="ron+eng",
                                          config=r'--psm 8 -c tessedit_char_whitelist=0123456789').strip()

    cnp_digits = re.sub(r'\D', '', raw_cnp)  # Eliminăm orice non-cifră
    if len(cnp_digits) == 13:
        return cnp_digits, roi_coords, raw_cnp
    return "necunoscut", roi_coords, raw_cnp


# --- MAIN LOGIC ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_cv_color = np.array(image)
    img_gray = cv2.cvtColor(img_cv_color, cv2.COLOR_RGB2GRAY)

    TARGET_WIDTH = 3000
    height, width = img_gray.shape
    scale = TARGET_WIDTH / width
    new_height = int(height * scale)
    img_resized_gray = cv2.resize(img_gray, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_CUBIC)
    img_resized_color = cv2.resize(img_cv_color, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_CUBIC)

    # --- Extragere & Afișare ---
    final_data = {}
    raw_texts = {}
    img_with_boxes = img_resized_color.copy()

    # Extragem fiecare câmp folosind funcția sa dedicată

    urgenta_text, (x, y, w, h), raw_urgenta = extract_urgenta(img_resized_gray)
    final_data["Urgenta Medicochirurgicala"] = urgenta_text
    raw_texts["Urgenta"] = raw_urgenta
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Galben pentru urgenta

    checkbox_val, (x, y, w, h) = detect_checkbox_continuare(img_resized_gray)
    final_data["In Continuare"] = checkbox_val
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Albastru pentru checkbox

    serie_text, (x, y, w, h) = extract_serie(img_resized_gray)
    final_data["Seria Certificat"] = serie_text if serie_text else "necunoscut"
    raw_texts["Serie"] = serie_text
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

    numar_text, (x, y, w, h) = extract_numar(img_resized_gray)
    final_data["Numar Certificat"] = numar_text if numar_text else "necunoscut"
    raw_texts["Numar"] = numar_text
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cod_text, (x, y, w, h), raw_cod = extract_cod(img_resized_gray)
    final_data["Cod Indemnizatie"] = cod_text
    raw_texts["Cod"] = raw_cod
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cnp_copil_text, (x, y, w, h), raw_cnp_copil = extract_cnp_copil(img_resized_gray)
    final_data["CNP Copil"] = cnp_copil_text
    raw_texts["CNP Copil"] = raw_cnp_copil
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 165, 0), 3)  # Portocaliu pentru CNP Copil

    # --- Procesare și citire pe TOATĂ imaginea ---
    processed_full_image = treat_print(img_resized_gray)
    full_extracted_text = pytesseract.image_to_string(processed_full_image, lang="ron+eng", config=r'--psm 6').strip()

    # --- Afișare rezultate ---
    st.subheader("🧾 Text Brut Extras din Toată Imaginea")
    st.text_area("Text complet", full_extracted_text, height=300)

    st.subheader("🏆 Rezultate Finale")
    st.dataframe(pd.DataFrame([final_data]))

    st.subheader("✅ Verificare Vizuală Chirurgicală")
    st.image(img_with_boxes, caption="Zonele de interes operate")

    # ... (codul de export Excel)
