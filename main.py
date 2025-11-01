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
st.title("🩺 OPERAȚIUNEA: Bisturiul (v2)")
st.write("Se aplică tratament specializat pe fiecare zonă de interes.")

uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"])

# --- "Schema" pentru Codul de Indemnizație ---
CHAR_MAP = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'
}

def translate_chars_to_digits(text):
    for char, digit in CHAR_MAP.items():
        text = text.replace(char.upper(), digit).replace(char.lower(), digit)
    return text

def extract_and_draw(image_color, image_gray):
    final_data = {"Seria Certificat": "necunoscut", "Numar Certificat": "necunoscut", "Cod Indemnizatie": "necunoscut", "In Continuare": "nu"}
    raw_texts = {}
    img_with_boxes = image_color.copy()

    roi_coords = {
        "Serie": (1988, 574, 256, 56),
        "Numar": (2391, 559, 375, 74),
        "Cod": (2236, 648, 185, 105),
        "Checkbox_Continuare": (2340, 187, 76, 91)
    }

    # --- Analiza Checkbox "In Continuare" ---
    x, y, w, h = roi_coords["Checkbox_Continuare"]
    roi_checkbox = image_gray[y:y+h, x:x+w]
    # Binarizăm pentru a număra pixelii negri
    _, thresh = cv2.threshold(roi_checkbox, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black_pixels = cv2.countNonZero(thresh)
    total_pixels = roi_checkbox.size
    # Dacă mai mult de 10% din pixeli sunt negri, e bifat
    if (black_pixels / total_pixels) > 0.1:
        final_data["In Continuare"] = "da"
    # Desenăm chenarul de verificare (albastru)
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # --- Operația pe SERIE ---
    x, y, w, h = roi_coords["Serie"]
    roi_serie = image_gray[y:y+h, x:x+w]
    processed_roi_serie = treat_print(roi_serie)
    text_serie = pytesseract.image_to_string(processed_roi_serie, lang="ron+eng", config=r'--psm 7').strip()
    raw_texts["Serie"] = text_serie
    if text_serie:
        final_data["Seria Certificat"] = text_serie
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # --- Operația pe NUMĂR ---
    x, y, w, h = roi_coords["Numar"]
    roi_numar = image_gray[y:y+h, x:x+w]
    processed_roi_numar = treat_print(roi_numar)
    text_numar = pytesseract.image_to_string(processed_roi_numar, lang="ron+eng", config=r'--psm 7 -c tessedit_char_whitelist=0123456789').strip()
    raw_texts["Numar"] = text_numar
    if text_numar:
        final_data["Numar Certificat"] = text_numar
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # --- Operația pe COD ---
    x, y, w, h = roi_coords["Cod"]
    roi_cod = image_gray[y:y+h, x:x+w]
    processed_roi_cod = treat_handwriting(roi_cod)
    raw_code = pytesseract.image_to_string(processed_roi_cod, lang="ron+eng", config=r'--psm 8').strip()
    raw_texts["Cod"] = raw_code
    if raw_code:
        translated_code = translate_chars_to_digits(raw_code.upper())
        try:
            num = int(re.sub(r'\D', '', translated_code))
            if 1 <= num <= 17:
                final_data["Cod Indemnizatie"] = f"{num:02d}"
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
        except (ValueError, IndexError):
            pass

    return final_data, img_with_boxes, raw_texts

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

    final_data, img_with_boxes, raw_texts = extract_and_draw(img_resized_color, img_resized_gray)

    # --- Procesare și citire pe TOATĂ imaginea ---
    processed_image = treat_print(img_resized_gray)
    full_extracted_text = pytesseract.image_to_string(processed_image, lang="ron+eng", config=r'--psm 6').strip()

    # --- Afișare rezultate ---
    st.subheader("🧾 Text Brut Extras din Toată Imaginea")
    st.text_area("Text complet", full_extracted_text, height=300)

    st.subheader("🏆 Rezultate Finale")
    st.dataframe(pd.DataFrame([final_data]))

    st.subheader("✅ Verificare Vizuală Chirurgicală")
    st.image(img_with_boxes, caption="Zonele de interes operate")

    st.subheader("🧾 Text Brut Extras (Debug)")
    st.text_area("Text din zona SERIE", raw_texts.get("Serie", ""), height=50)
    st.text_area("Text din zona NUMAR", raw_texts.get("Numar", ""), height=50)
    st.text_area("Text din zona COD", raw_texts.get("Cod", ""), height=50)

    # ... (codul de export Excel)
