import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import io
import re
from image_processing import PROCESSING_ALGORITHMS

# Configurare Tesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(layout="wide")
st.title("🩺 Certificate Medicale OCR (Demo robust)")
st.write("Încarcă o imagine cu un certificat medical pentru a extrage textul automat.")

uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"])

# --- "Schema" pentru Codul de Indemnizație ---
CHAR_MAP = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'
}

def translate_chars_to_digits(text):
    for char, digit in CHAR_MAP.items():
        text = text.replace(char.upper(), digit).replace(char.lower(), digit)
    return text

def extract_from_text(text, data_dict):
    # Seria și Numărul
    if data_dict.get("Seria Certificat", "necunoscut") == "necunoscut":
        match = re.search(r"seria\s+([A-Z0-9]+)", text, re.IGNORECASE)
        if match:
            data_dict["Seria Certificat"] = match.group(1).strip()

    if data_dict.get("Numar Certificat", "necunoscut") == "necunoscut":
        match = re.search(r"nr\.\s*([0-9]+)", text, re.IGNORECASE)
        if match:
            data_dict["Numar Certificat"] = match.group(1).strip()

    # Codul de Indemnizație cu "Schema" extinsă
    if data_dict.get("Cod Indemnizatie", "necunoscut") == "necunoscut":
        cod_patterns = [
            r"indemnizatie\s*\(1-17\):?\s*\[(\d{1,2})\]",
            r"indemnizatie\s*\(1-17\):?\s*([A-Z0-9]{2})",
        ]
        for pattern in cod_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_code = match.group(1).strip().upper()
                translated_code = translate_chars_to_digits(raw_code)
                try:
                    num = int(translated_code)
                    if 1 <= num <= 17:
                        data_dict["Cod Indemnizatie"] = f"{num:02d}"
                        break
                except (ValueError, IndexError):
                    continue
    return data_dict

def draw_boxes(image, data, ocr_df):
    img_with_boxes = image.copy()
    for key, value in data.items():
        if value != "necunoscut":
            exact_matches = ocr_df[ocr_df['text'].str.strip().str.upper() == value.upper()]
            if not exact_matches.empty:
                row = exact_matches.iloc[0]
                (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return img_with_boxes

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_cv_color = np.array(image)
    img_gray = cv2.cvtColor(img_cv_color, cv2.COLOR_RGB2GRAY)

    # Redimensionare la lățimea de aur
    TARGET_WIDTH = 2500
    height, width = img_gray.shape
    scale = TARGET_WIDTH / width
    new_height = int(height * scale)
    img_resized_gray = cv2.resize(img_gray, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_CUBIC)
    img_resized_color = cv2.resize(img_cv_color, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_CUBIC)

    st.subheader("🖼️ Variante de Imagine Procesată")
    cols = st.columns(len(PROCESSING_ALGORITHMS))
    processed_images = {}
    for col, (name, func) in zip(cols, PROCESSING_ALGORITHMS.items()):
        with col:
            st.write(f"**{name}**")
            img_p = func(img_resized_gray)
            processed_images[name] = img_p
            st.image(img_p, use_container_width=True)

    st.subheader("🔍 Rezultate OCR & Extragere")
    final_data = {"Seria Certificat": "necunoscut", "Numar Certificat": "necunoscut", "Cod Indemnizatie": "necunoscut"}
    best_ocr_df = None
    best_image_name = None
    found_count = 0

    for name, img_p in processed_images.items():
        with st.expander(f"🔬 Rezultate detaliate pentru algoritmul: **{name}**"):
            for psm in [6, 11, 12]:
                custom_config = f'--psm {psm}'
                ocr_df = pytesseract.image_to_data(img_p, lang="ron+eng", config=custom_config, output_type=pytesseract.Output.DATAFRAME)
                ocr_df = ocr_df[ocr_df.conf > 30]
                extracted_text = " ".join(ocr_df['text'].dropna())
                
                if len(extracted_text.strip()) > 5:
                    st.text_area(f"Text extras (psm={psm})", extracted_text, height=150, key=f"{name}_psm_{psm}")
                    current_found_count = len([v for v in final_data.values() if v != "necunoscut"])
                    final_data = extract_from_text(extracted_text, final_data)
                    new_found_count = len([v for v in final_data.values() if v != "necunoscut"])
                    
                    if new_found_count > found_count:
                        found_count = new_found_count
                        best_ocr_df = ocr_df
                        best_image_name = name

        if all(val != "necunoscut" for val in final_data.values()):
            break

    st.subheader("📊 Date Finale Extrase")
    st.dataframe(pd.DataFrame([final_data]))

    if best_ocr_df is not None and best_image_name is not None:
        st.subheader("✅ Verificare Vizuală")
        # Folosim imaginea originală redimensionată pentru a desena chenarele
        img_with_boxes = draw_boxes(img_resized_color, final_data, best_ocr_df)
        st.image(img_with_boxes, caption=f"Valori identificate pe imaginea originală (bazat pe {best_image_name})")

    # ... (codul de export Excel)
