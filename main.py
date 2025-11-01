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

def extract_from_text(text, data_dict):
    """Încearcă să completeze câmpurile goale din data_dict folosind textul extras."""

    # Seria și Numărul
    if data_dict.get("Seria Certificat", "necunoscut") == "necunoscut":
        pattern_seria = r"seria\s+([A-Z0-9]+)"
        match_seria = re.search(pattern_seria, text, re.IGNORECASE)
        if match_seria:
            data_dict["Seria Certificat"] = match_seria.group(1).strip()

    if data_dict.get("Numar Certificat", "necunoscut") == "necunoscut":
        pattern_numar = r"nr\.\s*([0-9]+)"
        match_numar = re.search(pattern_numar, text, re.IGNORECASE)
        if match_numar:
            data_dict["Numar Certificat"] = match_numar.group(1).strip()

    # Codul de indemnizație (vânătoare flexibilă cu validare)
    if data_dict.get("Cod Indemnizatie", "necunoscut") == "necunoscut":
        # Lista de modele, de la cel mai specific la cel mai general
        cod_patterns = [
            r"Cod\s+indemnizatie\s*\(1-17\):?\s*(\d{2})",  # Modelul ideal
            r"indemnizatie\s*\(1-17\):?\s*(\d{2})",       # Fără "Cod"
            r"\(1-17\):?\s*(\d{2})",                       # Doar "(1-17)"
            r"(\d{2})"                                     # Orice două cifre, ca ultimă soluție
        ]

        for pattern in cod_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    num = int(match.group(1).strip())
                    # Validare: codul trebuie să fie între 1 și 17
                    if 1 <= num <= 17:
                        # Formatare la două cifre (ex: 6 -> "06")
                        data_dict["Cod Indemnizatie"] = f"{num:02d}"
                        break  # Am găsit un cod valid, oprim căutarea
                except (ValueError, IndexError):
                    continue # Ignorăm dacă nu putem converti în număr
    
    return data_dict

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # --- Redimensionare la o rezoluție mai mare ---
    max_dim = 2600
    height, width = img_gray.shape
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_gray = cv2.resize(img_gray, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_AREA)

    st.subheader("🖼️ Variante de Imagine Procesată")
    cols = st.columns(len(PROCESSING_ALGORITHMS))
    processed_images = {}

    for col, (name, func) in zip(cols, PROCESSING_ALGORITHMS.items()):
        with col:
            st.write(f"**{name}**")
            img_p = func(img_gray)
            processed_images[name] = img_p
            st.image(img_p, use_container_width=True)

    # --- Extragere în cascadă ---
    st.subheader("🔍 Rezultate OCR & Extragere")
    final_data = {
        "Seria Certificat": "necunoscut",
        "Numar Certificat": "necunoscut",
        "Cod Indemnizatie": "necunoscut"
    }

    for name, img_p in processed_images.items():
        with st.expander(f"🔬 Rezultate detaliate pentru algoritmul: **{name}**"):
            st.write(f"Se rulează OCR pe imaginea procesată cu **{name}**...")
            
            # Încercăm diferite moduri de segmentare
            for psm in [6, 11, 12]:
                custom_config = f'--psm {psm}'
                extracted_text = pytesseract.image_to_string(img_p, lang="ron+eng", config=custom_config)
                
                # Verificăm dacă textul extras conține ceva relevant
                if len(extracted_text.strip()) > 5:
                    st.text_area(f"Text extras (psm={psm})", extracted_text, height=150, key=f"{name}_psm_{psm}")
                    final_data = extract_from_text(extracted_text, final_data)

        # Oprim dacă am găsit toate datele
        if all(val != "necunoscut" for val in final_data.values()):
            st.success(f"Toate datele au fost găsite folosind algoritmul {name}! 🎉")
            break

    # --- Afișare rezultate finale ---
    st.subheader("📊 Date Finale Extrase")
    df = pd.DataFrame([final_data])
    st.dataframe(df)

    # Buton pentru export Excel
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Certificat")
    towrite.seek(0)
    st.download_button(
        label="💾 Descarcă Excel",
        data=towrite,
        file_name="certificat_medical.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.write("---")
st.caption("Proiect demonstrativ – utilizare Tesseract OCR pentru extragerea datelor din certificate medicale.")
