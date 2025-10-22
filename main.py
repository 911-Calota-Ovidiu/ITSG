import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import io
import re

# Configurare Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(layout="wide")
st.title("🩺 Certificate Medicale OCR (Demo robust)")
st.write("Încarcă o imagine cu un certificat medical pentru a extrage textul automat.")

uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"])

def extract_certificate_data(text):
    """Funcție care primește textul OCR și returnează DataFrame cu mai multe coloane."""
    
    def find_first(patterns, text, group=1):
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(group).strip()
        return "necunoscut"

    # Nume pacient
    nume_patterns = [
        r"Nume\s*[:\-]?\s*([A-ZĂÂÎȘȚa-zăâîșț\s]+)",
        r"Angajat\s*[:\-]?\s*([A-ZĂÂÎȘȚa-zăâîșț\s]+)",
        r"Se adevereste ea\s*([A-ZĂÂÎȘȚa-zăâîșț\s]+)"
    ]
    nume = find_first(nume_patterns, text)

    # CNP
    cnp_patterns = [r"\b\d{13}\b"]
    cnp = find_first(cnp_patterns, text, group=0)

    # Cod boală
    cod_boala_patterns = [
        r"[A-Z]\d{2}\.\d",
        r"[A-Z]\d{2}\s\d"
    ]
    cod_boala = find_first(cod_boala_patterns, text, group=0)

    # Data început concediu
    data_inceput_patterns = [
        r"Data început\s*[:\-]?\s*(\d{2}[./]\d{2}[./]\d{4})",
        r"(\d{2}[./]\d{2}[./]\d{4})",
        r"(\d{4}-\d{2}-\d{2})"
    ]
    data_inceput = find_first(data_inceput_patterns, text)

    # Data sfârșit concediu
    data_sfarsit_patterns = [
        r"Data sfârșit\s*[:\-]?\s*(\d{2}[./]\d{2}[./]\d{4})",
        r"Până la\s*(\d{2}[./]\d{2}[./]\d{4})"
    ]
    data_sfarsit = find_first(data_sfarsit_patterns, text)

    # Medic
    medic_patterns = [
        r"Medic\s*[:\-]?\s*([A-ZĂÂÎȘȚa-zăâîșț\s]+)",
        r"Semnătura medicului\s*([A-ZĂÂÎȘȚa-zăâîșț\s]+)"
    ]
    medic = find_first(medic_patterns, text)

    # Seria certificatului
    seria_patterns = [
        r"Seria\s*[:\-]?\s*([A-Z0-9]+)",
        r"Nr\s*[:\-]?\s*([A-Z0-9]+)"
    ]
    seria = find_first(seria_patterns, text)

    data_dict = {
        "Nume pacient": [nume],
        "CNP": [cnp],
        "Cod boală": [cod_boala],
        "Data început concediu": [data_inceput],
        "Data sfârșit concediu": [data_sfarsit],
        "Medic": [medic],
        "Seria certificatului": [seria]
    }

    return pd.DataFrame(data_dict)

if uploaded_file is not None:
    # Citim imaginea
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagine încărcată", use_container_width=True)

    # Convertim în format OpenCV
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # --- Redimensionare moderată pentru imagini mari ---
    max_dim = 1200
    height, width = img_gray.shape
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_gray = cv2.resize(img_gray, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_CUBIC)

    # --- Binarizare simplă globală ---
    _, img_gray = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # --- Detectare și corectare orientare (mai sigur) ---
    try:
        osd = pytesseract.image_to_osd(img_gray)
        rotation_angle = int(re.search(r'Rotate:\s(\d+)', osd).group(1))
        if rotation_angle != 0:
            st.write(f"Rotire detectată: {rotation_angle}° → corectăm...")
            M = cv2.getRotationMatrix2D((img_gray.shape[1]/2, img_gray.shape[0]/2), -rotation_angle, 1)
            img_gray = cv2.warpAffine(img_gray, M, (img_gray.shape[1], img_gray.shape[0]))
    except:
        pass  # dacă nu poate detecta orientarea, continuăm fără rotație

    st.write("🔍 Procesăm imaginea cu Tesseract OCR...")

    # OCR cu Page Segmentation Mode potrivit pentru blocuri de text
    custom_config = r'--psm 6'  # treat image as a block of text
    extracted_text = pytesseract.image_to_string(img_gray, lang="ron", config=custom_config)

    # Afișăm textul extras
    st.subheader("🧾 Text extras:")
    st.text_area("Rezultat OCR", extracted_text, height=200)

    # --- Extragem datele folosind funcția ---
    df = extract_certificate_data(extracted_text)

    st.subheader("📊 Date extrase:")
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
