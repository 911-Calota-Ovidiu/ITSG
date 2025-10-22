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

st.title("🩺 Certificate Medicale OCR (Demo robust)")
st.write("Încarcă o imagine cu un certificat medical pentru a extrage textul automat.")

uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"])

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

    # --- Extragere simplificată câteva câmpuri cu regex ---
    nume = re.search(r"Nume\s*[:\-]?\s*([A-ZĂÂÎȘȚa-zăâîșț\s]+)", extracted_text)
    cnp = re.search(r"\b\d{13}\b", extracted_text)
    cod_boala = re.search(r"[A-Z]\d{2}\.\d", extracted_text)
    data_inceput = re.search(r"\d{2}\.\d{2}\.\d{4}", extracted_text)

    data_dict = {
        "Nume pacient": [nume.group(1).strip() if nume else "necunoscut"],
        "CNP": [cnp.group(0) if cnp else "necunoscut"],
        "Cod boală": [cod_boala.group(0) if cod_boala else "necunoscut"],
        "Data început concediu": [data_inceput.group(0) if data_inceput else "necunoscut"]
    }

    df = pd.DataFrame(data_dict)
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
