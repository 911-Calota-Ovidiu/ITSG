import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("🩺 Certificate Medicale OCR (Demo simplu)")

st.write("Încarcă o imagine cu un certificat medical pentru a extrage textul automat.")

uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Citim imaginea
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagine încărcată", use_container_width=True)

    # Convertim imaginea în format pentru OpenCV
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    st.write("🔍 Procesăm imaginea cu Tesseract OCR...")

    # Aplicăm OCR
    extracted_text = pytesseract.image_to_string(img_gray, lang="ron")

    # Afișăm textul extras
    st.subheader("🧾 Text extras:")
    st.text_area("Rezultat OCR", extracted_text, height=200)

    # Extragere simplificată câteva câmpuri cu regex (superficial)
    import re
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
