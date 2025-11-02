import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import io
import re
from image_processing import treat_print, treat_handwriting

st.set_page_config(layout="wide", page_title="Operațiunea: Bisturiul (OCR)")
st.title("🩺 Recunoaștere automată OCR")
st.caption("Aplicație pentru extragerea automată a datelor din certificate medicale. "
           "Toate câmpurile sunt prelucrate prin funcții specializate pentru scris de mână și text tipărit.")

uploaded_file = st.file_uploader("📷 Încarcă o imagine (JPG/PNG)", type=["jpg", "jpeg", "png"])

CHAR_MAP = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'
}

def translate_chars_to_digits(text):
    for char, digit in CHAR_MAP.items():
        text = text.replace(char.upper(), digit).replace(char.lower(), digit)
    return text

def extract_serie(image_gray, x_offset=0, y_offset=0):
    roi_coords = (1982, 573, 269, 59)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_print(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 7').strip()
    return text, (x, y, w, h)

def extract_numar(image_gray, x_offset=0, y_offset=0):
    roi_coords = (2382, 557, 394, 78)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_print(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng",
                                       config=r'--psm 7 -c tessedit_char_whitelist=0123456789').strip()
    return text, (x, y, w, h)

def extract_cod(image_gray, x_offset=0, y_offset=0):
    roi_coords = (2236, 648, 185, 105)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    raw_code = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 8').strip()
    translated_code = translate_chars_to_digits(raw_code.upper())
    try:
        num = int(re.sub(r'\D', '', translated_code))
        if 1 <= num <= 17:
            return f"{num:02d}", (x, y, w, h), raw_code
    except (ValueError, IndexError):
        pass
    return "necunoscut", (x, y, w, h), raw_code

def detect_checkbox_continuare(image_gray, x_offset=0, y_offset=0):
    roi_coords = (2340, 187, 76, 91)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi_checkbox = image_gray[y:y + h, x:x + w]
    inner_x_start, inner_y_start = w // 4, h // 4
    inner_w, inner_h = w // 2, h // 2
    roi_center = roi_checkbox[inner_y_start:inner_y_start + inner_h, inner_x_start:inner_x_start + inner_w]
    _, thresh = cv2.threshold(roi_center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black_pixels_ratio = cv2.countNonZero(thresh) / roi_center.size
    if black_pixels_ratio > 0.15:
        return "da", (x, y, w, h)
    return "nu", (x, y, w, h)

def extract_urgenta(image_gray, x_offset=0, y_offset=0):
    roi_coords = (1580, 95, 250, 80)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    raw_text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 8').strip()
    translated_text = translate_chars_to_digits(raw_text.upper())
    digits = re.sub(r'\D', '', translated_text)
    if len(digits) == 2:
        return digits, (x, y, w, h), raw_text
    return "necunoscut", (x, y, w, h), raw_text

def extract_cnp_copil(image_gray, x_offset=0, y_offset=0):
    roi_coords = (1255, 1030, 960, 100)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    raw_cnp = pytesseract.image_to_string(processed_roi, lang="ron+eng",
                                          config=r'--psm 8 -c tessedit_char_whitelist=0123456789').strip()
    cnp_digits = re.sub(r'\D', '', raw_cnp)
    if len(cnp_digits) == 13:
        return cnp_digits, (x, y, w, h), raw_cnp
    return "necunoscut", (x, y, w, h), raw_cnp

def extract_ambulator_internat_sectia(image_gray, x_offset=0, y_offset=0):
    roi_coords = (180, 1550, 580, 100)
    x, y, w, h = roi_coords
    x += x_offset
    y += y_offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 6').strip()
    return text, (x, y, w, h)

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
    aligned_img_color = img_resized_color.copy()
    aligned_img_gray = img_resized_gray.copy()
    original_width = aligned_img_gray.shape[1]
    original_height = aligned_img_gray.shape[0]
    d = pytesseract.image_to_data(aligned_img_gray, lang="ron+eng", config=r'--psm 3', output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    print(f"[DEBUG] Tesseract detected words: {d['text']}")
    all_text_min_x = float('inf')
    all_text_max_x = float('-inf')
    all_text_min_y = float('inf')
    all_text_max_y = float('-inf')
    found_any_text = False
    for i in range(n_boxes):
        if int(d['conf'][i]) > 30 and d['text'][i].strip() != '':
            all_text_min_x = min(all_text_min_x, d['left'][i])
            all_text_max_x = max(all_text_max_x, d['left'][i] + d['width'][i])
            all_text_min_y = min(all_text_min_y, d['top'][i])
            all_text_max_y = max(all_text_max_y, d['top'][i] + d['height'][i])
            found_any_text = True
    alignment_shift_x = 0
    alignment_shift_y = 0
    if found_any_text:
        text_block_center_x = (all_text_min_x + all_text_max_x) // 2
        text_block_center_y = (all_text_min_y + all_text_max_y) // 2
        image_center_x = original_width // 2
        image_center_y = original_height // 2
        shift_x = image_center_x - text_block_center_x
        shift_y = image_center_y - text_block_center_y
        print(f"[DEBUG] Overall text block found at x: {all_text_min_x}-{all_text_max_x}, y: {all_text_min_y}-{all_text_max_y}.")
        print(f"[DEBUG] Text block center: ({text_block_center_x}, {text_block_center_y}). Image center: ({image_center_x}, {image_center_y}).")
        print(f"[DEBUG] Calculated shift_x: {shift_x}, shift_y: {shift_y}")
        pad_left = max(0, -shift_x)
        pad_top = max(0, -shift_y)
        tx_for_warpAffine = shift_x + pad_left
        ty_for_warpAffine = shift_y + pad_top
        new_output_width = original_width + abs(shift_x)
        new_output_height = original_height + abs(shift_y)
        M = np.float32([[1, 0, tx_for_warpAffine], [0, 1, ty_for_warpAffine]])
        aligned_img_color = cv2.warpAffine(img_resized_color, M, (new_output_width, new_output_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        aligned_img_gray = cv2.warpAffine(img_resized_gray, M, (new_output_width, new_output_height), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        alignment_shift_x = tx_for_warpAffine
        alignment_shift_y = ty_for_warpAffine
        print(f"[DEBUG] warpAffine tx: {tx_for_warpAffine}, ty: {ty_for_warpAffine}")
        print(f"[DEBUG] new_output_width: {new_output_width}, new_output_height: {new_output_height}")
        print(f"[DEBUG] alignment_shift_x (for ROIs): {alignment_shift_x}, alignment_shift_y (for ROIs): {alignment_shift_y}")
        cv2.rectangle(aligned_img_color, 
                      (int(all_text_min_x + tx_for_warpAffine), int(all_text_min_y + ty_for_warpAffine)), 
                      (int(all_text_max_x + tx_for_warpAffine), int(all_text_max_y + ty_for_warpAffine)), 
                      (255, 0, 255), 5)
    else:
        st.warning(f"Nu a fost detectat suficient text pentru aliniere. Procesarea continuă fără aliniere.")
        print(f"[DEBUG] No significant text found for alignment.")
        print(f"[DEBUG] Full Tesseract data when no text found:")
        for i in range(n_boxes):
            if d['text'][i].strip() != '':
                print(f"  Word: '{d['text'][i]}', BBox: ({d['left'][i]}, {d['top'][i]}, {d['width'][i]}, {d['height'][i]}), Conf: {d['conf'][i]}")
    img_resized_color = aligned_img_color
    img_resized_gray = aligned_img_gray
    final_data = {}
    raw_texts = {}
    img_with_boxes = img_resized_color.copy()
    urgenta_text, (x, y, w, h), raw_urgenta = extract_urgenta(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["Urgenta Medicochirurgicala"] = urgenta_text
    raw_texts["Urgenta"] = raw_urgenta
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 3)
    checkbox_val, (x, y, w, h) = detect_checkbox_continuare(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["In Continuare"] = checkbox_val
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 3)
    serie_text, (x, y, w, h) = extract_serie(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["Seria Certificat"] = serie_text if serie_text else "necunoscut"
    raw_texts["Serie"] = serie_text
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
    numar_text, (x, y, w, h) = extract_numar(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["Numar Certificat"] = numar_text if numar_text else "necunoscut"
    raw_texts["Numar"] = numar_text
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cod_text, (x, y, w, h), raw_cod = extract_cod(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["Cod Indemnizatie"] = cod_text
    raw_texts["Cod"] = raw_cod
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cnp_copil_text, (x, y, w, h), raw_cnp_copil = extract_cnp_copil(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["CNP Copil"] = cnp_copil_text
    raw_texts["CNP Copil"] = raw_cnp_copil
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 165, 0), 3)
    ambulator_text, (x, y, w, h) = extract_ambulator_internat_sectia(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["Ambulator/Internat in spital Sectia"] = ambulator_text
    raw_texts["Ambulator/Internat in spital Sectia"] = ambulator_text
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 3)
    processed_full_image = treat_print(img_resized_gray)
    full_extracted_text = pytesseract.image_to_string(processed_full_image, lang="ron+eng", config=r'--psm 6').strip()
    st.success("✅ Extragerea a fost realizată cu succes!")
    st.subheader("🧾 Text brut extras")
    st.text_area("Rezultatul OCR complet", full_extracted_text, height=300)
    st.subheader("📋 Date structurate extrase")
    df_results = pd.DataFrame([final_data])
    st.dataframe(df_results, width='stretch')
    st.subheader("🖼️ Verificare vizuală (zone analizate)")
    st.image(img_with_boxes, caption="Zonele de interes procesate", width=1000)
    st.subheader("📤 Export rezultate")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Rezultate_OCR")
    st.download_button(
        label="💾 Descarcă fișier Excel",
        data=excel_buffer.getvalue(),
        file_name="rezultate_OCR.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
