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

def extract_serie(image_gray, x_offset=0, y_offset=0):
    roi_coords = (1982, 573, 269, 59)
    x, y, w, h = roi_coords
    x += x_offset  # Apply x offset
    y += y_offset  # Apply y offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_print(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 7').strip()
    return text, (x, y, w, h)  # Return adjusted coords


def extract_numar(image_gray, x_offset=0, y_offset=0):
    roi_coords = (2382, 557, 394, 78)
    x, y, w, h = roi_coords
    x += x_offset  # Apply x offset
    y += y_offset  # Apply y offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_print(roi)
    text = pytesseract.image_to_string(processed_roi, lang="ron+eng",
                                       config=r'--psm 7 -c tessedit_char_whitelist=0123456789').strip()
    return text, (x, y, w, h)  # Return adjusted coords


def extract_cod(image_gray, x_offset=0, y_offset=0):
    roi_coords = (2236, 648, 185, 105)
    x, y, w, h = roi_coords
    x += x_offset  # Apply x offset
    y += y_offset  # Apply y offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    raw_code = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 8').strip()

    translated_code = translate_chars_to_digits(raw_code.upper())
    try:
        num = int(re.sub(r'\D', '', translated_code))
        if 1 <= num <= 17:
            return f"{num:02d}", (x, y, w, h), raw_code  # Return adjusted coords
    except (ValueError, IndexError):
        pass
    return "necunoscut", (x, y, w, h), raw_code  # Return adjusted coords


def detect_checkbox_continuare(image_gray, x_offset=0, y_offset=0):
    roi_coords = (2340, 187, 76, 91)
    x, y, w, h = roi_coords
    x += x_offset  # Apply x offset
    y += y_offset  # Apply y offset
    roi_checkbox = image_gray[y:y + h, x:x + w]

    inner_x_start, inner_y_start = w // 4, h // 4
    inner_w, inner_h = w // 2, h // 2
    roi_center = roi_checkbox[inner_y_start:inner_y_start + inner_h, inner_x_start:inner_x_start + inner_w]

    _, thresh = cv2.threshold(roi_center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black_pixels_ratio = cv2.countNonZero(thresh) / roi_center.size

    if black_pixels_ratio > 0.15:
        return "da", (x, y, w, h)  # Return adjusted coords
    return "nu", (x, y, w, h)  # Return adjusted coords


def extract_urgenta(image_gray, x_offset=0, y_offset=0):
    # Am mărit aria de căutare
    roi_coords = (1580, 95, 250, 80)  # (x, y, w, h) - ajustat
    x, y, w, h = roi_coords
    x += x_offset  # Apply x offset
    y += y_offset  # Apply y offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    # Forțăm să caute un singur cuvânt format din 2 caractere
    raw_text = pytesseract.image_to_string(processed_roi, lang="ron+eng", config=r'--psm 8').strip()

    translated_text = translate_chars_to_digits(raw_text.upper())
    # Păstrăm doar cifrele
    digits = re.sub(r'\D', '', translated_text)
    if len(digits) == 2:
        return digits, (x, y, w, h), raw_text  # Return adjusted coords
    return "necunoscut", (x, y, w, h), raw_text  # Return adjusted coords


def extract_cnp_copil(image_gray, x_offset=0, y_offset=0):
    # Coordonate estimative. VA TREBUI SA LE AJUSTEZI TU!
    # Am pus o zona larga, bazata pe pozitia CNP-ului adultului din harta anterioara
    roi_coords = (794, 955, 400, 100)  # (x, y, w, h) - estimativ
    x, y, w, h = roi_coords
    x += x_offset  # Apply x offset
    y += y_offset  # Apply y offset
    roi = image_gray[y:y + h, x:x + w]
    processed_roi = treat_handwriting(roi)
    # Căutăm exact 13 cifre
    raw_cnp = pytesseract.image_to_string(processed_roi, lang="ron+eng",
                                          config=r'--psm 8 -c tessedit_char_whitelist=0123456789').strip()

    cnp_digits = re.sub(r'\D', '', raw_cnp)  # Eliminăm orice non-cifră
    if len(cnp_digits) == 13:
        return cnp_digits, (x, y, w, h), raw_cnp  # Return adjusted coords
    return "necunoscut", (x, y, w, h), raw_cnp  # Return adjusted coords


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

    # --- Image Alignment ---
    target_phrase = "CERTIFICAT DE CONCEDIU MEDICAL"
    target_words = [word.lower() for word in target_phrase.split()]

    aligned_img_color = img_resized_color.copy()
    aligned_img_gray = img_resized_gray.copy()

    original_width = aligned_img_gray.shape[1]
    original_height = aligned_img_gray.shape[0]

    # Perform OCR on the resized grayscale image to find the phrase
    d = pytesseract.image_to_data(aligned_img_gray, lang="ron+eng", config=r'--psm 3', output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])

    print(f"[DEBUG] Tesseract detected words: {d['text']}")

    phrase_found = False
    alignment_shift_x = 0  # This will be the shift applied to ROIs
    alignment_shift_y = 0  # This will be the shift applied to ROIs

    # Iterate through the detected words to find the phrase
    for i in range(n_boxes - len(target_words) + 1):
        current_sequence = [d['text'][i + j].lower() for j in range(len(target_words))]

        if current_sequence == target_words:
            # Phrase found, calculate its bounding box
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')

            for j in range(len(target_words)):
                word_index = i + j
                min_x = min(min_x, d['left'][word_index])
                max_x = max(max_x, d['left'][word_index] + d['width'][word_index])
                min_y = min(min_y, d['top'][word_index])
                max_y = max(max_y, d['top'][word_index] + d['height'][word_index])

            if min_x != float('inf'):  # If words were found
                phrase_center_x = (min_x + max_x) // 2
                phrase_center_y = (min_y + max_y) // 2

                # Define target position for the top-left corner of the phrase
                target_x_position = original_width // 20  # 5% from left edge
                target_y_position = original_height // 25 # 4% from top edge

                # Calculate the required shift for the content
                shift_x = target_x_position - min_x
                shift_y = target_y_position - min_y

                print(f"[DEBUG] Phrase '{target_phrase}' found at x: {min_x}-{max_x}, y: {min_y}-{max_y}.")
                print(f"[DEBUG] Target position: ({target_x_position}, {target_y_position}). Calculated shift_x: {shift_x}, shift_y: {shift_y}")

                # Calculate padding needed to prevent cropping for negative shifts
                pad_left = max(0, -shift_x)
                pad_top = max(0, -shift_y)

                # Total translation for warpAffine (includes padding)
                tx_for_warpAffine = shift_x + pad_left
                ty_for_warpAffine = shift_y + pad_top

                new_output_width = original_width + abs(shift_x)
                new_output_height = original_height + abs(shift_y)

                M = np.float32([[1, 0, tx_for_warpAffine], [0, 1, ty_for_warpAffine]])

                aligned_img_color = cv2.warpAffine(img_resized_color, M, (new_output_width, new_output_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                aligned_img_gray = cv2.warpAffine(img_resized_gray, M, (new_output_width, new_output_height), borderMode=cv2.BORDER_CONSTANT, borderValue=255)  # White border for gray image

                # The actual shift of the original image's top-left corner within the new canvas
                # This is the amount by which all original ROI coordinates need to be shifted
                alignment_shift_x = tx_for_warpAffine
                alignment_shift_y = ty_for_warpAffine

                print(f"[DEBUG] warpAffine tx: {tx_for_warpAffine}, ty: {ty_for_warpAffine}")
                print(f"[DEBUG] new_output_width: {new_output_width}, new_output_height: {new_output_height}")
                print(f"[DEBUG] alignment_shift_x (for ROIs): {alignment_shift_x}, alignment_shift_y (for ROIs): {alignment_shift_y}")

                # Draw rectangle around the detected phrase on the aligned image for visual debugging
                # The coordinates for drawing the rectangle need to be adjusted by the alignment_shift_x/y
                cv2.rectangle(aligned_img_color, 
                              (int(min_x + tx_for_warpAffine), int(min_y + ty_for_warpAffine)), 
                              (int(max_x + tx_for_warpAffine), int(max_y + ty_for_warpAffine)), 
                              (255, 0, 255), 5) # Magenta color, thicker line


            phrase_found = True
            break  # Phrase found and image aligned, exit loop

    if not phrase_found:
        st.warning(f"Fraza \'{target_phrase}\' nu a fost găsită pentru aliniere. Procesarea continuă fără aliniere.")
        print(f"[DEBUG] Phrase '{target_phrase}' NOT found in detected words.")
        print(f"[DEBUG] Full Tesseract data when phrase not found:")
        for i in range(n_boxes):
            if d['text'][i].strip() != '': # Only print non-empty words
                print(f"  Word: '{d['text'][i]}', BBox: ({d['left'][i]}, {d['top'][i]}, {d['width'][i]}, {d['height'][i]}), Conf: {d['conf'][i]}")

    # Use the aligned images for further processing
    img_resized_color = aligned_img_color
    img_resized_gray = aligned_img_gray

    # --- Extragere & Afișare ---
    final_data = {}
    raw_texts = {}
    img_with_boxes = img_resized_color.copy()

    # Extragem fiecare câmp folosind funcția sa dedicată

    urgenta_text, (x, y, w, h), raw_urgenta = extract_urgenta(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["Urgenta Medicochirurgicala"] = urgenta_text
    raw_texts["Urgenta"] = raw_urgenta
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Galben pentru urgenta

    checkbox_val, (x, y, w, h) = detect_checkbox_continuare(img_resized_gray, alignment_shift_x, alignment_shift_y)
    final_data["In Continuare"] = checkbox_val
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Albastru pentru checkbox

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

    # Display images side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(aligned_img_color, caption="Imagine Aliniată (cu chenar frază țintă)")
    with col2:
        st.image(img_with_boxes, caption="Zonele de interes operate")

    # ... (codul de export Excel)
