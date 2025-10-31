import cv2
import numpy as np
import easyocr
import re
from rapidfuzz import fuzz
from datetime import datetime

# ---------- Config ----------
EXPECTED_LABELS = {
    "medical_emergency_code": ["urgenta medico-", "urgență medico-", "urgență medico"],
    "continued": ["in continuare", "în continuare"],
    "series": ["seria", "serie", "Seria", "CERTIFICAT DE CONCEDIU MEDICAL Seria"],
    "number": ["nr", "Nr", "nr.", "Nr."],
    "indemnification_code": ["cod indemnizatie", "Cod indemnizație (1-17)"],
    "CNP": ["cnp", "cod numeric personal:", "cod numeric", "CNP:"],
    "child_CNP": ["Cod numeric personal al copilului bolnav/", "Cod numeric personal al copilului bolnav/ pacientului cu afectiuni oncologice"],
    "start_date": ["data de la", "de la", "data inceput", "data început", "data debut", "de la zi/luna/an"],
    "end_date": ["pana la", "data pana la", "data sfarsit", "data sfârșit", "data final", "pana la zi/luna/an", "până la zi/luna/an", "până la"],
    "diagnosis": ["diagnostic", "Cod diagnostic"],
    "with_CAS": ["cu CAS"],
    "doctor_name": ["medic", "nume medic", "semnatura medic", "doctor", "medic semnatura", "parafa si semnatura medicului", "parafa medic", "Medic/Semnatura/Parafa"],
}

LABEL_MATCH_THRESHOLD = 70 # Lowered from 75 to be more forgiving

# Initialize EasyOCR reader for Romanian only
eocr_reader = easyocr.Reader(['ro'], gpu=False)

# ---------- Utilities ----------
def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r'[^\w\s\d\.\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def find_label_boxes(words, expected_labels=EXPECTED_LABELS, threshold=LABEL_MATCH_THRESHOLD):
    search_pool = {i: normalize_text(w['text']) for i, w in enumerate(words)}
    found = {}
    for label_key, label_variants in expected_labels.items():
        best_match = None
        best_score = 0
        for i, ocr_text in search_pool.items():
            if len(ocr_text) < 3: continue
            for variant in label_variants:
                score = fuzz.token_set_ratio(variant, ocr_text)
                if score > best_score:
                    best_score = score
                    best_match = (i, ocr_text, score)
        if best_match and best_score >= threshold:
            # Avoid matching the same word to multiple labels
            if best_match[0] not in [f.get('word_idx') for f in found.values()]:
                i, text, score = best_match
                found[label_key] = {'word_idx': i, 'ocr_text': text, 'score': score, 'word': words[i]}
    return found

def extract_value_for_label(words, label_entry, max_horizontal_gap_ratio=0.5, max_vertical_gap_ratio=0.2):
    idx = label_entry['word_idx']
    lx, ly, lw, lh = words[idx]['bbox']
    lcx, lcy = words[idx]['cx'], words[idx]['cy']
    img_w = max([w['bbox'][0] + w['bbox'][2] for w in words]) if words else 1

    candidates = []
    for j, w in enumerate(words):
        if j == idx: continue
        x, y, ww, hh = w['bbox']
        cx, cy = w['cx'], w['cy']
        if x > lcx and abs(cy - lcy) < lh * 1.5 and (x - (lx + lw)) < img_w * max_horizontal_gap_ratio:
            candidates.append((j, w))
    
    if candidates:
        candidates = sorted(candidates, key=lambda t: t[1]['bbox'][0])
        selected_words_indices = [candidates[0][0]]
        last_word_idx = candidates[0][0]
        
        for (j, went) in candidates[1:]:
            last_word_bbox = words[last_word_idx]['bbox']
            last_x_end = last_word_bbox[0] + last_word_bbox[2]
            gap = went['bbox'][0] - last_x_end
            
            # New robust gap logic: gap should be less than ~2x the height of the last word.
            if gap < (last_word_bbox[3] * 2.5):
                selected_words_indices.append(j)
                last_word_idx = j
            else:
                break
        text = " ".join([words[k]['text'] for k in selected_words_indices])
        return text, selected_words_indices

    return None, []

def postprocess_field(field_key, raw_text):
    if raw_text is None: return ''
    s = raw_text.strip()
    if field_key == 'CNP' or field_key == 'child_CNP':
        digits = re.sub(r'\D', '', s)
        if len(digits) >= 13:
            return digits[:13]
        return digits
    if field_key == 'number':
        # Find a 7-digit number specifically
        numbers = re.findall(r'\b(\d{7})\b', s)
        return numbers[0] if numbers else s
    if 'date' in field_key.lower() or field_key in ('start_date','end_date'):
        s2 = re.sub(r'[^0-9./-]', '', s)
        for fmt in ['%d.%m.%Y','%d.%m.%y','%d/%m/%Y','%d/%m/%y','%d-%m-%Y','%d-%m-%y']:
            try:
                dt = datetime.strptime(s2, fmt)
                return dt.strftime('%d.%m.%Y')
            except:
                pass
        return s
    return s

def easyocr_words(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    
    result = eocr_reader.readtext(image, detail=1, paragraph=False, mag_ratio=1.7)

    words = []
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        x = int(min(tl[0], bl[0]))
        y = int(min(tl[1], tr[1]))
        wbox = int(max(tr[0], br[0]) - x)
        hbox = int(max(bl[1], br[1]) - y)
        words.append({
            'text': text.strip(),
            'conf': int(prob * 100),
            'bbox': (x, y, wbox, hbox),
            'cx': x + wbox/2,
            'cy': y + hbox/2
        })
    return words

def get_bounding_box_for_indices(words, indices):
    if not indices:
        return None
    bboxes = [words[i]['bbox'] for i in indices]
    min_x = min(b[0] for b in bboxes)
    min_y = min(b[1] for b in bboxes)
    max_x_w = max(b[0] + b[2] for b in bboxes)
    max_y_h = max(b[1] + b[3] for b in bboxes)
    return (min_x, min_y, max_x_w - min_x, max_y_h - min_y)

def auto_extract_structured_fields(image_path_or_array):
    words = easyocr_words(image_path_or_array)
    found_labels = find_label_boxes(words)
    results = {}
    all_text = " ".join([w['text'] for w in words])

    # --- Special, robust handling for CNP ---
    if 'CNP' in found_labels:
        cnp_label_entry = found_labels['CNP']
        label_word = cnp_label_entry['word']
        lx, ly, lw, lh = label_word['bbox']
        
        search_strip_top = ly - lh
        search_strip_bottom = ly + lh
        
        line_words_indices = []
        for i, word in enumerate(words):
            if word['bbox'][1] < search_strip_bottom and word['bbox'][1] + word['bbox'][3] > search_strip_top and word['cx'] > label_word['cx']:
                line_words_indices.append(i)
                
        line_words_indices.sort(key=lambda i: words[i]['cx'])
        line_text = " ".join([words[i]['text'] for i in line_words_indices])
        
        cleaned_line = re.sub(r'[^\d]', '', line_text)
        cnp_match = re.search(r'\b\d{13}\b', cleaned_line)
        
        val_text = cnp_match.group(0) if cnp_match else None
        processed = postprocess_field('CNP', val_text)
        # For CNP, we'll just box the whole line for simplicity
        value_bbox = get_bounding_box_for_indices(words, line_words_indices)

        results['CNP'] = {'raw': val_text, 'value': processed, 'label_score': cnp_label_entry['score'], 'bbox': value_bbox}
        if 'CNP' in found_labels: del found_labels['CNP']

    # --- Generic handling for all other labels ---
    for key, entry in found_labels.items():
        val_text, idxs = extract_value_for_label(words, entry)
        processed = postprocess_field(key, val_text)
        value_bbox = get_bounding_box_for_indices(words, idxs)
        results[key] = {'raw': val_text, 'value': processed, 'label_score': entry['score'], 'bbox': value_bbox}
        
    return results
