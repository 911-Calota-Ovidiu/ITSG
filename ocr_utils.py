import cv2
import numpy as np
import pytesseract
import easyocr
import re
from rapidfuzz import fuzz
from datetime import datetime

# ---------- Config ----------
EXPECTED_LABELS = {
    "patient_name": ["nume", "numele si prenumele", "nume și prenume", "nume/prenume", "nume si prenume"],
    "CNP": ["cnp", "cod numeric personal", "cod numeric", "CNP:"],
    "birth_date": ["data nasterii", "data nașterii", "data naşterii"],
    "start_date": ["data de la", "de la", "data inceput", "data început", "data debut"],
    "end_date": ["pana la", "data pana la", "data sfarsit", "data sfârșit", "data final"],
    "diagnosis": ["diagnostic", "motiv concediu", "diagnostice"],
    "doctor_name": ["medic", "nume medic", "semnatura medic", "doctor", "medic semnatura"],
    "ccmat": ["seria", "CCMAT", "Seria CCMAT", "Nr CCMAT", "nr CCMAT", "ccmat nr"]
}
LABEL_MATCH_THRESHOLD = 75

# Initialize EasyOCR reader
eocr_reader = easyocr.Reader(['ro', 'en'], gpu=False)

# ---------- Utilities ----------
def image_to_words(image_path_or_array, lang='ron+eng'):
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang=lang)
    words = []
    n = len(data['text'])
    for i in range(n):
        text = str(data['text'][i]).strip()
        if text == "":
            continue
        try:
            conf = int(float(data['conf'][i]))
        except:
            conf = -1
        x, y, wbox, hbox = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        words.append({
            'text': text,
            'conf': conf,
            'bbox': (x, y, wbox, hbox),
            'cx': x + wbox/2,
            'cy': y + hbox/2
        })
    return words, gray

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
            for variant in label_variants:
                score = fuzz.token_set_ratio(variant, ocr_text)
                if score > best_score:
                    best_score = score
                    best_match = (i, ocr_text, score)
        if best_match and best_score >= threshold:
            i, text, score = best_match
            found[label_key] = {'word_idx': i, 'ocr_text': text, 'score': score, 'word': words[i]}
    return found

def extract_value_for_label(words, label_entry, max_horizontal_gap=0.7, max_vertical_gap=0.25):
    idx = label_entry['word_idx']
    lx, ly, lw, lh = words[idx]['bbox']
    img_w = max([w['bbox'][0] + w['bbox'][2] for w in words])
    img_h = max([w['bbox'][1] + w['bbox'][3] for w in words])
    lcx, lcy = words[idx]['cx'], words[idx]['cy']
    max_hgap = max_horizontal_gap * img_w
    max_vgap = max_vertical_gap * img_h

    # --- If this is a date, look below label only ---
    if 'date' in label_entry['ocr_text'].lower() or 'de la' in label_entry['ocr_text'].lower() or 'pana la' in label_entry['ocr_text'].lower():
        below = []
        for j,w in enumerate(words):
            if j == idx: continue
            x,y,ww,hh = w['bbox']
            if y > ly + lh and abs((x + ww/2) - (lx + lw/2)) < lw*2 and (y - (ly+lh)) < max_vgap:
                below.append((j, w))
        if below:
            below = sorted(below, key=lambda t: (t[1]['bbox'][1], t[1]['bbox'][0]))
            idxs = [b[0] for b in below]
            text = " ".join([words[k]['text'] for k in idxs])
            return text, idxs

    # fallback: try right and below (original)
    candidates = []
    for j, w in enumerate(words):
        if j == idx: continue
        x,y,ww,hh = w['bbox']
        cx, cy = w['cx'], w['cy']
        if x > lx + lw and abs(cy - lcy) < lh*1.5 + hh*0.5 and (x - (lx + lw)) < max_hgap:
            candidates.append((j, w))
    if candidates:
        candidates = sorted(candidates, key=lambda t: (t[1]['bbox'][0], t[1]['bbox'][1]))
        selected = [candidates[0]]
        last_x_end = candidates[0][1]['bbox'][0] + candidates[0][1]['bbox'][2]
        for (j,went) in candidates[1:]:
            gap = went['bbox'][0] - last_x_end
            if gap < max(20, 0.06 * img_w):
                selected.append((j,went))
                last_x_end = went['bbox'][0] + went['bbox'][2]
            else:
                break
        idxs = [s[0] for s in selected]
        text = " ".join([words[k]['text'] for k in idxs])
        return text, idxs

    # fallback below
    below = []
    for j,w in enumerate(words):
        if j == idx: continue
        x,y,ww,hh = w['bbox']
        cx, cy = w['cx'], w['cy']
        if y > ly + lh and abs(cx - (lx + lw/2)) < lw*2 and (y - (ly+lh)) < max_vgap:
            below.append((j, w))
    if below:
        below = sorted(below, key=lambda t: (t[1]['bbox'][1], t[1]['bbox'][0]))
        idxs = [b[0] for b in below]
        text = " ".join([words[k]['text'] for k in idxs])
        return text, idxs

    return None, []

def postprocess_field(field_key, raw_text):
    if raw_text is None: return ''
    s = raw_text.strip()
    if field_key == 'CNP':
        digits = re.sub(r'\D', '', s)
        if len(digits) >= 13:
            return digits[:13]
        return digits
    if 'date' in field_key.lower() or field_key in ('start_date','end_date','birth_date'):
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
    result = eocr_reader.readtext(img)
    words = []
    for bbox, text, prob in result:
        pts = np.array(bbox)
        x = int(pts[:,0].min())
        y = int(pts[:,1].min())
        wbox = int(pts[:,0].max() - x)
        hbox = int(pts[:,1].max() - y)
        words.append({
            'text': text.strip(),
            'conf': int(prob * 100),
            'bbox': (x, y, wbox, hbox),
            'cx': x + wbox/2,
            'cy': y + hbox/2
        })
    return words

def auto_extract_structured_fields(image_path_or_array):
    words_t, gray = image_to_words(image_path_or_array)
    words_e = easyocr_words(image_path_or_array)
    words = words_t + words_e

    found_labels = find_label_boxes(words)
    results = {}
    all_text = " ".join([w['text'] for w in words])

    for key, entry in found_labels.items():
        val_text, idxs = extract_value_for_label(words, entry)

        # fallback regex
        if (val_text is None or val_text.strip() == ''):
            if key == 'CNP':
                m = re.search(r'\b\d{13}\b', all_text)
                if m: val_text = m.group(0)
            elif key == 'ccmat':
                m = re.search(r'(Seria\s*CCMAT\s*Nr\.?\s*\d+)', all_text, re.IGNORECASE)
                if m: val_text = m.group(1)
            elif 'date' in key:
                m = re.search(r'\b\d{2}[./-]\d{2}[./-]\d{2}\b', all_text)
                if m: val_text = m.group(0)

        processed = postprocess_field(key, val_text)
        results[key] = {'raw': val_text, 'value': processed, 'label_score': entry['score']}
    return results
