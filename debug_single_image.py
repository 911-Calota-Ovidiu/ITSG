import os
from ocr_utils import easyocr_words, find_label_boxes, normalize_text

IMAGE_PATH = "/home/tchibo/Documents/ITSG/training_data/1384563.png"

def debug_cnp():
    """Runs a detailed analysis on a single image to debug CNP extraction."""
    print(f"--- Analyzing CNP extraction for: {os.path.basename(IMAGE_PATH)} ---")

    words = easyocr_words(IMAGE_PATH)
    if not words:
        print("ERROR: No text could be detected on the image.")
        return

    found_labels = find_label_boxes(words)

    if 'CNP' in found_labels:
        print("\nSUCCESS: The 'CNP' label was found.")
        cnp_label_entry = found_labels['CNP']
        label_word = cnp_label_entry['word']
        lx, ly, lw, lh = label_word['bbox']
        print(f"  - Label Text: '{label_word['text']}'")
        print(f"  - Label BBox: {label_word['bbox']}")

        # Define a horizontal search strip on the same line as the label
        search_strip_top = ly - (lh * 0.5)
        search_strip_bottom = ly + (lh * 1.5)
        
        # Find all words on the same line to the right of the label
        line_words = []
        for word in words:
            word_y_center = word['cy']
            if search_strip_top < word_y_center < search_strip_bottom and word['cx'] > label_word['cx']:
                line_words.append(word)
                
        line_words.sort(key=lambda w: w['cx'])
        line_text = " ".join([w['text'] for w in line_words])

        print("\n--- Text found on the same line as CNP label ---")
        if line_text:
            print(f">>> {line_text}")
        else:
            print("No text was found to the right of the CNP label on the same line.")

    else:
        print("\nFAILURE: The 'CNP' label was NOT found.")
        print("--- All detected words (and why they didn't match) ---")
        for i, word in enumerate(words):
            normalized = normalize_text(word['text'])
            print(f"  - Word {i}: '{word['text']}' (Normalized: '{normalized}')")

if __name__ == "__main__":
    debug_cnp()
