# test_my_app.py
import unittest
import numpy as np
from unittest.mock import patch
from main import (
    translate_chars_to_digits, 
    extract_cod, 
    detect_checkbox_continuare,
    extract_cnp_copil
)

#RULEAZA ASA python -m unittest tests.py

class TestAppLogic(unittest.TestCase):

    def test_translate_chars(self):
        """Tests the character-to-digit replacement logic."""
        print("Running test: test_translate_chars")
        self.assertEqual(translate_chars_to_digits("O"), "0")
        self.assertEqual(translate_chars_to_digits("i"), "1")
        self.assertEqual(translate_chars_to_digits("GBS"), "685")
        self.assertEqual(translate_chars_to_digits("abc"), "abc")
        self.assertEqual(translate_chars_to_digits("Test ZI"), "Test 21")
        self.assertEqual(translate_chars_to_digits(""), "")

    @patch('my_app_logic.pytesseract.image_to_string')
    def test_extract_cod_logic(self, mock_pytesseract):
        """Tests the logic of 'extract_cod' by mocking Tesseract."""
        print("Running test: test_extract_cod_logic")
        
        # We need a dummy image, even though it won't be used by the mock
        dummy_image = np.zeros((3000, 3000), dtype=np.uint8)

        # Test case 1: Tesseract returns a valid number (e.g., "G")
        mock_pytesseract.return_value = "G"
        result, _, _ = extract_cod(dummy_image)
        self.assertEqual(result, "06")

        # Test case 2: Tesseract returns a valid number as a string
        mock_pytesseract.return_value = "12"
        result, _, _ = extract_cod(dummy_image)
        self.assertEqual(result, "12")

        # Test case 3: Tesseract returns a number out of range (e.g., "18")
        mock_pytesseract.return_value = "18"
        result, _, _ = extract_cod(dummy_image)
        self.assertEqual(result, "necunoscut")

        # Test case 4: Tesseract returns letters that get translated but are > 17
        mock_pytesseract.return_value = "SI"
        result, _, _ = extract_cod(dummy_image)
        self.assertEqual(result, "necunoscut") # "51" is > 17
        
        # Test case 5: Tesseract returns garbage text
        mock_pytesseract.return_value = "abc"
        result, _, _ = extract_cod(dummy_image)
        self.assertEqual(result, "necunoscut")
        
        # Test case 6: Tesseract returns single-digit
        mock_pytesseract.return_value = "5"
        result, _, _ = extract_cod(dummy_image)
        self.assertEqual(result, "05")

    def test_detect_checkbox_logic(self):
        """Tests the checkbox detection logic using fake images."""
        print("Running test: test_detect_checkbox_logic")
        
        # Test case 1: An "empty" box (all white)
        # 100x100 array filled with 255 (white)
        empty_box = np.full((3000, 3000), 255, dtype=np.uint8)
        result, _ = detect_checkbox_continuare(empty_box)
        self.assertEqual(result, "nu")

        # Test case 2: A "checked" box (mostly black)
        # 100x100 array filled with 0 (black)
        checked_box = np.zeros((3000, 3000), dtype=np.uint8)
        # Fill the center black so the check passes
        checked_box[500:1000, 500:1000] = 0
        result, _ = detect_checkbox_continuare(checked_box)
        self.assertEqual(result, "da")

        # Test case 3: A "partially checked" box (30% black)
        partial_box = np.full((3000, 3000), 255, dtype=np.uint8)
        # Put black pixels in the center region
        partial_box[500:800, 500:1500] = 0 # 30% black
        result, _ = detect_checkbox_continuare(partial_box)
        self.assertEqual(result, "da") # 30% > 15% threshold

    @patch('my_app_logic.pytesseract.image_to_string')
    def test_extract_cnp_copil_logic(self, mock_pytesseract):
        """Tests the logic of 'extract_cnp_copil' by mocking Tesseract."""
        print("Running test: test_extract_cnp_copil_logic")
        
        dummy_image = np.zeros((3000, 3000), dtype=np.uint8)

        # Test case 1: Valid 13-digit CNP
        mock_pytesseract.return_value = "1234567890123"
        result, _, _ = extract_cnp_copil(dummy_image)
        self.assertEqual(result, "1234567890123")

        # Test case 2: Valid CNP with spaces/noise
        mock_pytesseract.return_value = "123 456 789 0123"
        result, _, _ = extract_cnp_copil(dummy_image)
        self.assertEqual(result, "1234567890123")

        # Test case 3: Invalid - too short
        mock_pytesseract.return_value = "12345"
        result, _, _ = extract_cnp_copil(dummy_image)
        self.assertEqual(result, "necunoscut")
        
        # Test case 4: Invalid - garbage
        mock_pytesseract.return_value = "abc"
        result, _, _ = extract_cnp_copil(dummy_image)
        self.assertEqual(result, "necunoscut")

if __name__ == '__main__':
    unittest.main()