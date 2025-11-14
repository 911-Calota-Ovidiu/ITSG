# 🩺 Medical Certificate OCR -- Intelligent Data Extraction

### Automatic Extraction of data from Medical Leave Certificate (Certificat de concediu medical)

This project implements an intelligent OCR pipeline for extracting
structured information from Medical "Certificat de concediu medical"
forms.
It is designed for HR departments, clinics, and automation tools that
need to digitize thousands of scanned certificates each month.

The system uses **Python, OpenCV, and Tesseract OCR**, with an
The system uses Python, OpenCV, and Tesseract OCR, with an interactive interface built in Streamlit, capable of handling both printed and handwritten fields through customized preprocessing rules.

------------------------------------------------------------------------

## 🚀 Features

### ✔ Intelligent OCR

-   Hybrid approach: classical computer vision + OCR + rule-based
    post-processing
-   Separate processing pipelines for:
    -   **Printed text** (contrast enhancement + sharpening)
    -   **Handwritten text** (adaptive thresholding + dilation)

### ✔ Automatic Field Extraction

Extracts the main medical certificate fields: - Serie -
Numar - Cod Indemnizatie - CNP - Urgenta Medicochirurgicala - In Continuare - Ambulator/Internat --
Sectia - Date range: De la / Pana la - Cod Diagnostic -
Unitate sanitara emitenta

### ✔ Image Auto-Alignment

-   Detects text block center using Tesseract
-   Recenters form automatically
-   Allows for reliable ROI extraction even when scan is shifted

### ✔ Visualization & Debug Tools

-   Bounding boxes for all ROIs
-   Full raw OCR text viewer
-   Raw handwritten/printed values kept for debugging

### ✔ Export

-  Alows for 

------------------------------------------------------------------------

## 📦 Project Structure

    /main.py                # Streamlit application + OCR pipeline + field extraction
    /image_processing.py    # Preprocessing functions (printed/handwriting)
    /Raport.pdf             # Technical report (algorithm description & validation)

------------------------------------------------------------------------

## 🛠 Technology Stack

-   **Python 3.x**
-   **OpenCV** -- image preprocessing
-   **Pillow** -- image loading
-   **pytesseract** -- OCR engine
-   **NumPy / Pandas** -- data handling
-   **Streamlit** -- web interface
-   **openpyxl** -- Excel export

------------------------------------------------------------------------

## ⚙ Installation

### 1. Clone the repo

``` bash
git clone https://github.com/911-Calota-Ovidiu/ITSG.git
git checkout varianta-cibo-tesseract
```

### 2. Create a venv and activate it

``` 
py -m venv .venv
./.venv/Scripts/activate.bat
```

### 3. Install required libraries

``` bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

-   **Windows:** https://github.com/UB-Mannheim/tesseract/wiki

-   **macOS:**

    ``` bash
    brew install tesseract
    ```

-   **Linux (Ubuntu):**

    ``` bash
    sudo apt install tesseract-ocr
    ```

Ensure languages **ron** (Romanian) and **eng** are installed.

------------------------------------------------------------------------

## ▶ Running the App

``` bash
streamlit run main.py
```

------------------------------------------------------------------------

## 🔍 How It Works (Pipeline Overview)

1.  **Upload scanned certificate (JPG/PNG).**
2.  **Preprocessing:**
    -   Convert to grayscale
    -   Resize to standard width
    -   Detect main text block
    -   Auto-align the form
3.  **ROI-based extraction:**
    -   Printed fields → `treat_print()`
    -   Handwritten fields → `treat_handwriting()`
4.  **OCR recognition (Tesseract):**
    -   Character whitelists for numeric fields
    -   `psm 8` for small text regions
5.  **Post-processing:**
    -   O→0, I→1, Z→2, etc.
    -   Validations:
        -   CNP must have 13 digits
        -   Cod Indemnizatie must be 1--17
6.  **Export:**
    -   Results displayed in table
    -   Excel exported by clicking **Descarca Fisier Excel**


------------------------------------------------------------------------

## 📈 Future Improvements

-   CNN/Transformer models for handwriting
-   Automatic template detection
-   Multi-certificate batch processing
-   Cloud OCR integration (Azure, Google Vision)

------------------------------------------------------------------------

## 👥 Contributors (Team Meseriașii -- ITSG 2024--2025)

-   Badea Dan-Nicolai
-   Badu Cătălin Ioachim
-   Calotă Ovidiu
-   Cibotariu Andrei
