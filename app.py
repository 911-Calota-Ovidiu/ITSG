"""import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, send_file, redirect, url_for
from ocr_utils import auto_extract_structured_fields  # your OCR utility

# -------------------------------
# CONFIGURATION
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

APP_ROOT = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# HTML TEMPLATES
# -------------------------------
INDEX_HTML = 
<!doctype html>
<title>MedCert OCR</title>
<h2>Upload medical certificate (adeverinta / certificat medical)</h2>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<hr>


RESULT_HTML = 
<!doctype html>
<title>Result</title>
<h2>OCR result for {{filename}}</h2>
<form method="post" action="/save">
<table>
{% for k,v in results.items() %}
  <tr>
    <td><b>{{k}}</b></td>
    <td><input type="text" name="{{k}}" value="{{v}}" size="60"></td>
  </tr>
{% endfor %}
</table>
<input type="hidden" name="file_name" value="{{filename}}">
<br>
<button type="submit">Save and Export to Excel</button>
</form>
<br>
<a href="/">Process another</a>


# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    if not f:
        return redirect(url_for('index'))

    path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(path)

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    # --- Automatic field extraction ---
    ocr_results = auto_extract_structured_fields(img)

    # Convert to simple dict: field -> value
    results = {k: v['value'] for k, v in ocr_results.items()}

    return render_template_string(RESULT_HTML, results=results, filename=f.filename)

@app.route('/save', methods=['POST'])
def save():
    form = dict(request.form)
    filename = form.pop('file_name', '')

    df = pd.DataFrame([form])
    out_path = os.path.join(APP_ROOT, 'labeled')
    os.makedirs(out_path, exist_ok=True)

    excel_path = os.path.join(out_path, filename + '.xlsx')
    df.to_excel(excel_path, index=False)

    csv_path = os.path.join(out_path, filename + '.csv')
    df.to_csv(csv_path, index=False)

    return send_file(excel_path, as_attachment=True)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
"""
import os
import cv2
import pytesseract
from flask import Flask, request, render_template_string, send_file, redirect, url_for
import pandas as pd
from ocr_utils import auto_extract_structured_fields
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

APP_ROOT = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# HTML TEMPLATES
# -------------------------------
INDEX_HTML = """
<!doctype html>
<title>MedCert OCR</title>
<h2>Upload medical certificate (adeverinta / certificat medical)</h2>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<hr>
"""

RESULT_HTML = """
<!doctype html>
<title>Result</title>
<h2>OCR result for {{filename}}</h2>
<form method="post" action="/save">
<table>
{% for k,v in results.items() %}
  <tr>
    <td><b>{{k}}</b></td>
    <td><input type="text" name="{{k}}" value="{{v}}" size="60"></td>
  </tr>
{% endfor %}
</table>
<input type="hidden" name="file_name" value="{{filename}}">
<br>
<button type="submit">Save and Export to Excel</button>
</form>
<br>
<a href="/">Process another</a>
"""

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    if not f:
        return redirect(url_for('index'))
    path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(path)

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    # --- Automatic field extraction ---
    ocr_results = auto_extract_structured_fields(img)

    # convert to dict: field -> value
    results = {k: v['value'] for k, v in ocr_results.items()}

    return render_template_string(RESULT_HTML, results=results, filename=f.filename)

@app.route('/save', methods=['POST'])
def save():
    form = dict(request.form)
    filename = form.pop('file_name', '')
    df = pd.DataFrame([form])
    out_path = os.path.join(APP_ROOT, 'labeled')
    os.makedirs(out_path, exist_ok=True)
    excel_path = os.path.join(out_path, filename + '.xlsx')
    df.to_excel(excel_path, index=False)
    csv_path = os.path.join(out_path, filename + '.csv')
    df.to_csv(csv_path, index=False)
    return send_file(excel_path, as_attachment=True)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
