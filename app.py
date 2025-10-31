import os
import cv2
# import pytesseract
from flask import Flask, request, render_template_string, send_file, redirect, url_for
import pandas as pd
from ocr_utils import auto_extract_structured_fields
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
<style>
  body { font-family: sans-serif; margin: 2em; background-color: #f9f9f9; color: #333; }
  h2 { color: #333; text-align: center; }
  .container { max-width: 600px; margin: 0 auto; background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
  form { display: flex; flex-direction: column; align-items: center; }
  input[type=file] { border: 2px dashed #ccc; padding: 2em; border-radius: 8px; cursor: pointer; }
  input[type=submit] { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 1em; }
  input[type=submit]:hover { background-color: #0056b3; }
</style>
<div class="container">
  <h2>Upload medical certificate (adeverinta / certificat medical)</h2>
  <form method=post enctype=multipart/form-data action="/upload">
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
</div>
"""

RESULT_HTML = """
<!doctype html>
<title>Result</title>
<style>
  body { font-family: sans-serif; margin: 2em; background-color: #f9f9f9; color: #333; }
  h2 { color: #333; text-align: center; }
  .container { max-width: 800px; margin: 0 auto; background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
  td, th { border: 1px solid #ddd; padding: 8px; }
  tr:nth-child(even) { background-color: #f2f2f2; }
  input[type=text] { width: 100%; padding: 5px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
  button { background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; }
  button:hover { background-color: #218838; }
  .actions { display: flex; justify-content: space-between; align-items: center; }
  a { color: #007bff; text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
<div class="container">
  <h2>OCR result for {{filename}}</h2>
  <form method="post" action="/save">
  <table>
  {% for k,v in results.items() %}
    <tr>
      <td><b>{{k}}</b></td>
      <td><input type="text" name="{{k}}" value="{{v}}"></td>
    </tr>
  {% endfor %}
  </table>
  <input type="hidden" name="file_name" value="{{filename}}">
  <div class="actions">
    <button type="submit">Save and Export to Excel</button>
    <a href="/">Process another</a>
  </div>
  </form>
</div>
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
