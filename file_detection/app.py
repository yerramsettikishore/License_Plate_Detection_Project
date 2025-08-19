from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pytesseract
import uuid
from werkzeug.utils import secure_filename
from detection import detect_plate  # Importing the detection function

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Tesseract Path (Windows Only)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def process_image(file_path, filename):
    """ Process image for license plate detection and OCR """
    img = cv2.imread(file_path)

    if img is None:
        return None, None, "Invalid image file."

    # Perform license plate detection
    output_img, plate, detected_text = detect_plate(img)

    if detected_text == "No Plate Detected":
        return None, None, "No License Plate Detected."

    # Save processed and cropped images
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, output_img)

    cropped_filename = f"cropped_{filename}"
    cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
    cv2.imwrite(cropped_path, plate)

    return processed_filename, cropped_filename, detected_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        if file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            processed_filename, cropped_filename, extracted_text = process_image(file_path, unique_filename)

            if "Invalid" in extracted_text or "No License Plate" in extracted_text:
                return render_template("index.html", error=extracted_text)

            return render_template("result.html", 
                                   result=extracted_text, 
                                   uploaded_image=unique_filename, 
                                   processed_image=processed_filename, 
                                   cropped_image=cropped_filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5001, debug=True)
