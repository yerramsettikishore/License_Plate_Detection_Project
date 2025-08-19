from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)

plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')
plate_text_global = "No Plate Detected"

def preprocess_plate(plate):
    """Apply preprocessing techniques for better OCR accuracy."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def detect_plate(img):
    """Detect the license plate and extract its text."""
    global plate_text_global

    plate_img = img.copy()
    plate_text = "No Plate Detected"
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)

    if len(plate_rects) == 0:
        return plate_img, None, plate_text

    for (x, y, w, h) in plate_rects:
        plate = img[y:y+h, x:x+w]
        processed_plate = preprocess_plate(plate)
        custom_config = r'--oem 3 --psm 7'
        plate_text = pytesseract.image_to_string(processed_plate, config=custom_config).strip()

        plate_text_global = plate_text if plate_text else "No Plate Detected"
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (51, 181, 155), 3)

        return plate_img, plate, plate_text_global

    return plate_img, None, plate_text

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        processed_frame, _, plate_text = detect_plate(frame)
        cv2.putText(processed_frame, f"Plate: {plate_text}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_detection')
def live_detection():
    return render_template('result.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plate_text')
def get_plate_text():
    return jsonify({"plate_text": plate_text_global})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
