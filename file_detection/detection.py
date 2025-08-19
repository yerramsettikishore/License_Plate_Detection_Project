import cv2
import numpy as np
import pytesseract

# Load the Haar cascade for Indian license plates
plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

def detect_plate(img):
    plate_img = img.copy()
    roi = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)

    # Initialize plate and plate_text
    plate = None
    plate_text = "No Plate Detected"

    if len(plate_rects) == 0:
        return plate_img, plate, plate_text  # Return None for plate if no detection

    for (x, y, w, h) in plate_rects:
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x, y), (x+w, y+h), (51, 181, 155), 3)

        # Convert to grayscale for OCR
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, binary_plate = cv2.threshold(gray_plate, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use Tesseract OCR to extract text
        plate_text = pytesseract.image_to_string(binary_plate, config='--psm 8').strip()
        break  # Process only the first detected plate

    return plate_img, plate, plate_text if plate_text else "No Plate Detected"
