import cv2
import numpy as np
import pytesseract

# Load the pre-trained Haar Cascade for license plate detection
plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

def preprocess_plate(plate):
    """Apply preprocessing techniques for better OCR accuracy."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Adaptive thresholding
    return binary

def detect_plate(img):
    """Detect the license plate and extract its text."""
    plate_img = img.copy()
    plate_text = "No Plate Detected"
    detected_plate = None

    # Detect license plates
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)

    if len(plate_rects) == 0:
        return plate_img, None, plate_text  # No plate detected

    # Sort by area (width * height) and select the largest one
    plate_rects = sorted(plate_rects, key=lambda rect: rect[2] * rect[3], reverse=True)
    x, y, w, h = plate_rects[0]  # Take the largest detected plate

    detected_plate = img[y:y+h, x:x+w]  # Extract plate region
    processed_plate = preprocess_plate(detected_plate)  # Apply preprocessing

    # Perform OCR with optimized settings
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    extracted_text = pytesseract.image_to_string(processed_plate, config=custom_config).strip()

    # Draw a rectangle around the detected plate
    cv2.rectangle(plate_img, (x, y), (x + w, y + h), (51, 181, 155), 3)

    return plate_img, detected_plate, extracted_text if extracted_text else "No Plate Detected"

def test_detection(image_path):
    """Test license plate detection on a given image."""
    img = cv2.imread(image_path)
    detected_img, plate, extracted_text = detect_plate(img)

    # Show detected image
    cv2.imshow("License Plate Detection", detected_img)
    if plate is not None:
        cv2.imshow("Extracted Plate", plate)

    print(f"Detected License Plate Text: {extracted_text}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uncomment this line to test on an image
# test_detection("test_image.jpg")
