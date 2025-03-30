import cv2
import numpy as np
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt
from glob import glob

# Set Tesseract path if not in PATH environment variable
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
import easyocr


def preprocess_image(image):
    """
    Preprocess the image for better number plate detection.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(blurred)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    return threshold, edges


def detect_number_plate(image):
    """
    Detect the number plate region using contour analysis.
    """
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Loop over contours to find the best rectangular contour
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is rectangular
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Check if the aspect ratio matches a number plate
            if 2 < aspect_ratio < 6:
                return x, y, w, h

    return None


def segment_characters(plate_image):
    """
    Segment characters from the number plate image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 51, 1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5x5 矩形核
    # # 开运算：先腐蚀后膨胀
    # opened_image = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    opened_image = threshold.copy()
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv.CHAIN_APPROX_NONE
    # cv2.drawContours(threshold, contours, -1, 255, 2)
    # Filter out small contours (noise)
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Adjust based on character size
            characters.append((x, y, w, h))

    # Sort characters from left to right
    characters = sorted(characters, key=lambda x: x[0])

    return characters, opened_image


def recognize_characters_tesseract(plate_image, characters):
    """
    Recognize characters using Tesseract OCR.
    """
    recognized_text = ""
    for (x, y, w, h) in characters:
        # Extract the character region
        x = max(x - 3, 0)
        y = max(y - 3, 0)
        char_image = plate_image[y:y + h + 3, x:x + w + 3]
        plt.imshow(char_image)
        plt.title('character')
        plt.show()
        # Use Tesseract to recognize the character
        char_text = pytesseract.image_to_string(char_image,
                                                config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # print(char_text)
        recognized_text += char_text.strip()

    return recognized_text


def recognize_characters_easyocr(plate_image, characters):
    reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
    recognized_text = ""
    for (x, y, w, h) in characters:
        # Extract the character region
        x = max(x - 3, 0)
        y = max(y - 3, 0)
        char_image = plate_image[y:y + h + 3, x:x + w + 3]
        plt.imshow(char_image)
        plt.title('character')
        plt.show()
        # Use Tesseract to recognize the character
        char_text = reader.readtext(char_image, detail=0)  # print(char_text)
        char_text = [i for i in char_text if i.isalnum()]
        recognized_text += "".join(char_text)

    return recognized_text


def detect_plate_by_YOLO(image):
    from ultralytics import YOLO

    license_plate_detector = YOLO('/yolov11/yolo11n.pt').to('cuda')

    license_plates = license_plate_detector(image)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        if score > 0.5:
            # cv2.imwrite(f'yolo_plate_{int(x1)}_{int(x2)}.png', image[int(y1):int(y2), int(x1):int(x2), :])
            return int(x1), int(y1), int(x2), int(y2)
    return None


def main():
    # Load the input image
    for image_path in glob(r"D:\Personal\Chenguang Wang\ocr\dataset\images\*.png"):
        # image_path = r"D:\Personal\Chenguang Wang\dataset\images\Cars6.png"  # Replace with your image path
        image = cv2.imread(image_path)

        plate_region = detect_plate_by_YOLO(image)

        # # Preprocess the image
        # threshold, edges = preprocess_image(image)
        #
        # # Detect the number plate region
        # plate_region = detect_number_plate(edges)

        if plate_region is not None:
            x, y, w, h = plate_region

            # Extract the number plate
            # plate_image = image[y:y + h, x:x + w]
            plate_image = image[y:h, x:w]
            plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            plt.title("Plate")
            plt.show()

            # Segment characters from the number plate
            characters, opened_image = segment_characters(plate_image)
            plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            plt.title("plate_image")
            plt.show()

            # Recognize characters using Tesseract OCR
            # recognized_text = recognize_characters_tesseract(plate_image, characters)

            recognized_text = recognize_characters_easyocr(plate_image, characters)

            # Display the results
            print(f"Recognized Number Plate: {recognized_text}")

            # Draw bounding box around the number plate
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the image with the detected number plate
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Detected Number Plate")
            plt.show()
        else:
            print("Number plate not detected.")


if __name__ == "__main__":
    main()
