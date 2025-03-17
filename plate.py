import cv2
import numpy as np
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt
from glob import glob
# Set Tesseract path if not in PATH environment variable
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"


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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5x5 矩形核
    # 开运算：先腐蚀后膨胀
    opened_image = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv.CHAIN_APPROX_NONE
    # cv2.drawContours(threshold, contours, -1, 255, 2)
    # Filter out small contours (noise)
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5 and h/w>1:  # Adjust based on character size
            characters.append((x, y, w, h))

    # Sort characters from left to right
    characters = sorted(characters, key=lambda x: x[0])

    return characters, opened_image


def recognize_characters(plate_image, characters):
    """
    Recognize characters using Tesseract OCR.
    """
    recognized_text = ""
    for (x, y, w, h) in characters:
        # Extract the character region
        x = max(x-7,0)
        y = max(y-7,0)
        char_image = plate_image[y:y + h+7, x:x + w+7]
        plt.imshow(char_image)
        plt.title('character')
        plt.show()
        # Use Tesseract to recognize the character
        char_text = pytesseract.image_to_string(char_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # print(char_text)
        recognized_text += char_text.strip()

    return recognized_text

def detect_plate_by_YOLO(image):
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n-seg.pt")  # load an official model
    # Predict with the model
    # results = model(image)  # predict on an image
    results = model(image)

    # Access the results
    for result in results:
        xy = result.masks.xy  # mask in polygon format
        xyn = result.masks.xyn  # normalized
        masks = result.masks.data  # mask in matrix format (num_objects x H x W)



def main():
    # Load the input image
    for image_path in glob(r"D:\Personal\Chenguang Wang\dataset\images\*.png"):
        # image_path = r"D:\Personal\Chenguang Wang\dataset\images\Cars6.png"  # Replace with your image path
        image = cv2.imread(image_path)

        # detect_plate_by_YOLO(image)

        # Preprocess the image
        threshold, edges = preprocess_image(image)

        # Detect the number plate region
        plate_region = detect_number_plate(edges)

        if plate_region:
            x, y, w, h = plate_region

            # Extract the number plate
            plate_image = image[y:y + h, x:x + w]
            plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            plt.title("Plate")
            plt.show()

            # Segment characters from the number plate
            characters, opened_image = segment_characters(plate_image)
            plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            plt.title("plate_image")
            plt.show()


            # Recognize characters using Tesseract OCR
            recognized_text = recognize_characters(plate_image, characters)

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
