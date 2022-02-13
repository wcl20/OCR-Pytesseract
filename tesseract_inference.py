import cv2
import imutils
import numpy as np
import os
import pytesseract
from pprint import pprint
from pytesseract import Output

def preprocess(image):
    # Reference: https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image


def main():

    # Reference: https://nanonets.com/blog/ocr-with-tesseract/#ocrwithpytesseractandopencv
    # Reference: https://www.pyimagesearch.com/2021/08/16/installing-tesseract-pytesseract-and-python-ocr-packages-on-your-system/

    image = cv2.imread(os.path.join("images", "concrete-text.jpg"))
    image = imutils.resize(image, width=1280)

    # Only consider this character set
    whitelist = "abcdefghijklmnopqrstuvwxyz"
    whitelist += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    whitelist += "-0123456789"
    tesseract_config = r"-l eng"
    tesseract_config = rf"-c tessedit_char_whitelist={whitelist}"
    tesseract_config += r"--oem 1"
    tesseract_config += r"--psm 6"

    preprocessed = preprocess(image)
    # output = pytesseract.image_to_string(image, config=r"--oem 1 --psm 6")

    data = pytesseract.image_to_data(preprocessed, config=tesseract_config, output_type=Output.DICT)
    for i in range(len(data["text"])):
        # Get score of each text
        score = int(data["conf"][i])
        label = data["text"][i]

        # Skip text with low confidence / too short
        if score < 0: continue
        if len(label) < 5: continue

        # Draw bounding box and text
        color = list(np.random.random(size=3) * 255)
        x1 = data["left"][i]
        y1 = data["top"][i]
        x2 = x1 + data["width"][i]
        y2 = y1 + data["height"][i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    cv2.imshow("image", image)
    cv2.waitKey(0)








if __name__ == '__main__':
    main()
