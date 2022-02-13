import cv2
import imutils
import os
import pytesseract
import numpy as np

def preprocess(image):
    # Reference: https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (1, 1), 0)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def main():

    # Reference: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/?_ga=2.240811526.467560829.1637216418-926914685.1637216418
    # Reference: https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

    image = cv2.imread(os.path.join("images", "concrete-text.jpg"))
    image = imutils.resize(image, width=1280)
    image_height, image_width = image.shape[:2]

    model_path = os.path.join("models", "frozen_east_text_detection.pb")
    input_dim = 1280
    net = cv2.dnn.readNet(model_path)

    preprocessed = cv2.resize(image, (input_dim, input_dim))
    blob = cv2.dnn.blobFromImage(preprocessed, 1.0, (input_dim, input_dim), [123.68, 116.78, 103.94], True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    geometry, scores = net.forward(output_layers)

    rows, cols = scores.shape[2:]
    bboxs = []
    confs = []

    # Iterate each cell of the feature map, each cell gives one bounding box prediction
    for i in range(rows):
        for j in range(cols):
            score = scores[0, 0, i, j]
            if score < 0.5: continue

            # Compute offset as the feature maps are 4x smaller than the input
            offset_x = j * 4
            offset_y = i * 4

            # d0 is the distance from center to top edge
            d0 = geometry[0, 0, i, j]
            # d1 is the distance from center to right edge
            d1 = geometry[0, 1, i, j]
            # d2 is the distance from center to bottom edge
            d2 = geometry[0, 2, i, j]
            # d3 is the distance from center to left edge
            d3 = geometry[0, 3, i, j]
            # angle is the rotated angle of the bounding box
            angle = geometry[0, 4, i, j]

            x2 = int(offset_x + (np.cos(angle) * d1) + (np.sin(angle) * d2))
            y2 = int(offset_y - (np.sin(angle) * d1) + (np.cos(angle) * d2))

            h = int(d0 + d2)
            w = int(d1 + d3)
            x = int(x2 - w)
            y = int(y2 - h)
            bboxs.append([x, y, w, h])
            confs.append(float(score))

    indices = cv2.dnn.NMSBoxes(bboxs, confs, score_threshold=0.5, nms_threshold=0.4)
    copy = image.copy()
    for i in indices:
        color = list(np.random.random(size=3) * 255)

        bbox = bboxs[i[0]]
        x1 = int(bbox[0] / input_dim * image_width)
        y1 = int(bbox[1] / input_dim * image_height)
        x2 = int((bbox[0] + bbox[2]) / input_dim * image_width)
        y2 = int((bbox[1] + bbox[3]) / input_dim * image_height)

        # Extract ROI (Add padding)
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.25)
        roi_x1 = max(0, x1 - pad_x)
        roi_y1 = max(0, y1 - pad_y)
        roi_x2 = min(image_width, x2 + pad_x)
        roi_y2 = min(image_height, y2 + pad_y)
        roi = preprocess(image[roi_y1:roi_y2, roi_x1:roi_x2])

        text = pytesseract.image_to_string(roi, config=r"-l eng --oem 1 --psm 7")
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

        cv2.rectangle(copy, (x1, y1), (x2, y2), color, 1)
        cv2.putText(copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    cv2.imshow("Image", copy)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
