

import cv2
import numpy as np


def imgByteToNumpy(img):
    image_byte = img
    img_np = cv2.imdecode(np.fromstring(
        image_byte, np.uint8), cv2.IMREAD_COLOR)
    return img_np


def background_subtraction(img):

    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([55, 100, 50])
    upper = np.array([65, 220, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    result = cv2.bitwise_and(img, img, mask=mask)

    return result
