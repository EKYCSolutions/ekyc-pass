import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class TeamMatch():
    def __init__(self) -> None:
        self.knn = KNeighborsClassifier(n_neighbors=2)

    def match(self, img,  team):
        data = []
        label = []

        home, away = team["home"], team["away"]
        for h in home:
            x, y, w, h = h.values()
            image = img[y: y+h, x: x+w]
            data.append(self.getHist(image))
            label.append(0)

        for a in away:
            x, y, w, h = a.values()
            image = img[y: y+h, x: x+w]
            data.append(self.getHist(image))
            label.append(1)

        data, label = np.array(data), np.array(label)
        data, label = self.unison_shuffled_copies(data, label)

        data = data[:, :, 0]

        x_train, x_test, y_train, y_test = train_test_split(
            data, label, test_size=0.2, stratify=label, shuffle=True)

        self.knn.fit(x_train, y_train)

    def getHist(self, img):
        if (type(img) == str):
            img = cv2.imread(img)
        img = self.background_subtraction(img)
        # Split the image into its RGB channels
        r, g, b = cv2.split(img)

        # Calculate the histogram for each channel
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

        hist1 = np.concatenate([hist_r, hist_g, hist_b])
        return hist1

    def background_subtraction(self, img):

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
        # cv2.imwrite('pills_result.jpg', result)

        # cv2.imshow('res', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        return result

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))

        return np.array(a[p]),  np.array(b[p])
