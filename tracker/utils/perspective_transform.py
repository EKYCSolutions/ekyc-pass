import cv2
import json
import numpy as np


def get_points(json_file):
    f = open(json_file, 'r')
    data = json.load(f)
    tl, tr, br, bl = data["shapes"][0]["points"]
    return np.float32([tl, tr, br, bl])


def warp(img, src, dst, odim, show=False):

    # odim = (width,  height)
    matrix = cv2.getPerspectiveTransform(src, dst)

    imgOutput = cv2.warpPerspective(
        img, matrix, odim, cv2.INTER_LINEAR)

    if show:
        cv2.imshow("warp", imgOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return imgOutput
