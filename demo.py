import numpy as np
import cv2
from multiprocessing import Process
import time


def webcam_video():

    cap = cv2.VideoCapture(
        "/Users/menghang/Desktop/ml-dev/football-server/demo_video/template_2d.avi")
    cv2.namedWindow("template", cv2.WINDOW_NORMAL)
    cv2.moveWindow("template", 600, 500)

    while (True):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('template', frame)
            cv2.resizeWindow("template", 768, 360)

            if cv2.waitKey(35) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def local_video():
    cap = cv2.VideoCapture(
        "/Users/menghang/Desktop/ml-dev/football-server/demo_video/concat.avi")
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("detection", 600, 10)

    while (True):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('detection', frame)
            cv2.resizeWindow("detection", 768, 360)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def blue():
    cap = cv2.VideoCapture(
        "/Users/menghang/Desktop/ml-dev/football-server/demo_video/blue.avi")
    cv2.namedWindow("blue", cv2.WINDOW_NORMAL)
    cv2.moveWindow("blue", 40, 10)

    while (True):
        ret, frame = cap.read()
        if ret == True:

            cv2.imshow('blue', frame)
            cv2.resizeWindow("blue", 400, 400)
            if cv2.waitKey(395) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def white():
    cap = cv2.VideoCapture(
        "/Users/menghang/Desktop/ml-dev/football-server/demo_video/white.avi")
    cv2.namedWindow("white", cv2.WINDOW_NORMAL)
    cv2.moveWindow("white", 40, 400)

    while (True):
        ret, frame = cap.read()
        if ret == True:

            cv2.imshow('white', frame)
            cv2.resizeWindow("white", 400, 400)

            if cv2.waitKey(395) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    p1 = Process(target=local_video)
    p2 = Process(target=webcam_video)
    p3 = Process(target=blue)
    p4 = Process(target=white)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
