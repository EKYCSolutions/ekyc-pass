import cv2
import numpy as np

result = cv2.VideoWriter('panorama_perspective.mp4',
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                         60, (1920, 1080))

corners_3d = np.array([[
    157.57894736842104,
    160.8421052631579],
    [
    7.578947368421041,
    1842.421052631579
],
    [
    2810.2105263157896,
    1821.3684210526314
],
    [
    2665.4736842105262,
    174.0
]])
width, height = 1080, 1920
corners_2d = np.array([[0, 0],
                       [width - 1, 0],
                       [width - 1, height - 1],
                       [0, height - 1]])

cap = cv2.VideoCapture("panorama.mp4")

i = 0
while cap.isOpened():
    status, frame = cap.read()

    if status:
        M = cv2.getPerspectiveTransform(corners_3d.astype(
            np.float32), corners_2d.astype(np.float32))
        top_down_view = cv2.warpPerspective(frame, M, (width, height))
        top_down_view = cv2.rotate(top_down_view, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("perspective", top_down_view)
        result.write(top_down_view)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
result.release()
