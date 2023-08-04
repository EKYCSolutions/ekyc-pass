import cv2
import stitching
import imutils
import numpy as np
c1 = cv2.VideoCapture(
    "/Users/menghang/Downloads/Sequences/Film Role-0 ID-1 T-2 m00s00-000-m00s00-185.avi")
c2 = cv2.VideoCapture(
    "/Users/menghang/Downloads/Sequences/Film Role-0 ID-3 T-2 m00s00-000-m00s00-185.avi")
c3 = cv2.VideoCapture(
    "/Users/menghang/Downloads/Sequences/Film Role-0 ID-5 T-2 m00s00-000-m00s00-185.avi")

settings = {"detector": "sift", "confidence_threshold": 0.3,
            }

stitcher = stitching.Stitcher(**settings)


result = cv2.VideoWriter('panorama.mp4',
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                         60, (1080, 1920))


while c1.isOpened() and c2.isOpened() and c3.isOpened():
    r1, f_right = c1.read()
    r2, f_middle = c2.read()
    r3, f_left = c3.read()

    print(f_right.shape, f_middle.shape, f_left.shape)

    if r1 and r2 and r3:

        f_right = cv2.resize(f_right, (1080, 1920))
        f_middle = cv2.resize(f_middle, (1080, 1920))
        f_left = cv2.resize(f_left, (1080, 1920))

        # f_left = f_left[:, :-220]
        # f_right = f_right[:, 200:]

        output = np.hstack((f_left, f_middle, f_right))
        cv2.imshow("ouptut", output)
        cv2.waitKey(0)
        # print(output.shape)
        # # result.write(output)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        pass

    else:
        break

c1.release()
c2.release()
c3.release()
# result.release()

# Closes all the frames
cv2.destroyAllWindows()
