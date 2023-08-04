# import cv2
# import json
# import numpy as np

# points = []

# json_files = ["frame/f1.json", "frame/f2.json", "frame/f3.json"]
# img_files = ["frame/f1.jpg", "frame/f2.jpg", "frame/f3.jpg"]

# out_shape = (300, 300)

# for i, file in enumerate(json_files):

#     f = open(file, 'r')
#     img = cv2.imread(img_files[i])
#     h, w, channel = img.shape

#     data = json.load(f)
#     tl, tr, br, bl = data["shapes"][0]["points"]

#     src = np.array([tl, tr, br, bl], np.float32)

#     tlpx = tl[0] / w
#     tlpy = tl[1] / h

#     new_tlx = tlpx * out_shape[0]
#     new_tly = tlpy * out_shape[1]

#     trpx = tr[0] / w
#     trpy = tr[1] / h
#     new_trx = trpx * out_shape[0]
#     new_try = new_tly

#     height_diff = (bl[1] - tl[1]) / h

#     new_blx = new_tlx
#     new_bly = new_tly + (height_diff * out_shape[1])

#     new_brx = new_trx
#     new_bry = new_try + (height_diff * out_shape[1])

#     dst = np.float32([[new_tlx, new_tly], [new_trx, new_try],
#                      [new_brx, new_bry], [new_blx, new_bly]])

#     print(dst)
#     matrix = cv2.getPerspectiveTransform(src, dst)

#     imgOutput = cv2.warpPerspective(
#         img, matrix, out_shape, cv2.INTER_LINEAR)

#     cv2.imshow(f"img{i}", imgOutput)
#     cv2.waitKey(0)

import matplotlib.pyplot as plt
import cv2

img_template = cv2.imread('/Users/menghang/Desktop/Screenshot 2023-08-04 at 12.51.14 AM.png')

plt.imshow(img_template)
plt.show()