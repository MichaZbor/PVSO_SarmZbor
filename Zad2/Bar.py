import cv2
import numpy as np


def nothing(x):
    pass


def something(x):
    img[:] = [x, 0, 0]


cv2.namedWindow("ImageWindow")
img = np.zeros((300, 512, 3), np.uint8)
cv2.createTrackbar("B", "ImageWindow", 0, 255, something)

switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'ImageWindow', 0, 1, nothing)

while (1):
    cv2.imshow('ImageWindow', img)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break

    # get current positions of four trackbars
    b = cv2.getTrackbarPos('B', 'ImageWindow')
    s = cv2.getTrackbarPos(switch, 'ImageWindow')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, 0, 0]

cv2.destroyAllWindows()


