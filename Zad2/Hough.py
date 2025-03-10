import numpy as np
import cv2

p1 = 80
p2 = 30


def tb1(x):
    global p1
    p1 = x


def tb2(x):
    global p2
    p2 = x


cv2.namedWindow("ImageWindow")

img = np.ones((300, 300, 3), np.uint8)*50

img = cv2.circle(img, (190, 150), 100, (100, 0, 100), -1)
img[80:180, 90:180] = [0, 0, 0]
img = cv2.circle(img, (50, 50), 20, (250, 0, 0), -1)
img = cv2.resize(img, (600, 600))

cv2.createTrackbar("P1", "ImageWindow", 0, 100, tb1)
cv2.createTrackbar("P2", "ImageWindow", 0, 100, tb2)


img = cv2.medianBlur(img, 5)
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while 1:
    img2 = img.copy()
    #circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=30, minRadius=1, maxRadius=-1)
    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1, 10, param1=p1, param2=p2, minRadius=1, maxRadius=-1)

    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img2, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img2, (i[0], i[1]), 2, (0, 0, 255), 3)
    except TypeError:
        pass

    cv2.imshow('ImageWindow', img2)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break







