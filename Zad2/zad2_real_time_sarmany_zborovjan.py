from ximea import xiapi
import cv2
import numpy as np

from zad2_spracovanie_sarmany_zborovjan import Param

p1 = 80
p2 = 30
p3 = 10
p4 = 10
p5 = 0

def tb1(x):
    global p1
    p1 = x


def tb2(x):
    global p2
    p2 = x

def tb3(x):
    global p3
    p3 = x


def tb4(x):
    global p4
    p4 = x

def tb5(x):
    global p5
    p5 = x

cv2.namedWindow("ImageWindow")
cv2.createTrackbar("P1", "ImageWindow", p1, 200, tb1)
cv2.createTrackbar("P2", "ImageWindow", p2, 200, tb2)
cv2.createTrackbar("min Dist", "ImageWindow", p3, 200, tb3)
cv2.createTrackbar("min Radius", "ImageWindow", p4, 200, tb4)
cv2.createTrackbar("max Radius", "ImageWindow", p5, 200, tb5)

cam = xiapi.Camera()

#start communication
#to open specific device, use:
#cam.open_device_by_SN('41305651')
#(open by serial number)
print('Opening first camera...')
cam.open_device()

#settings
cam.set_exposure(100000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' %cam.get_exposure())

#create instance of Image to store image data and metadata
img = xiapi.Image()

# dst = cv2.undistort(img, Param["mtx"], Param["dist"], None,Param["newcameramtx"])
# x, y, w, h = Param["roi"]
# dst = dst[y:y + h, x:x + w]

mtx = Param["mtx"]
print(f"f_x: {mtx[0, 0]}\nf_y: {mtx[1, 1]}\nc_x: {mtx[0, 2]}\nc_y: {mtx[1, 2]}")

#  start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()


while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (240, 240))  # for this size was found distortion
    dst = cv2.undistort(image, Param["mtx"], Param["dist"], None, Param["newcameramtx"])
    x, y, w, h = Param["roi"]
    dst = dst[y:y + h, x:x + w]
    dst = cv2.resize(dst, (390, 300))

    gimg = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    dst2 = dst.copy()
    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1, p3, param1=p1, param2=p2, minRadius=p4, maxRadius=p5)
    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(dst2, (i[0], i[1]), i[2], (0, 255, 0), 2)  # draw the outer circle
            cv2.circle(dst2, (i[0], i[1]), 2, (0, 0, 255), 3)  # draw the center of the circle
            cv2.putText(dst2, str(2*i[2]), (i[0], i[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
    except TypeError:
        pass

    cv2.imshow("test", dst)
    cv2.imshow("circles", dst2)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')

cv2.waitKey()

