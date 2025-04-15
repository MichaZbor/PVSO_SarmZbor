import time

from ximea import xiapi
import cv2
import numpy as np
from zad3_spracovanie_sarmany_zborovjan import convolve2d, difference_of_gaussians, imread_grayscale

WIDTH = 240
HEIGHT = 240

cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(100000)
cam.set_param('imgdataformat', 'XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' % cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

image_count = 0

dog_kernel = difference_of_gaussians( 5, 4, 8)

# Aplikácia DoG filtra na obrázok
while True:
    start_time = time.time()
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (240, 240))
    dog_image = imread_grayscale(image)

    dog_image = convolve2d(dog_image, dog_kernel)
    cv2.imshow("test", image)
    cv2.imshow("DoG", dog_image)
    cv2.imshow("DoG aplikovaný na obrázok", cv2.normalize(dog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    print("Elapsed time: %f s" % (time.time() - start_time))
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