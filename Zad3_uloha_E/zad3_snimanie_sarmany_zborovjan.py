from ximea import xiapi
import cv2
import numpy as np

WIDTH = 240
HEIGHT = 240

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

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

image_count = 0

while cv2.waitKey() != ord('q'):

        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image, (240, 240))
        cv2.imshow("test", image)
        filename = f"obrazok.png"
        cv2.imwrite(filename, image)
        print(f"Image saved: {filename}")
        break


#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

print('Done.')

cv2.waitKey()

