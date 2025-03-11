from ximea import xiapi
import cv2
import numpy as np

# termination criteria (end if accuracy improvement less 0.001 or after 30 iterations)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5 * 7, 3), np.float32)  # corner points in 3D space - 35 points with xyz coordinates
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)  # populate coordinates xy with numbers 0-6 and 0-4 in form of grid reshaped

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = [f"Pics/ob{i}.png" for i in range(1, 22)]
for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7, 5), corners2, ret)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread('Pics/ob5.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

Param = {"mtx":mtx, "dist":dist, "newcameramtx":newcameramtx,"roi":roi}

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('Pics/calibresult.png', dst)



img = cv2.resize(img, (600, 600))
cv2.imshow('Pics/ob5.png', img)

img2 = cv2.imread('Pics/calibresult.png')
img2 = cv2.resize(img2, (600, 600))
cv2.imshow('calibresult', img2)
cv2.waitKey()