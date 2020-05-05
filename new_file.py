# import packages
import cv2 as cv
import os
import datetime as dt
import PIL
import numpy as np
import glob
import matplotlib.pyplot as plt
#
#
# calculate camera matrix and distortion coefficients
# example from: https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
def camera_calibration(rel_path):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # (6,8) are inner squares of chessboard
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for fname in images:
        # load image
        img = cv.imread(fname)
        # covert to gray scale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (8,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(50)
        else:
            print('No Chessboard detected.')
    cv.destroyAllWindows()
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # set back cwd
    os.chdir(org_path)
    
    return mtx, dist

# undistort any image taken by the same camera
def undistort_images(rel_path, mtx, dist):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    #
    for image in images:
        img = cv.imread(image)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        un_img = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        un_img = un_img[y:y+h, x:x+w]
        cv.imwrite(os.path.join('undistorted_images', image), un_img)
        cv.destroyAllWindows()
        
    os.chdir(org_path)
# run functions
mtx, dist = camera_calibration('calibration_images_originals')
undistort_images('test_undistort', mtx, dist)