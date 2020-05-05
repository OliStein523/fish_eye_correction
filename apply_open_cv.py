#
#
#
# different applications of open CV for image processing

import cv2 as cv
import os
import datetime as dt
import PIL
import numpy as np
import glob
import matplotlib.pyplot as plt
#
# relative path of image directory
path_calibration = 'calibration_images_originals'
#
# image path to work with
path_images = path_calibration
#
template = cv.imread('minipatch.jpg')
image = cv.imread('with_patch.jpg')


def cut_area(rel_path, threshold):
    org_path = os.getcwd()
    template = cv.imread('minipatch.jpg',0)
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".jpg"]
    #
    for image in images:
        img = cv.imread(image,0)
        match = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        areas = np.where(match >= threshold)
        
        for pt in zip(*areas[::-1]):
            cv.rectangle(img, pt, (pt[0] + 50, pt[1] + 50), (0,255,255), 1)
            
        min_ypixel = np.min(areas[0])
        min_xpixel = np.min(areas[1])
        max_ypixel = np.max(areas[0])
        max_xpixel = np.max(areas[1])
        cv.rectangle(img, (min_xpixel, min_ypixel), (max_xpixel+50, max_ypixel+50), (0,0,255), 1)
        # Show the final image with the matched area. 
        cv.imshow('Detected',img) 
        cv.waitKey(0)
    os.chdir(org_path)

#cut_area('test_image', 0.8)

def image_processing(rel_path):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".jpg"]
    print(images)
    for image in images:
        # imread reads the image (1 for colour image, 0 for grey scale image, -1 for unchanged image)
        img = cv.imread(image,0)
        # crop image
        # img[y:y+h, x:x+w]
        crop = img[50:200, 200:400]
        # show image
        plt.hist(crop.ravel(), 256, [0,256])
        plt.show()
        cv.imshow("cropeed", crop)
        cv.waitKey(0)
        
        
    os.chdir(org_path)

#image_processing('test_image')

# camera calibration
def camera_calibration(rel_path):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for fname in images:
        img = cv.imread(fname)
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
            cv.waitKey(500)
    cv.destroyAllWindows()
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # set back cwd
    os.chdir(org_path)
    
    return mtx, dist

def undistort_images(rel_path, mtx, dist):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    
    for image in images:
        img = cv.imread(image)
        h, w = img.shape[:2]
        newcamtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undist_img = cv.undistort(img, mtx, dist, None, newcamtx)
        x, y, w, h = roi
        undist_img = undist_img[y:y+h, x:x+w]
        cv.imwrite('undist_' + image, undist_img)
        cv.destroyAllWindows()
        
    os.chdir(org_path)

mtx, dist = camera_calibration('calibration_images_originals')
undistort_images('test_undistort', mtx, dist)


# resize images
# image information are lost after decrease
def decrease_images(path, percentage):       
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    for image in images:
        img = cv.imread(image, cv.IMREAD_UNCHANGED)
        print(f'Original Dimensions of {os.path.basename(image)} are: {img.shape}')
        #
        scale_percent = percentage
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        #
        # resize image
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        print(f'Resized Dimension of {os.path.basename(image)} are: {resized.shape}')
        #
    #    cv.imshow("Resized image:", resized)
    #    cv.waitKey(0)
        savepath = os.path.split(image)[0]
        savename, ext = os.path.split(image)[1].split(".")
        new_savepath = os.path.join(savepath, savename+f'_reduced_{scale_percent}'+f'.{ext}')
        cv.imwrite(new_savepath, resized)
        cv.destroyAllWindows()
    os.chdir(org_path)
        
def remove_scaled_images(rel_path):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    for image in images:
        if "reduced" not in os.path.basename(image):
            os.remove(image)
    os.chdir(org_path)
            
def add_time_date(rel_path):
    org_path = os.getcwd()
    new_path = os.chdir(os.path.join(os.getcwd(), rel_path))
    images = [files for files in os.listdir(new_path) if os.path.splitext(files)[1] == ".JPG"]
    for image in images:
        # get metadata from Image
        img = PIL.Image.open(os.path.basename(image))
        exifdata = img.getexif()
        #
        # get date and time (tag = 36867)
        img_date = exifdata.get(36867)
#        img_width = exifdata.get(40962)
#        img_height = exifdata.get(40963)
#                https://www.thepythoncode.com/article/extracting-image-metadata-in-python
#                Tag ID for ExifVersion is 36864
#                Tag ID for ComponentsConfiguration is 37121
#                Tag ID for CompressedBitsPerPixel is 37122
#                Tag ID for DateTimeOriginal is 36867
#                Tag ID for DateTimeDigitized is 36868
#                Tag ID for ShutterSpeedValue is 37377
#                Tag ID for ApertureValue is 37378
#                Tag ID for ExposureBiasValue is 37380
#                Tag ID for MaxApertureValue is 37381
#                Tag ID for SubjectDistance is 37382
#                Tag ID for MeteringMode is 37383
#                Tag ID for LightSource is 37384
#                Tag ID for Flash is 37385
#                Tag ID for FocalLength is 37386
#                Tag ID for ColorSpace is 40961
#                Tag ID for ExifImageWidth is 40962
#                Tag ID for ExifImageHeight is 40963
#                Tag ID for Contrast is 41992
#                Tag ID for Saturation is 41993
#                Tag ID for Sharpness is 41994
#                Tag ID for DeviceSettingDescription is 41995
#                Tag ID for ExposureIndex is 41493
#                Tag ID for ImageDescription is 270
#                Tag ID for SensingMethod is 41495
#                Tag ID for Make is 271
#                Tag ID for FileSource is 41728
#                Tag ID for ExposureTime is 33434
#                Tag ID for ExifInteroperabilityOffset is 40965
#                Tag ID for XResolution is 282
#                Tag ID for FNumber is 33437
#                Tag ID for SceneType is 41729
#                Tag ID for YResolution is 283
#                Tag ID for ExposureProgram is 34850
#                Tag ID for CustomRendered is 41985
#                Tag ID for ISOSpeedRatings is 34855
#                Tag ID for ResolutionUnit is 296
#                Tag ID for ExposureMode is 41986
#                Tag ID for FlashPixVersion is 40960
#                Tag ID for WhiteBalance is 41987
#                Tag ID for BodySerialNumber is 42033
#                Tag ID for Software is 305
#                Tag ID for DateTime is 306
#                Tag ID for DigitalZoomRatio is 41988
#                Tag ID for FocalLengthIn35mmFilm is 41989
#                Tag ID for SceneCaptureType is 41990
#                Tag ID for GainControl is 41991
#                Tag ID for Model is 272
#                Tag ID for SubjectDistanceRange is 41996
#                Tag ID for Orientation is 274
#                Tag ID for ExifOffset is 34665
#                Tag ID for YCbCrPositioning is 531
#                Tag ID for MakerNote is 37500
#        img_day = str(img_date).split(" ")[0]
#        img_day = img_day.replace(":",".")
#        img_time = str(img_date).split(" ")[1]
        # write info to image
        img = cv.imread(image)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, f"{img_date}", (10,10), font,  1.5, (255,255,255), 5, cv.LINE_AA)
        cv.imshow("", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    os.chdir(org_path)
