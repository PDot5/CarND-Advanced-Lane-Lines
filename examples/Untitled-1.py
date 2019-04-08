#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'examples'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ## First, I'll compute the camera calibration using chessboard images

#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

def cal_undistort(img, objpoints, imgpoints):
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(10)
        
        cv2.destroyAllWindows()   

    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)
        

print('ok')
print(objp.shape)
print(corners.shape)

img_size = (img.shape[1], img.shape[0])
print(img_size)

%matplotlib inline
plt.figure(figsize=(10.,8))
img = mpimg.imread("../camera_cal/calibration5.jpg")
# Undistort using mtx and dist
undist = cv2.undistort(img, mtx, dist, None, mtx)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Undistorted')
fig = plt.imshow(undist)

#%% [markdown]




