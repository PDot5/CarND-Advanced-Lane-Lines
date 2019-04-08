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
from moviepy.editor import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib qt
get_ipython().run_line_magic('matplotlib', 'inline')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
nx = 9
ny = 6

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        cv2.imshow('Calibrate Camara Images',img)
        cv2.waitKey(10)

cv2.destroyAllWindows()


def cal_undistort(img, objpoints, imgpoints):

    # Return the camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

def unwarp_corners(img, nx, ny, mtx, dist):

# Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        
        # Choose offset from image corners to plot detected corners
        offset = 100 # offset for dst points
        # Grab the image shape
        img_n = (gray.shape[1], gray.shape[0])

        # Graph source points from the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # Get destination points for displaying warped result
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Un_M = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_NEAREST)
        # Return the resulting image and matrix
        return warped, M

def transform_image(img):
    
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    
    # Resources: https://medium.com/intro-to-artificial-intelligence
    # Used for helping to determine upper and lower points

    # Manually input points for upper and lower points
    left_upper  = [568,470]
    right_upper = [717,470]
    left_lower  = [260,680]
    right_lower = [1043,680]
    #Graph source points from the outer four detected corners
    src = np.float32([left_upper, left_lower, right_upper, right_lower])
    dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    un_M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    unwarp = cv2.warpPerspective(img, un_M, img_size, flags=cv2.INTER_NEAREST)

    return warped, M, unwarp, un_M

# Define HLS color threshold
def hls_select(img, chan='H', thresh=(0, 255)):
    # Convert img to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Separate HLS Channels
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    if chan == 'H':
        # Apply a threshold to the H channel and return the binary image of theshold result
        thresh = [15, 110]
        binary_output = np.zeros_like(H)
        binary_output[(H > thresh[0]) & (H <= thresh[1])] = 1
    if chan == 'L':
        # Apply a threshold to the S channel and return the binary image of theshold result
        thresh = [100, 145]
        binary_output = np.zeros_like(L)
        binary_output[(L > thresh[0]) & (L <= thresh[1])] = 1
    if chan == 'S':
        # Apply a threshold to the  channel and return the binary image of theshold result
        thresh = [90, 255]
        binary_output = np.zeros_like(S)
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

# Define a function that applies Sobel x and y and computes
# the magnitude of the gradient and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the x and y derivitave of the image using Sobel Operator (Sobelx and Sobely)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # Calculate the magnitude
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0, 255) and convert to type = np.uint8
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_mag)
    # Return the mask as binary_output image
    binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    # Return result
    return binary_output

######################

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):   
    
    # Covnvert to HLS color
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    
    # Take the x and y derivitave of the image using Sobel Operator (Sobelx and Sobely)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the absolute of both sobelx and sobely
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # Calculate the direction using the acrtan2 function
    dirs = np.arctan2(abs_sobely, abs_sobelx)
    
    #create a copy of the binary_output image
    binary_output = np.zeros_like(dirs)
    
    # Return a mask as the binary_output image
    binary_output[(dirs >= thresh[0]) & (dirs <= thresh[1])] = 1
    
    return binary_output

###################### Combined Threshold

def combined_thresh(img, sobel_kernel=3, abs_thresh=(15,255), _mag_thresh=(15,255), dir_thresh=(0, np.pi/2)):
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255)

    # Using both magnitude and direction, create a mask of the
    # compined thresholds and return the combined output
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))

    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) ) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined

#########################

# Define a function that applies Sobel x or y and takes
# the absolute value and applies a threshold
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with OpenCV Sobel() function and take absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8-bit int
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Using inclusive thresholds
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_min)] = 1
    # Return result
    return binary_output

###########################

# Define a histogram function to find peaks of where
# binary activations occur across the image
def hist(img):
    # Lines most likely to be on bottom half of the image
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels veritcally and make sure to set 'axis'
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def find_lane_pixels(img, nwindows = 9, margin = 100, minpix = 50):
    
    binary_warp = hls_select(top_view, chan='S', thresh=(0,255))
    # Take the histogram of the bottom half of the binary image
    histogram = np.sum(binary_warp[binary_warp.shape[0]//2:,:], axis=0)
    # Create output image to draw on and to visualize results
    out_img = np.dstack((binary_warp, binary_warp, binary_warp))*255
    # Find peak of the left and right halves of the histogram... starting points
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Set height of windows based on nwindows above and image shape
    window_height = np.int(binary_warp.shape[0]//nwindows)
    # Identify the x and y positions of the all nonzero pixels in the image
    nonzero = binary_warp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current position to be updated
    leftx_curr = leftx_base
    rightx_curr = rightx_base

    # Create empty list to recieve left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows 1X1
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warp.shape[0] - (window + 1)*window_height
        win_y_high = binary_warp.shape[0] - window*window_height
        win_xleft_low = leftx_curr - margin
        win_xleft_high = leftx_curr + margin
        win_xright_low = rightx_curr - margin
        win_xright_high = rightx_curr + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in the x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter the next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_curr = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_curr = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previous was list of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

left_fit = np.array([2.57257e-04, -3.14207980e-01,  3.174e+02])
right_fit = np.array([2.6479348e-04, -3.42848953e-01,  1.1616170e+03])

def fit_poly_one(binary_warp):
    top_view, M_pers, un_W, un_M = transform_image(img)
    # Find the lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warp)
    # Fit a second order polynomial to each useing 'np.polyfit'
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warp.shape[0]-1, binary_warp.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if 'left_fit' and 'right_fit' are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Visualization... Determine colors in left and right lane regins
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img

def fit_poly_two(img, leftx, lefty, rightx, righty):
    
    binary_warp = hls_select(top_view, chan='S', thresh=(0,255))
    # Find the lane pixels first
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warp.shape[0]-1, binary_warp.shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty
    
def search_around_poly(img):

    binary_warp = hls_select(top_view, chan='S', thresh=(0,255))
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly_two(binary_warp.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warp, binary_warp, binary_warp))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result

def generate_data(ym_per_pix, xm_per_pix):
  
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    ##### TO-DO: Fit new polynomials to x,y in world space #####
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    
    return ploty, left_fit_cr, right_fit_cr

def gen_data():
    
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    right_fit = np.polyfit(ploty, rightx, 2)
    
    return ploty, left_fit, right_fit
    
def measure_curvature_pixels(img, left_fit, right_fit):
   
    # Feed in real data
    ploty, left_fit, right_fit = gen_data()
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Calculate vehicle center
    # left_lane and right_lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    center_img = 640
    center = (left_lane_bottom + right_lane_bottom)/2

    return left_curverad, right_curverad, center


def measure_curvature_real():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle center
    #left_lane and right lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_img = 640
    center = (lane_center - center_img)*xm_per_pix # Convert to meters

    return left_curverad, right_curverad, center
#########################

def reverse_pers_trans(img, binary_warp, left_fit, right_fit, un_M):

    ploty = np.linspace(0, binary_warp.shape[0]-1, binary_warp.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Fit new polynomials to x,y in world space
    left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
    right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2] 
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane lines onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (un_M)
    newwarp = cv2.warpPerspective(color_warp, un_M, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    # plt.imshow(result)

    return result

######################

def draw_data(img, left_curverad, right_curverad, center):
    new_img = np.copy(img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_text = 'Left Curve Radius: ' + '{:04.2f}'.format(left_curverad) + 'm'
    r_text = 'Right Curve Radius: ' + '{:04.2f}'.format(right_curverad) + 'm'
    cv2.putText(new_img, l_text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    cv2.putText(new_img, r_text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center > 0:
        direction = 'right'
    elif center < 0:
        direction = 'left'
    abs_center_dist = abs(center)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,180), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    
    return new_img

def process_image(img):
    #video pipeline
    calibrated = cal_undistort(img, objpoints, imgpoints)
    undist = unwarp_corners(calibrated, nx, ny, mtx, dist)
    out_img = find_lane_pixels(undist)
    fitPoly = search_around_poly(out_img)
    combined = combined_thresh(fitPoly, sobel_kernel=3, abs_thresh=(15,255))
    result = reverse_pers_trans(calibrated, combined, left_fit, right_fit, un_M)
    final_result = draw_data(result, left_curverad, right_curverad, center)
    
    return final_result

#######################

print(objp.shape)
print(corners.shape)

img_size = (img.shape[1], img.shape[0])
print(img_size)

%matplotlib inline
plt.figure(figsize=(10,8))
img = mpimg.imread("../camera_cal/calibration5.jpg")

######################## Undistort using mtx and dist

undistorted = cal_undistort(img, objpoints, imgpoints)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Undistorted')
fig = plt.imshow(undistorted)

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
undistorted = cal_undistort(img, objpoints, imgpoints)

plt.subplot(2,2,3)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,4)
plt.title('Undistorted')
fig = plt.imshow(undistorted)

########################## Unwarp Corners

# Return the camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

plt.figure(figsize=(10,8))
img = mpimg.imread("../camera_cal/calibration3.jpg")
top_down, perspective_M = unwarp_corners(img, nx, ny, mtx, dist)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Undistorted & Warped')
fig = plt.imshow(top_down)

plt.figure(figsize=(10,8))
img = mpimg.imread("../camera_cal/calibration3.jpg")

top_down, perspective_M = unwarp_corners(img, nx, ny, mtx, dist)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Perspective Transform')
fig = plt.imshow(perspective_M)

########################### Bird Eye View Transform

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
# Get Bird Eye View using getPerspectiveTransform
top_view, M_pers, un_W, un_M = transform_image(img)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Bird Eye View')
fig = plt.imshow(top_view)

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test4.jpg")
# Get Bird Eye View using getPerspectiveTransform
top_view, M_pers, un_W, un_M = transform_image(img)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Bird Eye View')
fig = plt.imshow(top_view)

############################## Unpersective


plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test2.jpg")
# Get Bird Eye View using getPerspectiveTransform
top_view, M_pers, un_W, un_M = transform_image(img)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Unperspective')
fig = plt.imshow(un_W)

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test3.jpg")
# Get Bird Eye View using getPerspectiveTransform
top_view, unpers, un_W, un_M = transform_image(img)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Unperspective')
fig = plt.imshow(un_W)

###############################

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
# Binary image of threshold result
hls_binary = hls_select(img, chan='H', thresh=(0, 255))

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Binary H')
fig = plt.imshow(hls_binary, cmap='gray')

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
# Binary image of threshold result
hls_binary = hls_select(img, chan='L', thresh=(0, 255))

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Binary L')
fig = plt.imshow(hls_binary, cmap='gray')

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
# Binary image of threshold result
hls_binary = hls_select(img, chan='S', thresh=(0, 255))

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Binary S')
fig = plt.imshow(hls_binary, cmap='gray')

###############################

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
# Binary image of threshold result
mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(20, 180))

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Threshold Magnitude')
fig = plt.imshow(mag_binary, cmap='gray')

######################### Binary image of threshold result

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
abs_sobel_binary = abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=60)

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Threshold Gradient X')
fig = plt.imshow(abs_sobel_binary, cmap='gray')

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
abs_sobel_binary = abs_sobel_thresh(img, orient='y', thresh_min=0, thresh_max=60)

plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Threshold Gradient Y')
fig = plt.imshow(abs_sobel_binary, cmap='gray')

#################### Create histogram and visualize the result

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test4.jpg")
histogram = hist(img)

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Histogram')
plt.plot(histogram)

##########################

binary_warp = hls_select(top_view, chan='S', thresh=(0, 255))
plt.figure(figsize=(10,8))
hist_img = hist(binary_warp)

# Plot the result
plt.subplot(2,2,1)
plt.title('Bird Eye View')
fig = plt.imshow(top_view, cmap="gray")

plt.subplot(2,2,2)
plt.title('Binary Bird Eye View')
fig = plt.imshow(binary_warp, cmap="gray")

plt.subplot(2,2,3)
plt.title('Histogram')
plt.plot(hist_img)
############################ Poly Fit One

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test4.jpg")
out_img = fit_poly_one(img)

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Sliding Window Polynomial Fit')
fig = plt.imshow(out_img)


########################### Poly Fit Two Using Previous Fitted Lines

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test5.jpg")
result = search_around_poly(binary_warp)

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)


plt.subplot(2,2,2)
plt.title('Search Previous Poly Fit')
ploty = np.linspace(0, binary_warp.shape[0]-1, binary_warp.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
plt.plot(left_fitx, ploty, color='yellow', linewidth=2)
plt.plot(right_fitx, ploty, color='yellow', linewidth=2)

fig = plt.imshow(result)

#########################
print("Left & Right and Center Radius of Curvature:")
left_curverad, right_curverad, center = measure_curvature_pixels(img, left_fit, right_fit)
print(left_curverad, right_curverad, center)

print("Conversion to Real World Space:")
# Calculate the radius of curvature in meters for both lane lines
left_curverad, right_curverad, center = measure_curvature_real()
print(left_curverad, 'm', right_curverad, 'm', center, 'm')

########################## Reverse Trasnform

combined = combined_thresh(img, sobel_kernel=3, abs_thresh=(15,255))

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test4.jpg")
results = reverse_pers_trans(img, combined, left_fit, right_fit, un_M)

# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Overlay Warp')
fig = plt.imshow(results)

#########################

plt.figure(figsize=(10,8))
img = mpimg.imread("../test_images/test4.jpg")
overlay = process_image(img)
# Plot the result
plt.subplot(2,2,1)
plt.title('Original')
fig = plt.imshow(img)

plt.subplot(2,2,2)
plt.title('Overlay Warp')
fig = plt.imshow(overlay)

######################### Process Video

Output_Video = '../project_video_output.mp4'
Input_Video = '../project_video.mp4'
clip1 = VideoFileClip(Input_Video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(Output_Video, audio=False)

#%% [markdown]

