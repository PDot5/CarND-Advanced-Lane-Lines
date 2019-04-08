# Self Driving Car Lane Detection

---
## Goals:

The goals of this project are as follows:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### 1. Camera Calibration

The camera calibration will return the camera matrix, distortion coefficients, rotation and translation vectors. By making a list of images to step through, the camera calibration will search for corners, select, and display the points.

### 2. Undistort and Warp
Using the openCV undistort function, the image will be undistorted and then converted to grayscale. Then we can search for corners in the grayscale image. Choose an offset from the image and plot detected corners and graph the source points for the outer four corners. Next, get destination points for displaying warped result. Given src and dst points, calculate the perspective transform matrix and then warp the image with OpenCV warpPerspective(). You can them transform the image persective to get a bird-eye-view.

![](../images_p2/undistorted.png)

![](../images_p2/undistort_warp.png)

![](../images_p2/unperspective.png)

![](../images_p2/bird_eye_view.png)

### 3. Color Threshold

Define the HLS color threshold by converting the RGB image to HLS and apply the threshold to the specified channel and return the binary threshold result.

![](../images_p2/convert_HLS.png)

### 4. Magintude Threshold

Define a function that applies Sobel x and Sobely which then computes the magnitude of the gradient and applies it to the threshold. Next, calculate the magnitude and scale to 8-bit. Create a binary mask where the magnitude thresholds are met and return the binary output image.

### 5. Directional Threshold

Using this function, we are able to take the derivative of the image by using the sobel operator and then calculate the absolute value of both sobelx and sobely. Next, calculate the direction by using the arctan2 function.By creating a copy of the image, we can a mask as the binary_output

![](../images_p2/threshold_gradient.png)

### 6. Combined Threshold

By using both magnitude and direction, create a mask of the compined thresholds and return the combined output.

### 7. Absolute Threshold
Define a function that applies sobel x or y and take the absolute value and applies a threshold. Apply x or y gradient with the OpenCV Sobel() function and take the absolute value of the orientation you choose. Rescale back to 8-bit int and then create a copy of the image and apply the threshold. Using inclusive thresholds, return the result.

### 8. Histogram

Create a function that finds the peaks of where the binary activations occur across the image by first choosing the bottom half of the image which is where we are interested in for line detection. Sum across the image pixels veritcally since this is the most likely for lane lines and make sure to set 'axis' and return the result.

![](../images_p2/histogram.png)

### 9. Finding Lane Line Pixels

By using the histogram function, we can create an output image to draw and visualize the findings. Find the peak of the left and right halves of the histogram starting points. Next, set height of windows based on the number of windows (nwindows) above and image shape and identify the x and y positions by determing all of the nonzero pixels in the image.

Create an empty list in order to receive the left and right lane pixel indicies. By stepping through the windows, we can identify the boundaries in the x and y directions as well as the left and right halves. By identifying the nonzero pixels in the x and y found in the window number, we can append these indicies to a list and then recenter the next window to the mean position. Concatenate the list of indicies from the previous list of pixels and then we can extract the left and right pixel position.

### 10. Polynomial Fit

After finding the lane line pixels, generate the x and y values for plotting. Use a calculation for both polynomials using ploty for left_fit and right_fit.

![](../images_p2/poly_fit.png)

### 11. Search Polynomial

Creating a search function, tune the width of the margin around the previous polynomial for the search. Get the activated pixels and extract the left and right line pixel positions. Next, fit the new polynomial and create a new image to draw on and an image to show the selection window. After this step, you can color in the left and right pixels. Generate a polygon to illustrate the search window area and recast the x and y points into usable format for cv2.fillPoly(). Creating a warped blank image, you can display the polynomial lines.

### 12. Generate Data

For each y position generate x position within +/-50 pix of the line base position in each case (x=200 for left, and x=900 for right). For the left and right x positions, reverse to match top-top-to-bottom in the y direction and then fit a second order polynomial to pixel positions in each fake lane line. 

### Measure Curvature Pixels

Now you can feed in the real data and choose the y-values of where you want the radius of curvature. We'll choose the maximum y-value, corresponding to the bottom of the image. Use a calculation in order to determing the radius curvature. Another calculation will be used to determine the center of the vehicle and the left_lane and right_lane bottom in pixels.

### Measure Curvature Real

Given the previous found information, we can now use a conversion to calculate meters for real world use. Next, feed in the real data and implement the calculations of the radius of the curve. Calculate the center of the vehicle and the left_lane and right_lane bottom in pixels. Next, use the conversion calculation to convert from pixels to meters.

### Reverse Perspective Transformation

In this step we will take in the information and fit a new polynomial to the x and y in world space by creating a new image to draw on. Recast the x and y points into a usable format for the cv.fillPoly() function. The next step will be to draw the lane lines on to the warped blank image and then warp the blank image back to the original image space using the inverse persective matrix (un_M). Now you can combined this with the orginal image and return the result.

![](../images_p2/overlay_warp.png)

### 2. Potential shortcomings within current pipeline

## Video Pipleine

### Process Image

Create the pipleline in order to step through each frame of the video in order to produce an output video with the overlaying mask of the lane line detection by:

* Calibrating and undistoring the image
* Unwarp the image
* Find lane pixels
* Use the search_around_poly function to et the activated pixels and extract the left and right line pixel positions
* Combined found thresholds
* Reverse prespective transformation allows for the calibrated image and the combined thresholds to fit the right and left lanes on an image
* Return the final result.


## Potential Shortcomings

The left_fit and right_fit can be adjusted to gain a better fit on the image mask overlay. Additionally, the thresholds can be adjusted some in order to better detect the pixels in the magnitude and direction thresholds.


## Suggest possible improvements to your pipeline

The pipeline can possible be utilized better, but I will have to revisit that to determine if I am calling the correct functions and double check the calculations in order to gain the best desired output.
