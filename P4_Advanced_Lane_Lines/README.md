## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**_It is worth noting, that the class has been 100% python-based. However, there were no code requirements for this project so I decided to implement the project in C++ to famailiarize myself with this API_**

[//]: # (Image References)

[image1]: ./output_images/distortion_correction1.jpg "Undistorted Chessboard"
[image2]: ./output_images/distortion_correction2.jpg "Undistorted Road"
[image3]: ./output_images/perspective_transform.jpg "perspective transform"
[image4]: ./output_images/binary_threshold.jpg "filtered"
[image5]: ./output_images/lane_detection.jpg "detection"
[image6]: ./output_images/final_image.jpg "processed frame"



1. Camera Calibration:
---
![alt text][image1]

2. Pipeline (single images)
---
#### 2.1 Distortion-corrected Image
![alt text][image2]

#### 2.2  Describe how (and identify where in your code) you performed a perspective transform
On line 628 of P4_Advanced_Lane_Lines.cpp, I call openCV's warpPerspective function with the input being an undistorted image and the transformed image in warpedFrame. M was calculated in an initital step in the function initWarpImg and does not change throughout the lfetime of the program
```python
		//apply perspective transform
		warpPerspective(undistortedFrame, warpedFrame, M, undistortedFrame.size());
```
![alt text][image3]

#### 2.3 Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  
The _applyFilters_ function in the main pipeline loop takes in a warped image and outputs a filtered binary image. I experiemented with several different strategies. Inititally, I focused on using only HSV and HSL thresholding, but I was not capturing the white lines to the extent that was necessary. Therefore, I added sobel threshold in the x gradient. This sobel gradient did introduce a little noise so I applied a gaussianBlur prior to the input of the sobel threshold. In areas of the video that were in shadows or bright light, my algorithm had a tendency to lose the yellow lane. I resolved this be applying a LAB filter with thresholds that were empirically determined during testing.  The 4 filtered output and the sobel output is combined into one final image via a call top openCV's bitwiseOr

![alt text][image4]


#### 2.4 Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I used a sliding window strategy similar to the one suggested in the lectures to detect the location of each line within. In the case where the maximum pixel count in a column is the same for multiple columns in a window then I take an average of the columns. If there is no lane detected in a window then I use the previous location.  As discussed in class I then generate fake data around around each predicted lane index which is then fit into the 2nd order polynomial fit equation 
![alt text][image5]

#### 2.5 Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
This occurs in the insertCurvatureOffset function which is the final processing stage in video pipeline. I calculate both the center offset and the curvature in the manner suggested in the lectures. 



#### 2.6 Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
![alt text][image6]

3. Pipeline (video)
---
Here's a [link to my video result](./project_video_solution.mp4). The code snippet below is my main processing loop



4. Discussion
---
Implementing the project in C++, rather than python, did introduce some extra work. In particular, not having numpy to calculate the polynomial fit.  A byproduct of this project was that I created a GUI based tool that allows me to apply different filters on real-time video while modifying there threshold values. I expect this software to be a useful personal tool for future endeavors

I do not expect my intitial algorithm to perform consistently at different times of the day due to my filters dependency on light and color (which is also dependent upon light). Therefore, I believe an edge detection based algorithm would has greater potential. 

My algorithm could be made more reliable with addiotional testing and error checking. For example, I would not be comfortable letting this algorithm control my vehicle knowing that there are no gaurds in the code (other than my moving average) that would prevent a drastic change in the predicted lane location from one frame to the next whichc ould possibly occur if a false positive were to occur. 



