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
The _applyFilters_ function in the main pipeline loop takes in a warped image and outputs a filtered binary image. I experiemented with several different strategies. Inititally, I focused on using only HSV and HSL thresholding, but I was not capturing the white lines to the extent that was necessary. Therefore, I added sobel detection to the HSV/HSL filtering strategy for my final implementatoin. The 4 filtered output and the sobel output is combined into one final image via a call top openCV's bitwiseOr
```python
void applyFilters(Mat in, Mat & out)
{
	Mat tmp,tmpHSV,tmpHLS,gray,sobelx;

	cv::cvtColor(in, tmpHLS, CV_RGB2HLS);
	cv::cvtColor(in, tmpHSV, CV_RGB2HSV);

	//Apply light filter in HLS space
	inRange(tmpHLS, cv::Scalar(1, 1, 180), cv::Scalar(255, 255, 255), tmp);

	out = tmp;
	
	//Apply HSV Filters to attempt to extract white and yellow lines
	inRange(tmpHSV, cv::Scalar(0, 50, 240), cv::Scalar(180, 255, 255), tmp);
	bitwise_or(tmp, out, out);

	inRange(tmpHSV, cv::Scalar(0, 0, 240), cv::Scalar(40, 23, 255), tmp);
	bitwise_or(tmp, out, out);

	inRange(tmpHSV, cv::Scalar(150, 0, 230), cv::Scalar(180, 10, 255), tmp);
	bitwise_or(tmp, out, out);

	//Apply sobel detection which is similar to Canny edge detection
	cvtColor(in, gray, CV_RGB2GRAY);
	Sobel(gray, sobelx, CV_8U, 1, 0);

	//if a matrix value is above threshold of 50 then make it 255 which is akin to 1 in a binary image
	threshold(sobelx, sobelx, 50, 255, CV_THRESH_BINARY);

	//combine sobel with HLS and HSV filtered matrix
	bitwise_or(sobelx, out, out);
}
```
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

```python
	while (notFinished == true)
	{
		//init frame is pristine and not manipulated
		initFrame = camFrame.clone();

		//undistort current frame 
		undistort(camFrame, undistortedFrame, intrinsic, distCoeff);
	
		//apply perspective transform
		warpPerspective(undistortedFrame, warpedFrame, M, undistortedFrame.size());
		
		//apply filters and thresholding on warped image
		applyFilters(warpedFrame, binaryWarpedFrame);
		
		//perform lane detection algorithm. Lane points contains the points 
		//corresponding to the estimated equations for the left and right lanes
		lanePoints = detectLanes(binaryWarpedFrame);
		
		const cv::Point* elementPoints[1] = { &lanePoints[0] };
		int numberOfPoints = (int)lanePoints.size();

		colorWarpedFrame = Mat::zeros(binaryWarpedFrame.size(), CV_8UC3);

		//color the space in between the 2 detected lanes
		fillPoly(colorWarpedFrame, elementPoints, &numberOfPoints, 1, Scalar(0, 255, 0), 8);

		//unwarp image with highlighted lane back to original 
		warpPerspective(colorWarpedFrame, colorWarpedFrame, Minv, colorWarpedFrame.size());

		//overlay highlighted lane on original image
		addWeighted(initFrame, 1, colorWarpedFrame, .3, 0, processedFrame);

		//calculate and insert curvature and offset information 
		insertCurvatureOffset(processedFrame);

		imshow("processed frame", processedFrame);
		waitKey(1);
		
		//read the next frame
		notFinished = camStream.read(camFrame);
	}
```


4. Discussion
---
Doing the project in C++ rather than python did introduce some extra work. In particular, not having numpy to calculate the polynomial fit. However, I am confident that in the future I can develop in C+ + for openCV-based projects

