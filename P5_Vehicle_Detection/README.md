## Udacity CarND

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[searchwindow0]: ./output_images/searchwindows1.png
[searchwindow1]: ./output_images/searchwindows2.png
[searchwindow2]: ./output_images/searchwindows3.png
[searchwindow3]: ./output_images/searchwindows4.png
[searchwindowAll]: ./output_images/searchwindowsAll.png
[pipeline0]: ./output_images/pipeline0.png
[pipeline1]: ./output_images/pipeline1.png
[pipeline2]: ./output_images/pipeline2.png
[pipeline3]: ./output_images/pipeline3.png
[pipeline4]: ./output_images/pipeline4.png
[pipeline5]: ./output_images/pipeline5.png
[heatmap0]: ./output_images/heat-map0.png
[heatmap1]: ./output_images/heat-map1.png
[heatmap2]: ./output_images/heat-map2.png
[heatmap3]: ./output_images/heat-map3.png
[heatmap4]: ./output_images/heat-map4.png
[heatmap5]: ./output_images/heat-map5.png
[video1]: ./processed_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
 Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### 1 Writeup / README

#### 1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### 2 Histogram of Oriented Gradients (HOG)

#### 2.1 Explain how (and identify where in your code) you extracted HOG features from the training images.
HOG features were extracted via the lesson supplied _get_hog_features_ function.  The code for this step is contained in the 2nd code cell. get_hog_featurees is called by the extract_features which is both used to train the classifier as well as performing vehicle detection 

#### 2.2 Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

```Python
#HOG Params
hog_feat = True
orient = 12  # HOG orientations.  
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block 
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

#spatial binning params
spatial_feat = True # Spatial features on or off
spatial = 32
spatial_size = (spatial,spatial) # Spatial binning dimensions 

#color histogram params
hist_feat = False # Histogram features on or off
hist_bins = 32    # Number of histogram bins
```

In the "Hog Classify" lesson from the course, I tried various combinations of parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) in order to find the values that maximized the classifier's accuracy. My final HOG settings can be found in the "Parameter Initialization" cell. Color features were not used in my final algorithm in order to reduce my computational overhead

The classifier is trained in the "Classifier" section of the notebook. I modified my original version that used the default settings for LinearSVC such that GridSearchCV is used to tune the classifier in order to improve accuracy. The tuned classifier's accuracy is compared to the default classifier accuracy and the better of the 2 performing classifiers are chosen as the classifier moving forward. Testing shows that the tuned model always has a better accuracy than the default model by .2-.3%
### 3 Sliding Window Search

#### 3.1 Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I inititalized the search window parameters in the "Parameter Initialization" cell in the notebook. The sliding window search, which was extracted from a lesson, can be found in the "Search Windows" cell
![alt text][searchwindow0]
![alt text][searchwindow1]
![alt text][searchwindow2]
![alt text][searchwindow3]
![alt text][searchwindowAll]

#### 3.2 Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][pipeline0]
![alt text][pipeline1]
![alt text][pipeline2]
![alt text][pipeline3]
![alt text][pipeline4]
![alt text][pipeline5]
---

### 4 Video Implementation

#### 4.1
Here's a [link to my video result][video1]


#### 4.2  Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are the calculated heatmaps and their corresponding bounding boxes:

![alt text][heatmap0]
![alt text][heatmap1]
![alt text][heatmap2]
![alt text][heatmap3]
![alt text][heatmap4]
![alt text][heatmap5]





---

### 5 Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### Problems
* Processing of the final video took ~45minutes thus prohibiting rapid devleopment. In order to reduce the processing overhead I suggest creating video snippets that focus on s subsection of the final video
* False positives are still present. False positives could be further reduced by fusing different classification schemes via a Kalman filter into one final output that is more accurate than a stand-alone classification scheme
* The algorithm is easily parrallelizable. If deploying on a real-time system then hardware with a poarallel architecture should be chosen (ie Jetson TX2)
* The project suggested using a linear svm. More complex classifiers, such as a Decision Tree, may provide for more accurate results
* My results indicated the classifying with color histogram did not improve performance. I personally feel that this is useful information and will improve accuracy. Further investigation is necessary to determine why color binning did not improve the model

