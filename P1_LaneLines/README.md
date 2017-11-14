# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

    1. color_filter
    2. gaussian_blur
    3. grayscale
    4. canny  
    5. region_of_interest
    6 hough_lines
    7. draw_lines
    8. weighted_img

1. Applied a color filter using inRange. The thresholds were determined by examining them in a 3rd party software and determining the range of bgr valuesof the lines

2. Gausian Blur with a kernel size of 7

3. grayscale - self explanatory

4. canny filter on grayed image with htreshold range of 5-15

5. Region of Interest - Ised a polygon with coordinates of [ [440,329],[538,329], [1000,600],[0,600]] for my ROI. Testing confirmed that ROI had to be applied after canny or else canny would detect an edge on the border of the ROI

6. Houghlines - I used the HoughLines opencv call rather than houghLinesP because I found the line output to be more friendly for averaging/extrapolating. I used a threshold of 30 that was determined be empirical analysis 

7. I drew the lines on a tmp black image. I modified the draw lines function in order to classify lines based on the 
angle(quadrant | or ||). A slight filter is applied in order to remove extreme values. Post angle filter, each  of the 2 quadrant's points are averaged. In order to extrapolate the line segments detected I compute y=mx+b for each quadrant and use this to compute the intersection of the 2 lines. Finally, I draw 2 lines each starting with an extreme value that resides off screen and ends at the intersection of the 2 lines

8. I use weighted_img to combine the lines with the original image



### 2. Identify potential shortcomings with your current pipeline


1. Cars in the ROI would result in lines being detected that did not pertain to lane lines
2. Construction zones that did not have a middle line to seperate 2 lanes would be detected as one big lane
3. snow/rain on the road might result n lines that were difficult to detect with the given pipeline
4. region of interest might be incorrect whenever sharper turns occur 

### 3. Suggest possible improvements to your pipeline

Several that come to mind
1. Moving average could be applied to the lines to reduce noise
2. horizon detection could be used  to dynimaclly calculate ROI
3. object recognition to filter out cars in ROI
