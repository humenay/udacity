# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_new_images/dataVis.png "Visualization"
[image2]: ./my_new_images/1.png "Traffic Sign 1"
[image3]: ./my_new_images/2.png "Traffic Sign 2"
[image4]: ./my_new_images/3.png "Traffic Sign 3"
[image5]: ./my_new_images/4.png "Traffic Sign 4"
[image6]: ./my_new_images/5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/humenay/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3)
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

###Design and Test a Model Architecture

The following preprocessing techniques were applied:
1)Converting rgb to grayscale
2)Normalization

I chose these preprocessing techniques because they were suggested in the course material. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.). My model was based of the suggested LeNet architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
|:---------------------:|:---------------------------------------------:|
| Convolution 1     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Input = 28x28x6. Output = 14x14x6.|
|:---------------------:|:---------------------------------------------:|
| Convolution 2	        | 1x1 stride, output 10x10x16      			    |
| RELU		            | 									            |
| Max pooling	      	| 2x2 stride output 5x5x16                      |
| Flatten					| 	Output 400								|
|:---------------------:|:---------------------------------------------:|
| Fully Connected		| Input = 400. Output = 120.                    |
| RELU		            | 									            |
|:---------------------:|:---------------------------------------------:|
| Fully Connected		| Input = 120. Output = 84                      |
| RELU		            | 									            |
|:---------------------:|:---------------------------------------------:|
| Dropout		| dropout ratio is 50%. Input = 84. Output = 43         |
| RELU		            | 									            |


 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters.
EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 
I orignally implemented the normalization routine. 
The second sthing I did was apply the gray scale preprocessing technique and I saw that it had a positive effect on the accuracy.
Third thing I did was try several different learning rates and I saw that this did not have much of an impact on accuracy
The fourth thing I did is experimented with the hyperbolic tangent activation function and I saw that this did not have a positive impact on accuracy. 
The 5th thing I did was to implement a dropout layer. This had a positive impact on accuracy so I kept it. 
Last, I increased my epoch's to 20


My final model results were:
* validation set accuracy of 93%



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 80km/hr      		    | 30km/hr   									| 
| No entry     			| No entry 										|
| Yield					| Yield											|
| Stop Sign	      		| Stop Sign					 				|
| Construction			| Construction      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80% which is less than what I was expecting

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 



Sign 1 (index is 5):
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 1   									| 
| 3.19921645e-09     				| 2 										|
| 8.30630231e-10					| 5											|
| 1.33716135e-10	      			| 3					 				|
| 7.24599010e-14				    | 14      							|

Sign 2 (index is 17):
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 17   									| 
| 5.53777944e-15     				| 33 										|
| 5.67539537e-17					| 14											|
| 2.94318611e-26	      			| 26					 				|
| 4.37442169e-27				    | 8      							|

Sign 3 (index is 13):
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 13   									| 
|5.98226955e-21     				| 35 										|
| 4.96197377e-21					| 9											|
| 1.14820736e-24	      			| 15					 				|
| 1.12493738e-26				    | 33      							|

Sign 4 (index is 14):
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|9.99999762e-01         			| 14   									| 
|2.35856689e-07     				| 33 										|
|3.21441078e-08					| 13										|
|1.71886316e-09	      			| 35					 				|
|1.87380680e-10				    | 25      							|

Sign 5 (index is 25):
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|9.99999762e-01         			| 25   									| 
|1.58199946e-05     				| 22 										|
|1.87849434e-06					| 31										|
1.61548701e-06	      			| 29					 				|
|1.03638996e-07				    | 39      							|

The 80km/hr sign was not classified correctly. It is worth noting that the top 5 softmax values were all numbers with signs so apparently 
the classifier is having an issue with numbers


