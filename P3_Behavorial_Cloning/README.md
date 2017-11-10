# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./submission_images/cnn_architecture.png "Model Visualization"
[image2]: ./submission_images/car1.png "center driving"
[image3]: ./submission_images/car2.png "Recovery Image"
[image4]: ./submission_images/car3.png "Recovery Image"


## 1 Files Submitted & Code Quality

### 1.1 Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results 

### 1.2 Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 1.3 Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## 2 Model Architecture and Training Strategy

### 2.1 An appropriate model architecture has been employed

My model consists of a convolution neural network that was presented in  NVIDIA's "End to End Learning for Self-Driving Cars" paper

Input into the model was preprocessed as follows:
* by normalizing it to being between -.5 and .5 centered at 0
* images were cropped in order to reduce the top portion of the images that was primarily sky

### 2.2 Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting.


### 2.3 Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

### 2.4 Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

## 3. Architecture and Training Documentation

### 3.1. Solution Design Approach

It quickly became apparent that the crux of the problem was not designing/coding the convolutional neural network, but rather gaining quality data for the training set. 
The following were all choices that I made that helped me converge upon a working model with minimal overhead
* my software design and data collection effort was a piece-meal approach. This allowed me to incrementally add trainng data in order to understand the training data's impact
* I used a joystick for training. This allowed me to have smoother steering


### 3.2. Final Model Architecture

The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper

Here is a visualization of the architecture

![alt text][image1]

### 3.3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded data going the opposite way on the track in order to introduce diversity into the data set. 

I reran the model in the simulator and noticed my car was veering off the road for sharp curves so I gathered training data that was limited to only curves. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recenter itself if ventured from the center of the road

![alt text][image3]
![alt text][image4]
