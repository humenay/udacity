import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


#global variables
images = []
measurements = []

def readInFile(filePath):
	lines = []
	cnt = 0
	with open(filePath) as csvfile:
		#read in csv file
		reader = csv.reader(csvfile)
		for line in reader:
			if(cnt != 0):
				lines.append(line)
			cnt = cnt + 1
			
		#load images that are defined in csv file
		for line in lines:
			src_path = line[0]
			if filePath.find("./data/") >= 0:
				filename = src_path.split('/')[-1]
				current_path = './data/IMG/' + filename
			elif filePath.find("./saved_data/") >= 0:
				filename = src_path.split('\\')[-1]
				current_path = './saved_data/IMG/' + filename
			else:
				current_path = src_path
				
			img = cv2.imread(current_path)
			images.append(img)
			meas = float(line[3])
			measurements.append(meas)
			
			
			
readInFile('./data/driving_log.csv')
readInFile('./saved_data/joystick1.csv')
readInFile('./sim_data/run1/driving_log.csv')
readInFile('./sim_data/backwards/driving_log.csv')	
readInFile('./sim_data/turns/driving_log.csv')
readInFile('./sim_data/recovery/driving_log.csv')



#load images into training dataset (X_train) and the measurements into label dataset (y_train)
X_train = np.array(images)
y_train = np.array(measurements)


# model based off nvidia end to end driving model 
model = Sequential()
model.add(Lambda(lambda x: (x/255.0) -.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

#train it!
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2)


#save model
model.save('model.h5')
exit()