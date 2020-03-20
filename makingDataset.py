import numpy as np
import os
import cv2
import random

trainingDir = 'train/'
testingDir = 'test/'
imageSize = 49

def CreateLabel(imagename):

	label = imagename.split('.')[0][-1]
	if (label == '1'):
		return np.array([1,0])
	elif (label == '0'):
		return np.array([0,1])

def createTrainData():

	trainingData = []
	dirs = os.listdir(trainingDir)

	for img in dirs:
		pathOfImage = os.path.join(trainingDir, img)
		imageData = cv2.imread(pathOfImage)
		imageData = cv2.resize(imageData, (imageSize, imageSize))
		trainingData.append([np.array(imageData), CreateLabel(img)])

	random.shuffle(trainingData)

	x = []
	y = []
	for data,label in trainingData:
		x.append(data)
		y.append(label)

	x = np.array(x).reshape(-1,imageSize,imageSize,3)
	x = x / 255.0	
	np.save('xData.npy',x)
	np.save('yData.npy',y)
	return trainingData

def createTestData():

	testingData = []
	dirs = os.listdir(testingDir)

	for img in dirs:
		pathOfImage = os.path.join(testingDir, img)
		imageData = cv2.imread(pathOfImage)
		imageData = cv2.resize(imageData, (imageSize, imageSize))
		x = np.array(imageData)
		x = x / 255.0 
		testingData.append([x, CreateLabel(img)])

	random.shuffle(testingData)
	np.save('testingData.npy', testingData)
	return testingData

createTrainData()
createTestData()




