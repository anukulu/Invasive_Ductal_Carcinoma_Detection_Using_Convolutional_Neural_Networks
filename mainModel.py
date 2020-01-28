import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import png

# This is a CNN model (biases are still to be added during activation and batch normalization has yet to be implemented)

class CNN:
	def __init__(self, learningRate, image, category):
		self.learningRate = learningRate
		self.maxPoolFilterSize = 2
		self.maxPoolStride = 2
		self.image = image
		self.category = category
		self.epsilon = 0.01

		# declarations for layer 1
		self.numberOfFiltersLayer1 = 32
		self.filterSizeLayer1 = 3
		self.filterDepthLayer1 = 3
		self.filtersLayer1 = np.random.randn(self.numberOfFiltersLayer1,self.filterSizeLayer1,self.filterSizeLayer1,self.filterDepthLayer1)
		self.convoultionSizeLayer1 = self.FindSize(self.image.shape[0],1,self.filterSizeLayer1,2)
		self.convolutionLayer1 = np.zeros((self.convoultionSizeLayer1,self.convoultionSizeLayer1,self.numberOfFiltersLayer1))
		self.reluLayer1 = np.zeros(self.convolutionLayer1.shape)
		self.maxpooledSizeLayer1 = self.FindSize(self.convolutionLayer1.shape[0],0,self.maxPoolFilterSize,self.maxPoolStride)
		self.maxpooledLayer1 = np.zeros((self.maxpooledSizeLayer1,self.maxpooledSizeLayer1,self.numberOfFiltersLayer1))
		
		# declarations for layer 2
		self.numberOfFiltersLayer2 = 128
		self.filterSizeLayer2 = 3
		self.filterDepthLayer2 = self.numberOfFiltersLayer1
		self.filtersLayer2 = np.random.randn(self.numberOfFiltersLayer2,self.filterSizeLayer2,self.filterSizeLayer2,self.filterDepthLayer2)
		self.convoultionSizeLayer2 = self.FindSize(self.maxpooledLayer1.shape[0],1,self.filterSizeLayer2,1)
		self.convolutionLayer2 = np.zeros((self.convoultionSizeLayer2,self.convoultionSizeLayer2,self.numberOfFiltersLayer2))
		self.reluLayer2 = np.zeros(self.convolutionLayer2.shape)
		self.maxpooledSizeLayer2 = self.FindSize(self.convolutionLayer2.shape[0],0,self.maxPoolFilterSize,self.maxPoolStride)
		self.maxpooledLayer2 = np.zeros((self.maxpooledSizeLayer2,self.maxpooledSizeLayer2,self.numberOfFiltersLayer2))

		# declarations for fully connected layer 1
		self.heightOfFlattenedLayer = np.prod(list(self.maxpooledLayer2.shape))
		self.weightsFCLayer1 = np.random.randn(128, self.heightOfFlattenedLayer)
		self.fullyconnectedLayer1 = np.zeros((128,1))
		self.activationFCLayer1 = np.zeros(self.fullyconnectedLayer1.shape)

		# declarations for fully connected layer 2
		self.weightsFCLayer2 = np.random.randn(2,128)
		self.fullyconnectedLayer2 = np.zeros((2,1))
		self.softmaxFCLayer2 = np.zeros(self.fullyconnectedLayer2.shape)

		# Loss
		self.Loss = 0


	def FindSize(self, n, p, f, s):
		return (int(((n+ (2 * p)- f)/s) + 1))

	def ForwardPass(self):

		# for layer 1
		self.convolutionLayer1 = self.LimitingPixels(self.Convolution3D(self.image, self.filtersLayer1, 1, 2))
		self.reluLayer1 = self.LimitingPixels(self.ReluActivation(self.convolutionLayer1))
		self.maxpooledLayer1 = self.MaxPool(self.reluLayer1, self.maxPoolStride)

		# for layer 2
		self.convolutionLayer2 = self.LimitingPixels(self.Convolution3D(self.maxpooledLayer1, self.filtersLayer2, 1, 1))
		self.reluLayer2 = self.LimitingPixels(self.ReluActivation(self.convolutionLayer2))
		self.maxpooledLayer2 = self.MaxPool(self.reluLayer2, self.maxPoolStride)

		# for flattening the last convoluted volume
		self.flattenedLayer = self.maxpooledLayer2.flatten()

		# for fully connected layer 1 
		self.fullyconnectedLayer1 = np.dot(self.weightsFCLayer1, self.flattenedLayer)
		self.activationFCLayer1 = self.ReluActivation(self.fullyconnectedLayer1)

		# for fully connected layer 2 (also apply softmax in this layer)
		self.fullyconnectedLayer2 = np.dot(self.weightsFCLayer2, self.activationFCLayer1)
		self.softmaxFCLayer2 = self.Softmax(self.fullyconnectedLayer2)
		print(self.softmaxFCLayer2)

		# evaluating the loss for the model
		if (self.softmaxFCLayer2[np.where(self.category == 1)] == 0):
			self.Loss = -1 * np.log(self.epsilon + self.softmaxFCLayer2[np.where(self.category == 1)])
		else:
			self.loss = -1 * np.log(self.softmaxFCLayer2[np.where(self.category == 1)])
		print(self.Loss)

	def Softmax(self, array):

		temp = array - np.max(array)
		exponents = np.exp(temp)
		sumOfExponents = np.sum(exponents)
		exponents = exponents / sumOfExponents

		return exponents


	def Convolution2D(self, fltr, layer, stride, pad):

		outputHeight = int(((layer.shape[0] - fltr.shape[0] ) / stride) + 1) 
		outputWidth = int(((layer.shape[1] - fltr.shape[1] )/ stride) + 1)

		convolutedImage = np.zeros((outputHeight, outputWidth))
		widthIndex = 0
		heightIndex = 0

		for i in range(0, int(layer.shape[0] - fltr.shape[0]), stride):
			widthIndex = 0
			for j in range(0, int(layer.shape[1] - fltr.shape[1]), stride):

				convolutedImage[heightIndex, widthIndex] = np.sum(fltr * layer[i:(i + fltr.shape[0]) , j: (j + fltr.shape[1])])
				widthIndex = widthIndex + 1
			
			heightIndex = heightIndex + 1
		return convolutedImage

	def Convolution3D(self, originalImage, fltr, pad, stride):

		# new image after padding each of the layers
		image = np.zeros((originalImage.shape[0] + 2 * pad, originalImage.shape[1] + 2 * pad, originalImage.shape[2]))

		for i in range(originalImage.shape[2]):
			# padding each layer and assigning to the new image
			image[:,:,i] = np.pad(originalImage[:,:,i], (pad,pad), mode='constant', constant_values = 0)

		outputHeight = self.FindSize(originalImage.shape[0], pad, fltr.shape[1], stride) 
		outputWidth = self.FindSize(originalImage.shape[1], pad, fltr.shape[2], stride)
		outputDepth = fltr.shape[0]

		convolutedVolume = np.zeros((outputHeight, outputWidth, outputDepth))

		for x in range(fltr.shape[0]):

			convolutedImage = np.zeros((outputHeight, outputWidth))

			for y in range(fltr.shape[3]):

				convolutedImage += self.Convolution2D(fltr[x, :, :, y], image[:,:,y], stride, pad)

			convolutedVolume[:,:,x] = convolutedImage

		return convolutedVolume

	def MaxPool(self, convolutedVolume, stride):	# We don't generally use padding while pooling

		outputHeight = int(((convolutedVolume.shape[0] - self.maxPoolFilterSize)/ stride) + 1)
		outputWidth = int(((convolutedVolume.shape[1] - self.maxPoolFilterSize)/ stride) + 1)
		outputDepth = convolutedVolume.shape[2]

		maxPooledVolume = np.zeros((outputHeight, outputWidth, outputDepth))
		maxPool2D = np.zeros((outputHeight, outputWidth))

		for x in range(convolutedVolume.shape[2]):	# for each layer of the total volume
			i = 0
			for y in range(0, convolutedVolume.shape[0] - self.maxPoolFilterSize, stride):
				j = 0
				for z in range(0, convolutedVolume.shape[1] - self.maxPoolFilterSize, stride):
					sliceOfVolume = convolutedVolume[:,:,x]
					maxPool2D[i, j] = np.max(sliceOfVolume[y:y+ self.maxPoolFilterSize, z:z + self.maxPoolFilterSize])
					j = j + 1

				i = i + 1

			maxPooledVolume[:,:,x] = maxPool2D

		return maxPooledVolume

	def ReluActivation(self, volumeData):
		# r = np.zeros_like(volumeData) # making the same dimension tensor as the provided volume

		activation = np.where(volumeData > 0, volumeData, 0)
		return activation 

	def LimitingPixels(self, volume):

		temp = np.where(volume>255, 255, volume)
		return temp

	def ShowLayer3D(self, volume, indexOfSlice):
		plt.imshow(volume[:, :, indexOfSlice])
		plt.show()

	def ShowLayer4D(Self, volume, indexOf3DVolume, indexOfSlice):
		plt.imshow(volume[indexOf3DVolume, :, :, indexOfSlice])
		plt.show()

testImage = Image.open("test_image.png")
img = np.array(testImage)
classify = np.array([1,0])

cnn = CNN(0.01, img, classify)
cnn.ForwardPass()
cnn.ShowLayer3D(cnn.convolutionLayer1, 2)


