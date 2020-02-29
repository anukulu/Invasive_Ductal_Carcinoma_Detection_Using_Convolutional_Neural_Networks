import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import png
import time

# This is a CNN model (only batch normalization is yet to be implemented)

class CNN:
	def __init__(self, learningRate, regularizationFactor):
		self.learningRate = learningRate
		self.regularizationFactor = regularizationFactor
		self.maxPoolFilterSize = 2
		self.maxPoolStride = 2
		self.imageSize = 50
		self.epsilon = 0.01	# threshold value added so that the result of the distribution is greater than zero

		# declarations for layer 1
		self.numberOfFiltersLayer1 = 32
		self.filterSizeLayer1 = 3
		self.filterDepthLayer1 = 3
		self.filtersLayer1 = np.random.randn(self.numberOfFiltersLayer1,self.filterSizeLayer1,self.filterSizeLayer1,self.filterDepthLayer1)
		self.convoultionSizeLayer1 = self.FindSize(self.imageSize,1,self.filterSizeLayer1,2)
		self.convolutionLayer1 = np.zeros((self.convoultionSizeLayer1,self.convoultionSizeLayer1,self.numberOfFiltersLayer1))
		self.biasesLayer1 = np.random.randn(*(self.convolutionLayer1.shape))

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
		self.biasesLayer2 = np.random.randn(*(self.convolutionLayer2.shape))

		self.reluLayer2 = np.zeros(self.convolutionLayer2.shape)
		self.maxpooledSizeLayer2 = self.FindSize(self.convolutionLayer2.shape[0],0,self.maxPoolFilterSize,self.maxPoolStride)
		self.maxpooledLayer2 = np.zeros((self.maxpooledSizeLayer2,self.maxpooledSizeLayer2,self.numberOfFiltersLayer2))

		# declarations for fully connected layer 1
		self.heightOfFlattenedLayer = np.prod(list(self.maxpooledLayer2.shape))
		self.weightsFCLayer1 = np.random.randn(128, self.heightOfFlattenedLayer)
		self.fullyconnectedLayer1 = np.zeros((128,1))
		self.biasesFCLayer1 = np.random.randn(*(self.fullyconnectedLayer1.shape))

		self.activationFCLayer1 = np.zeros(self.fullyconnectedLayer1.shape)

		# declarations for fully connected layer 2
		self.weightsFCLayer2 = np.random.randn(2,128)
		self.fullyconnectedLayer2 = np.zeros((2,1))
		self.biasesFCLayer2 = np.random.randn(*(self.fullyconnectedLayer2.shape))
		self.softmaxFCLayer2 = np.zeros(self.fullyconnectedLayer2.shape)

		# Loss
		self.Loss = 0

	def SetImage(self, image, category):
		self.image = image
		self.category = category

	def FindSize(self, n, p, f, s):
		return (int(((n+ (2 * p)- f)/s) + 1))

	def ForwardPass(self):

		# for layer 1
		self.convolutionLayer1 = self.LimitingPixels(np.add(self.Convolution3D(self.image, self.filtersLayer1, 1, 2), self.biasesLayer1))
		self.reluLayer1 = self.LimitingPixels(self.ReluActivation(self.convolutionLayer1))
		self.maxpooledLayer1 = self.MaxPool(self.reluLayer1, self.maxPoolStride)


		# for layer 2
		self.convolutionLayer2 = self.LimitingPixels(np.add(self.Convolution3D(self.maxpooledLayer1, self.filtersLayer2, 1, 1) , self.biasesLayer2))
		self.reluLayer2 = self.LimitingPixels(self.ReluActivation(self.convolutionLayer2))
		self.maxpooledLayer2 = self.MaxPool(self.reluLayer2, self.maxPoolStride)

		# for flattening the last convoluted volume
		self.flattenedLayer = self.maxpooledLayer2.flatten()
		# print(self.flattenedLayer.shape)
		self.flattenedLayer = np.reshape(self.flattenedLayer, (self.flattenedLayer.shape[0], 1))
		
		# ok upto here

		# for fully connected layer 1 
		z = np.dot(self.weightsFCLayer1, self.flattenedLayer)
		o = np.reshape(z, (z.shape[0],1))
		self.fullyconnectedLayer1 = o + self.biasesFCLayer1
		self.activationFCLayer1 = self.ReluActivation(self.fullyconnectedLayer1)

		# for fully connected layer 2 (also apply softmax in this layer)
		self.fullyconnectedLayer2 = np.add(np.dot(self.weightsFCLayer2, self.activationFCLayer1)  , self.biasesFCLayer2)
		# print(self.fullyconnectedLayer2)
		self.softmaxFCLayer2 = self.Softmax(self.fullyconnectedLayer2)
		# print(self.softmaxFCLayer2)

		# summing all the square of the weights for L2 regularization

		squareSumWeightsLayer1 = np.sum(self.Square(self.filtersLayer1))
		squareSumWeightsLayer2 = np.sum(self.Square(self.filtersLayer2))
		squareSumWeightsFCLayer1 = np.sum(self.Square(self.weightsFCLayer1))
		squareSumWeightsFCLayer2 = np.sum(self.Square(self.weightsFCLayer2))

		L2Regularization = squareSumWeightsLayer1 + squareSumWeightsLayer2 + squareSumWeightsFCLayer1 + squareSumWeightsFCLayer2
		# print(L2Regularization)

		# evaluating the loss for the model
		if (self.softmaxFCLayer2[np.where(self.category == 1)][0][0] == 0):
			self.Loss = 0.5 * (-1 * np.log(self.epsilon + self.softmaxFCLayer2[np.where(self.category == 1)][0][0]) + (self.regularizationFactor * L2Regularization) )
		else:
			self.Loss = 0.5 * (-1 * np.log(self.softmaxFCLayer2[np.where(self.category == 1)][0][0]) + (self.regularizationFactor * L2Regularization) )
		print(self.Loss)

		return self.Loss

	def BackwardPass(self):
		
		# defining the gradients for each layer starting from the end

		# gradient for output layer
		threshold = 0.005
		tempOutputG = np.where(self.softmaxFCLayer2 > threshold, self.softmaxFCLayer2, threshold)
		tempOutputG = -1 / tempOutputG
		self.outputG = np.multiply(tempOutputG, np.transpose(np.array([self.category])))
		# print(self.outputG)

		# gradient for weights connected to  output layer
		self.weightsFCLayer2G = np.zeros(self.weightsFCLayer2.shape)
		summMatrix = []
		for i in range(self.weightsFCLayer2.shape[0]):
			summ = 0
			for j in range(self.outputG.shape[0]):
				dij = 1 if i == j else 0
				value = self.softmaxFCLayer2[j][0] * (dij - self.softmaxFCLayer2[i][0]) * self.outputG[j][0]
				# print(value)
				summ = summ + value
			summMatrix.append(summ)
		for x in range(self.weightsFCLayer2G.shape[0]):
			self.weightsFCLayer2G[x,:] = summMatrix[x] * np.transpose(self.activationFCLayer1)
		
		# gradient for x in fully connected layer 1
		self.activationFCLayer1G = np.zeros(self.activationFCLayer1.shape)
		summMatrix = []
		for i in range(self.activationFCLayer1G.shape[0]):
			summ = 0
			for j in range(self.outputG.shape[0]):
				product = np.sum([self.softmaxFCLayer2[k][0] * self.weightsFCLayer2[k][i] for k in range(self.softmaxFCLayer2.shape[0])])
				value = self.softmaxFCLayer2[j][0] * (self.weightsFCLayer2[j][i] - product) * self.outputG[j][0]
				summ = summ + value
			self.activationFCLayer1G[i, :] = [summ]
		self.activationFCLayer1G = np.where(self.activationFCLayer1 > 0, self.activationFCLayer1G, 0)

		# gradient for weights in first fully connected layer
		self.weightsFCLayer1G = np.zeros(self.weightsFCLayer1.shape)
		for i in range(self.weightsFCLayer1.shape[0]):
			self.weightsFCLayer1G[i, :] = self.activationFCLayer1G[i][0] * np.transpose(self.flattenedLayer)

		# gradient for flattened layer 
		self.flattenedLayerG = np.zeros(self.flattenedLayer.shape)
		for i in range(self.flattenedLayerG.shape[0]):
			self.flattenedLayerG[i, :] = [np.sum(np.array([self.weightsFCLayer1[k][i] * self.activationFCLayer1G[k][0] for k in range(self.weightsFCLayer1.shape[0])]))]

		# gradient for second maxpooled volume
		self.maxpooledLayer2G = np.zeros(self.maxpooledLayer2.shape)
		self.maxpooledLayer2G = np.reshape(self.flattenedLayerG, self.maxpooledLayer2.shape)
		
		# gradient for second convoluted volume
		self.reluLayer2G = self.InverseMaxpool(self.maxpooledLayer2G, self.reluLayer2, 2)
		self.reluLayer2G = np.where(self.reluLayer2 > 0, self.reluLayer2G, 0)

		# gradient for second layer filter
		self.filtersLayer2G = np.zeros(self.filtersLayer2.shape)
		for x in range(self.filtersLayer2.shape[0]):
			for y in range(self.maxpooledLayer1.shape[2]):
				paddedInput = np.pad(self.maxpooledLayer1[:,:,y], (1,1), mode='constant', constant_values = 0)
				self.filtersLayer2G[x, :, :, y] = self.Convolution2D(self.reluLayer2G[:,:,x], paddedInput, 1, 1)

		# gradient for first maxpooled volume
		sizeOfGradient = (self.maxpooledLayer1.shape[0] + 2 , self.maxpooledLayer1.shape[1] + 2, self.maxpooledLayer1.shape[2])
		self.maxpooledLayer1G = np.zeros(sizeOfGradient)
		tempFilters = self.FlipVolume(self.filtersLayer2)
		padSize = self.filtersLayer2.shape[1] - 1
		paddedGradient = np.pad(self.reluLayer2G, (padSize, padSize), mode='constant', constant_values = 0)
		for x in range(self.filtersLayer2.shape[0]):
			for y in range(self.maxpooledLayer1.shape[2]):
				self.maxpooledLayer1G[:,:,y] = self.Convolution2D(tempFilters[x, :,:, y], paddedGradient[:,:,x], 1, 1)
		self.maxpooledLayer1G = self.RemovePadding(self.maxpooledLayer1G)

		# gradient for first convoluted volume
		self.reluLayer1G = self.InverseMaxpool(self.maxpooledLayer1G, self.reluLayer1, 2)
		self.reluLayer1G = np.where(self.reluLayer1 > 0, self.reluLayer1G, 0)
		self.tempReluLayer1G = self.reluLayer1G
		self.reluLayer1G = self.Dilation(self.reluLayer1G, 2)
		# print(self.reluLayer1G.shape)

		# gradient for first layer filter
		self.filtersLayer1G = np.zeros(self.filtersLayer1.shape)
		for x in range(self.filtersLayer1.shape[0]):
			for y in range(self.image.shape[2]):
				paddedInput = np.pad(self.image[:,:,y], (1,1), mode='constant', constant_values = 0)
				self.filtersLayer1G[x, :, :, y] = self.Convolution2D(self.reluLayer1G[:,:,x], paddedInput, 1, 1)
		# print(self.filtersLayer1G)

		# gradients of the biases

		# gradient for bias of final output layer
		self.biasesFCLayer2G = np.zeros(self.biasesFCLayer2.shape)
		constantForBias = np.sum(np.exp(self.fullyconnectedLayer2 - 0.999 * self.fullyconnectedLayer2))
		for  i in range(self.biasesFCLayer2.shape[0]):
			temp = 0
			for j in range(self.softmaxFCLayer2.shape[0]):
				dij = 1 if i == j else 0
				temp = ((dij - self.softmaxFCLayer2[j][0]) / constantForBias) * (self.outputG[j][0])
			self.biasesFCLayer2G[i][0] = temp

		# gradient for bias of FC layer 1
		self.biasesFCLayer1G = np.zeros(self.biasesFCLayer1.shape)
		self.biasesFCLayer1G = self.activationFCLayer1G

		#gradient for convolution layer 2 bias
		self.biasesLayer2G = np.zeros(self.biasesLayer2.shape)
		self.biasesLayer2G = self.reluLayer2G

		#gradient for convolution layer 1 bias
		self.biasesLayer1G = np.zeros(self.biasesLayer1.shape)
		self.biasesLayer1G = self.tempReluLayer1G

	def GradientUpdate(self):

		# filters and biases layer 1
		self.filtersLayer1 = np.subtract(self.filtersLayer1, self.learningRate * np.add(self.filtersLayer1G, self.filtersLayer1))
		self.biasesLayer1 = np.subtract(self.biasesLayer1, self.learningRate * self.biasesLayer1G)

		# filters and biases layer 2
		self.filtersLayer2 = np.subtract(self.filtersLayer2, self.learningRate * np.add(self.filtersLayer2G, self.filtersLayer2))
		self.biasesLayer2 = np.subtract(self.biasesLayer2, self.learningRate * self.biasesLayer2G)

		# weights and biases FC layer 1
		self.weightsFCLayer1 = np.subtract(self.weightsFCLayer1, self.learningRate * np.add(self.weightsFCLayer1G, self.weightsFCLayer1))
		self.biasesFCLayer1 = np.subtract(self.biasesFCLayer1, self.learningRate * self.biasesFCLayer1G)

		# weights and biases FC layer 2
		self.weightsFCLayer2 = np.subtract(self.weightsFCLayer2, self.learningRate * np.add(self.weightsFCLayer2G, self.weightsFCLayer2))
		self.biasesFCLayer2 = np.subtract(self.biasesFCLayer2, self.learningRate * self.biasesFCLayer2G)

	def Dilation(self,volume, dilation):
		sizeAfterDilation = volume.shape[0] + ((volume.shape[0] - 1) * (dilation - 1))
		# print(sizeAfterDilation)
		tempVolume = np.zeros((sizeAfterDilation, sizeAfterDilation, volume.shape[2]))

		for x in range(volume.shape[2]):
			a = volume[:,:,x]
			# print(a.shape)
			column = np.zeros((1, a.shape[0]))
			for i in range(sizeAfterDilation):
				if (i % 2 != 0):
					a = np.hstack((a[:,:i], np.transpose(column), a[:,i:]))	
			a = np.transpose(a)
			# print(a.shape)
			row = np.zeros((1, a.shape[0]))
			for i in range(sizeAfterDilation):
				if(i % 2 != 0):
					a = np.hstack((a[:,:i], np.transpose(row), a[:,i:]))
			a = np.transpose(a)
			# print(a.shape)
			tempVolume[:,:,x] = a
		return tempVolume

	def RemovePadding(self, volume):
		tempVolume1 = np.zeros((volume.shape[0]-2, volume.shape[1], volume.shape[2]))
		tempVolume2 = np.zeros((volume.shape[0]-2, volume.shape[1]-2, volume.shape[2]))
		for i in range(volume.shape[2]):
			tempVolume1[:,:,i] = np.delete(volume[:,:,i], [0, volume.shape[0]-1], axis=0)
			tempVolume2[:,:,i] = np.delete(tempVolume1[:,:,i], [0, tempVolume1.shape[1]-1], axis=1)
		return tempVolume2

	def FlipVolume(self, volume):
		tempvolume = np.zeros(volume.shape)
		for x in range(volume.shape[0]):
			for y in range(volume.shape[3]):
				tempvolume[x,:,:,y] = np.flip(volume[x,:,:,y], (0,1))
		return tempvolume

	def Square(self, volume):
		temp = np.multiply(volume, volume)
		return temp

	def Softmax(self, array):

		constant = 0.9999 * array
		temp = array - constant
		# print(temp)
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
			for y in range(0, convolutedVolume.shape[0] - self.maxPoolFilterSize + 1, stride):
				j = 0
				for z in range(0, convolutedVolume.shape[1] - self.maxPoolFilterSize + 1, stride):
					sliceOfVolume = convolutedVolume[:,:,x]
					maxPool2D[i, j] = np.max(sliceOfVolume[y:y+ self.maxPoolFilterSize, z:z + self.maxPoolFilterSize])
					j = j + 1

				i = i + 1

			maxPooledVolume[:,:,x] = maxPool2D

		return maxPooledVolume

	def InverseMaxpool(self, gradientVolume, convolutedVolume, stride):
		gradientForConvolutedVolume = np.zeros(convolutedVolume.shape)

		for x in range(convolutedVolume.shape[2]):	# for each layer of the total volume
			i = 0
			for y in range(0, convolutedVolume.shape[0] - self.maxPoolFilterSize + 1, stride):
				j = 0
				for z in range(0, convolutedVolume.shape[1] - self.maxPoolFilterSize + 1, stride):
					sliceOfVolume = convolutedVolume[:,:,x]
					smallSlice = sliceOfVolume[y:y+ self.maxPoolFilterSize, z:z + self.maxPoolFilterSize]
					relativePositionOfMax = np.where(smallSlice == np.max(smallSlice))
					realPosition = np.array([relativePositionOfMax[0][0] + y, relativePositionOfMax[1][0] + z])
					gradientForConvolutedVolume[realPosition[0], realPosition[1], x] = gradientVolume[i,j,x]
					j = j + 1

				i = i + 1
		return gradientForConvolutedVolume

	def ReluActivation(self, volumeData):
		# r = np.zeros_like(volumeData) # making the same dimension tensor as the provided volume

		activation = np.where(volumeData > 0, volumeData, 0)
		return activation 

	def LimitingPixels(self, volume):

		temp = np.where(volume>255, 255, volume)
		return temp

	def ShowLayer3D(self, volume, indexOfSlice, typeOfShow):

		if(typeOfShow == 1):
			plt.imshow(volume[:, :,indexOfSlice])
			plt.show()
		else:
			print(volume[:,:,indexOfSlice])

	def ShowLayer4D(Self, volume, indexOf3DVolume, indexOfSlice, typeOfShow):

		if(typeOfShow == 1):
			plt.imshow(volume[indexOf3DVolume, :, :, indexOfSlice])
			plt.show()
		else:
			print(volume[indexOf3DVolume, :, :, indexOfSlice])


start = time.time()

testImage = Image.open("test_image.png")
testImg = np.array(testImage) / 255.0
img = np.zeros((testImg.shape[0]-1, testImg.shape[1]-1, testImg.shape[2]))
classify = np.array([0,1])

# creating a 49 * 49 * 3 image to feedin CNN
for x in range(img.shape[2]):
	temp = np.delete(testImg[:,:,x], 0, axis=0)
	img[:,:,x] = np.delete(temp, 0, axis=1) 
print(img.shape)

# classify is a dummy one hot encoded vector for the different classes of the output and only used for testing (temporary)
cnn = CNN(0.005, 0.00001) # learning rate and regularization factor 
cnn.SetImage(img, classify) 
cnn.ForwardPass()
# cnn.ShowLayer4D(cnn.filtersLayer1, 1, 1, 1)
# cnn.ShowLayer3D(cnn.convolutionLayer1, 25, 1)
cnn .BackwardPass()
cnn.GradientUpdate()

end = time.time()
difference = end - start
print(difference)

