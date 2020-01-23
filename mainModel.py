import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import png

class CNN:
	def __init__(self, learningRate):
		self.learningRate = learningRate
		self.maxPoolFilterSize = 2

	def Convolution2D(self, fltr, layer, stride, pad):

		outputHeight = int(((layer.shape[0] - fltr.shape[0] + 2 * pad) / stride) + 1) 
		outputWidth = int(((layer.shape[1] - fltr.shape[1] + 2 * pad)/ stride) + 1)

		convolutedImage = np.zeros((outputHeight, outputWidth))

		for i in range(int(layer.shape[0] - 2 * pad)):
			for j in range(0, int(layer.shape[1] - 2 * pad), stride):

				convolutedImage[i,j] = np.sum(fltr * layer[i:(i + fltr.shape[0]) , j: (j + fltr.shape[1])])

		return convolutedImage

	def Convolution3D(self, originalImage, fltr, pad, stride):

		# new image after padding each of the layers
		image = np.zeros((originalImage.shape[0] + 2 * pad, originalImage.shape[1] + 2 * pad, originalImage.shape[2]))

		for i in range(originalImage.shape[2]):
			# padding each layer and assigning to the new image
			image[:,:,i] = np.pad(originalImage[:,:,i], (pad,pad), mode='constant', constant_values = 0)

		outputHeight = int(((originalImage.shape[0] - fltr.shape[1] + 2 * pad) / stride) + 1) 
		outputWidth = int(((originalImage.shape[1] - fltr.shape[2] + 2 * pad)/ stride) + 1)
		outputDepth = fltr.shape[0]

		convolutedVolume = np.zeros((outputHeight, outputWidth, outputDepth))

		for x in range(fltr.shape[0]):

			convolutedImage = np.zeros((outputHeight, outputWidth))

			for y in range(fltr.shape[3]):

				convolutedImage += self.Convolution2D(fltr[x, :, :, y], originalImage[:,:,y], stride, pad)

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

	def reluActivation(self, volumeData):
		r = np.zeros_like(volumeData) # making the same dimension tensor as the provided volume

		activation = np.where(volumeData > 0, volumeData, 0)
		return activation 



cnn = CNN(0.01)

testImage = Image.open("test_image.jpg")
img = np.array(testImage)

filters = np.random.randn(3,3,3,3)

convoultion = cnn.Convolution3D(img, filters, 1, 1)
print(convoultion.shape)
plt.imshow(convoultion)
plt.show()

activate = cnn.reluActivation(convoultion)
print(activate.shape)
plot2 = plt.imshow(activate)
plt.show()

maxpool = cnn.MaxPool(activate, 2)
print(maxpool.shape)
plot = plt.imshow(maxpool)
plt.show()





